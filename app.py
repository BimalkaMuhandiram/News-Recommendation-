from fastapi import FastAPI, HTTPException, Depends  # Import FastAPI framework, HTTPException for handling errors, and Depends for dependency injection
from pydantic import BaseModel  # Import Pydantic's BaseModel for defining request and response data models with validation
from sqlalchemy.orm import sessionmaker, Session  # Import SQLAlchemy's sessionmaker and Session to handle database interactions
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine  # Import AsyncSession for asynchronous database interaction and create_async_engine for async database engine creation
from sqlalchemy.future import select  # For SQLAlchemy queries in async mode
from sqlalchemy.sql import text  # For running raw SQL queries
from joblib import load  # For loading pre-trained models
from sklearn.neighbors import NearestNeighbors  # Import 'NearestNeighbors' from scikit-learn for content-based filtering
import random  # For shuffling recommendations
import pandas as pd  # For handling data in DataFrame format
from datetime import datetime  # For working with timestamps
from apscheduler.schedulers.background import BackgroundScheduler  # For scheduling tasks asynchronously
import asyncio  # For managing asynchronous tasks
from models import User, RecFeedback, RecItems, RecUser  # Import database models that represent tables in the database
from database import get_db  # For database interaction
import hashlib # To provides a way to securely hash data using a variety of cryptographic hash functions
import logging # For debugging and monitoring, and can log messages at various severity levels
from sqlalchemy.orm import aliased
from sqlalchemy import or_, desc
from models import News
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# Initialize FastAPI app
app = FastAPI()

# Database Configuration
DATABASE_URL = "postgresql+asyncpg://postgres:abcd1234@localhost:5432/news_app"
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Feedback mapping dictionary
FEEDBACK_MAPPING = {
    1: "click",
    2: "read",
    3: "like",
    4: "comment"
}

# Adjust weights based on feedback type
FEEDBACK_WEIGHTS = {
    "click": 0.05,
    "read": 0.1,
    "like": 0.2,
    "comment": 0.3
}

# Initialize geolocator once
geolocator = Nominatim(user_agent="news_location_app")

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Model and data version control
def get_file_hash(filename):
    """Calculate the MD5 hash of a file"""
    with open(filename, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# Load models and data
def load_models():
    global tfidf_vectorizer, cosine_sim_matrix, cf_model, knn, tfidf_matrix, news_df, previous_model_hash

    tfidf_vectorizer = load('models/tfidf_vectorizer.pkl')
    cosine_sim_matrix = load('models/cosine_similarity_matrix.pkl')
    cf_model = load('models/svd_recommender_model.pkl')

    # Load datasets
    news_df = pd.read_csv('data/news.csv')
    news_df_preprocessed = pd.read_csv('data/news_df_preprocessed.csv')

    if 'combined' not in news_df_preprocessed.columns:
        raise KeyError("The 'combined' column is missing in the preprocessed news DataFrame.")

    # Initialize NearestNeighbors for Content-Based Filtering (CBF)
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    tfidf_matrix = tfidf_vectorizer.transform(news_df_preprocessed['combined'])
    knn.fit(tfidf_matrix)

    # Store model hash for version control
    current_model_hash = get_file_hash('models/tfidf_vectorizer.pkl')
    if current_model_hash != previous_model_hash:
        logger.info("Models have changed. Reloading models.")
        previous_model_hash = current_model_hash
    else:
        logger.info("No change in models. Skipping reload.")

# Load models on startup
previous_model_hash = ""
load_models()

# Schedule model reloading every 10 minutes if needed
scheduler = BackgroundScheduler()
scheduler.add_job(load_models, 'interval', minutes=10)
scheduler.start()

# API Request Models
class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 5
    location: str = None 
    user_latitude: float
    user_longitude: float

class FeedbackRequest(BaseModel):
    user_id: int
    item_id: int
    feedback_type: float

class RecUserRequest(BaseModel):
    user_id: int
    labels: str
    subscribe: bool
    comment: str
    created_by_id: int
    updated_by_id: int

# Fetch RecFeedback data from database
async def fetch_rec_feedback(user_id):
    async with AsyncSessionLocal() as session:
        await session.commit()  # Ensure previous transactions are committed
        await session.flush()    # Flush session before querying new data
        
        result = await session.execute(
            text("SELECT user_id, item_id, feedback_type FROM rec_feedback WHERE user_id = :user_id"),
            {"user_id": user_id}
        )
        feedback_df = pd.DataFrame(result.fetchall(), columns=result.keys())

        if feedback_df.empty:
            return None
        
        return feedback_df
    
def filter_articles_by_location(news_df, user_location, top_n=10):
    # Ensure 'location' column exists
    if 'location' not in news_df.columns:
        news_df['location'] = None

    # Filter only if we have non-null location values
    if news_df['location'].notna().any():
        loc_filtered = news_df[news_df['location'] == user_location]
        if not loc_filtered.empty:
            return loc_filtered.head(top_n)
        else:
            print(f"No articles found for location: {user_location}")
            return pd.DataFrame()
    else:
        print("No valid location data available in articles.")
        return pd.DataFrame()
    
# Add latitude and longitude columns if they don't exist
if 'latitude' not in news_df.columns or 'longitude' not in news_df.columns:
    news_df['latitude'] = None
    news_df['longitude'] = None

# Function to geocode a location name
async def geocode_location(location_name: str):
    try:
        location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception as e:
        print(f"Geocoding error for '{location_name}': {e}")
        return None, None
    
# Call this once at app startup to populate lat/lon for existing data
@app.on_event("startup")
async def startup_event():
    await update_missing_lat_lon()

# Function to update missing lat/lon for articles
async def update_missing_lat_lon():
    missing_coords = news_df[news_df['latitude'].isnull() | news_df['longitude'].isnull()]
    for idx, row in missing_coords.iterrows():
        lat, lon = await geocode_location(row['location'])
        news_df.at[idx, 'latitude'] = lat
        news_df.at[idx, 'longitude'] = lon
        await asyncio.sleep(1)  # sleep to respect geocoding API limits

# Filter news articles by proximity (within some radius in km)
def filter_articles_by_proximity(user_lat, user_lon, radius_km=50, top_n=10):
    def distance_from_user(row):
        if pd.notnull(row['latitude']) and pd.notnull(row['longitude']):
            return geodesic((user_lat, user_lon), (row['latitude'], row['longitude'])).km
        else:
            return float('inf')  # ignore articles without coordinates

    news_df['distance_km'] = news_df.apply(distance_from_user, axis=1)
    nearby_articles = news_df[news_df['distance_km'] <= radius_km]
    nearby_articles = nearby_articles.sort_values('distance_km').head(top_n)
    return nearby_articles

# Get top-N recommendations using Collaborative Filtering (CF)
async def get_top_n_recommendations(user_id, n=5):
    rec_feedback_df = await fetch_rec_feedback(user_id)
    if rec_feedback_df is None or rec_feedback_df.empty:
        return []

    # Get all unique items & filter out seen ones
    all_items = set(news_df['id'].unique())
    seen_items = set(rec_feedback_df[rec_feedback_df['user_id'] == user_id]['item_id'].tolist())
    unseen_items = list(all_items - seen_items)

    if not unseen_items:
        # If all items are seen, return popular ones
        popular_items = news_df.nlargest(n, 'shares')['id'].tolist()
        return popular_items

    # CF Predictions on unseen items
    predictions = [(item, cf_model.predict(user_id, item).est) for item in unseen_items]
    top_n_recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

    logger.info(f"CF Unseen Recommendations for User {user_id}: {top_n_recommendations}")
    return [item for item, _ in top_n_recommendations]


# Get user recommendations using Content-Based Filtering (CBF)
async def get_user_recommendations(user_id, top_n=5, similarity_threshold=0.1):
    rec_feedback_df = await fetch_rec_feedback(user_id)
    if rec_feedback_df is None or rec_feedback_df.empty:
        return []

    user_interactions = rec_feedback_df[rec_feedback_df['user_id'] == user_id]
    if user_interactions.empty:
        return []

    seen_articles = set(user_interactions['item_id'].unique())
    article_indices = news_df[news_df['id'].isin(seen_articles)].index.tolist()

    if not article_indices:
        return []

    # Get similar articles using KNN
    distances, indices = knn.kneighbors(tfidf_matrix[article_indices], n_neighbors=top_n + 10)
    sim_scores = sorted(zip(indices.flatten(), distances.flatten()), key=lambda x: x[1])

    recommended_articles = []
    for idx, dist in sim_scores:
        article_id = news_df.iloc[idx]['id']
        if article_id not in seen_articles and dist <= (1 - similarity_threshold):
            recommended_articles.append(article_id)
        if len(recommended_articles) == top_n:
            break

    logger.info(f"CBF Unseen Recommendations for User {user_id}: {recommended_articles}")
    return recommended_articles


# Hybrid recommendation function: Switching between CF and CBF
async def switching_hybrid_recommend(user_id, top_n=5, threshold=5):
    rec_feedback_df = await fetch_rec_feedback(user_id)

    if rec_feedback_df is None or rec_feedback_df.empty:
        logger.info(f"New User {user_id}. Showing popular articles.")
        recommendations = news_df.nlargest(top_n, 'shares')['id'].tolist()
        return news_df[news_df['id'].isin(recommendations)][['id', 'title']]

    user_interactions = rec_feedback_df[rec_feedback_df['user_id'] == user_id]
    seen_articles = set(user_interactions['item_id'].tolist())

    if len(user_interactions) >= threshold:
        recommendations = await get_top_n_recommendations(user_id, top_n)
    else:
        recommendations = await get_user_recommendations(user_id, top_n)

    # Remove already seen articles
    recommendations = [item for item in recommendations if item not in seen_articles]

    if not recommendations:
        logger.info(f"CF & CBF failed. Providing popular articles.")
        recommendations = news_df.nlargest(top_n, 'shares')['id'].tolist()

    return news_df[news_df['id'].isin(recommendations)][['id', 'title']]


async def weighted_hybrid_recommend(user_id, top_n=5, base_cf_weight=0.6, base_cbf_weight=0.4):
    """Combine CF and CBF recommendations using weighted scores, adjusted by feedback."""
    
    # Fetch feedback data for the user
    rec_feedback_df = await fetch_rec_feedback(user_id)

    if rec_feedback_df is None or rec_feedback_df.empty:
        logger.info(f"No feedback found for User {user_id}. Providing default recommendations.")
        # If no feedback, fallback to top N popular articles
        recommendations = news_df.nlargest(top_n, 'shares')['id'].tolist()
        return news_df[news_df['id'].isin(recommendations)][['id', 'title']]

    # Adjust weights based on feedback data
    feedback_counts = rec_feedback_df['feedback_type'].value_counts().to_dict()
    
    # Dynamically adjust weights based on feedback types (for example)
    for feedback, count in feedback_counts.items():
        weight_adjustment = FEEDBACK_WEIGHTS.get(feedback, 0)
        if feedback in ["like", "comment"]:  
            base_cbf_weight += weight_adjustment * count  # Increase content-based weight
            base_cf_weight -= weight_adjustment * count  # Decrease collaborative filtering weight
        elif feedback in ["click", "read"]:  
            base_cf_weight += weight_adjustment * count  # Increase collaborative filtering weight
            base_cbf_weight -= weight_adjustment * count  # Decrease content-based weight

    # Normalize weights to ensure they sum to 1
    total_weight = base_cf_weight + base_cbf_weight
    base_cf_weight = max(0, min(base_cf_weight, 1))
    base_cbf_weight = 1 - base_cf_weight

    logger.info(f"Adjusted Weights for User {user_id}: CF={base_cf_weight}, CBF={base_cbf_weight}")

    # Get recommendations from both CF and CBF
    cf_recommendations = await get_top_n_recommendations(user_id, top_n * 2)  # More items for merging
    cbf_recommendations = await get_user_recommendations(user_id, top_n * 2)  # More items for merging

    # If no CF or CBF recommendations, fall back to the other
    if not cf_recommendations:
        return news_df[news_df['id'].isin(cbf_recommendations[:top_n])][['id', 'title']]
    if not cbf_recommendations:
        return news_df[news_df['id'].isin(cf_recommendations[:top_n])][['id', 'title']]

    # Merge CF and CBF recommendations by their adjusted weights
    cf_scores = {item: base_cf_weight for item in cf_recommendations}
    cbf_scores = {item: base_cbf_weight for item in cbf_recommendations}

    combined_scores = {}
    for item, score in cf_scores.items():
        combined_scores[item] = combined_scores.get(item, 0) + score
    for item, score in cbf_scores.items():
        combined_scores[item] = combined_scores.get(item, 0) + score

    # Sort and get top N recommendations
    sorted_recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = [item for item, _ in sorted_recommendations[:top_n]]

    logger.info(f"Weighted Hybrid Recommendations for User {user_id}: {top_recommendations}")
    return news_df[news_df['id'].isin(top_recommendations)][['id', 'title']]


# Function to get category-based recommendations
async def get_category_based_recommendations(categories, top_n=5, db: Session = Depends(get_db)):
    """Fetch recommendations based on the user's selected categories from the database."""
    
    if not categories:
        return []

    try:
        # Create aliases for RecItems and News table
        rec_items_alias = aliased(RecItems)
        news_alias = aliased(News) 

        # Construct query to fetch item_ids based on category labels and non-zero shares from the news table
        query = (
            select(rec_items_alias.item_id, news_alias.shares)
            .join(news_alias, news_alias.id == rec_items_alias.item_id)  # Join RecItems with News by item_id
            .where(
                or_(*[rec_items_alias.labels.like(f"%{category}%") for category in categories])
            )
            .where(news_alias.shares > 0)  # Ensure shares are greater than zero
            .order_by(desc(news_alias.shares))  # Order by descending shares
            .limit(top_n)  # Limit to top N recommendations
        )

        # Execute the query and fetch results
        result = await db.execute(query)
        articles = result.fetchall()

        # Extract the item_ids from the result
        recommendations = [article[0] for article in articles]  # Get item_ids

        return recommendations

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching category-based recommendations: {str(e)}")

# FastAPI Routes
@app.post("/switching_recommendations/")
async def switching_recommendations(request: RecommendationRequest):
    """Get recommendations using the Switching Hybrid approach."""
    recommendations = await switching_hybrid_recommend(request.user_id, request.top_n)

    if isinstance(recommendations, pd.DataFrame):
        recommendations = recommendations.to_dict(orient="records")

    return {"user_id": request.user_id, "recommended_articles": recommendations}

@app.post("/weighted_recommendations/")
async def weighted_recommendations(request: RecommendationRequest):
    """Get recommendations using the Weighted Hybrid approach with dynamic feedback weighting."""
    recommendations = await weighted_hybrid_recommend(request.user_id, request.top_n)

    if isinstance(recommendations, pd.DataFrame):
        recommendations = recommendations.to_dict(orient="records")

    return {"user_id": request.user_id, "recommended_articles": recommendations}

@app.post("/location_recommendations")
async def location_weighted_recommendations(request: RecommendationRequest):
    try:
        # Await async recommendation call
        recommended_articles = await weighted_hybrid_recommend(request.user_id)

        # Filter based on location
        loc_filtered = recommended_articles[recommended_articles['location'] == request.user_location]

        # Limit results
        loc_filtered_ids = loc_filtered.head(request.top_n)

        return {"articles": loc_filtered_ids.to_dict(orient="records")}

    except Exception as e:
        print("Error filtering by location:", e)

        # Fallback recommendation
        fallback_articles = await weighted_hybrid_recommend(request.user_id)
        fallback_ids = fallback_articles.head(request.top_n)

        return {"articles": fallback_ids.to_dict(orient="records")}
    
# API endpoint for location-based recommendations
@app.post("/location_recommendations")
async def location_recommendations(request: RecommendationRequest):
    # Filter by proximity
    filtered_articles = filter_articles_by_proximity(
        user_lat=request.user_latitude,
        user_lon=request.user_longitude,
        top_n=request.top_n
    )

    # If no articles found nearby, fallback to default top_n articles
    if filtered_articles.empty:
        filtered_articles = news_df.head(request.top_n)

    # Return article IDs or full article info as needed
    result = filtered_articles.to_dict(orient='records')
    return {"recommendations": result}

@app.get("/user_feedback/")
async def get_user_feedback(user_id: int, db: Session = Depends(get_db)):
    feedback_data = await fetch_rec_feedback(user_id)
    
    if feedback_data is None:
        raise HTTPException(status_code=404, detail="No feedback found for this user")
    
    return feedback_data.to_dict(orient="records")

# Add feedback logic
@app.post("/add_feedback/")
async def add_feedback(request: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    try:
        # Mapping feedback type
        feedback_label = FEEDBACK_MAPPING.get(int(request.feedback_type), "unknown")

        # Ensure the user exists or create a new user
        user = await db.execute(select(User).filter(User.id == request.user_id))
        user = user.scalars().first()

        if not user:
            new_user = User(id=request.user_id, user_name=f"user_{request.user_id}",
                            created_at=datetime.utcnow(), updated_at=datetime.utcnow())
            db.add(new_user)
            await db.commit()
            await db.refresh(new_user)

        # Ensure the article exists
        article = await db.execute(select(RecItems).filter(RecItems.item_id == request.item_id))
        article = article.scalars().first()

        if not article:
            raise HTTPException(status_code=404, detail="Article not found")

        # Update RecUser labels based on article labels
        article_labels = set(article.labels.split(", ")) if article.labels else set()

        rec_user = await db.execute(select(RecUser).filter(RecUser.user_id == request.user_id))
        rec_user = rec_user.scalars().first()

        if rec_user:
            existing_labels = set(rec_user.labels.split(", ")) if rec_user.labels else set()
            updated_labels = existing_labels.union(article_labels)
            rec_user.labels = ", ".join(updated_labels)
            rec_user.updated_at = datetime.utcnow()
        else:
            rec_user = RecUser(user_id=request.user_id, labels=", ".join(article_labels),
                               created_at=datetime.utcnow(), updated_at=datetime.utcnow())
            db.add(rec_user)

        # Insert new feedback entry
        feedback_entry = RecFeedback(user_id=request.user_id, item_id=request.item_id,
                                      feedback_type=feedback_label, created_at=datetime.utcnow())
        db.add(feedback_entry)

        # Commit changes to the database
        await db.commit()

        # Refresh rec_user to get the updated data after commit
        await db.refresh(rec_user)

        # Wait for database consistency before fetching updated feedback
        await asyncio.sleep(1)

        # Fetch updated feedback
        updated_feedback = await fetch_rec_feedback(request.user_id)

        if updated_feedback is None:
            raise HTTPException(status_code=500, detail="Failed to fetch updated feedback.")

        # Generate new recommendations after the feedback is added
        # This is for weighted hybrid recommendations
        new_recommendations_weighted = await weighted_hybrid_recommend(request.user_id, top_n=5)

        # Get category-based recommendations for the user
        rec_user = await db.execute(select(RecUser).filter(RecUser.user_id == request.user_id))
        rec_user = rec_user.scalars().first()

        if not rec_user:
            raise HTTPException(status_code=404, detail="User not found")

        # Get the user's preferred categories (labels)
        user_categories = rec_user.labels.split(", ") if rec_user.labels else []

        new_recommendations_category = await get_category_based_recommendations(user_categories, top_n=5, db=db)

        # If recommendations are in DataFrame format, convert to dict
        if isinstance(new_recommendations_weighted, pd.DataFrame):
            new_recommendations_weighted = new_recommendations_weighted.to_dict(orient="records")

        # Return response with feedback info and both types of recommendations
        return {
            "message": "Feedback added successfully",
            "new_labels": rec_user.labels,
            "new_recommendations_weighted": new_recommendations_weighted,
            "new_recommendations_category": new_recommendations_category
        }

    except Exception as e:
        await db.rollback()  # Rollback any changes if error occurs
        raise HTTPException(status_code=500, detail=f"Error adding feedback: {str(e)}")