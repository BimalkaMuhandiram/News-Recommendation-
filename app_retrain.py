
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime
import pandas as pd
import joblib
import os
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV
from models import NewsPreprocessed, RecFeedback  
from database import get_db  
import random
from surprise import accuracy

# Initialize FastAPI app
#app = FastAPI()

# Directory to store models
MODEL_DIR = "new_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Global variable to store model training status
training_status = {
    "cbf_model": "Not Started",  
    "cf_model": "Not Started" 
}

# Utility Functions
def save_model(model, model_type):
    """Save model with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{MODEL_DIR}/{model_type}_{timestamp}.pkl"
    joblib.dump(model, filename)
    return filename

def load_latest_model(model_type):
    """Load the latest version of a model."""
    files = glob.glob(f"{MODEL_DIR}/{model_type}_*.pkl")
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    return joblib.load(latest_file)

# CBF Training
async def train_cbf_model(db: AsyncSession):
    try:
        # Start training, update status
        training_status["cbf_model"] = "Training Started"
        
        result = await db.execute(select(NewsPreprocessed.id, NewsPreprocessed.title, NewsPreprocessed.description, NewsPreprocessed.combined))
        rows = result.fetchall()
        df = pd.DataFrame(rows, columns=["id", "title", "description", "combined"])
        
        if df.empty:
            print("No data available for CBF training.")
            training_status["cbf_model"] = "Error: No data"
            return None, None
        
        df["combined"].fillna("", inplace=True)
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['combined'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        tfidf_path = save_model(tfidf, "tfidf_vectorizer")
        cosine_path = save_model(cosine_sim, "cosine_similarity")
        print(f"CBF Models Saved: {tfidf_path}, {cosine_path}")
        
        # After training, update the status
        training_status["cbf_model"] = "Training Finished"
        return tfidf, cosine_sim
    except Exception as e:
        print(f"Error in CBF model training: {e}")
        training_status["cbf_model"] = "Error"
        return None, None

# CF Training with Hyperparameter Tuning
async def train_cf_model(db: AsyncSession):
    try:
        training_status["cf_model"] = "Training Started"
        
        # Fetch feedback data
        result = await db.execute(select(RecFeedback.user_id, RecFeedback.item_id, RecFeedback.feedback_type, RecFeedback.time_stamp))
        rows = result.fetchall()
        df = pd.DataFrame(rows, columns=['user_id', 'item_id', 'feedback_type', 'time_stamp'])

        if df.empty:
            print("No feedback records found.")
            training_status["cf_model"] = "Error: No data"
            return None

        # Handle missing or invalid feedback types (if any)
        df = df[df['feedback_type'].notna()]
        df['weight'] = df['feedback_type'].map({'click': 1, 'read': 2, 'like': 3, "comment": 4}).fillna(0)
        
        # Ensure user_id and item_id are strings
        df['user_id'] = df['user_id'].astype(str)
        df['item_id'] = df['item_id'].astype(str)

        # Remove overly popular items (reduce popularity bias)
        item_counts = df['item_id'].value_counts()
        threshold = np.percentile(item_counts, 95)  
        df = df[df['item_id'].map(item_counts) < threshold]

        # Remove users with very few interactions 
        user_counts = df['user_id'].value_counts()
        df = df[df['user_id'].map(user_counts) >= 3] 

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['user_id', 'item_id', 'weight']], reader)

        # Use train-test split for better generalization
        trainset, testset = train_test_split(data, test_size=0.2)

        # Perform grid search on Surprise SVD
        param_grid = {
            'n_factors': [10, 20, 30, 40],  # More factors to capture diversity
            'n_epochs': [20, 30, 40, 50],  # More epochs for better model convergence
            'lr_all': [0.003, 0.005, 0.007, 0.01],  # Experiment with different learning rates
            'reg_all': [0.1, 0.2, 0.3, 0.5]  # Tuning regularization to avoid overfitting
        }
        grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
        grid_search.fit(data)

        # Print best parameters and RMSE
        best_params = grid_search.best_params['rmse']
        print(f"Best parameters extracted: {best_params}")
        print(f"Best RMSE: {grid_search.best_score['rmse']}")

        # Use the best parameters from grid search
        model = SVD(n_factors=best_params['n_factors'],
                    n_epochs=best_params['n_epochs'],
                    lr_all=best_params['lr_all'],
                    reg_all=best_params['reg_all'])

        model.fit(trainset)

        # Evaluate model on test set
        test_predictions = model.test(testset)
        test_rmse = accuracy.rmse(test_predictions)
        print(f"Test RMSE: {test_rmse}")

        # Save trained model
        svd_path = save_model(model, "svd_recommender")
        print(f"CF Model Saved: {svd_path}")

        # After training, update the status
        training_status["cf_model"] = "Training Finished"
        return model

    except Exception as e:
        print(f"Error in CF model training: {e}")
        training_status["cf_model"] = "Error"
        return None


# Endpoints
@app.post("/train-cbf-model/")
async def train_cbf_model_endpoint(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    background_tasks.add_task(train_cbf_model, db)
    return {"message": "CBF model training started in the background."}

@app.post("/train-cf-model/")
async def train_cf_model_endpoint(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    background_tasks.add_task(train_cf_model, db)
    return {"message": "CF model training started in the background."}

@app.get("/model-training-status/")
async def get_model_training_status():
    """Get the current training status for CBF and CF models."""
    return {
        "cbf_model_status": training_status["cbf_model"],
        "cf_model_status": training_status["cf_model"]
    }

# List Models
@app.get("/models/list")
async def list_models():
    """List all available saved models."""
    models = glob.glob(f"{MODEL_DIR}/*.pkl")
    return {"models": models}

# Content-Based Filtering (CBF) Model Testing
@app.get("/test-cbf-model/")
async def test_cbf_model(user_id: int, db: AsyncSession = Depends(get_db)):
    """Test the latest Content-Based Filtering (CBF) model for a specific user."""
    tfidf = load_latest_model("tfidf_vectorizer")
    cosine_sim = load_latest_model("cosine_similarity")

    if tfidf is None or cosine_sim is None:
        raise HTTPException(status_code=404, detail="CBF models not found. Train the models first.")

    # Load articles from the database
    result = await db.execute(select(NewsPreprocessed.id, NewsPreprocessed.combined))
    rows = result.fetchall()
    df = pd.DataFrame(rows, columns=["id", "combined"])

    if df.empty:
        raise HTTPException(status_code=404, detail="No articles found in the database.")
    
    # Fetch the articles that the user has interacted with (feedback data)
    result_feedback = await db.execute(select(RecFeedback.user_id, RecFeedback.item_id)
                                       .filter(RecFeedback.user_id == user_id))
    rows_feedback = result_feedback.fetchall()
    interacted_articles = [row[1] for row in rows_feedback]

    if not interacted_articles:
        raise HTTPException(status_code=404, detail="No feedback found for this user.")

    # Find the articles that are most similar to the user's interacted articles
    all_similar_articles = []
    for article_id in interacted_articles:
        # Check if the article exists in the dataframe before proceeding
        matching_article_indices = df.index[df["id"] == article_id]

        if not matching_article_indices.empty:
            article_index = matching_article_indices[0]
            sim_scores = list(enumerate(cosine_sim[article_index]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get top 5 similar articles, excluding the interacted article itself
            top_indices = [i[0] for i in sim_scores[1:6]]  # exclude the first one as it will be the article itself
            all_similar_articles.extend(df.iloc[top_indices]["id"].tolist())

    # Remove duplicates and limit to top 5 recommendations
    all_similar_articles = list(set(all_similar_articles))
    top_recommendations = all_similar_articles[:5]

    return {"user_id": int(user_id), "recommended_articles": top_recommendations}

# Collaborative Filtering (CF) Model Testing
@app.get("/test-cf-model/")
async def test_cf_model(user_id: int, db: AsyncSession = Depends(get_db)):
    model = load_latest_model("svd_recommender")
    if model is None:
        raise HTTPException(status_code=404, detail="CF model not found. Train the model first.")

    # Fetch feedback data
    result = await db.execute(select(RecFeedback.user_id, RecFeedback.item_id, RecFeedback.feedback_type))
    rows = result.fetchall()
    df = pd.DataFrame(rows, columns=['user_id', 'item_id', 'feedback_type'])

    if df.empty:
        raise HTTPException(status_code=404, detail="No feedback records found.")

    # Convert IDs to strings for Surprise compatibility
    df['user_id'] = df['user_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    user_str = str(user_id)

    if user_str not in df['user_id'].unique():
        return {"message": f"User {user_id} not found in training data."}

    # Get user's interacted items
    interacted_items = set(df[df['user_id'] == user_str]['item_id'])
    all_items = set(df['item_id'].unique())
    candidate_items = list(all_items - interacted_items)

    if not candidate_items:
        return {"message": "No new items to recommend for this user."}

    # Predict scores for all non-interacted items
    predictions = []
    for item_id in candidate_items:
        try:
            pred = model.predict(user_str, item_id)
            predictions.append((item_id, pred.est))
        except Exception:
            continue  # Skip any prediction errors

    # Remove globally popular items (optional but useful)
    top_popular_items = df['item_id'].value_counts().head(10).index.tolist()
    predictions = [p for p in predictions if p[0] not in top_popular_items]

    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Format results
    top_items = predictions[:5]
    recommended_items = [int(item_id) for item_id, _ in top_items]
    raw_preds = [(int(item_id), round(score, 3)) for item_id, score in top_items]

    print(f"Predictions for user {user_id}: {raw_preds}")

    return {
        "user_id": int(user_id),
        "recommended_items": recommended_items
    }