from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
import pandas as pd
from surprise import SVD, Dataset, Reader
import pickle
from celery import Celery
from models import RecFeedback  # Ensure the model is correctly defined in models.py

# Celery setup
celery_app = Celery(
    "tasks",
    backend="rpc://",
    broker="pyamqp://guest@localhost//"
)

# Database URL and SQLAlchemy setup
DATABASE_URL = "postgresql://postgres:abcd1234@localhost:5432/news_app"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

@celery_app.task
def train_recommendation_model():
    """Train and update the recommendation model asynchronously."""
    try:
        # Open a synchronous database session using the context manager to handle session properly
        with SessionLocal() as session:
            # Fetch user feedback data
            result = session.execute(select(RecFeedback.user_id, RecFeedback.item_id, RecFeedback.feedback_type))
            rows = result.fetchall()

            # Check if rows are empty
            if not rows:
                return "No feedback data found. Model training skipped."

            # Convert data to DataFrame
            df = pd.DataFrame(rows, columns=["user_id", "item_id", "feedback_type"])

            # Debugging: Print DataFrame to verify it looks correct
            print(f"DataFrame:\n{df}")

            # Convert categorical feedback into numerical values (adjust weights)
            feedback_mapping = {"click": 1, "read": 2, "like": 3, "comment": 4}
            df["feedback_type"] = df["feedback_type"].map(feedback_mapping).fillna(1)

            # Prepare data for training
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(df[['user_id', 'item_id', 'feedback_type']], reader)
            trainset = data.build_full_trainset()

            # Train model
            model = SVD()
            model.fit(trainset)

            # Save the trained model
            with open("recommendation_model.pkl", "wb") as f:
                pickle.dump(model, f)

            return "Model retrained successfully!"
    
    except Exception as e:
        return f"Error training model: {str(e)}"