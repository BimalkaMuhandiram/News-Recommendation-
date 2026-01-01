from pydantic import BaseModel # For data validation and serialization/deserialization of data
from datetime import datetime # To handle dates and times
from typing import List, Optional  # For type hints

# Pydantic model for 'RecommendationRequest' (Input for recommendations endpoint)
class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 5

# Pydantic model for 'FeedbackRequest' (Input for feedback endpoint)
class FeedbackRequest(BaseModel):
    user_id: int
    item_id: int
    feedback_type: str

# Pydantic model for 'News' (Response model)
class NewsResponse(BaseModel):
    id: int
    title: str
    description: str
    published_date: datetime
    breaking_news: bool
    source_url: str
    shares: int
    comment_count: int
    type: str

    class Config:
        orm_mode = True  # Allows to serialize SQLAlchemy models to Pydantic models

# Schema for Recommendation Response
class RecommendationResponse(BaseModel):
    user_id: int
    recommended_articles: List[NewsResponse]