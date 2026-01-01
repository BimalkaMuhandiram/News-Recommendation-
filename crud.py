from sqlalchemy.orm import Session # To interact with the database in a transactional manner
from models import User, News, RecUser, RecItem, RecFeedback, NewsPreprocessed # For querying and interacting with data in the database
from datetime import datetime # To handle date and time-related operations
from typing import Optional # To indicate that a variable or field can be of a specific type or 'None'

# CRUD operations for 'User' model
def create_user(db: Session, user_email: str, user_name: str, app_version: str, device_id: Optional[str] = None):
    db_user = User(
        user_email=user_email,
        user_name=user_name,
        app_version=app_version,
        device_id=device_id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def update_user(db: Session, user_id: int, **kwargs):
    db_user = get_user(db, user_id)
    if db_user:
        for key, value in kwargs.items():
            setattr(db_user, key, value)
        db_user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_user)
    return db_user

def delete_user(db: Session, user_id: int):
    db_user = get_user(db, user_id)
    if db_user:
        db.delete(db_user)
        db.commit()
    return db_user

# CRUD operations for 'News' model
def create_news(db: Session, title: str, description: str, published_date: datetime, breaking_news: bool):
    db_news = News(
        title=title,
        description=description,
        published_date=published_date,
        breaking_news=breaking_news,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(db_news)
    db.commit()
    db.refresh(db_news)
    return db_news

def get_news(db: Session, news_id: int):
    return db.query(News).filter(News.id == news_id).first()

# CRUD operations for 'NewsPreprocessed' model
def create_news_preprocessed(db: Session, title: str, description: str, published_date: datetime, breaking_news: bool, source_url: str):
    combined_text = f"{title} {description}"
    
    db_news = NewsPreprocessed(
        title=title,
        description=description,
        published_date=published_date,
        breaking_news=breaking_news,
        source_url=source_url,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        published_at=datetime.utcnow(),
        combined=combined_text
    )
    db.add(db_news)
    db.commit()
    db.refresh(db_news)
    return db_news

# Get a single news_preprocessed entry by ID
def get_news_preprocessed(db: Session, news_id: int):
    return db.query(NewsPreprocessed).filter(NewsPreprocessed.id == news_id).first()

# Get all news_preprocessed entries
def get_all_news_preprocessed(db: Session, skip: int = 0, limit: int = 10):
    return db.query(NewsPreprocessed).offset(skip).limit(limit).all()

# Update a news_preprocessed entry
def update_news_preprocessed(db: Session, news_id: int, title: str = None, description: str = None, breaking_news: bool = None):
    db_news = db.query(NewsPreprocessed).filter(NewsPreprocessed.id == news_id).first()
    
    if not db_news:
        return None  # News not found

    if title:
        db_news.title = title
    if description:
        db_news.description = description
    if breaking_news is not None:
        db_news.breaking_news = breaking_news

    db_news.combined = f"{db_news.title} {db_news.description}"  # Recalculate combined field
    db_news.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(db_news)
    return db_news

# Delete a news_preprocessed entry
def delete_news_preprocessed(db: Session, news_id: int):
    db_news = db.query(NewsPreprocessed).filter(NewsPreprocessed.id == news_id).first()
    
    if not db_news:
        return None  # News not found

    db.delete(db_news)
    db.commit()
    return db_news

# CRUD operations for 'RecUser' model
def create_rec_user(db: Session, user_id: int, labels: list, subscribe: list, comment: list):
    db_rec_user = RecUser(
        user_id=user_id,
        labels=labels,
        subscribe=subscribe,
        comment=comment,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(db_rec_user)
    db.commit()
    db.refresh(db_rec_user)
    return db_rec_user

def get_rec_user(db: Session, rec_user_id: int):
    return db.query(RecUser).filter(RecUser.id == rec_user_id).first()

# CRUD operations for 'RecItem' model
def create_rec_item(db: Session, item_id: int, is_hidden: bool, categories: list, labels: list, comment: str):
    db_rec_item = RecItem(
        item_id=item_id,
        is_hidden=is_hidden,
        categories=categories,
        labels=labels,
        comment=comment,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(db_rec_item)
    db.commit()
    db.refresh(db_rec_item)
    return db_rec_item

def get_rec_item(db: Session, rec_item_id: int):
    return db.query(RecItem).filter(RecItem.id == rec_item_id).first()

# CRUD operations for 'RecFeedback' model
def create_feedback(db: Session, user_id: int, item_id: int, feedback_type: str, comment: Optional[str] = None):
    db_feedback = RecFeedback(
        user_id=user_id,
        item_id=item_id,
        feedback_type=feedback_type,
        comment=comment,
        time_stamp=datetime.utcnow(),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback

def get_feedback(db: Session, feedback_id: int):
    return db.query(RecFeedback).filter(RecFeedback.id == feedback_id).first()