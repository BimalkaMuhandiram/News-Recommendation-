from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean, Float, func #To define columns in database tables and perform operations such as querying or calculating
from sqlalchemy.orm import relationship # To define relationships between models (tables) in SQLAlchemy
from database import Base # For defining models, allowing SQLAlchemy to map them to database tables

# Define 'users' table model
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    created_by_id = Column(Integer, ForeignKey('users.id'))
    updated_by_id = Column(Integer, ForeignKey('users.id'))
    firebase_access_token = Column(String)
    user_email = Column(String, unique=True, index=True)
    user_name = Column(String)
    user_image_url = Column(String)
    type = Column(String)
    app_version = Column(String)

    created_by = relationship("User", foreign_keys=[created_by_id])
    updated_by = relationship("User", foreign_keys=[updated_by_id])

# Define 'news' table model
class News(Base):
    __tablename__ = 'news'

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(String)
    published_date = Column(DateTime)
    breaking_news = Column(Boolean)
    blob_image = Column(String)
    source_url = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    published_at = Column(DateTime)
    created_by_id = Column(Integer, ForeignKey('users.id'))
    updated_by_id = Column(Integer, ForeignKey('users.id'))
    shares = Column(Integer)
    comment_count = Column(Integer)
    type = Column(String)

    created_by = relationship("User", foreign_keys=[created_by_id])
    updated_by = relationship("User", foreign_keys=[updated_by_id])

# Define 'news_preprocessed' table model
class NewsPreprocessed(Base):
    __tablename__ = 'news_preprocessed'

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    published_date = Column(DateTime, nullable=False)
    breaking_news = Column(Boolean, default=False)
    source_url = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    published_at = Column(DateTime, nullable=False)
    combined = Column(String, nullable=False)

# Define 'rec_users' table model
class RecUser(Base):
    __tablename__ = 'rec_users'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    labels = Column(String)
    subscribe = Column(Boolean)
    comment = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    created_by_id = Column(Integer, ForeignKey('users.id'))
    updated_by_id = Column(Integer, ForeignKey('users.id'))

    user = relationship("User", foreign_keys=[user_id])
    created_by = relationship("User", foreign_keys=[created_by_id])
    updated_by = relationship("User", foreign_keys=[updated_by_id])

# Define 'rec_items' table model
class RecItems(Base):
    __tablename__ = "rec_items"

    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, nullable=False)
    is_hidden = Column(Integer, default=0)
    categories = Column(String)
    time_stamp = Column(String)
    labels = Column(String)
    comment = Column(String)
    created_at = Column(String)
    updated_at = Column(String)
    created_by_id = Column(Integer, ForeignKey("users.id"))
    updated_by_id = Column(Integer, ForeignKey("users.id"))

    created_by = relationship("User", foreign_keys=[created_by_id])
    updated_by = relationship("User", foreign_keys=[updated_by_id])

# Define 'rec_feedbacks' table model
class RecFeedback(Base):
    __tablename__ = 'rec_feedback'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True) 
    feedback_type = Column(String, nullable=False)
    time_stamp = Column(DateTime, default=func.now()) 
    comment = Column(String, nullable=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    item_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    updated_by_id = Column(Integer, ForeignKey('users.id'), nullable=True)

    user = relationship("User", foreign_keys=[user_id])
    created_by = relationship("User", foreign_keys=[created_by_id])
    updated_by = relationship("User", foreign_keys=[updated_by_id])