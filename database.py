from sqlalchemy import create_engine # To create a standard (synchronous) SQLAlchemy engine
from sqlalchemy.ext.declarative import declarative_base # To create a base class for models using the declarative system
from sqlalchemy.orm import sessionmaker # # To create session classes for interacting with the database
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession # For creating engines and managing sessions in asynchronous applications
from sqlalchemy.future import select # For future-style queries (works with both synchronous and asynchronous engines)
from sqlalchemy.pool import NullPool # For disables connection pooling.

# Database URL 
DATABASE_URL = "postgresql+asyncpg://postgres:abcd1234@localhost:5432/news_app"

# Create an asynchronous engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    poolclass=NullPool,  # Disables connection pooling, suitable for async apps
)

# Create asynchronous session factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,  # Use AsyncSession for asynchronous database queries
    expire_on_commit=False,
)

# Declare a base class for your models
Base = declarative_base()

# Dependency to get the database session for FastAPI route handlers
async def get_db():
    async with AsyncSessionLocal() as db:
        try:
            yield db
        finally:
            await db.close()