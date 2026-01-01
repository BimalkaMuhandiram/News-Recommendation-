# retrain_models.py
import asyncio
from app.database import AsyncSessionLocal
from app.train import retrain_models 

async def retrain():
    # Use async session
    async with AsyncSessionLocal() as db:
        await retrain_models(db)

if __name__ == "__main__":
    # Run the retrain function as an async task
    asyncio.run(retrain())