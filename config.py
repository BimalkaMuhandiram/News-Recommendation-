# To interact with the OS, such as accessing environment variables
import os 

class Settings:
    # Database settings
    DATABASE_URL = os.getenv("postgresql+asyncpg://postgres:abcd1234@localhost:5432/news_app")

# Create an instance of the Settings class and assign it to the 'settings' variable
settings = Settings()