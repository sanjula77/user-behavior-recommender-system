import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MONGO_URI: str = os.getenv("MONGO_URI")
    MONGO_DB: str = os.getenv("MONGO_DB")

settings = Settings()
