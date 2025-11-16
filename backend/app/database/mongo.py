from pymongo import MongoClient
from app.core.config import settings

class MongoDB:
    _client = None

    @staticmethod
    def get_client():
        if MongoDB._client is None:
            MongoDB._client = MongoClient(settings.MONGO_URI)
        return MongoDB._client

    @staticmethod
    def get_db():
        client = MongoDB.get_client()
        return client[settings.MONGO_DB]  # New database name
