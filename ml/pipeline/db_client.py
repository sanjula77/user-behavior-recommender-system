# Connect to MongoDB using backend .env
# ml/pipeline/db_client.py
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load .env from backend directory or project root
backend_env = os.path.join(os.path.dirname(__file__), '..', '..', 'backend', '.env')
root_env = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
if os.path.exists(backend_env):
    load_dotenv(backend_env)
elif os.path.exists(root_env):
    load_dotenv(root_env)
else:
    load_dotenv()  # Try current directory as fallback

MONGO_URI = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI")
MONGO_DB = os.getenv("MONGO_DB") or os.getenv("MONGO_DB_NAME") or os.getenv("MONGO_DATABASE") or "web_behavior_insight"

if not MONGO_URI:
    raise RuntimeError("MONGO_URI/MONGODB_URI not found in environment. Set it in backend/.env")

_client = None

def get_db():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client[MONGO_DB]

def get_events_collection():
    db = get_db()
    return db["events"]
