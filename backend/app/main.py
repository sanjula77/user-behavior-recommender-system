from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from bson import ObjectId
from app.api.events import router as event_router

# Monkey patch jsonable_encoder to handle ObjectId
_original_jsonable_encoder = jsonable_encoder

def _jsonable_encoder(obj, *args, **kwargs):
    if isinstance(obj, ObjectId):
        return str(obj)
    return _original_jsonable_encoder(obj, *args, **kwargs)

# Replace the function in the module
import fastapi.encoders
fastapi.encoders.jsonable_encoder = _jsonable_encoder

app = FastAPI(title="Web Behavior Insight Engine - Backend")

app.include_router(event_router)

@app.get("/")
def health_check():
    return {"status": "running"}
