# Load events from MongoDB into DataFrame
# ml/pipeline/extractor.py
import pandas as pd
from .db_client import get_events_collection
from typing import Optional
from datetime import datetime

def load_events(limit: Optional[int] = None, query: dict = None) -> pd.DataFrame:
    """
    Load events from MongoDB collection into a pandas DataFrame.
    :param limit: optional limit for development
    :param query: Mongo query dict (e.g. {"event_type": "page_view"})
    """
    coll = get_events_collection()
    cursor = coll.find(query or {}).sort("timestamp", 1)
    if limit:
        cursor = cursor.limit(limit)

    docs = list(cursor)
    if not docs:
        return pd.DataFrame(columns=[
            "event_id","user_id","session_id","event_type","page_url","timestamp","metadata","_id"
        ])

    df = pd.DataFrame(docs)
    # normalize timestamp columns
    # some events may have timestamp strings; convert robustly
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        df["timestamp"] = pd.NaT

    # ensure some columns exist
    for col in ["user_id","session_id","event_type","page_url","metadata","event_id"]:
        if col not in df.columns:
            df[col] = None

    return df
