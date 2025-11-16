# Cleaning & normalization functions
# ml/pipeline/preprocess.py
import pandas as pd
from urllib.parse import urlparse

def normalize_url(url: str) -> str:
    if not url or pd.isna(url):
        return ""
    try:
        parsed = urlparse(url)
        # keep path + query, drop scheme+host for normalization
        path = parsed.path or "/"
        q = f"?{parsed.query}" if parsed.query else ""
        return (path + q).lower()
    except Exception:
        return str(url).lower()

def fill_missing_user_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure user_id and session_id exist. Keep None as string if missing.
    """
    df["user_id"] = df["user_id"].fillna("anonymous")
    df["session_id"] = df["session_id"].fillna(df["user_id"] + "_sess")
    return df

def flatten_metadata_column(df: pd.DataFrame, metadata_fields: list = None) -> pd.DataFrame:
    """
    Optionally extract some common metadata fields into top-level columns.
    metadata_fields: list of keys to extract if present in metadata JSON.
    """
    if metadata_fields is None:
        metadata_fields = ["button","element","scroll"]

    def extract(meta, key):
        if not meta or pd.isna(meta):
            return None
        if isinstance(meta, dict):
            return meta.get(key)
        try:
            # if stored as stringified JSON
            import json
            parsed = json.loads(meta)
            return parsed.get(key)
        except Exception:
            return None

    for key in metadata_fields:
        df[f"meta_{key}"] = df["metadata"].apply(lambda m: extract(m, key))
    return df

def preprocess_events(df):
    # drop duplicates by event_id or _id + event_type + timestamp
    if "event_id" in df.columns:
        df = df.drop_duplicates(subset=["event_id"], keep="last")
    else:
        df = df.drop_duplicates(subset=["_id"], keep="last")

    df = fill_missing_user_session(df)
    df["page_url_norm"] = df["page_url"].apply(normalize_url)
    df = flatten_metadata_column(df)
    # remove events with no timestamp (optional)
    df = df[~df["timestamp"].isna()].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
