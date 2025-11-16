# Build user-item interaction matrix + item metadata
# ml/phase2_hybrid/data_prep.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

def load_sessions():
    path = os.path.join(OUTPUT_DIR, "sessions_features.csv")
    return pd.read_csv(path, parse_dates=["start_time","end_time"])

def load_events(limit=None):
    # load original events by connecting to ml/pipeline/extractor if you prefer,
    # but here we read events from Mongo export or the backend collection was saved.
    # For simplicity, try to read events CSV if exists, otherwise return empty
    events_csv = os.path.join(OUTPUT_DIR, "events_export.csv")
    if os.path.exists(events_csv):
        return pd.read_csv(events_csv, parse_dates=["timestamp"])
    else:
        # fallback: try to load from mongodb via extractor (if available)
        try:
            from ml.pipeline.extractor import load_events
            df = load_events(limit=limit)
            return df
        except Exception:
            return pd.DataFrame()

def build_user_item(events_df):
    """
    Build implicit user-item interaction counts.
    Returns:
      user_item_df: pandas DataFrame with index=user_id and columns=item_id (page_url_norm)
    """
    if events_df.empty:
        return pd.DataFrame()

    events_df["item"] = events_df["page_url_norm"].astype(str)
    # weight clicks more than page views
    def event_weight(row):
        if row.get("event_type") == "click":
            return 2.0
        return 1.0

    events_df["weight"] = events_df.apply(event_weight, axis=1)

    # aggregate weights per user-item
    agg = events_df.groupby(["user_id", "item"])["weight"].sum().reset_index()
    users = agg["user_id"].unique().tolist()
    items = agg["item"].unique().tolist()

    user_index = {u:i for i,u in enumerate(users)}
    item_index = {it:i for i,it in enumerate(items)}

    # build sparse matrix later; for simplicity create DataFrame
    data = defaultdict(dict)
    for _, row in agg.iterrows():
        u = row["user_id"]
        it = row["item"]
        data[u][it] = row["weight"]

    user_item_df = pd.DataFrame.from_dict(data, orient="index").fillna(0)
    return user_item_df

def train_test_split_interactions(user_item_df, test_size=0.2, seed=42):
    """
    Simple user-level split: keep some users for test and train.
    For production use time-based or leave-one-out splits.
    """
    users = user_item_df.index.tolist()
    train_users, test_users = train_test_split(users, test_size=test_size, random_state=seed)
    train_df = user_item_df.loc[train_users]
    test_df = user_item_df.loc[test_users]
    return train_df, test_df
