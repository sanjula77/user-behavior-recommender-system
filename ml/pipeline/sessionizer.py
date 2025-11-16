# Build sessions from events
# ml/pipeline/sessionizer.py
import pandas as pd
from datetime import timedelta

SESSION_TIMEOUT = timedelta(minutes=30)

def sessionize_events(df: pd.DataFrame, timeout: timedelta = SESSION_TIMEOUT) -> pd.DataFrame:
    """
    Build session-level grouping if raw session_id may be unreliable.
    We assume df is sorted by timestamp ascending.
    Returns events DataFrame with an ensured `session_group` column (string).
    """
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column required")

    # use existing session_id but split further when gap > timeout
    session_groups = []
    prev_user = None
    prev_ts = None
    group_id = None
    for idx, row in df.iterrows():
        user = row.get("user_id", "anonymous")
        ts = row["timestamp"]
        if prev_user != user or prev_ts is None or (ts - prev_ts) > timeout:
            # start new group
            group_id = f"{user}_{ts.isoformat()}"
        session_groups.append(group_id)
        prev_user = user
        prev_ts = ts

    df["session_group"] = session_groups
    return df

def build_session_table(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate session-level features from sessionized events.
    Returns DataFrame where each row is a session.
    """
    def session_agg(group, session_group_key):
        times = group["timestamp"]
        duration = (times.max() - times.min()).total_seconds()
        pages = group["page_url_norm"].nunique()
        total_events = len(group)
        clicks = (group["event_type"] == "click").sum()
        page_views = (group["event_type"] == "page_view").sum()
        bounce = 1 if total_events == 1 and page_views == 1 else 0
        return {
            "user_id": group["user_id"].iloc[0],
            "session_id": session_group_key,  # Use the groupby key instead of accessing from group
            "start_time": times.min(),
            "end_time": times.max(),
            "duration_sec": duration,
            "pages": pages,
            "total_events": total_events,
            "clicks": int(clicks),
            "page_views": int(page_views),
            "bounce": bounce
        }

    # Iterate over groups to avoid the FutureWarning about grouping columns
    sessions = []
    for session_group, group in events_df.groupby("session_group", group_keys=False):
        result = session_agg(group, session_group)
        sessions.append(result)
    
    sessions_df = pd.DataFrame(sessions)
    return sessions_df
