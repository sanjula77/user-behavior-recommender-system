# Create session-level & user-level features
# ml/pipeline/feature_engineer.py
import pandas as pd

def user_level_features(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    From session table produce user-level aggregated features.
    """
    agg = sessions_df.groupby("user_id").agg(
        total_sessions = ("session_id", "nunique"),
        avg_session_duration = ("duration_sec", "mean"),
        median_pages = ("pages", "median"),
        bounce_rate = ("bounce", "mean"),
        avg_clicks = ("clicks", "mean"),
    ).reset_index()

    # fill NaNs
    agg = agg.fillna(0)
    return agg

def session_feature_transform(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Additional derived features on sessions (examples).
    """
    df = sessions_df.copy()
    # engagement score example (simple)
    df["engagement_score"] = df["pages"] * 0.6 + df["clicks"] * 0.4 - df["bounce"] * 2
    return df
