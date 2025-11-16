# Orchestrator: extract -> preprocess -> features -> save
# ml/run_pipeline.py
import sys
import os

# Add project root to Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.pipeline.extractor import load_events
from ml.pipeline.preprocess import preprocess_events
from ml.pipeline.sessionizer import sessionize_events, build_session_table
from ml.pipeline.feature_engineer import session_feature_transform, user_level_features
from ml.pipeline.save_features import save_df

def run(limit=None):
    print("Loading events...")
    df = load_events(limit=limit)
    print(f"Loaded {len(df)} events")
    if df.empty:
        print("No events found. Exiting.")
        return

    print("Preprocessing...")
    df_clean = preprocess_events(df)
    print(f"{len(df_clean)} events after preprocessing")

    print("Sessionizing...")
    df_sessions_events = sessionize_events(df_clean)
    print(f"{df_sessions_events['session_group'].nunique()} session groups")

    print("Building session table...")
    sessions = build_session_table(df_sessions_events)
    sessions = session_feature_transform(sessions)
    print(f"{len(sessions)} sessions aggregated")

    print("Building user-level features...")
    users = user_level_features(sessions)
    print(f"{len(users)} users aggregated")

    print("Saving features...")
    save_df(sessions, "sessions_features")
    save_df(users, "user_features")

    print("Pipeline complete.")

if __name__ == "__main__":
    # optional: pass a small limit for dev debugging
    run(limit=1000)
