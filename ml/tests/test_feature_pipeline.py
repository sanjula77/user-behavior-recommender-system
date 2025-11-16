# Lightweight tests for feature pipeline
# ml/tests/test_feature_pipeline.py
import os
import pandas as pd
from ml.pipeline.extractor import load_events
from ml.pipeline.preprocess import preprocess_events
from ml.pipeline.sessionizer import sessionize_events, build_session_table

def test_pipeline_runs_small_sample():
    # Try to pull a couple events; if none exist, create a dummy DF to run through pipeline
    df = load_events(limit=10)
    if df.empty:
        # create small synthetic
        import datetime
        docs = []
        user = "test_user"
        for i in range(3):
            docs.append({
                "event_id": f"e{i}",
                "user_id": user,
                "session_id": f"sess{i//2}",
                "event_type": "page_view" if i % 2 == 0 else "click",
                "page_url": f"https://example.com/page{i}",
                "timestamp": pd.to_datetime(datetime.datetime.utcnow()),
                "metadata": {}
            })
        df = pd.DataFrame(docs)

    df_clean = preprocess_events(df)
    df_s = sessionize_events(df_clean)
    sessions = build_session_table(df_s)
    assert isinstance(sessions, pd.DataFrame)
    # at least one session produced
    assert len(sessions) >= 1
