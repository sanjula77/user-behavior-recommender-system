"""
Daily Trends Analysis
Generates daily aggregated metrics from sessions, events, and ML predictions.
"""

import pandas as pd
import json
from pathlib import Path
import sys

# ============================
# PROJECT ROOT DETECTION
# ============================
def get_project_root():
    """Detect project root directory"""
    current_dir = Path(__file__).resolve().parent
    # Go up from src/insights to project root
    if current_dir.name == "insights" and current_dir.parent.name == "src":
        return current_dir.parent.parent
    # If running from project root
    if (current_dir / "data").exists():
        return current_dir
    # Fallback
    return Path.cwd()

PROJECT_ROOT = get_project_root()
BASE_DIR = PROJECT_ROOT / "data"
RAW_DIR = BASE_DIR / "raw"
PRED_DIR = BASE_DIR / "predictions"
OUTPUT_DIR = BASE_DIR / "insights"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# VALIDATION
# ============================
def validate_files():
    """Check if required files exist"""
    required_files = {
        "sessions": RAW_DIR / "sessions_raw.csv",
        "events": RAW_DIR / "events_raw.csv",
        "pred_rf": PRED_DIR / "y_pred_convert_rf.csv",
        "pred_xgb": PRED_DIR / "y_pred_convert_xgb.csv",
        "pred_lr": PRED_DIR / "y_pred_convert_lr.csv",
        "pred_bounce": PRED_DIR / "y_pred_bounce.csv"
    }
    
    missing = []
    for name, path in required_files.items():
        if not path.exists():
            missing.append(f"{name}: {path}")
    
    if missing:
        print("ERROR: Missing required files:")
        for item in missing:
            print(f"  - {item}")
        sys.exit(1)
    
    return required_files

# ============================
# LOAD DATA
# ============================
print("=" * 60)
print("Daily Trends Generation")
print("=" * 60)
print(f"Project root: {PROJECT_ROOT}")
print(f"Output directory: {OUTPUT_DIR}")

print("\nValidating files...")
file_paths = validate_files()

print("\nLoading data...")
try:
    sessions = pd.read_csv(file_paths["sessions"])
    events = pd.read_csv(file_paths["events"])
    pred_rf = pd.read_csv(file_paths["pred_rf"])
    pred_xgb = pd.read_csv(file_paths["pred_xgb"])
    pred_lr = pd.read_csv(file_paths["pred_lr"])
    pred_bounce = pd.read_csv(file_paths["pred_bounce"])
    print(f"✓ Loaded {len(sessions)} sessions, {len(events)} events")
    print(f"✓ Loaded predictions: {len(pred_rf)} users")
except Exception as e:
    print(f"ERROR loading data: {e}")
    sys.exit(1)

# ============================
# PREPROCESS TIMESTAMPS
# ============================
print("\nProcessing timestamps...")
try:
    sessions["start_time"] = pd.to_datetime(sessions["start_time"])
    sessions["date"] = sessions["start_time"].dt.date
    
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    events["date"] = events["timestamp"].dt.date
    
    print(f"✓ Date range: {sessions['date'].min()} to {sessions['date'].max()}")
except Exception as e:
    print(f"ERROR processing timestamps: {e}")
    sys.exit(1)

# ============================
# DAILY BASIC METRICS
# ============================
print("\nComputing daily metrics...")
try:
    daily_sessions = sessions.groupby("date").size().rename("sessions")
    daily_events = events.groupby("date").size().rename("events")
    print(f"✓ Daily sessions and events computed")
except Exception as e:
    print(f"ERROR computing basic metrics: {e}")
    sys.exit(1)

# ============================
# DAILY CONVERSION PROBABILITY
# ============================
print("Computing conversion probabilities...")
try:
    # Check required columns
    required_cols = ["user_id", "conversion_probability"]
    for df_name, df in [("pred_rf", pred_rf), ("pred_xgb", pred_xgb), ("pred_lr", pred_lr)]:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{df_name} missing columns: {missing}")
    
    # Average probability from all 3 conversion models
    convert_probs = (
        pred_rf[["user_id", "conversion_probability"]]
        .rename(columns={"conversion_probability": "prob_rf"})
        .merge(
            pred_xgb[["user_id", "conversion_probability"]].rename(columns={"conversion_probability": "prob_xgb"}),
            on="user_id",
            how="outer"
        )
        .merge(
            pred_lr[["user_id", "conversion_probability"]].rename(columns={"conversion_probability": "prob_lr"}),
            on="user_id",
            how="outer"
        )
    )
    
    # Compute average (handle missing values)
    convert_probs["conversion_prob_avg"] = convert_probs[["prob_rf", "prob_xgb", "prob_lr"]].mean(axis=1, skipna=True)
    
    # Map to session dates (use first session date per user)
    user_dates = sessions.groupby("user_id")["date"].first().reset_index()
    convert_probs = convert_probs.merge(user_dates, on="user_id", how="left")
    
    # Filter out users without dates
    convert_probs = convert_probs.dropna(subset=["date"])
    
    daily_conversion = (
        convert_probs.groupby("date")["conversion_prob_avg"]
        .mean()
        .rename("avg_conversion_probability")
    )
    print(f"✓ Conversion probabilities computed for {len(convert_probs)} users")
except Exception as e:
    print(f"ERROR computing conversion probabilities: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================
# DAILY BOUNCE PROBABILITY
# ============================
print("Computing bounce probabilities...")
try:
    if "bounce_probability" not in pred_bounce.columns:
        raise ValueError("pred_bounce missing 'bounce_probability' column")
    
    bounce_probs = pred_bounce[["user_id", "bounce_probability"]].copy()
    bounce_probs = bounce_probs.merge(user_dates, on="user_id", how="left")
    bounce_probs = bounce_probs.dropna(subset=["date"])
    
    daily_bounce = (
        bounce_probs.groupby("date")["bounce_probability"]
        .mean()
        .rename("avg_bounce_probability")
    )
    print(f"✓ Bounce probabilities computed for {len(bounce_probs)} users")
except Exception as e:
    print(f"ERROR computing bounce probabilities: {e}")
    sys.exit(1)

# ============================
# MERGE EVERYTHING
# ============================
print("\nMerging daily trends...")
try:
    # Convert to DataFrames with date as index
    daily_sessions_df = daily_sessions.to_frame()
    daily_events_df = daily_events.to_frame()
    daily_conversion_df = daily_conversion.to_frame()
    daily_bounce_df = daily_bounce.to_frame()
    
    # Merge all metrics
    daily_trends = (
        daily_sessions_df
        .join(daily_events_df, how="outer")
        .join(daily_conversion_df, how="outer")
        .join(daily_bounce_df, how="outer")
        .fillna(0)
        .sort_index()
    )
    
    # Reset index to get date as column
    daily_trends = daily_trends.reset_index()
    daily_trends.rename(columns={"index": "date"}, inplace=True)
    
    # Ensure date is datetime for JSON serialization
    daily_trends["date"] = pd.to_datetime(daily_trends["date"])
    
    print(f"✓ Merged {len(daily_trends)} days of data")
    print(f"  Columns: {list(daily_trends.columns)}")
except Exception as e:
    print(f"ERROR merging trends: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================
# SAVE OUTPUTS
# ============================
print("\nSaving outputs...")
try:
    # CSV - convert date back to string for CSV
    daily_trends_csv = daily_trends.copy()
    daily_trends_csv["date"] = daily_trends_csv["date"].dt.strftime("%Y-%m-%d")
    daily_trends_csv.to_csv(OUTPUT_DIR / "daily_trends.csv", index=False)
    
    # JSON - convert to records format
    daily_json = daily_trends.copy()
    daily_json["date"] = daily_json["date"].dt.strftime("%Y-%m-%d")
    daily_json_dict = daily_json.to_dict(orient="records")
    
    with open(OUTPUT_DIR / "daily_trends.json", "w") as f:
        json.dump(daily_json_dict, f, indent=2, default=str)
    
    print(f"✓ Saved: {OUTPUT_DIR / 'daily_trends.csv'}")
    print(f"✓ Saved: {OUTPUT_DIR / 'daily_trends.json'}")
    print(f"\nDaily Trends Summary:")
    print(f"  Total days: {len(daily_trends)}")
    print(f"  Avg sessions/day: {daily_trends['sessions'].mean():.1f}")
    print(f"  Avg events/day: {daily_trends['events'].mean():.1f}")
    print(f"  Avg conversion prob: {daily_trends['avg_conversion_probability'].mean():.3f}")
    print(f"  Avg bounce prob: {daily_trends['avg_bounce_probability'].mean():.3f}")
except Exception as e:
    print(f"ERROR saving outputs: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Daily Trends Generation Complete!")
print("=" * 60)
