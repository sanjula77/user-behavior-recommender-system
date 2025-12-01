"""
Segment Insights Analysis
Analyzes user behavior trends by segment over time with anomaly detection.
"""

import pandas as pd
from pathlib import Path
import sys

# ============================
# PROJECT ROOT DETECTION
# ============================
def get_project_root():
    """Detect project root directory"""
    current_dir = Path(__file__).resolve().parent
    if current_dir.name == "insights" and current_dir.parent.name == "src":
        return current_dir.parent.parent
    if (current_dir / "data").exists():
        return current_dir
    return Path.cwd()

PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
INSIGHT_DIR = DATA_DIR / "insights"
INSIGHT_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# CONFIG
# ============================
SEGMENTS_FILE = DATA_DIR / "processed/user_segments_final.csv"
PREDICTIONS_FILE = DATA_DIR / "predictions/all_predictions.csv"
SESSIONS_FILE = DATA_DIR / "raw/sessions_raw.csv"
ANOMALY_STD_MULTIPLIER = 2

# ============================
# VALIDATION
# ============================
print("=" * 60)
print("Segment Insights Generation")
print("=" * 60)
print(f"Project root: {PROJECT_ROOT}")

required_files = {
    "segments": SEGMENTS_FILE,
    "predictions": PREDICTIONS_FILE,
    "sessions": SESSIONS_FILE
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

# ============================
# LOAD DATA
# ============================
print("\nLoading data...")
try:
    segments = pd.read_csv(SEGMENTS_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE)
    sessions = pd.read_csv(SESSIONS_FILE, parse_dates=['start_time'])
    
    print(f"✓ Loaded {len(segments)} users with segments")
    print(f"✓ Loaded {len(predictions)} predictions")
    print(f"✓ Loaded {len(sessions)} sessions")
    
    # Check required columns
    if 'behavior_segment' not in segments.columns:
        raise ValueError("segments file missing 'behavior_segment' column")
    if 'user_id' not in predictions.columns:
        raise ValueError("predictions file missing 'user_id' column")
    if 'user_id' not in sessions.columns:
        raise ValueError("sessions file missing 'user_id' column")
        
except Exception as e:
    print(f"ERROR loading data: {e}")
    sys.exit(1)

# ============================
# MERGE DATA WITH DATES
# ============================
print("\nMerging data with session dates...")
try:
    # Extract date from sessions
    sessions['date'] = sessions['start_time'].dt.date
    
    # Get first session date per user (or use all sessions for more granularity)
    user_dates = sessions[['user_id', 'date']].drop_duplicates()
    
    # Merge segment info with predictions
    data = predictions.merge(
        segments[['user_id', 'behavior_segment']], 
        on='user_id', 
        how='left'
    )
    
    # Merge with dates
    data = data.merge(user_dates, on='user_id', how='left')
    
    # Filter out users without dates
    data = data.dropna(subset=['date'])
    data['date'] = pd.to_datetime(data['date'])
    
    print(f"✓ Merged data: {len(data)} user-date combinations")
    print(f"  Segments: {data['behavior_segment'].value_counts().to_dict()}")
except Exception as e:
    print(f"ERROR merging data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================
# ADD WEEK COLUMN
# ============================
print("\nCreating week groupings...")
try:
    data['week'] = data['date'].dt.to_period('W').apply(lambda r: r.start_time)
    print(f"✓ Created week column")
    print(f"  Total weeks: {data['week'].nunique()}")
except Exception as e:
    print(f"ERROR creating week column: {e}")
    sys.exit(1)

# ============================
# AGGREGATE WEEKLY BY SEGMENT
# ============================
print("\nAggregating weekly metrics by segment...")
try:
    # Check which columns exist in predictions
    available_cols = data.columns.tolist()
    
    # Use correct column names from all_predictions.csv
    convert_prob_col = 'convert_prob_rf' if 'convert_prob_rf' in available_cols else 'conversion_probability'
    bounce_prob_col = 'bounce_prob' if 'bounce_prob' in available_cols else 'bounce_probability'
    convert_pred_col = 'convert_pred_rf' if 'convert_pred_rf' in available_cols else 'y_convert_pred'
    bounce_pred_col = 'bounce_pred_optimal' if 'bounce_pred_optimal' in available_cols else 'y_bounce_pred_optimal'
    
    print(f"  Using columns: convert_prob={convert_prob_col}, bounce_prob={bounce_prob_col}")
    
    segment_weekly = data.groupby(['behavior_segment', 'week']).agg(
        total_users=('user_id', 'nunique'),
        total_conversions=('y_convert_actual', 'sum') if 'y_convert_actual' in available_cols else ('user_id', 'count'),
        avg_conversion_prob=('convert_prob_rf', 'mean') if 'convert_prob_rf' in available_cols else ('conversion_probability', 'mean'),
        total_bounces=('y_bounce_actual', 'sum') if 'y_bounce_actual' in available_cols else ('user_id', 'count'),
        avg_bounce_prob=('bounce_prob', 'mean') if 'bounce_prob' in available_cols else ('bounce_probability', 'mean')
    ).reset_index()
    
    # Compute rates
    segment_weekly['conversion_rate'] = segment_weekly['avg_conversion_prob']  # Already probability
    segment_weekly['bounce_rate'] = segment_weekly['avg_bounce_prob']  # Already probability
    
    print(f"✓ Aggregated {len(segment_weekly)} segment-week combinations")
    print(f"  Segments: {segment_weekly['behavior_segment'].unique().tolist()}")
except Exception as e:
    print(f"ERROR aggregating metrics: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================
# SEGMENT-LEVEL ANOMALY DETECTION
# ============================
print("\nDetecting anomalies by segment...")
try:
    for col in ['conversion_rate', 'bounce_rate']:
        segment_weekly[f'{col}_anomaly'] = 0
        
        for seg in segment_weekly['behavior_segment'].unique():
            seg_data = segment_weekly[segment_weekly['behavior_segment'] == seg]
            
            if len(seg_data) < 2:
                continue  # Need at least 2 data points
            
            mean = seg_data[col].mean()
            std = seg_data[col].std()
            
            if std == 0 or pd.isna(std):
                continue
            
            indices = seg_data.index
            segment_weekly.loc[indices, f'{col}_anomaly'] = seg_data[col].apply(
                lambda x: 1 if (x > mean + ANOMALY_STD_MULTIPLIER*std) or (x < mean - ANOMALY_STD_MULTIPLIER*std) else 0
            )
            
            anomaly_count = segment_weekly.loc[indices, f'{col}_anomaly'].sum()
            if anomaly_count > 0:
                print(f"  ✓ {seg} - {col}: {anomaly_count} anomalies")
    
    # Overall anomaly flag
    segment_weekly['has_anomaly'] = (
        segment_weekly['conversion_rate_anomaly'] | 
        segment_weekly['bounce_rate_anomaly']
    ).astype(int)
    
    total_anomalies = segment_weekly['has_anomaly'].sum()
    print(f"  ✓ Total segment-weeks with anomalies: {total_anomalies}")
except Exception as e:
    print(f"ERROR detecting anomalies: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================
# SAVE OUTPUTS
# ============================
print("\nSaving outputs...")
try:
    # Convert week to string for CSV
    segment_weekly_csv = segment_weekly.copy()
    segment_weekly_csv['week'] = pd.to_datetime(segment_weekly_csv['week']).dt.strftime('%Y-%m-%d')
    segment_weekly_csv.to_csv(INSIGHT_DIR / "segment_weekly_trends.csv", index=False)
    
    # JSON with ISO date format
    segment_weekly_json = segment_weekly.copy()
    segment_weekly_json['week'] = pd.to_datetime(segment_weekly_json['week'])
    segment_weekly_json.to_json(
        INSIGHT_DIR / "segment_weekly_trends.json", 
        orient='records', 
        date_format='iso'
    )
    
    print(f"✓ Saved: {INSIGHT_DIR / 'segment_weekly_trends.csv'}")
    print(f"✓ Saved: {INSIGHT_DIR / 'segment_weekly_trends.json'}")
except Exception as e:
    print(f"ERROR saving outputs: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Segment Insights Generation Complete!")
print("=" * 60)
