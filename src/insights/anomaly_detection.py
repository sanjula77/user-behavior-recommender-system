"""
Anomaly Detection Analysis
Detects anomalies in daily metrics using rolling window and z-score methods.
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
PREDICTIONS_DIR = DATA_DIR / "predictions"
INSIGHT_DIR = DATA_DIR / "insights"
INSIGHT_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# CONFIG
# ============================
SESSIONS_FILE = DATA_DIR / "raw/sessions_raw.csv"
ALL_PRED_FILE = PREDICTIONS_DIR / "all_predictions.csv"
OUTPUT_JSON = INSIGHT_DIR / "anomaly_insights.json"
OUTPUT_CSV = INSIGHT_DIR / "anomaly_insights.csv"

# Parameters
ROLLING_WINDOW = 7  # days
Z_THRESHOLD = 2  # standard deviations

# ============================
# VALIDATION
# ============================
print("=" * 60)
print("Anomaly Detection Analysis")
print("=" * 60)
print(f"Project root: {PROJECT_ROOT}")

if not SESSIONS_FILE.exists():
    print(f"ERROR: Sessions file not found: {SESSIONS_FILE}")
    sys.exit(1)

if not ALL_PRED_FILE.exists():
    print(f"ERROR: Predictions file not found: {ALL_PRED_FILE}")
    print("Please run 04_supervised_ml_models.ipynb Cell 9 first!")
    sys.exit(1)

# ============================
# LOAD DATA
# ============================
print("\nLoading data...")
try:
    sessions = pd.read_csv(SESSIONS_FILE, parse_dates=['start_time'])
    preds = pd.read_csv(ALL_PRED_FILE)
    
    print(f"✓ Loaded {len(sessions)} sessions")
    print(f"✓ Loaded {len(preds)} predictions")
    
    # Check required columns
    if 'user_id' not in sessions.columns:
        raise ValueError("sessions missing 'user_id' column")
    if 'user_id' not in preds.columns:
        raise ValueError("predictions missing 'user_id' column")
    
    # Check which probability columns exist
    available_cols = preds.columns.tolist()
    print(f"  Available prediction columns: {[c for c in available_cols if 'prob' in c.lower()]}")
    
except Exception as e:
    print(f"ERROR loading data: {e}")
    sys.exit(1)

# ============================
# MERGE DATA
# ============================
print("\nMerging predictions with session timestamps...")
try:
    # Extract date from sessions
    sessions['date'] = sessions['start_time'].dt.date
    
    # Merge predictions with session timestamps
    # Use first session date per user for simplicity
    user_dates = sessions.groupby('user_id')['date'].first().reset_index()
    data = preds.merge(user_dates, on='user_id', how='left')
    
    # Filter out users without dates
    data = data.dropna(subset=['date'])
    data['date'] = pd.to_datetime(data['date'])
    
    print(f"✓ Merged data: {len(data)} user-date combinations")
    
except Exception as e:
    print(f"ERROR merging data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================
# AGGREGATE DAILY METRICS
# ============================
print("\nAggregating daily metrics...")
try:
    # Determine which columns to use
    convert_rf_col = 'convert_prob_rf' if 'convert_prob_rf' in data.columns else None
    convert_xgb_col = 'convert_prob_xgb' if 'convert_prob_xgb' in data.columns else None
    convert_lr_col = 'convert_prob_lr' if 'convert_prob_lr' in data.columns else None
    bounce_col = 'bounce_prob' if 'bounce_prob' in data.columns else 'bounce_probability'
    
    print(f"  Using columns:")
    if convert_rf_col:
        print(f"    RF conversion: {convert_rf_col}")
    if convert_xgb_col:
        print(f"    XGB conversion: {convert_xgb_col}")
    if convert_lr_col:
        print(f"    LR conversion: {convert_lr_col}")
    print(f"    Bounce: {bounce_col}")
    
    # Build aggregation dictionary
    agg_dict = {}
    
    if convert_rf_col:
        agg_dict['avg_conversion_rf'] = (convert_rf_col, 'mean')
        agg_dict['high_value_users_rf'] = (convert_rf_col, lambda x: (x > 0.7).sum())
    
    if convert_xgb_col:
        agg_dict['avg_conversion_xgb'] = (convert_xgb_col, 'mean')
        agg_dict['high_value_users_xgb'] = (convert_xgb_col, lambda x: (x > 0.7).sum())
    
    if convert_lr_col:
        agg_dict['avg_conversion_lr'] = (convert_lr_col, 'mean')
    
    if bounce_col:
        agg_dict['avg_bounce'] = (bounce_col, 'mean')
        agg_dict['high_bounce_users'] = (bounce_col, lambda x: (x >= 0.5327).sum())
    
    daily_metrics = data.groupby('date').agg(**agg_dict).reset_index()
    
    print(f"✓ Aggregated {len(daily_metrics)} days of metrics")
    
except Exception as e:
    print(f"ERROR aggregating metrics: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================
# COMPUTE ROLLING BASELINE & Z-SCORES
# ============================
print("\nComputing rolling statistics and z-scores...")
try:
    # Metrics to analyze
    metrics_to_analyze = []
    if 'avg_conversion_rf' in daily_metrics.columns:
        metrics_to_analyze.append('avg_conversion_rf')
    if 'avg_conversion_xgb' in daily_metrics.columns:
        metrics_to_analyze.append('avg_conversion_xgb')
    if 'avg_conversion_lr' in daily_metrics.columns:
        metrics_to_analyze.append('avg_conversion_lr')
    if 'avg_bounce' in daily_metrics.columns:
        metrics_to_analyze.append('avg_bounce')
    
    print(f"  Analyzing {len(metrics_to_analyze)} metrics")
    
    for col in metrics_to_analyze:
        # Rolling statistics
        daily_metrics[f'{col}_roll_mean'] = daily_metrics[col].rolling(
            ROLLING_WINDOW, 
            min_periods=1
        ).mean()
        
        daily_metrics[f'{col}_roll_std'] = daily_metrics[col].rolling(
            ROLLING_WINDOW, 
            min_periods=1
        ).std().fillna(0)
        
        # Z-scores
        daily_metrics[f'{col}_zscore'] = (
            (daily_metrics[col] - daily_metrics[f'{col}_roll_mean']) / 
            (daily_metrics[f'{col}_roll_std'] + 1e-6)  # Add small epsilon to avoid division by zero
        )
        
        # Anomaly flag
        daily_metrics[f'{col}_anomaly'] = daily_metrics[f'{col}_zscore'].abs() > Z_THRESHOLD
        
        # Count anomalies
        anomaly_count = daily_metrics[f'{col}_anomaly'].sum()
        print(f"  ✓ {col}: {anomaly_count} anomalies detected")
    
    # Overall anomaly flag
    anomaly_cols = [f'{col}_anomaly' for col in metrics_to_analyze]
    if anomaly_cols:
        daily_metrics['has_anomaly'] = daily_metrics[anomaly_cols].any(axis=1).astype(int)
        total_anomalies = daily_metrics['has_anomaly'].sum()
        print(f"  ✓ Total days with anomalies: {total_anomalies}")
    
except Exception as e:
    print(f"ERROR computing statistics: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================
# FLAG SPIKES/DROPS
# ============================
print("\nFlagging spike/drop types...")
try:
    def anomaly_type(row, col):
        """Determine if anomaly is a spike or drop"""
        if not row[f'{col}_anomaly']:
            return None
        return "spike" if row[f'{col}_zscore'] > 0 else "drop"
    
    for col in metrics_to_analyze:
        daily_metrics[f'{col}_anomaly_type'] = daily_metrics.apply(
            lambda row: anomaly_type(row, col), 
            axis=1
        )
    
    print(f"✓ Anomaly types flagged")
    
except Exception as e:
    print(f"ERROR flagging anomaly types: {e}")
    sys.exit(1)

# ============================
# SAVE OUTPUTS
# ============================
print("\nSaving outputs...")
try:
    # Convert date to string for CSV
    daily_metrics_csv = daily_metrics.copy()
    daily_metrics_csv['date'] = pd.to_datetime(daily_metrics_csv['date']).dt.strftime('%Y-%m-%d')
    daily_metrics_csv.to_csv(OUTPUT_CSV, index=False)
    
    # JSON with ISO date format
    daily_metrics_json = daily_metrics.copy()
    daily_metrics_json['date'] = pd.to_datetime(daily_metrics_json['date'])
    daily_metrics_json.to_json(
        OUTPUT_JSON, 
        orient='records', 
        date_format='iso'
    )
    
    print(f"✓ Saved: {OUTPUT_CSV}")
    print(f"✓ Saved: {OUTPUT_JSON}")
    
    # Print summary
    if 'has_anomaly' in daily_metrics.columns:
        anomaly_days = daily_metrics[daily_metrics['has_anomaly'] == 1]
        if len(anomaly_days) > 0:
            print(f"\nAnomaly Summary:")
            print(f"  Total anomaly days: {len(anomaly_days)}")
            print(f"  Date range: {anomaly_days['date'].min()} to {anomaly_days['date'].max()}")
    
except Exception as e:
    print(f"ERROR saving outputs: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Anomaly Detection Complete!")
print("=" * 60)
