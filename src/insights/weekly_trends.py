"""
Weekly Trends Analysis
Aggregates daily trends into weekly summaries with anomaly detection.
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
DAILY_FILE = INSIGHT_DIR / "daily_trends.csv"
ANOMALY_STD_MULTIPLIER = 2  # Threshold for anomaly detection (mean ± 2*std)

# ============================
# VALIDATION
# ============================
print("=" * 60)
print("Weekly Trends Generation")
print("=" * 60)
print(f"Project root: {PROJECT_ROOT}")
print(f"Daily trends file: {DAILY_FILE}")

if not DAILY_FILE.exists():
    print(f"ERROR: Daily trends file not found: {DAILY_FILE}")
    print("Please run daily_trends.py first!")
    sys.exit(1)

# ============================
# LOAD DAILY TRENDS
# ============================
print("\nLoading daily trends...")
try:
    daily = pd.read_csv(DAILY_FILE, parse_dates=["date"])
    print(f"✓ Loaded {len(daily)} days of data")
    print(f"  Date range: {daily['date'].min()} to {daily['date'].max()}")
    
    # Check required columns
    required_cols = ["date", "sessions", "events", "avg_conversion_probability", "avg_bounce_probability"]
    missing_cols = [col for col in required_cols if col not in daily.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"  Columns: {list(daily.columns)}")
except Exception as e:
    print(f"ERROR loading daily trends: {e}")
    sys.exit(1)

# ============================
# CREATE WEEK COLUMN
# ============================
print("\nCreating week groupings...")
try:
    # Week starting on Monday
    daily['week'] = daily['date'].dt.to_period('W').apply(lambda r: r.start_time)
    print(f"✓ Created week column")
    print(f"  Total weeks: {daily['week'].nunique()}")
except Exception as e:
    print(f"ERROR creating week column: {e}")
    sys.exit(1)

# ============================
# WEEKLY AGGREGATION
# ============================
print("\nAggregating weekly metrics...")
try:
    weekly_summary = daily.groupby('week').agg(
        total_sessions=('sessions', 'sum'),
        total_events=('events', 'sum'),
        avg_conversion_prob=('avg_conversion_probability', 'mean'),
        avg_bounce_prob=('avg_bounce_probability', 'mean')
    ).reset_index()
    
    # Compute rates (using sessions as denominator)
    weekly_summary['conversion_rate'] = (
        weekly_summary['avg_conversion_prob']  # Already a probability, can use directly
    )
    weekly_summary['bounce_rate'] = (
        weekly_summary['avg_bounce_prob']  # Already a probability
    )
    
    # Add computed metrics
    weekly_summary['avg_sessions_per_day'] = weekly_summary['total_sessions'] / 7
    weekly_summary['avg_events_per_day'] = weekly_summary['total_events'] / 7
    
    print(f"✓ Aggregated {len(weekly_summary)} weeks")
    print(f"  Avg sessions/week: {weekly_summary['total_sessions'].mean():.1f}")
    print(f"  Avg conversion rate: {weekly_summary['conversion_rate'].mean():.3f}")
    print(f"  Avg bounce rate: {weekly_summary['bounce_rate'].mean():.3f}")
except Exception as e:
    print(f"ERROR aggregating weekly metrics: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================
# ANOMALY DETECTION
# ============================
print("\nDetecting anomalies...")
try:
    for col in ['conversion_rate', 'bounce_rate']:
        if col not in weekly_summary.columns:
            print(f"  Warning: {col} not found, skipping anomaly detection")
            continue
            
        mean = weekly_summary[col].mean()
        std = weekly_summary[col].std()
        
        if std == 0 or pd.isna(std):
            print(f"  Warning: {col} has zero variance, skipping anomaly detection")
            weekly_summary[f'{col}_anomaly'] = 0
        else:
            weekly_summary[f'{col}_anomaly'] = weekly_summary[col].apply(
                lambda x: 1 if (x > mean + ANOMALY_STD_MULTIPLIER*std) or (x < mean - ANOMALY_STD_MULTIPLIER*std) else 0
            )
            
            anomaly_count = weekly_summary[f'{col}_anomaly'].sum()
            print(f"  ✓ {col}: {anomaly_count} anomalies detected (mean={mean:.3f}, std={std:.3f})")
    
    # Overall anomaly flag
    weekly_summary['has_anomaly'] = (
        weekly_summary['conversion_rate_anomaly'] | 
        weekly_summary['bounce_rate_anomaly']
    ).astype(int)
    
    total_anomalies = weekly_summary['has_anomaly'].sum()
    print(f"  ✓ Total weeks with anomalies: {total_anomalies}")
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
    weekly_summary_csv = weekly_summary.copy()
    weekly_summary_csv['week'] = pd.to_datetime(weekly_summary_csv['week']).dt.strftime('%Y-%m-%d')
    weekly_summary_csv.to_csv(INSIGHT_DIR / "weekly_trends.csv", index=False)
    
    # JSON with ISO date format
    weekly_summary_json = weekly_summary.copy()
    weekly_summary_json['week'] = pd.to_datetime(weekly_summary_json['week'])
    weekly_summary_json.to_json(
        INSIGHT_DIR / "weekly_trends.json", 
        orient='records', 
        date_format='iso'
    )
    
    print(f"✓ Saved: {INSIGHT_DIR / 'weekly_trends.csv'}")
    print(f"✓ Saved: {INSIGHT_DIR / 'weekly_trends.json'}")
except Exception as e:
    print(f"ERROR saving outputs: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Weekly Trends Generation Complete!")
print("=" * 60)
