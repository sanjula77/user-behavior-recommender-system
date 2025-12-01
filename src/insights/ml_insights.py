"""
ML Insights Analysis
Generates insights from ML model predictions by segment.
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
ALL_PRED_FILE = PREDICTIONS_DIR / "all_predictions.csv"
SEGMENT_FILE = DATA_DIR / "processed/user_segments_final.csv"

# Thresholds for insights
HIGH_CONVERSION_PROB = 0.7
MEDIUM_CONVERSION_PROB = 0.5
HIGH_BOUNCE_PROB = 0.5327  # optimal threshold used in models

# ============================
# VALIDATION
# ============================
print("=" * 60)
print("ML Insights Generation")
print("=" * 60)
print(f"Project root: {PROJECT_ROOT}")

if not ALL_PRED_FILE.exists():
    print(f"ERROR: Predictions file not found: {ALL_PRED_FILE}")
    print("Please run 04_supervised_ml_models.ipynb Cell 9 first!")
    sys.exit(1)

if not SEGMENT_FILE.exists():
    print(f"ERROR: Segments file not found: {SEGMENT_FILE}")
    print("Please run 03_user_clustering.ipynb first!")
    sys.exit(1)

# ============================
# LOAD DATA
# ============================
print("\nLoading data...")
try:
    all_preds = pd.read_csv(ALL_PRED_FILE)
    segments = pd.read_csv(SEGMENT_FILE)
    
    print(f"✓ Loaded {len(all_preds)} predictions")
    print(f"✓ Loaded {len(segments)} user segments")
    
    # Check required columns
    if 'user_id' not in all_preds.columns:
        raise ValueError("predictions missing 'user_id' column")
    if 'user_id' not in segments.columns:
        raise ValueError("segments missing 'user_id' column")
    if 'behavior_segment' not in segments.columns:
        raise ValueError("segments missing 'behavior_segment' column")
    
    # Check which probability columns exist
    available_cols = all_preds.columns.tolist()
    print(f"  Available columns: {available_cols[:10]}...")
    
except Exception as e:
    print(f"ERROR loading data: {e}")
    sys.exit(1)

# ============================
# MERGE SEGMENTS
# ============================
print("\nMerging segments with predictions...")
try:
    data = all_preds.merge(
        segments[['user_id', 'behavior_segment']], 
        on='user_id', 
        how='left'
    )
    
    # Fill missing segments
    data['behavior_segment'] = data['behavior_segment'].fillna('Unknown')
    
    print(f"✓ Merged data: {len(data)} records")
    print(f"  Segments: {data['behavior_segment'].value_counts().to_dict()}")
except Exception as e:
    print(f"ERROR merging data: {e}")
    sys.exit(1)

# ============================
# ML INSIGHTS AGGREGATION
# ============================
print("\nComputing segment-level ML insights...")
try:
    # Determine which columns to use
    convert_rf_col = 'convert_prob_rf' if 'convert_prob_rf' in data.columns else 'conversion_probability'
    convert_xgb_col = 'convert_prob_xgb' if 'convert_prob_xgb' in data.columns else 'conversion_probability'
    convert_lr_col = 'convert_prob_lr' if 'convert_prob_lr' in data.columns else 'conversion_probability'
    bounce_col = 'bounce_prob' if 'bounce_prob' in data.columns else 'bounce_probability'
    
    print(f"  Using columns:")
    print(f"    RF: {convert_rf_col}")
    print(f"    XGB: {convert_xgb_col}")
    print(f"    LR: {convert_lr_col}")
    print(f"    Bounce: {bounce_col}")
    
    # Segment-level summary
    agg_dict = {
        'total_users': ('user_id', 'nunique')
    }
    
    # Add conversion probability aggregations if columns exist
    if convert_rf_col in data.columns:
        agg_dict['rf_high_prob'] = (convert_rf_col, lambda x: (x > HIGH_CONVERSION_PROB).sum())
    if convert_xgb_col in data.columns:
        agg_dict['xgb_high_prob'] = (convert_xgb_col, lambda x: (x > HIGH_CONVERSION_PROB).sum())
    if convert_lr_col in data.columns:
        agg_dict['lr_high_prob'] = (convert_lr_col, lambda x: (x > HIGH_CONVERSION_PROB).sum())
    if bounce_col in data.columns:
        agg_dict['bounce_risk'] = (bounce_col, lambda x: (x >= HIGH_BOUNCE_PROB).sum())
    
    segment_summary = data.groupby('behavior_segment').agg(**agg_dict).reset_index()
    
    # Compute percentages
    for col in ['rf_high_prob', 'xgb_high_prob', 'lr_high_prob', 'bounce_risk']:
        if col in segment_summary.columns:
            segment_summary[f'{col}_pct'] = (
                segment_summary[col] / segment_summary['total_users'] * 100
            )
    
    print(f"✓ Computed insights for {len(segment_summary)} segments")
    print(f"\nSegment Summary:")
    for _, row in segment_summary.iterrows():
        print(f"  {row['behavior_segment']}: {row['total_users']} users")
        if 'rf_high_prob_pct' in row:
            print(f"    High conversion (RF): {row['rf_high_prob_pct']:.1f}%")
        if 'bounce_risk_pct' in row:
            print(f"    At bounce risk: {row['bounce_risk_pct']:.1f}%")
    
except Exception as e:
    print(f"ERROR computing insights: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================
# USER-TARGET INSIGHTS
# ============================
print("\nIdentifying high-value and at-risk users...")
try:
    # High-value users for campaigns
    high_value_rf = None
    high_value_xgb = None
    at_risk_bounce = None
    
    if convert_rf_col in data.columns:
        high_value_rf = data[data[convert_rf_col] > HIGH_CONVERSION_PROB].copy()
        print(f"✓ High-value users (RF): {len(high_value_rf)}")
    
    if convert_xgb_col in data.columns:
        high_value_xgb = data[data[convert_xgb_col] > HIGH_CONVERSION_PROB].copy()
        print(f"✓ High-value users (XGB): {len(high_value_xgb)}")
    
    if bounce_col in data.columns:
        at_risk_bounce = data[data[bounce_col] >= HIGH_BOUNCE_PROB].copy()
        print(f"✓ At-risk users (bounce): {len(at_risk_bounce)}")
    
except Exception as e:
    print(f"ERROR identifying users: {e}")
    sys.exit(1)

# ============================
# SAVE OUTPUTS
# ============================
print("\nSaving outputs...")
try:
    # Segment summary
    segment_summary.to_csv(INSIGHT_DIR / "ml_segment_summary.csv", index=False)
    segment_summary.to_json(
        INSIGHT_DIR / "ml_segment_summary.json", 
        orient='records', 
        date_format='iso'
    )
    print(f"✓ Saved: {INSIGHT_DIR / 'ml_segment_summary.csv'}")
    
    # High-value and at-risk users
    if high_value_rf is not None and len(high_value_rf) > 0:
        output_cols = ['user_id', convert_rf_col]
        if 'behavior_segment' in high_value_rf.columns:
            output_cols.append('behavior_segment')
        high_value_rf[output_cols].to_csv(
            INSIGHT_DIR / "high_value_users_rf.csv", 
            index=False
        )
        print(f"✓ Saved: {INSIGHT_DIR / 'high_value_users_rf.csv'}")
    
    if high_value_xgb is not None and len(high_value_xgb) > 0:
        output_cols = ['user_id', convert_xgb_col]
        if 'behavior_segment' in high_value_xgb.columns:
            output_cols.append('behavior_segment')
        high_value_xgb[output_cols].to_csv(
            INSIGHT_DIR / "high_value_users_xgb.csv", 
            index=False
        )
        print(f"✓ Saved: {INSIGHT_DIR / 'high_value_users_xgb.csv'}")
    
    if at_risk_bounce is not None and len(at_risk_bounce) > 0:
        output_cols = ['user_id', bounce_col]
        if 'behavior_segment' in at_risk_bounce.columns:
            output_cols.append('behavior_segment')
        at_risk_bounce[output_cols].to_csv(
            INSIGHT_DIR / "at_risk_users_bounce.csv", 
            index=False
        )
        print(f"✓ Saved: {INSIGHT_DIR / 'at_risk_users_bounce.csv'}")
    
except Exception as e:
    print(f"ERROR saving outputs: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("ML Insights Generation Complete!")
print("=" * 60)
