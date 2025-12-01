"""
LLM Summaries Generation
Generates human-readable insights using Google Gemini API from all insight data.
"""

import os
import json
from pathlib import Path
import sys

# ============================
# DEPENDENCY CHECKS
# ============================
def check_dependencies():
    """Check if required packages are installed"""
    missing = []
    
    try:
        import pandas as pd
    except ImportError:
        missing.append("pandas")
    
    try:
        from dotenv import load_dotenv
    except ImportError:
        missing.append("python-dotenv")
    
    try:
        from google.generativeai import configure, GenerativeModel
    except ImportError:
        missing.append("google-generativeai")
    
    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

# Now safe to import
import pandas as pd
from dotenv import load_dotenv
from google.generativeai import configure, GenerativeModel

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
OUTPUT_FILE = INSIGHT_DIR / "llm_summary.json"
ENV_FILE = PROJECT_ROOT / ".env"

# Input files (optional - will use what's available)
INPUT_FILES = {
    "daily_trends": INSIGHT_DIR / "daily_trends.json",
    "weekly_trends": INSIGHT_DIR / "weekly_trends.json",
    "segment_trends": INSIGHT_DIR / "segment_weekly_trends.json",
    "ml_summary": INSIGHT_DIR / "ml_segment_summary.json",
    "anomaly_insights": INSIGHT_DIR / "anomaly_insights.json"
}

# ============================
# VALIDATION
# ============================
print("=" * 60)
print("LLM Summaries Generation")
print("=" * 60)
print(f"Project root: {PROJECT_ROOT}")
print(f"Output file: {OUTPUT_FILE}")

# Check .env file
if not ENV_FILE.exists():
    print(f"\nWARNING: .env file not found: {ENV_FILE}")
    print("Create a .env file with: GEMINI_API_KEY=your_key_here")
    sys.exit(1)

# Load API key
load_dotenv(ENV_FILE)
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    print("\nERROR: GEMINI_API_KEY not found in .env file")
    print("Add to .env file: GEMINI_API_KEY=your_api_key_here")
    sys.exit(1)

# Configure API
try:
    configure(api_key=gemini_api_key)
    print("✓ API key loaded and configured")
except Exception as e:
    print(f"ERROR configuring API: {e}")
    sys.exit(1)

# ============================
# LOAD INSIGHT DATA
# ============================
print("\nLoading insight data...")
available_data = {}
missing_files = []

for name, file_path in INPUT_FILES.items():
    if file_path.exists():
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                available_data[name] = data
                print(f"✓ Loaded {name}: {len(data)} records")
        except Exception as e:
            print(f"  Warning: Could not load {name}: {e}")
            missing_files.append(name)
    else:
        missing_files.append(name)
        print(f"  Warning: {name} not found: {file_path}")

if not available_data:
    print("\nERROR: No insight data files found!")
    print("Please run the insight generation scripts first:")
    print("  1. daily_trends.py")
    print("  2. weekly_trends.py")
    print("  3. segment_insights.py")
    print("  4. ml_insights.py")
    print("  5. anomaly_detection.py")
    sys.exit(1)

print(f"\n✓ Loaded {len(available_data)} data sources")

# ============================
# PREPARE DATA FOR PROMPT
# ============================
print("\nPreparing data for LLM prompt...")

# Build comprehensive data summary
data_summary = []

# 1. Daily Trends Summary
if "daily_trends" in available_data:
    try:
        daily_df = pd.DataFrame(available_data["daily_trends"])
        if len(daily_df) > 0:
            # Get last 30 days
            recent_daily = daily_df.tail(30)
            
            # Select relevant columns (with fallbacks)
            cols_to_show = []
            for col in ["date", "sessions", "events", "avg_conversion_probability", "avg_bounce_probability"]:
                if col in recent_daily.columns:
                    cols_to_show.append(col)
            
            if cols_to_show:
                daily_summary = recent_daily[cols_to_show].to_string(index=False)
                data_summary.append(f"=== Daily Trends (Last 30 Days) ===\n{daily_summary}\n")
    except Exception as e:
        print(f"  Warning: Could not process daily_trends: {e}")

# 2. Weekly Trends Summary
if "weekly_trends" in available_data:
    try:
        weekly_df = pd.DataFrame(available_data["weekly_trends"])
        if len(weekly_df) > 0:
            # Get last 8 weeks
            recent_weekly = weekly_df.tail(8)
            
            cols_to_show = []
            for col in ["week", "total_sessions", "conversion_rate", "bounce_rate", "has_anomaly"]:
                if col in recent_weekly.columns:
                    cols_to_show.append(col)
            
            if cols_to_show:
                weekly_summary = recent_weekly[cols_to_show].to_string(index=False)
                data_summary.append(f"=== Weekly Trends (Last 8 Weeks) ===\n{weekly_summary}\n")
    except Exception as e:
        print(f"  Warning: Could not process weekly_trends: {e}")

# 3. Segment Insights Summary
if "segment_trends" in available_data:
    try:
        segment_df = pd.DataFrame(available_data["segment_trends"])
        if len(segment_df) > 0:
            # Aggregate by segment (latest week)
            if "week" in segment_df.columns:
                latest_week = segment_df["week"].max()
                latest_segments = segment_df[segment_df["week"] == latest_week]
            else:
                latest_segments = segment_df
            
            cols_to_show = []
            for col in ["behavior_segment", "total_users", "conversion_rate", "bounce_rate", "has_anomaly"]:
                if col in latest_segments.columns:
                    cols_to_show.append(col)
            
            if cols_to_show:
                segment_summary = latest_segments[cols_to_show].to_string(index=False)
                data_summary.append(f"=== Segment Performance (Latest) ===\n{segment_summary}\n")
    except Exception as e:
        print(f"  Warning: Could not process segment_trends: {e}")

# 4. ML Insights Summary
if "ml_summary" in available_data:
    try:
        ml_df = pd.DataFrame(available_data["ml_summary"])
        if len(ml_df) > 0:
            cols_to_show = []
            for col in ["behavior_segment", "total_users", "rf_high_prob_pct", "bounce_risk_pct"]:
                if col in ml_df.columns:
                    cols_to_show.append(col)
            
            if cols_to_show:
                ml_summary = ml_df[cols_to_show].to_string(index=False)
                data_summary.append(f"=== ML Model Insights ===\n{ml_summary}\n")
    except Exception as e:
        print(f"  Warning: Could not process ml_summary: {e}")

# 5. Anomaly Insights Summary
if "anomaly_insights" in available_data:
    try:
        anomaly_df = pd.DataFrame(available_data["anomaly_insights"])
        if len(anomaly_df) > 0:
            # Get anomalies only
            anomaly_cols = [col for col in anomaly_df.columns if "anomaly" in col.lower() and "type" in col.lower()]
            if anomaly_cols:
                # Filter to rows with anomalies
                has_anomaly_col = None
                for col in ["has_anomaly", "avg_conversion_rf_anomaly", "avg_bounce_anomaly"]:
                    if col in anomaly_df.columns:
                        has_anomaly_col = col
                        break
                
                if has_anomaly_col:
                    anomalies = anomaly_df[anomaly_df[has_anomaly_col] == 1]
                    if len(anomalies) > 0:
                        cols_to_show = ["date"] + anomaly_cols[:3]  # Limit columns
                        cols_to_show = [c for c in cols_to_show if c in anomalies.columns]
                        if cols_to_show:
                            anomaly_summary = anomalies[cols_to_show].tail(10).to_string(index=False)
                            data_summary.append(f"=== Recent Anomalies ===\n{anomaly_summary}\n")
    except Exception as e:
        print(f"  Warning: Could not process anomaly_insights: {e}")

if not data_summary:
    print("\nERROR: Could not extract any data for prompt")
    sys.exit(1)

data_str = "\n".join(data_summary)
print(f"✓ Prepared data summary ({len(data_str)} characters)")

# ============================
# BUILD PROMPT
# ============================
prompt = f"""You are a business analyst specializing in user behavior analytics. 

Analyze the following user behavior metrics and generate actionable insights.

{data_str}

Please provide:
1. **Key Trends**: 3-5 sentences about overall patterns and trends
2. **Anomalies**: Any spikes, drops, or unusual patterns detected
3. **Segment Insights**: Differences between user segments
4. **Recommendations**: 3-5 actionable recommendations based on the data

Format your response in clear, concise sentences suitable for a business dashboard.
Focus on actionable insights that can drive business decisions.
"""

print(f"\n✓ Prompt prepared ({len(prompt)} characters)")

# ============================
# CALL GEMINI API
# ============================
print("\nCalling Gemini API...")

# Model selection - try in order of preference
# You can change the primary model here
MODEL_PREFERENCES = [
    "gemini-2.5-flash",      # Latest model (if available)
    "gemini-1.5-flash",      # Free tier compatible (faster, cheaper)
    "gemini-1.5-pro",        # More capable (may have rate limits)
]

# Try models in order until one works
model = None
model_name = None
gemini_response = None

for attempt_model in MODEL_PREFERENCES:
    try:
        print(f"  Attempting model: {attempt_model}...")
        model = GenerativeModel(attempt_model)
        model_name = attempt_model
        break
    except Exception as e:
        print(f"    Model {attempt_model} not available: {e}")
        continue

if model is None:
    print("\nERROR: None of the preferred models are available")
    print("Available models to try:")
    print("  - gemini-2.5-flash")
    print("  - gemini-1.5-flash")
    print("  - gemini-1.5-pro")
    print("\nCheck Google AI Studio for available models in your region/tier")
    sys.exit(1)

try:
    print(f"  ✓ Using model: {model_name}")
    print("  Sending request to Gemini API...")
    gemini_response = model.generate_content(prompt)
    
    
    if not gemini_response or not hasattr(gemini_response, 'text'):
        raise ValueError("Empty or invalid response from API")
    
    summary_text = gemini_response.text
    
    if not summary_text or len(summary_text.strip()) == 0:
        raise ValueError("Empty summary text received")
    
    print(f"✓ Received summary using {model_name} ({len(summary_text)} characters)")
    
except Exception as e:
    error_msg = str(e)
    print(f"\nERROR calling Gemini API: {error_msg}")
    print("\nPossible causes:")
    print("  - Invalid API key")
    print("  - API quota exceeded")
    print("  - Model not available in your region/tier")
    print("  - Network connection issue")
    print("  - API service unavailable")
    print("\nTry:")
    print("  1. Check your API key is valid")
    print("  2. Verify you have quota remaining")
    print("  3. Check Google AI Studio for available models")
    print("  4. Try a different model in MODEL_PREFERENCES list")
    sys.exit(1)

# ============================
# SAVE SUMMARY
# ============================
print("\nSaving summary...")
try:
    output_data = {
        "summary": summary_text,
        "metadata": {
            "data_sources_used": list(available_data.keys()),
            "data_sources_missing": missing_files,
            "summary_length": len(summary_text),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved: {OUTPUT_FILE}")
    
    # Print summary preview
    print("\n" + "=" * 60)
    print("SUMMARY PREVIEW")
    print("=" * 60)
    preview = summary_text[:500] + "..." if len(summary_text) > 500 else summary_text
    print(preview)
    print("\n" + "=" * 60)
    
except Exception as e:
    print(f"ERROR saving summary: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("LLM Summaries Generation Complete!")
print("=" * 60)
print(f"\nOutput saved to: {OUTPUT_FILE}")
print(f"Data sources used: {', '.join(available_data.keys())}")
if missing_files:
    print(f"Missing data sources: {', '.join(missing_files)}")
