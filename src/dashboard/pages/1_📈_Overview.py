import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_path))

import streamlit as st
import pandas as pd
from src.dashboard.services.api_client import APIClient
from src.dashboard.components.cards import kpi_grid
from src.dashboard.components.sidebar import render_sidebar

st.set_page_config(page_title="Overview", page_icon="📈", layout="wide")

# Sidebar
start_date, end_date = render_sidebar()

st.title("📈 Executive Overview")

# Fetch Data
with st.spinner("Loading insights..."):
    llm_summary = APIClient.get_llm_summary()
    daily_trends = APIClient.get_daily_trends(
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date) if end_date else None
    )

# AI Summary Section
if llm_summary and llm_summary.get("summary"):
    st.subheader("🤖 AI Insights")
    
    summary_text = llm_summary.get("summary", "")
    if summary_text:
        st.info(summary_text)
    
    recommendations = llm_summary.get("recommendations", [])
    if recommendations:
        with st.expander("View Recommendations"):
            if isinstance(recommendations, list):
                for rec in recommendations:
                    st.write(f"• {rec}")
            elif isinstance(recommendations, str):
                st.write(recommendations)
    
    # Show metadata if available
    metadata = llm_summary.get("metadata", {})
    if metadata:
        with st.expander("View Metadata"):
            st.json(metadata)
elif llm_summary is None:
    st.warning("⚠️ LLM summary not available. Make sure llm_summaries.py has been run.")
else:
    st.info("No AI insights available yet.")

st.divider()

# KPI Grid
if daily_trends:
    df = pd.DataFrame(daily_trends)
    
    # Calculate aggregates using actual column names from API
    total_sessions = df['sessions'].sum() if 'sessions' in df.columns else 0
    total_events = df['events'].sum() if 'events' in df.columns else 0
    avg_conversion_prob = df['avg_conversion_probability'].mean() * 100 if 'avg_conversion_probability' in df.columns else 0
    avg_bounce_prob = df['avg_bounce_probability'].mean() * 100 if 'avg_bounce_probability' in df.columns else 0
    
    # Calculate unique users estimate (sessions can approximate active users)
    unique_users_estimate = total_sessions  # Using sessions as proxy
    
    metrics = [
        {"title": "Total Sessions", "value": f"{total_sessions:,.0f}", "delta": None, "help": "Total number of user sessions"},
        {"title": "Total Events", "value": f"{total_events:,.0f}", "delta": None, "help": "Total number of events tracked"},
        {"title": "Avg Conversion Prob", "value": f"{avg_conversion_prob:.1f}%", "delta": None, "help": "Average conversion probability"},
        {"title": "Avg Bounce Prob", "value": f"{avg_bounce_prob:.1f}%", "delta": None, "help": "Average bounce probability"},
    ]
    
    kpi_grid(metrics)
    
    # Additional insights
    if len(df) > 1:
        st.subheader("📊 Trend Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'sessions' in df.columns:
                recent_sessions = df['sessions'].tail(7).sum()
                previous_sessions = df['sessions'].tail(14).head(7).sum() if len(df) >= 14 else recent_sessions
                session_change = ((recent_sessions - previous_sessions) / previous_sessions * 100) if previous_sessions > 0 else 0
                st.metric("Last 7 Days Sessions", f"{recent_sessions:,.0f}", f"{session_change:+.1f}%")
        
        with col2:
            if 'avg_conversion_probability' in df.columns:
                recent_conv = df['avg_conversion_probability'].tail(7).mean() * 100
                previous_conv = df['avg_conversion_probability'].tail(14).head(7).mean() * 100 if len(df) >= 14 else recent_conv
                conv_change = recent_conv - previous_conv
                st.metric("Last 7 Days Avg Conversion", f"{recent_conv:.1f}%", f"{conv_change:+.1f}%")
else:
    st.warning("No trend data available for the selected period.")
