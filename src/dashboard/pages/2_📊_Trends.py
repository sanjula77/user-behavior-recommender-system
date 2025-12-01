import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_path))

import streamlit as st
import pandas as pd
from src.dashboard.services.api_client import APIClient
from src.dashboard.components.charts import plot_trend_line, plot_bar_chart
from src.dashboard.components.sidebar import render_sidebar

st.set_page_config(page_title="Trends", page_icon="📊", layout="wide")

start_date, end_date = render_sidebar()

st.title("📊 User Behavior Trends")

with st.spinner("Loading trend data..."):
    daily_data = APIClient.get_daily_trends(
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date) if end_date else None
    )
    weekly_data = APIClient.get_weekly_trends()

tab1, tab2 = st.tabs(["Daily Trends", "Weekly Trends"])

with tab1:
    if daily_data:
        df_daily = pd.DataFrame(daily_data)
        
        # Ensure date column is datetime
        if 'date' in df_daily.columns:
            df_daily['date'] = pd.to_datetime(df_daily['date'])
        
        col1, col2 = st.columns(2)
        with col1:
            if 'sessions' in df_daily.columns:
                st.plotly_chart(plot_trend_line(df_daily, "date", "sessions", "Daily Sessions"), use_container_width=True)
            else:
                st.info("Sessions data not available")
        with col2:
            if 'events' in df_daily.columns:
                st.plotly_chart(plot_trend_line(df_daily, "date", "events", "Daily Events"), use_container_width=True)
            else:
                st.info("Events data not available")
            
        col3, col4 = st.columns(2)
        with col3:
            if 'avg_conversion_probability' in df_daily.columns:
                st.plotly_chart(plot_trend_line(df_daily, "date", "avg_conversion_probability", "Avg Conversion Probability"), use_container_width=True)
            else:
                st.info("Conversion probability data not available")
        with col4:
            if 'avg_bounce_probability' in df_daily.columns:
                st.plotly_chart(plot_trend_line(df_daily, "date", "avg_bounce_probability", "Avg Bounce Probability"), use_container_width=True)
            else:
                st.info("Bounce probability data not available")
    else:
        st.info("No daily data available.")

with tab2:
    if weekly_data:
        df_weekly = pd.DataFrame(weekly_data)
        
        # Ensure week column is datetime
        if 'week' in df_weekly.columns:
            df_weekly['week'] = pd.to_datetime(df_weekly['week'])
        
        col1, col2 = st.columns(2)
        with col1:
            if 'total_sessions' in df_weekly.columns:
                st.plotly_chart(plot_bar_chart(df_weekly, "week", "total_sessions", "Weekly Total Sessions"), use_container_width=True)
            else:
                st.info("Sessions data not available")
        with col2:
            if 'conversion_rate' in df_weekly.columns:
                st.plotly_chart(plot_trend_line(df_weekly, "week", "conversion_rate", "Weekly Conversion Rate"), use_container_width=True)
            else:
                st.info("Conversion rate data not available")
        
        col3, col4 = st.columns(2)
        with col3:
            if 'bounce_rate' in df_weekly.columns:
                st.plotly_chart(plot_trend_line(df_weekly, "week", "bounce_rate", "Weekly Bounce Rate"), use_container_width=True)
            else:
                st.info("Bounce rate data not available")
        with col4:
            if 'has_anomaly' in df_weekly.columns:
                anomaly_counts = df_weekly.groupby('week')['has_anomaly'].sum().reset_index()
                st.plotly_chart(plot_bar_chart(anomaly_counts, "week", "has_anomaly", "Weekly Anomaly Count"), use_container_width=True)
            else:
                st.info("Anomaly data not available")
    else:
        st.info("No weekly data available.")
