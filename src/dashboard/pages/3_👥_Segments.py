import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_path))

import streamlit as st
import pandas as pd
from src.dashboard.services.api_client import APIClient
from src.dashboard.components.charts import plot_segment_distribution, plot_multi_line
from src.dashboard.components.sidebar import render_sidebar

st.set_page_config(page_title="Segments", page_icon="👥", layout="wide")

start_date, end_date = render_sidebar()

st.title("👥 User Segments")

with st.spinner("Loading segment data..."):
    summary = APIClient.get_segment_summary()
    trends = APIClient.get_segment_trends(
        segment=None,
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date) if end_date else None
    )

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Segment Distribution")
    if summary:
        # Handle different summary response structures
        if isinstance(summary, list) and len(summary) > 0:
            df_summary = pd.DataFrame(summary)
        elif isinstance(summary, dict):
            # Check if it's a dict with 'segments' key
            if 'segments' in summary and isinstance(summary['segments'], list):
                df_summary = pd.DataFrame(summary['segments'])
            else:
                # Single segment dict - convert to DataFrame
                df_summary = pd.DataFrame([summary])
        else:
            df_summary = pd.DataFrame(summary)
        
        # Handle different column name variations
        if 'segment' in df_summary.columns and 'behavior_segment' not in df_summary.columns:
            df_summary = df_summary.rename(columns={'segment': 'behavior_segment'})
        
        # Calculate user counts from trends if not in summary
        if 'user_count' not in df_summary.columns and 'total_users' in df_summary.columns:
            df_summary['user_count'] = df_summary['total_users']
        elif 'user_count' not in df_summary.columns and trends:
            # Calculate from trends data
            df_trends_temp = pd.DataFrame(trends)
            if 'behavior_segment' in df_trends_temp.columns and 'total_users' in df_trends_temp.columns:
                user_counts = df_trends_temp.groupby('behavior_segment')['total_users'].sum().reset_index()
                user_counts.columns = ['behavior_segment', 'user_count']
                
                # Only merge if df_summary has behavior_segment column
                if 'behavior_segment' in df_summary.columns:
                    df_summary = df_summary.merge(user_counts, on='behavior_segment', how='left')
                else:
                    # If summary doesn't have behavior_segment, use trends data directly
                    df_summary = user_counts
        
        if not df_summary.empty and 'behavior_segment' in df_summary.columns and 'user_count' in df_summary.columns:
            st.plotly_chart(
                plot_segment_distribution(df_summary, "behavior_segment", "user_count", "Users by Segment"),
                use_container_width=True
            )
            
            # Display summary table
            display_cols = ["behavior_segment", "user_count"]
            if 'avg_conversion_rate' in df_summary.columns:
                display_cols.append("avg_conversion_rate")
            elif 'conversion_rate' in df_summary.columns:
                display_cols.append("conversion_rate")
            
            st.dataframe(
                df_summary[display_cols],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Segment distribution data not available.")
    else:
        st.info("No segment summary available.")

with col2:
    st.subheader("Segment Performance Trends")
    if trends:
        df_trends = pd.DataFrame(trends)
        
        # Ensure week column is datetime
        if 'week' in df_trends.columns:
            df_trends['week'] = pd.to_datetime(df_trends['week'])
        
        # Get available metrics
        available_metrics = []
        metric_options = {
            "Total Users": "total_users",
            "Conversion Rate": "conversion_rate",
            "Bounce Rate": "bounce_rate",
            "Avg Conversion Prob": "avg_conversion_prob",
            "Avg Bounce Prob": "avg_bounce_prob"
        }
        
        for label, col in metric_options.items():
            if col in df_trends.columns:
                available_metrics.append((label, col))
        
        if available_metrics and 'behavior_segment' in df_trends.columns:
            metric_label, metric_col = st.selectbox(
                "Select Metric",
                available_metrics,
                format_func=lambda x: x[0]
            )
            
            st.plotly_chart(
                plot_multi_line(df_trends, "week", metric_col, "behavior_segment", f"{metric_label} by Segment"),
                use_container_width=True
            )
        else:
            st.info("No valid metrics available for trend analysis.")
    else:
        st.info("No segment trend data available.")
