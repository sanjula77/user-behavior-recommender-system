import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_path))

import streamlit as st
import pandas as pd
from src.dashboard.services.api_client import APIClient
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.charts import plot_bar_chart

st.set_page_config(page_title="Anomalies", page_icon="🚨", layout="wide")

start_date, end_date = render_sidebar()

st.title("🚨 Anomaly Detection")

with st.spinner("Loading anomaly data..."):
    summary = APIClient.get_anomaly_summary()
    anomalies = APIClient.get_anomalies(
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date) if end_date else None
    )

if summary:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Anomaly Days", summary.get("total_anomaly_days", summary.get("total_anomalies", 0)))
    col2.metric("Anomaly Rate", f"{summary.get('anomaly_rate', 0):.2f}%")
    
    # Get most common anomaly type
    anomaly_types = summary.get("anomaly_types", {})
    if anomaly_types:
        most_common = max(anomaly_types.items(), key=lambda x: x[1].get("total", 0) if isinstance(x[1], dict) else 0)
        col3.metric("Most Common Type", most_common[0] if most_common else "N/A")
    else:
        col3.metric("Total Days", summary.get("total_days", 0))

st.divider()

if anomalies:
    df = pd.DataFrame(anomalies)
    
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    st.subheader("Detected Anomalies")
    
    # Filter anomalies - show only days with anomalies
    anomaly_cols = [col for col in df.columns if 'anomaly' in col.lower() and col != 'has_anomaly' and not col.endswith('_type')]
    
    if anomaly_cols:
        # Create a filter for anomaly types
        show_all = st.checkbox("Show all days (including non-anomaly)", value=False)
        
        if not show_all:
            # Filter to only anomaly days
            df = df[df['has_anomaly'] == True] if 'has_anomaly' in df.columns else df
    
    # Display key columns
    display_cols = ['date'] if 'date' in df.columns else []
    
    # Add anomaly columns
    for col in ['avg_conversion_rf_anomaly', 'avg_conversion_xgb_anomaly', 'avg_bounce_anomaly', 'has_anomaly']:
        if col in df.columns:
            display_cols.append(col)
    
    # Add value columns
    for col in ['avg_conversion_rf', 'avg_conversion_xgb', 'avg_bounce']:
        if col in df.columns:
            display_cols.append(col)
    
    if display_cols:
        st.dataframe(
            df[display_cols],
            use_container_width=True,
            column_config={
                "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            }
        )
    else:
        st.dataframe(df, use_container_width=True)
    
    # Visualizing anomalies over time
    if "date" in df.columns and 'has_anomaly' in df.columns:
        st.subheader("Anomalies Over Time")
        chart_data = df.groupby("date")['has_anomaly'].sum().reset_index(name="count")
        chart_data['date'] = pd.to_datetime(chart_data['date'])
        st.plotly_chart(
            plot_bar_chart(chart_data, "date", "count", "Daily Anomaly Count"),
            use_container_width=True
        )
    
    # Show anomaly breakdown by type
    if anomaly_cols:
        st.subheader("Anomaly Breakdown")
        anomaly_summary = {}
        for col in anomaly_cols:
            if col in df.columns:
                count = int(df[col].sum()) if df[col].dtype in ['bool', 'int64'] else 0
                if count > 0:
                    anomaly_summary[col.replace('_anomaly', '').replace('_', ' ').title()] = count
        
        if anomaly_summary:
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(pd.DataFrame(list(anomaly_summary.items()), columns=['Metric', 'Count']), x='Metric', y='Count')
        
else:
    st.info("No anomalies detected in the selected period.")
