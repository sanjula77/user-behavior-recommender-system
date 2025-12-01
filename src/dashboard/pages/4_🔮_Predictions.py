import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_path))

import streamlit as st
import pandas as pd
from src.dashboard.services.api_client import APIClient
from src.dashboard.components.sidebar import render_sidebar

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")

render_sidebar()

st.title("🔮 ML Predictions")

tab1, tab2 = st.tabs(["High Value Users", "At Risk Users"])

with tab1:
    st.subheader("Predicted High Value Users")
    st.markdown("Users predicted to make a purchase soon.")
    
    model = st.selectbox("Select Model", ["rf", "xgb", "lr"], format_func=lambda x: x.upper())
    
    with st.spinner(f"Loading predictions from {model.upper()} model..."):
        high_value = APIClient.get_high_value_users(model)
        
    if high_value:
        df = pd.DataFrame(high_value)
        
        # Determine probability column name
        prob_col = None
        for col in ['conversion_prob_rf', 'conversion_prob_xgb', 'conversion_prob_lr', 'prediction_prob', 'conversion_probability']:
            if col in df.columns:
                prob_col = col
                break
        
        # Configure column display
        column_config = {"user_id": "User ID"}
        if prob_col:
            column_config[prob_col] = st.column_config.ProgressColumn(
                "Conversion Probability",
                format="%.2f",
                min_value=0,
                max_value=1,
            )
        if 'behavior_segment' in df.columns:
            column_config['behavior_segment'] = "Segment"
        
        st.dataframe(
            df,
            use_container_width=True,
            column_config=column_config
        )
        
        # Show summary stats
        if prob_col:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Users", len(df))
            with col2:
                st.metric("Avg Probability", f"{df[prob_col].mean():.2%}")
            with col3:
                high_prob_count = len(df[df[prob_col] >= 0.7]) if prob_col else 0
                st.metric("High Prob Users (≥70%)", high_prob_count)
    else:
        st.info("No high-value predictions available.")

with tab2:
    st.subheader("Predicted At-Risk Users")
    st.markdown("Users predicted to bounce (leave without interacting).")
    
    with st.spinner("Loading predictions..."):
        at_risk = APIClient.get_at_risk_users()
        
    if at_risk:
        df = pd.DataFrame(at_risk)
        
        # Determine probability column name
        prob_col = None
        for col in ['bounce_prob', 'bounce_probability', 'prediction_prob']:
            if col in df.columns:
                prob_col = col
                break
        
        # Configure column display
        column_config = {"user_id": "User ID"}
        if prob_col:
            column_config[prob_col] = st.column_config.ProgressColumn(
                "Bounce Probability",
                format="%.2f",
                min_value=0,
                max_value=1,
            )
        if 'behavior_segment' in df.columns:
            column_config['behavior_segment'] = "Segment"
        
        st.dataframe(
            df,
            use_container_width=True,
            column_config=column_config
        )
        
        # Show summary stats
        if prob_col:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total At-Risk Users", len(df))
            with col2:
                st.metric("Avg Bounce Prob", f"{df[prob_col].mean():.2%}")
            with col3:
                high_risk_count = len(df[df[prob_col] >= 0.5]) if prob_col else 0
                st.metric("High Risk Users (≥50%)", high_risk_count)
    else:
        st.info("No at-risk predictions available.")
