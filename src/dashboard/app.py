import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path))

import streamlit as st
from src.dashboard.config import APP_TITLE, APP_ICON

# This file acts as a redirect to the main Overview page
# Streamlit multipage apps default to the first file alphabetically or the one specified here.
# However, the best practice for custom navigation is to have the pages in the `pages/` directory.
# This file can be used for a landing page or just to set config.

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Welcome to User Behavior Insights")

st.markdown(
    """
    ### 🚀 Getting Started
    
    Select a page from the sidebar to explore the data:
    
    - **📈 Overview**: High-level KPIs and AI-generated summaries.
    - **📊 Trends**: Deep dive into daily and weekly user behavior trends.
    - **👥 Segments**: Analyze user segments and their performance.
    - **🔮 Predictions**: View ML model predictions for high-value and at-risk users.
    - **🚨 Anomalies**: Monitor detected data anomalies.
    
    ---
    """
)

st.info("👈 Open the sidebar to navigate.")
