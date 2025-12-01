import streamlit as st
from datetime import date, timedelta
from src.dashboard.services.api_client import APIClient

def render_sidebar():
    """
    Renders the sidebar with navigation and global filters.
    """
    with st.sidebar:
        st.title("📊 User Insights")
        
        # API Status Indicator
        if APIClient.get_health():
            st.success("API Connected", icon="🟢")
        else:
            st.error("API Offline", icon="🔴")
            st.warning("Ensure the FastAPI server is running.")

        st.divider()

        # Date Range Filter
        st.subheader("📅 Date Range")
        today = date.today()
        default_start = today - timedelta(days=30)
        
        date_range = st.date_input(
            "Select Range",
            value=(default_start, today),
            max_value=today,
            help="Select a date range to filter the data"
        )
        
        start_date = None
        end_date = None

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date = date_range[0]
            end_date = date_range[1]
        elif isinstance(date_range, date):
            # Single date selected
            start_date = date_range
            end_date = date_range
        
        st.divider()
        
        st.info(
            """
            **About**
            
            This dashboard visualizes synthetic user behavior data processed by our ML pipeline.
            """
        )
        
        return start_date, end_date
