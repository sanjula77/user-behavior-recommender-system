import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_path))

import streamlit as st
import pandas as pd
from src.dashboard.services.api_client import APIClient
from src.dashboard.components.sidebar import render_sidebar

st.set_page_config(page_title="Recommendations", page_icon="🧠", layout="wide")

# Sidebar (keeps navigation consistent)
_, _ = render_sidebar()

st.title("🧠 Recommendation Engine")
st.caption("Personalized suggestions for users and actionable insights for admins.")

col_user, col_page = st.columns(2)

with col_user:
    st.subheader("User Recommendations")
    user_id = st.text_input("Enter User ID", placeholder="e.g., U001")
    limit_user = st.slider("Results", min_value=1, max_value=20, value=5, key="user_rec_limit")
    
    if user_id:
        with st.spinner(f"Fetching recommendations for {user_id}..."):
            user_recs = APIClient.get_user_recommendations(user_id=user_id, limit=limit_user)
        recommendations = user_recs.get("recommendations", []) if user_recs else []
        
        if recommendations:
            df_user = pd.DataFrame(recommendations)
            st.dataframe(
                df_user[["recommended_page_id", "score", "hybrid_score", "rank", "method"]],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No recommendations found. Make sure the pipeline has been executed.")
    else:
        st.info("Provide a user ID to see personalized results.")

with col_page:
    st.subheader("Similar Pages")
    page_id = st.text_input("Enter Page ID", placeholder="e.g., P005")
    limit_page = st.slider("Top Similar Pages", min_value=1, max_value=20, value=5, key="page_rec_limit")
    
    if page_id:
        with st.spinner(f"Finding pages related to {page_id}..."):
            page_recs = APIClient.get_page_recommendations(page_id=page_id, limit=limit_page)
        recommendations = page_recs.get("recommendations", []) if page_recs else []
        
        if recommendations:
            df_page = pd.DataFrame(recommendations)
            st.dataframe(
                df_page[["recommended_page_id", "score", "hybrid_score", "rank", "method"]],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No similar pages found. Ensure the content-based model has been trained.")
    else:
        st.info("Provide a page ID to view similar content.")

st.divider()

col_admin, col_metrics = st.columns([2, 1])

with col_admin:
    st.subheader("Admin Recommendations")
    admin_recs = APIClient.get_admin_recommendations()
    if admin_recs:
        df_admin = pd.DataFrame(admin_recs)
        st.dataframe(
            df_admin[["category", "message", "score"]],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Run the admin recommendation generator to populate insights.")

with col_metrics:
    st.subheader("Evaluation Metrics")
    metrics = APIClient.get_recommendation_metrics()
    if metrics:
        st.metric("Hit Rate", f"{metrics.get('hit_rate', 0):.2%}")
        st.metric("NDCG", f"{metrics.get('ndcg', 0):.2f}")
        st.metric("MRR", f"{metrics.get('mrr', 0):.2f}")
        st.metric("Users Evaluated", metrics.get("users_evaluated", 0))
    else:
        st.info("Metrics unavailable. Execute evaluation to generate results.")


