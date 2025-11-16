# Hybrid scoring & top-K recommend function
# ml/phase2_hybrid/hybrid.py
import numpy as np
import pandas as pd
import os
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

def load_models():
    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    sim_matrix = np.load(os.path.join(MODEL_DIR, "item_sim_matrix.npy"))
    svd = None
    try:
        svd = joblib.load(os.path.join(MODEL_DIR, "svd_model.joblib"))
    except Exception:
        # SVD model might not exist if there were insufficient items
        pass
    user_factors = pd.read_csv(os.path.join(MODEL_DIR, "user_factors.csv"), index_col=0)
    item_factors = pd.read_csv(os.path.join(MODEL_DIR, "item_factors.csv"), index_col=0)
    user_cluster = None
    try:
        user_cluster = pd.read_csv(os.path.join(OUTPUT_DIR, "user_features_with_cluster.csv"), index_col=False)
    except Exception:
        pass
    return tfidf, sim_matrix, svd, user_factors, item_factors, user_cluster

def recommend_for_user(user_id, user_item_df, top_k=10,
                       w_content=0.4, w_collab=0.4, w_behavior=0.2):
    """
    user_item_df: train user-item df (users x items)
    """

    tfidf, sim_matrix, svd, user_factors, item_factors, user_cluster = load_models()
    items = user_item_df.columns.tolist()

    # 1) content score: average similarity of candidate item to items user touched
    if user_id not in user_item_df.index:
        interacted_items = []
    else:
        row = user_item_df.loc[user_id]
        interacted_items = [it for it, val in row.to_dict().items() if val > 0]

    content_scores = np.zeros(len(items))
    if interacted_items:
        idx_map = {it:i for i,it in enumerate(items)}
        interacted_idx = [idx_map[it] for it in interacted_items if it in idx_map]
        if interacted_idx:
            # average similarity
            content_scores = sim_matrix[:, interacted_idx].mean(axis=1)

    # 2) collaborative score: dot(user_factors, item_factors)
    collab_scores = np.zeros(len(items))
    if user_id in user_factors.index:
        uvec = user_factors.loc[user_id].values
        collab_scores = item_factors.values.dot(uvec)
        # normalize
    # 3) behavior score: use simple proxy: total sessions or avg engagement
    behavior_scores = np.zeros(len(items))
    try:
        users_df = pd.read_csv(os.path.join(OUTPUT_DIR, "user_features.csv"))
        urow = users_df[users_df["user_id"] == user_id]
        if not urow.empty:
            engagement = float(urow["avg_session_duration"].iloc[0])  # example
            behavior_scores = np.full(len(items), engagement)
    except Exception:
        pass

    # normalize all scores
    def norm(x):
        if x.max() == x.min():
            return np.zeros_like(x)
        x = (x - x.min()) / (x.max() - x.min() + 1e-9)
        return x

    c = norm(content_scores)
    p = norm(collab_scores)
    b = norm(behavior_scores)

    final = w_content * c + w_collab * p + w_behavior * b
    top_idx = np.argsort(-final)[:top_k]
    recommendations = [items[i] for i in top_idx]
    scores = [float(final[i]) for i in top_idx]
    return list(zip(recommendations, scores))
