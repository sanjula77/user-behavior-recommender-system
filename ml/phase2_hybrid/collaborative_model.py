# TruncatedSVD-based latent factors
# ml/phase2_hybrid/collaborative_model.py
import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from .save_load import save_model

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_collaborative(train_user_item_df, n_components=50):
    """
    train_user_item_df: DataFrame users x items (implicit weights)
    Returns user_factors (DataFrame) and item_factors (DataFrame)
    """
    if train_user_item_df.empty:
        raise ValueError("Cannot train collaborative model with empty training data")
    
    # convert to matrix
    users = train_user_item_df.index.tolist()
    items = train_user_item_df.columns.tolist()
    X = train_user_item_df.values  # shape (n_users, n_items)

    # Handle edge cases with very few users/items
    n_items = len(items)
    n_users = len(users)
    
    # TruncatedSVD requires at least 2 features (items)
    if n_items < 2:
        # Create simple identity-like factors when only 1 item exists
        print(f"Warning: Only {n_items} item(s) found. Using simple identity factors instead of SVD.")
        n_comp = min(n_components, n_users)
        if n_comp < 1:
            n_comp = 1
        
        # Create simple factors based on the interaction weights
        user_latent = X  # users x 1 (just use the interaction values)
        item_latent = np.ones((n_items, n_comp))  # 1 x n_comp
        
        # Pad if needed for compatibility
        if user_latent.shape[1] < n_comp:
            padding = np.zeros((n_users, n_comp - user_latent.shape[1]))
            user_latent = np.hstack([user_latent, padding])
        
        svd = None  # No SVD model to save
        user_factors = pd.DataFrame(user_latent, index=users)
        user_factors.index.name = "user_id"
        item_factors = pd.DataFrame(item_latent, index=items)
        item_factors.index.name = "item"
        
        # Don't save SVD model, but save factors
        user_factors.to_csv(os.path.join(MODEL_DIR, "user_factors.csv"))
        item_factors.to_csv(os.path.join(MODEL_DIR, "item_factors.csv"))
        print("Saved simple collaborative factors (SVD skipped due to insufficient items).")
    else:
        max_components = min(n_components, min(X.shape) - 1, len(users) - 1, len(items) - 1)
        if max_components < 1:
            # Fallback to identity-like factors for very small datasets
            max_components = 1
        
        # SVD on items dimension -> item latent vectors = V, user latent = U*S
        svd = TruncatedSVD(n_components=max_components, random_state=42)
        user_latent = svd.fit_transform(X)  # users x n_components
        item_latent = svd.components_.T     # items x n_components

        user_factors = pd.DataFrame(user_latent, index=users)
        user_factors.index.name = "user_id"
        item_factors = pd.DataFrame(item_latent, index=items)
        item_factors.index.name = "item"

        save_model(svd, os.path.join(MODEL_DIR, "svd_model.joblib"))
        user_factors.to_csv(os.path.join(MODEL_DIR, "user_factors.csv"))
        item_factors.to_csv(os.path.join(MODEL_DIR, "item_factors.csv"))
        print("Saved collaborative SVD model and latent factors.")
    
    return user_factors, item_factors
