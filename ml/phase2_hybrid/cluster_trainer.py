# Train & save KMeans clusterer
# ml/phase2_hybrid/cluster_trainer.py
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .save_load import save_model

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_user_clusters(n_clusters=4):
    user_path = os.path.join(OUTPUT_DIR, "user_features.csv")
    if not os.path.exists(user_path):
        print(f"Warning: {user_path} not found. Skipping cluster training.")
        return None, None, pd.DataFrame()
    
    df = pd.read_csv(user_path)
    if df.empty:
        print("Warning: user_features.csv is empty. Skipping cluster training.")
        return None, None, pd.DataFrame()
    
    features = df.drop(columns=["user_id"], errors="ignore").fillna(0)
    
    # Adjust n_clusters if we have fewer users than clusters
    n_users = len(df)
    actual_clusters = min(n_clusters, n_users)
    
    if actual_clusters < 1:
        print("Warning: No users to cluster. Skipping cluster training.")
        return None, None, df
    
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=actual_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    df["cluster"] = labels

    # save models: scaler + kmeans + labeled users
    save_model(scaler, os.path.join(MODEL_DIR, "cluster_scaler.joblib"))
    save_model(kmeans, os.path.join(MODEL_DIR, "kmeans.joblib"))
    df.to_csv(os.path.join(OUTPUT_DIR, "user_features_with_cluster.csv"), index=False)
    print(f"Trained KMeans with {n_clusters} clusters, saved labeled users and models.")
    return kmeans, scaler, df
