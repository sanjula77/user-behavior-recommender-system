# model save/load helpers (joblib)
# ml/phase2_hybrid/save_load.py
import joblib
import os

def save_model(obj, path):
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    joblib.dump(obj, path)
