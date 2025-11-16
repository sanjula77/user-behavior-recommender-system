# Save features to CSV / Parquet
# ml/pipeline/save_features.py
import os
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_df(df: pd.DataFrame, name: str):
    csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    pq_path = os.path.join(OUTPUT_DIR, f"{name}.parquet")
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(pq_path, index=False)
    except Exception as e:
        print("Parquet save failed:", e)
    print(f"Saved {name} to {csv_path} and parquet (if supported).")
