# orchestrator: train everything, save models
# ml/phase2_hybrid/run_train_all.py
from .data_prep import load_events, build_user_item, train_test_split_interactions
from .cluster_trainer import train_user_clusters
from .content_model import train_content_model
from .collaborative_model import train_collaborative
from .evaluator import evaluate_all
import os

def run_all():
    print("Loading events...")
    events = load_events(limit=None)
    if events.empty:
        print("No raw events found; abort.")
        return

    print("Building user-item matrix...")
    ui = build_user_item(events)
    if ui.empty:
        print("No interactions found; abort.")
        return

    train_df, test_df = train_test_split_interactions(ui, test_size=0.2)
    print(f"Train users: {len(train_df)}, Test users: {len(test_df)}")

    print("Training clusters...")
    kmeans, scaler, users_labeled = train_user_clusters(n_clusters=4)

    print("Training content model...")
    item_list = ui.columns.tolist()
    vectorizer, sim_matrix = train_content_model(item_list)

    print("Training collaborative model...")
    user_factors, item_factors = train_collaborative(train_df, n_components=20)

    print("Evaluating...")
    metrics = evaluate_all(train_df, test_df, k=10)
    print("Evaluation metrics:", metrics)
    print("Training complete. Models saved in ml/phase2_hybrid/models/ and ml/outputs/")

if __name__ == "__main__":
    run_all()
