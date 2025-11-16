# precision@k, recall@k evaluation
# ml/phase2_hybrid/evaluator.py
from .hybrid import recommend_for_user
import numpy as np

def precision_at_k(recs, ground_truth, k=10):
    if not ground_truth:
        return 0.0
    rec_items = [r[0] for r in recs[:k]]
    hits = sum(1 for it in rec_items if it in ground_truth)
    return hits / k

def recall_at_k(recs, ground_truth, k=10):
    if not ground_truth:
        return 0.0
    rec_items = [r[0] for r in recs[:k]]
    hits = sum(1 for it in rec_items if it in ground_truth)
    return hits / len(ground_truth)

def evaluate_all(train_df, test_df, k=10, weights=(0.4,0.4,0.2)):
    if test_df.empty:
        return {
            "precision@K": 0.0,
            "recall@K": 0.0,
            "n_users": 0,
            "note": "Empty test set - cannot evaluate"
        }
    
    users = test_df.index.tolist()
    precisions = []
    recalls = []
    for user in users:
        gt = [it for it, val in test_df.loc[user].to_dict().items() if val > 0]
        recs = recommend_for_user(user, train_df, top_k=k, w_content=weights[0], w_collab=weights[1], w_behavior=weights[2])
        precisions.append(precision_at_k(recs, gt, k))
        recalls.append(recall_at_k(recs, gt, k))
    return {
        "precision@K": float(np.mean(precisions)),
        "recall@K": float(np.mean(recalls)),
        "n_users": len(users)
    }
