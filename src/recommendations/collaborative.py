"""
Collaborative filtering recommender.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .utils import RECOMMENDATIONS_DIR, get_logger, ensure_file_exists

logger = get_logger("recommendations.collaborative")


class CollaborativeRecommender:
    """User-based collaborative filtering using cosine similarity."""

    def __init__(self) -> None:
        self.interactions_path = RECOMMENDATIONS_DIR / "user_page_interactions.parquet"
        self.output_path = RECOMMENDATIONS_DIR / "collaborative_recommendations.parquet"

    def fit(self, top_n: int = 5) -> Path:
        """Train collaborative filtering model and persist recommendations."""
        ensure_file_exists(self.interactions_path)
        interactions = pd.read_parquet(self.interactions_path)

        pivot = interactions.pivot_table(
            index="user_id", columns="page_id", values="weight_sum", fill_value=0
        )
        if pivot.empty:
            raise ValueError("No interaction data available for collaborative model.")

        user_ids = pivot.index.tolist()
        page_ids = pivot.columns.tolist()
        matrix = pivot.values

        logger.info("Computing user-user cosine similarity for %s users", len(user_ids))
        similarity = cosine_similarity(matrix)

        recommendations: List[Dict[str, object]] = []
        for i, user in enumerate(user_ids):
            similar_users = similarity[i].copy()
            similar_users[i] = 0  # exclude self
            top_users_idx = similar_users.argsort()[::-1][:10]
            user_scores = np.zeros(len(page_ids))
            for idx in top_users_idx:
                weight = similar_users[idx]
                if weight <= 0:
                    continue
                user_scores += weight * matrix[idx]
            already_seen = matrix[i] > 0
            user_scores[already_seen] = 0
            top_items_idx = user_scores.argsort()[::-1][:top_n]
            for rank, item_idx in enumerate(top_items_idx, start=1):
                score = user_scores[item_idx]
                if score <= 0:
                    continue
                recommendations.append(
                    {
                        "user_id": user,
                        "recommended_page_id": page_ids[item_idx],
                        "score": float(score),
                        "rank": rank,
                        "method": "collaborative",
                    }
                )

        rec_df = pd.DataFrame(recommendations)
        rec_df.to_parquet(self.output_path, index=False)
        logger.info("Saved collaborative recommendations to %s", self.output_path)
        return self.output_path

    def recommend(self, user_id: str, top_n: int = 5) -> pd.DataFrame:
        """Return top-N recommended pages for a given user."""
        ensure_file_exists(self.output_path)
        df = pd.read_parquet(self.output_path)
        subset = df[df["user_id"] == user_id].nsmallest(top_n, "rank")
        return subset


if __name__ == "__main__":
    CollaborativeRecommender().fit()


