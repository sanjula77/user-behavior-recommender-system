"""
Evaluation metrics for the recommendation system.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict

from .utils import RECOMMENDATIONS_DIR, get_logger, ensure_file_exists, save_json

logger = get_logger("recommendations.evaluation")


class RecommendationEvaluator:
    def __init__(self) -> None:
        self.interactions_path = RECOMMENDATIONS_DIR / "user_page_interactions.parquet"
        self.collab_path = RECOMMENDATIONS_DIR / "collaborative_recommendations.parquet"
        self.hybrid_path = RECOMMENDATIONS_DIR / "hybrid_user_recommendations.parquet"
        self.output_path = RECOMMENDATIONS_DIR / "evaluation_metrics.json"

    def evaluate(self, k: int = 5, use_train_split: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Evaluate recommendations with proper train/test split to avoid data leakage.
        
        Args:
            k: Number of top recommendations to consider
            use_train_split: If True, generate recommendations from train data only (proper evaluation).
                           If False, use pre-computed recommendations (faster but may have leakage).
        """
        ensure_file_exists(self.interactions_path)

        interactions = pd.read_parquet(self.interactions_path)
        interactions["last_seen"] = pd.to_datetime(interactions.get("last_seen"))
        interactions = interactions.dropna(subset=["last_seen"])

        interactions = interactions.sort_values("last_seen")
        interaction_counts = interactions.groupby("user_id")["page_id"].transform("count")
        interactions = interactions[interaction_counts > 1]

        if interactions.empty:
            logger.warning("Insufficient interactions to evaluate recommendations.")
            empty_metrics = {
                "hit_rate": 0.0,
                "ndcg": 0.0,
                "mrr": 0.0,
                "users_evaluated": 0,
            }
            save_json(empty_metrics, self.output_path)
            return empty_metrics

        # Proper temporal split: last interaction per user = test
        test = interactions.groupby("user_id").tail(1)
        train = interactions.drop(test.index)
        
        if use_train_split:
            logger.info("Generating recommendations from train data only (no data leakage)")
            # Generate recommendations using ONLY training data
            from .collaborative import CollaborativeRecommender
            from .hybrid import HybridRecommender
            from .content_based import ContentBasedRecommender
            from .data_pipeline import RecommendationDataPipeline
            
            # Save original interactions and temporarily replace with train data
            import shutil
            backup_path = self.interactions_path.with_suffix(".parquet.backup")
            if not backup_path.exists():
                shutil.copy(self.interactions_path, backup_path)
            
            # Create train-only interactions file
            train.to_parquet(self.interactions_path, index=False)
            
            try:
                # Regenerate recommendations from train data only
                collab_train = CollaborativeRecommender()
                collab_train.fit()
                
                # Use existing content-based (doesn't depend on user interactions)
                content_train = ContentBasedRecommender()
                if not (RECOMMENDATIONS_DIR / "content_recommendations.parquet").exists():
                    content_train.fit()
                
                # Generate hybrid from train-based collaborative
                hybrid_train = HybridRecommender()
                hybrid_train.combine()
                
                collab_recs = pd.read_parquet(self.collab_path)
                hybrid_recs = pd.read_parquet(self.hybrid_path)
                
                # Restore original interactions
                shutil.copy(backup_path, self.interactions_path)
                
                logger.info("Train-based recommendations generated for evaluation")
            except Exception as e:
                # Restore original on error
                if backup_path.exists():
                    shutil.copy(backup_path, self.interactions_path)
                logger.warning(f"Failed to generate train-based recommendations: {e}")
                logger.warning("Falling back to pre-computed recommendations (may have leakage)")
                ensure_file_exists(self.collab_path)
                ensure_file_exists(self.hybrid_path)
                collab_recs = pd.read_parquet(self.collab_path)
                hybrid_recs = pd.read_parquet(self.hybrid_path)
        else:
            # Use pre-computed recommendations (faster but may include test data)
            logger.warning("Using pre-computed recommendations - may include test data (data leakage possible)")
            ensure_file_exists(self.collab_path)
            ensure_file_exists(self.hybrid_path)
            collab_recs = pd.read_parquet(self.collab_path)
            hybrid_recs = pd.read_parquet(self.hybrid_path)

        collab_metrics = self._evaluate_source(collab_recs, test, k)
        hybrid_metrics = self._evaluate_source(hybrid_recs, test, k)

        results = {
            "users_evaluated": collab_metrics["users_evaluated"],
            "collaborative": collab_metrics,
            "hybrid": hybrid_metrics,
            "hit_rate": hybrid_metrics["hit_rate"],
            "ndcg": hybrid_metrics["ndcg"],
            "mrr": hybrid_metrics["mrr"],
        }
        save_json(results, self.output_path)
        logger.info("Evaluation metrics saved to %s", self.output_path)
        return results

    def _evaluate_source(
        self, recommendations: pd.DataFrame, test_set: pd.DataFrame, k: int
    ) -> Dict[str, float]:
        if recommendations.empty:
            return {"hit_rate": 0.0, "ndcg": 0.0, "mrr": 0.0, "users_evaluated": 0}

        recs_topk = recommendations.sort_values("rank").groupby("user_id").head(k)
        hit_count = 0
        ndcg_sum = 0.0
        mrr_sum = 0.0
        users_evaluated = 0

        for user_id, grp in test_set.groupby("user_id"):
            relevant_items = set(grp["page_id"].tolist())
            user_recs = recs_topk[recs_topk["user_id"] == user_id]
            if user_recs.empty:
                continue
            users_evaluated += 1
            ranked_items = user_recs["recommended_page_id"].tolist()
            hit = any(item in relevant_items for item in ranked_items)
            if hit:
                hit_count += 1
            ndcg_sum += self._ndcg_at_k(ranked_items, relevant_items, k)
            mrr_sum += self._mrr_at_k(ranked_items, relevant_items)

        if users_evaluated == 0:
            return {"hit_rate": 0.0, "ndcg": 0.0, "mrr": 0.0, "users_evaluated": 0}

        return {
            "hit_rate": hit_count / users_evaluated,
            "ndcg": ndcg_sum / users_evaluated,
            "mrr": mrr_sum / users_evaluated,
            "users_evaluated": users_evaluated,
        }

    @staticmethod
    def _ndcg_at_k(ranked_items, relevant_items, k) -> float:
        dcg = 0.0
        for i, item in enumerate(ranked_items[:k]):
            if item in relevant_items:
                dcg += 1 / np.log2(i + 2)
        # ideal DCG equals 1 because we only have binary relevance
        return dcg

    @staticmethod
    def _mrr_at_k(ranked_items, relevant_items):
        for idx, item in enumerate(ranked_items, start=1):
            if item in relevant_items:
                return 1 / idx
        return 0.0


if __name__ == "__main__":
    RecommendationEvaluator().evaluate()


