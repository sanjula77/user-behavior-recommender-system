"""
Utility script to run the full recommendation pipeline.
"""
from __future__ import annotations

from .data_pipeline import RecommendationDataPipeline
from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender
from .hybrid import HybridRecommender
from .admin_insights import AdminRecommendations
from .evaluation import RecommendationEvaluator
from .utils import get_logger

logger = get_logger("recommendations.runner")


def run_pipeline() -> None:
    pipeline = RecommendationDataPipeline()
    pipeline.run()

    content = ContentBasedRecommender()
    content.fit()

    collab = CollaborativeRecommender()
    collab.fit()

    hybrid = HybridRecommender()
    hybrid.combine()

    admin = AdminRecommendations()
    admin.generate()

    evaluator = RecommendationEvaluator()
    evaluator.evaluate()

    logger.info("Recommendation system pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()


