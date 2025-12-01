"""
Content-based recommendation module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Avoid pulling in TensorFlow/Keras from transformers; we only need the PyTorch stack.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

try:
    from sentence_transformers import SentenceTransformer
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    SentenceTransformer = None  # type: ignore[misc, assignment]

from .utils import RECOMMENDATIONS_DIR, get_logger, ensure_file_exists

logger = get_logger("recommendations.content_based")


class ContentBasedRecommender:
    """Compute content similarity between pages."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.page_metadata_path = RECOMMENDATIONS_DIR / "page_metadata.parquet"
        self.output_path = RECOMMENDATIONS_DIR / "content_recommendations.parquet"
        self.model_name = model_name
        self._encoder: Optional["SentenceTransformer"] = None

    def fit(self, top_n: int = 5) -> Path:
        """Train similarity model and persist recommendations."""
        ensure_file_exists(self.page_metadata_path)
        df = pd.read_parquet(self.page_metadata_path)
        df = df.dropna(subset=["page_id", "page_text"])

        if df.empty:
            raise ValueError("No page metadata available for content-based model.")

        if SentenceTransformer is None:
            logger.warning(
                "SentenceTransformer not available. Falling back to TF-IDF vectors."
            )
            vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
            matrix = vectorizer.fit_transform(df["page_text"].str.lower())
            similarity_matrix = cosine_similarity(matrix)
        else:
            logger.info(
                "Computing sentence embeddings (%s) for %s pages",
                self.model_name,
                len(df),
            )
            encoder = self._get_encoder()
            embeddings = encoder.encode(
                df["page_text"].tolist(),
                normalize_embeddings=True,
                convert_to_tensor=False,
            )
            similarity_matrix = cosine_similarity(embeddings)

        recommendations: List[Dict[str, object]] = []
        page_ids = df["page_id"].tolist()
        for idx, page_id in enumerate(page_ids):
            similarities = similarity_matrix[idx]
            related_indices = similarities.argsort()[::-1][1 : top_n + 1]
            for rank, other_idx in enumerate(related_indices, start=1):
                recommendations.append(
                    {
                        "page_id": page_id,
                        "recommended_page_id": page_ids[other_idx],
                        "score": float(similarities[other_idx]),
                        "rank": rank,
                        "method": "content",
                    }
                )

        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df.to_parquet(self.output_path, index=False)
        logger.info("Saved content-based recommendations to %s", self.output_path)
        return self.output_path

    def _get_encoder(self) -> "SentenceTransformer":
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for embedding-based recommendations."
            )
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def recommend(self, page_id: str, top_n: int = 5) -> pd.DataFrame:
        """Return top-N similar pages for a given page_id."""
        ensure_file_exists(self.output_path)
        df = pd.read_parquet(self.output_path)
        subset = df[df["page_id"] == page_id].nsmallest(top_n, "rank")
        return subset


if __name__ == "__main__":
    ContentBasedRecommender().fit()


