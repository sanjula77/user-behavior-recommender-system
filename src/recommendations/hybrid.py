"""
Hybrid recommendation logic combining content and collaborative scores.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .utils import DATA_DIR, RECOMMENDATIONS_DIR, get_logger, ensure_file_exists

logger = get_logger("recommendations.hybrid")


class HybridRecommender:
    """Blend content-based and collaborative recommendations."""

    def __init__(self) -> None:
        self.content_path = RECOMMENDATIONS_DIR / "content_recommendations.parquet"
        self.collab_path = RECOMMENDATIONS_DIR / "collaborative_recommendations.parquet"
        self.output_user_path = RECOMMENDATIONS_DIR / "hybrid_user_recommendations.parquet"
        self.output_page_path = RECOMMENDATIONS_DIR / "hybrid_page_similarities.parquet"
        self.user_segments_path = DATA_DIR / "processed" / "user_segments_final.csv"
        self.segment_behavior_path = RECOMMENDATIONS_DIR / "segment_behavior.parquet"

    def combine(
        self,
        content_weight: float = 0.4,
        collab_weight: float = 0.6,
        fallback_weight: float = 0.3,
        fallback_limit: int = 5,
    ) -> Dict[str, pd.DataFrame]:
        """Normalize and merge scores from both recommenders."""
        ensure_file_exists(self.content_path)
        ensure_file_exists(self.collab_path)
        ensure_file_exists(self.user_segments_path)
        ensure_file_exists(self.segment_behavior_path)
        content_df = pd.read_parquet(self.content_path)
        collab_df = pd.read_parquet(self.collab_path)
        segments_df = pd.read_csv(self.user_segments_path)[
            ["user_id", "behavior_segment"]
        ]
        segment_popularity = pd.read_parquet(self.segment_behavior_path)

        if content_df.empty and collab_df.empty:
            raise ValueError("Content and collaborative recommendations are empty.")

        segment_groups, global_popularity = self._prepare_segment_popularity(
            segment_popularity, fallback_limit
        )

        user_recs = self._build_user_recommendations(
            collab_df=collab_df,
            collab_weight=collab_weight,
            segments_df=segments_df,
            segment_groups=segment_groups,
            global_popularity=global_popularity,
            fallback_weight=fallback_weight,
        )
        user_recs.to_parquet(self.output_user_path, index=False)

        if content_df.empty:
            logger.warning(
                "Content-based recommendations are empty. "
                "Hybrid page similarities will not be generated."
            )
            page_recs = pd.DataFrame(
                columns=["page_id", "recommended_page_id", "score", "hybrid_score", "rank", "method"]
            )
            page_recs.to_parquet(self.output_page_path, index=False)
        else:
            content_df = self._normalize_scores(
                content_df, group_cols=["page_id"], score_col="score"
            )
            content_df["hybrid_score"] = content_df["score_norm"] * content_weight
            page_recs = content_df.drop(columns=["score_norm"])
            page_recs.to_parquet(self.output_page_path, index=False)

        logger.info(
            "Hybrid recommendations saved (%s, %s)",
            self.output_user_path,
            self.output_page_path,
        )
        return {
            "user": user_recs,
            "page": page_recs,
        }

    @staticmethod
    def _normalize_scores(
        df: pd.DataFrame, group_cols: list, score_col: str
    ) -> pd.DataFrame:
        """Normalize scores per group to [0, 1]."""
        df = df.copy()
        df["score_norm"] = df.groupby(group_cols)[score_col].transform(
            lambda x: x / (x.max() or 1)
        )
        return df

    def _build_user_recommendations(
        self,
        collab_df: pd.DataFrame,
        collab_weight: float,
        segments_df: pd.DataFrame,
        segment_groups: Dict[str, List[Dict[str, object]]],
        global_popularity: List[Dict[str, object]],
        fallback_weight: float,
    ) -> pd.DataFrame:
        columns = [
            "user_id",
            "recommended_page_id",
            "score",
            "hybrid_score",
            "rank",
            "method",
        ]
        if collab_df.empty:
            logger.warning(
                "Collaborative recommendations are empty. Falling back to segment popularity."
            )
            user_recs = pd.DataFrame(columns=columns)
        else:
            collab_df = self._normalize_scores(
                collab_df, group_cols=["user_id"], score_col="score"
            )
            collab_df["hybrid_score"] = collab_df["score_norm"] * collab_weight
            user_recs = collab_df.drop(columns=["score_norm"])

        existing_users = set(user_recs["user_id"].unique())
        segment_lookup = dict(
            zip(segments_df["user_id"].tolist(), segments_df["behavior_segment"].tolist())
        )
        missing_users = [
            user_id
            for user_id in segment_lookup.keys()
            if user_id not in existing_users
        ]
        fallback_df = self._generate_fallback_recommendations(
            missing_users=missing_users,
            segment_lookup=segment_lookup,
            segment_groups=segment_groups,
            global_popularity=global_popularity,
            fallback_weight=fallback_weight,
        )
        if not fallback_df.empty:
            user_recs = (
                pd.concat([user_recs, fallback_df], ignore_index=True)
                if not user_recs.empty
                else fallback_df
            )

        if user_recs.empty:
            return pd.DataFrame(columns=columns)

        user_recs = (
            user_recs.sort_values(
                ["user_id", "hybrid_score"], ascending=[True, False]
            )
            .drop_duplicates(subset=["user_id", "recommended_page_id"])
        )
        user_recs["rank"] = user_recs.groupby("user_id").cumcount() + 1
        return user_recs

    def _prepare_segment_popularity(
        self, segment_behavior: pd.DataFrame, limit: int
    ) -> tuple[Dict[str, List[Dict[str, object]]], List[Dict[str, object]]]:
        if segment_behavior.empty:
            return {}, []
        sorted_behavior = segment_behavior.dropna(subset=["behavior_segment"]).sort_values(
            ["behavior_segment", "sessions_cnt"], ascending=[True, False]
        )
        segment_groups = {
            segment: grp.head(limit)[["page_id", "sessions_cnt"]].to_dict("records")
            for segment, grp in sorted_behavior.groupby("behavior_segment")
        }
        global_popularity = (
            segment_behavior.groupby("page_id")["sessions_cnt"]
            .sum()
            .sort_values(ascending=False)
            .head(limit)
            .reset_index()[["page_id", "sessions_cnt"]]
            .to_dict("records")
        )
        return segment_groups, global_popularity

    def _generate_fallback_recommendations(
        self,
        missing_users: List[str],
        segment_lookup: Dict[str, str],
        segment_groups: Dict[str, List[Dict[str, object]]],
        global_popularity: List[Dict[str, object]],
        fallback_weight: float,
    ) -> pd.DataFrame:
        if not missing_users:
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "recommended_page_id",
                    "score",
                    "hybrid_score",
                    "rank",
                    "method",
                ]
            )

        rows: List[Dict[str, object]] = []
        for user_id in missing_users:
            segment = segment_lookup.get(user_id)
            candidates = segment_groups.get(segment)
            method = "segment_popular"
            if not candidates:
                candidates = global_popularity
                method = "global_popular"
            if not candidates:
                continue
            for idx, candidate in enumerate(candidates, start=1):
                rows.append(
                    {
                        "user_id": user_id,
                        "recommended_page_id": candidate["page_id"],
                        "score": None,
                        "hybrid_score": fallback_weight * (1 / idx),
                        "rank": idx,
                        "method": method,
                    }
                )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "recommended_page_id",
                    "score",
                    "hybrid_score",
                    "rank",
                    "method",
                ]
            )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    HybridRecommender().combine()


