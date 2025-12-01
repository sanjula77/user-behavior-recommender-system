"""
Generate admin-focused recommendations based on behavioral analytics.
"""
from __future__ import annotations

import pandas as pd
from typing import List, Dict

from .utils import RECOMMENDATIONS_DIR, DATA_DIR, get_logger, ensure_file_exists

logger = get_logger("recommendations.admin")


class AdminRecommendations:
    """Produce actionable insights for admins."""

    def __init__(self) -> None:
        self.sessions_path = DATA_DIR / "raw" / "sessions_raw.csv"
        self.segment_behavior_path = RECOMMENDATIONS_DIR / "segment_behavior.parquet"
        self.output_path = RECOMMENDATIONS_DIR / "admin_recommendations.parquet"

    def generate(self) -> pd.DataFrame:
        ensure_file_exists(self.sessions_path)
        ensure_file_exists(self.segment_behavior_path)

        sessions = pd.read_csv(self.sessions_path)
        segments = pd.read_parquet(self.segment_behavior_path)

        insights: List[Dict[str, object]] = []

        # Recommendation 1: pages with high exit rate
        exit_counts = sessions.groupby("exit_page").size()
        entry_counts = sessions.groupby("entry_page").size()
        exit_rate = (exit_counts / entry_counts).fillna(0).sort_values(ascending=False)
        top_exit = exit_rate.head(5)
        for page_id, rate in top_exit.items():
            insights.append(
                {
                    "category": "Page Optimization",
                    "message": f"Optimize page {page_id} - exit rate {rate:.2f}.",
                    "score": float(min(rate, 1.0)),
                }
            )

        # Recommendation 2: segments with low conversion
        segment_summary = (
            segments.groupby("behavior_segment")
            .agg(avg_duration=("avg_duration", "mean"), sessions_cnt=("sessions_cnt", "sum"))
            .reset_index()
        )
        low_engagement = segment_summary.sort_values("avg_duration").head(5)
        for _, row in low_engagement.iterrows():
            insights.append(
                {
                    "category": "Segment Engagement",
                    "message": f"Add internal links for {row['behavior_segment']} (avg duration {row['avg_duration']:.1f}s).",
                    "score": 0.8,
                }
            )

        # Recommendation 3: device/channel issues
        device_exit = sessions.groupby("device_type")["exit_page"].count().reset_index()
        worst_device = device_exit.sort_values("exit_page", ascending=False).head(1)
        if not worst_device.empty:
            device = worst_device.iloc[0]["device_type"]
            insights.append(
                {
                    "category": "Experience",
                    "message": f"Improve performance for {device} users - highest exit count.",
                    "score": 0.7,
                }
            )

        df = pd.DataFrame(insights)
        df.to_parquet(self.output_path, index=False)
        logger.info("Saved admin recommendations to %s", self.output_path)
        return df


