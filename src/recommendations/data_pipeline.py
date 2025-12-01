"""
Data preparation pipeline for the recommendation system.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

from .utils import (
    PROJECT_ROOT,
    DATA_DIR,
    RECOMMENDATIONS_DIR,
    get_logger,
    ensure_file_exists,
)

PAGE_DESCRIPTIONS = {
    "P001": "Homepage with product highlights, customer stories, hero CTA, and links into pricing, blog, and category pages.",
    "P002": "Category explorer showcasing solution bundles, feature comparisons, and filters for industries or company sizes.",
    "P003": "Product detail page describing core capabilities, integrations, screenshots, and onboarding experience.",
    "P004": "Blog article hub covering best practices, thought leadership, and educational guides for digital teams.",
    "P005": "Pricing page listing plan tiers, usage limits, billing FAQs, ROI calculator, and upgrade prompts.",
    "P006": "Checkout funnel page summarizing cart items, payment details, trust badges, and purchase confirmation steps.",
}

PAGE_KEYWORDS = {
    "home": "homepage overview hero banner testimonials quick links featured content",
    "category": "solution catalog filter comparison industry persona use-cases",
    "product": "features integrations demo screenshots specification value proposition",
    "blog": "article insights best practices education resources thought leadership",
    "pricing": "plan tiers cost calculator free trial billing faq discount",
    "checkout": "cart payment form trust badges order summary confirmation security",
}

logger = get_logger("recommendations.data_pipeline")


class RecommendationDataPipeline:
    """Prepare cleaned datasets used by the recommenders."""

    def __init__(self) -> None:
        self.sessions_file = DATA_DIR / "raw" / "sessions_raw.csv"
        self.pages_file = DATA_DIR / "raw" / "pages.csv"
        self.page_features_file = DATA_DIR / "processed" / "page_features.csv"
        self.session_features_file = DATA_DIR / "processed" / "session_features.csv"
        self.segments_file = DATA_DIR / "processed" / "user_segments_final.csv"
        self.outputs: Dict[str, Path] = {
            "interactions": RECOMMENDATIONS_DIR / "user_page_interactions.parquet",
            "page_metadata": RECOMMENDATIONS_DIR / "page_metadata.parquet",
            "segment_behavior": RECOMMENDATIONS_DIR / "segment_behavior.parquet",
        }

    def run(self) -> Dict[str, Path]:
        """Execute the full pipeline."""
        logger.info("Starting recommendation data pipeline")
        self._validate_inputs()

        sessions = self._load_sessions()
        page_metadata = self._build_page_metadata()
        interactions = self._build_interactions(sessions)
        segment_behavior = self._build_segment_behavior(sessions)

        page_metadata.to_parquet(self.outputs["page_metadata"], index=False)
        interactions.to_parquet(self.outputs["interactions"], index=False)
        segment_behavior.to_parquet(self.outputs["segment_behavior"], index=False)

        logger.info("Data pipeline complete")
        return self.outputs

    def _validate_inputs(self) -> None:
        for path in [
            self.sessions_file,
            self.pages_file,
            self.page_features_file,
            self.session_features_file,
            self.segments_file,
        ]:
            ensure_file_exists(path)

    def _load_sessions(self) -> pd.DataFrame:
        df = pd.read_csv(self.sessions_file)
        session_features = pd.read_csv(
            self.session_features_file,
            usecols=[
                "session_id",
                "click",
                "scroll",
                "add_to_cart",
                "purchase",
                "page_view",
            ],
        )
        df = df.merge(session_features, on="session_id", how="left")
        required_cols = {"user_id", "entry_page", "exit_page", "behavior_label"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Sessions file missing columns: {missing}")
        event_cols = ["click", "scroll", "add_to_cart", "purchase", "page_view"]
        df[event_cols] = df[event_cols].fillna(0)
        df["start_time"] = pd.to_datetime(df["start_time"])
        df["date"] = df["start_time"].dt.date
        latest_ts = df["start_time"].max()
        recency_days = (latest_ts - df["start_time"]).dt.days
        df["recency_weight"] = np.exp(-(recency_days / 30))
        return df

    def _build_page_metadata(self) -> pd.DataFrame:
        pages = pd.read_csv(self.pages_file)
        page_features = pd.read_csv(self.page_features_file)
        page_features = page_features.rename(
            columns={
                "page_views": "page_views",
                "avg_time_on_page": "avg_time_on_page",
                "avg_scroll_depth": "avg_scroll_depth",
                "click_count": "click_count",
            }
        )
        metadata = pages.merge(page_features, on="page_id", how="left")
        metadata["page_description"] = metadata["page_id"].map(PAGE_DESCRIPTIONS).fillna(
            metadata["page_type"].astype(str)
        )
        metadata["page_keywords"] = metadata["page_type"].map(PAGE_KEYWORDS).fillna("")
        metadata["page_text"] = metadata.apply(
            lambda row: " ".join(
                [
                    str(row["url_path"]),
                    str(row["page_type"]),
                    str(row["page_description"]),
                    str(row["page_keywords"]),
                    f"avg dwell {row.get('avg_dwell_time_sec', 'NA')} seconds",
                    f"avg time on page {row.get('avg_time_on_page', 'NA')}",
                    f"avg scroll depth {row.get('avg_scroll_depth', 'NA')}",
                ]
            ),
            axis=1,
        )
        return metadata

    def _build_interactions(self, sessions: pd.DataFrame) -> pd.DataFrame:
        sessions = sessions.copy()
        engagement = (
            0.4 * sessions["click"]
            + 0.3 * sessions["scroll"]
            + 0.8 * sessions["add_to_cart"]
            + 1.2 * sessions["purchase"]
            + 0.2 * sessions["page_view"]
        )
        sessions["entry_weight"] = (1.0 + engagement) * sessions["recency_weight"]
        sessions["exit_weight"] = (0.7 + 0.5 * engagement) * sessions["recency_weight"]

        entry_df = sessions[["user_id", "entry_page", "date", "entry_weight"]].rename(
            columns={"entry_page": "page_id", "entry_weight": "interaction_weight"}
        )
        entry_df["interaction_type"] = "entry"

        exit_df = sessions[["user_id", "exit_page", "date", "exit_weight"]].rename(
            columns={"exit_page": "page_id", "exit_weight": "interaction_weight"}
        )
        exit_df["interaction_type"] = "exit"

        interactions_df = pd.concat([entry_df, exit_df], ignore_index=True)

        aggregated = (
            interactions_df.groupby(["user_id", "page_id"])
            .agg(
                interaction_count=("interaction_weight", "size"),
                weight_sum=("interaction_weight", "sum"),
                last_seen=("date", "max"),
            )
            .reset_index()
        )
        return aggregated

    def _build_segment_behavior(self, sessions: pd.DataFrame) -> pd.DataFrame:
        segments = pd.read_csv(self.segments_file)
        segments = segments[["user_id", "behavior_segment"]]
        merged = sessions.merge(segments, on="user_id", how="left")
        segment_behavior = (
            merged.groupby(["behavior_segment", "entry_page"])
            .agg(
                sessions_cnt=("session_id", "count"),
                avg_duration=("duration_sec", "mean"),
                exits=("exit_page", lambda x: (x == x.name[1]).sum()),
            )
            .reset_index()
        )
        segment_behavior = segment_behavior.rename(
            columns={"entry_page": "page_id"}
        )
        return segment_behavior


if __name__ == "__main__":
    RecommendationDataPipeline().run()


