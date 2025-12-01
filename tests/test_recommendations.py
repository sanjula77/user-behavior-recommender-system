"""
Comprehensive test suite for the recommendation system.
Validates data quality, recommendation accuracy, and system integrity.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.recommendations.utils import RECOMMENDATIONS_DIR, DATA_DIR


class RecommendationTester:
    """Test suite for recommendation system validation."""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
        
    def log_result(self, test_name: str, passed: bool, message: str = ""):
        """Log test result."""
        status = "[PASS]" if passed else "[FAIL]"
        result = f"{status}: {test_name}"
        if message:
            result += f" - {message}"
        self.test_results.append(result)
        if not passed:
            self.failed_tests.append(test_name)
        print(result)
    
    def test_data_pipeline_outputs(self):
        """Test data pipeline generates required files with valid schemas."""
        print("\n" + "="*60)
        print("Testing Data Pipeline Outputs")
        print("="*60)
        
        # Test page metadata
        page_meta_path = RECOMMENDATIONS_DIR / "page_metadata.parquet"
        if page_meta_path.exists():
            df = pd.read_parquet(page_meta_path)
            required_cols = ["page_id", "page_text"]
            has_cols = all(col in df.columns for col in required_cols)
            self.log_result(
                "Page metadata exists with required columns",
                has_cols,
                f"Found {len(df)} pages, columns: {list(df.columns)}" if has_cols else f"Missing: {set(required_cols) - set(df.columns)}"
            )
            self.log_result(
                "Page metadata non-empty",
                len(df) > 0,
                f"{len(df)} pages found"
            )
            self.log_result(
                "Page text is populated",
                df["page_text"].notna().all() and (df["page_text"].str.len() > 0).all(),
                f"All {len(df)} pages have non-empty text"
            )
        else:
            self.log_result("Page metadata file exists", False, "File not found")
        
        # Test interactions
        interactions_path = RECOMMENDATIONS_DIR / "user_page_interactions.parquet"
        if interactions_path.exists():
            df = pd.read_parquet(interactions_path)
            required_cols = ["user_id", "page_id", "weight_sum"]
            has_cols = all(col in df.columns for col in required_cols)
            self.log_result(
                "User-page interactions exist with required columns",
                has_cols,
                f"Found {len(df)} interactions" if has_cols else f"Missing: {set(required_cols) - set(df.columns)}"
            )
            self.log_result(
                "Interactions have valid weights",
                (df["weight_sum"] > 0).all() if "weight_sum" in df.columns else False,
                f"All {len(df)} interactions have positive weights" if "weight_sum" in df.columns else "weight_sum column missing"
            )
            self.log_result(
                "Interactions cover multiple users",
                df["user_id"].nunique() > 100,
                f"{df['user_id'].nunique()} unique users"
            )
        else:
            self.log_result("Interactions file exists", False, "File not found")
    
    def test_content_recommendations(self):
        """Test content-based recommendation quality."""
        print("\n" + "="*60)
        print("Testing Content-Based Recommendations")
        print("="*60)
        
        content_path = RECOMMENDATIONS_DIR / "content_recommendations.parquet"
        if content_path.exists():
            df = pd.read_parquet(content_path)
            
            self.log_result(
                "Content recommendations file exists",
                True,
                f"Found {len(df)} recommendations"
            )
            
            self.log_result(
                "Recommendations have valid scores",
                (df["score"] >= 0).all() and (df["score"] <= 1).all(),
                f"Score range: {df['score'].min():.4f} - {df['score'].max():.4f}"
            )
            
            self.log_result(
                "Non-zero similarity scores",
                (df["score"] > 0).any(),
                f"{len(df[df['score'] > 0])} recommendations with positive scores"
            )
            
            # Check each page has recommendations
            unique_pages = df["page_id"].nunique()
            page_meta_path = RECOMMENDATIONS_DIR / "page_metadata.parquet"
            if page_meta_path.exists():
                total_pages = len(pd.read_parquet(page_meta_path))
                self.log_result(
                    "All pages have recommendations",
                    unique_pages == total_pages,
                    f"{unique_pages}/{total_pages} pages have recommendations"
                )
            
            # Check rankings are valid
            rank_valid = df.groupby("page_id")["rank"].apply(
                lambda x: x.isin(range(1, len(x) + 1)).all()
            ).all()
            self.log_result(
                "Valid ranking structure",
                rank_valid,
                "Ranks are sequential per page"
            )
        else:
            self.log_result("Content recommendations file exists", False, "File not found")
    
    def test_collaborative_recommendations(self):
        """Test collaborative filtering recommendation quality."""
        print("\n" + "="*60)
        print("Testing Collaborative Recommendations")
        print("="*60)
        
        collab_path = RECOMMENDATIONS_DIR / "collaborative_recommendations.parquet"
        if collab_path.exists():
            df = pd.read_parquet(collab_path)
            
            self.log_result(
                "Collaborative recommendations file exists",
                True,
                f"Found {len(df)} recommendations"
            )
            
            self.log_result(
                "Recommendations have valid scores",
                (df["score"] >= 0).all(),
                f"Score range: {df['score'].min():.6f} - {df['score'].max():.6f}"
            )
            
            self.log_result(
                "User coverage",
                df["user_id"].nunique() > 100,
                f"{df['user_id'].nunique()} users have recommendations"
            )
            
            # Check ranking consistency
            rank_valid = df.groupby("user_id")["rank"].apply(
                lambda x: x.isin(range(1, len(x) + 1)).all() and x.nunique() == len(x)
            ).all()
            self.log_result(
                "Valid ranking per user",
                rank_valid,
                "Each user has unique, sequential ranks"
            )
        else:
            self.log_result("Collaborative recommendations file exists", False, "File not found")
    
    def test_hybrid_recommendations(self):
        """Test hybrid recommendation logic and fallback mechanisms."""
        print("\n" + "="*60)
        print("Testing Hybrid Recommendations")
        print("="*60)
        
        hybrid_user_path = RECOMMENDATIONS_DIR / "hybrid_user_recommendations.parquet"
        hybrid_page_path = RECOMMENDATIONS_DIR / "hybrid_page_similarities.parquet"
        
        # Test user recommendations
        if hybrid_user_path.exists():
            df = pd.read_parquet(hybrid_user_path)
            
            self.log_result(
                "Hybrid user recommendations exist",
                True,
                f"Found {len(df)} recommendations"
            )
            
            self.log_result(
                "Hybrid scores are valid",
                (df["hybrid_score"] >= 0).all() and (df["hybrid_score"] <= 1).all(),
                f"Hybrid score range: {df['hybrid_score'].min():.4f} - {df['hybrid_score'].max():.4f}"
            )
            
            # Check fallback mechanism works
            has_fallback = "segment_popular" in df["method"].values if "method" in df.columns else False
            self.log_result(
                "Fallback mechanism active",
                has_fallback,
                f"{len(df[df['method'].str.contains('segment', na=False)])} fallback recommendations" if has_fallback else "No fallback recommendations"
            )
            
            # Check method diversity
            methods = df["method"].unique() if "method" in df.columns else []
            self.log_result(
                "Multiple recommendation methods",
                len(methods) > 1,
                f"Methods: {list(methods)}"
            )
        else:
            self.log_result("Hybrid user recommendations exist", False, "File not found")
        
        # Test page similarities
        if hybrid_page_path.exists():
            df = pd.read_parquet(hybrid_page_path)
            self.log_result(
                "Hybrid page similarities exist",
                True,
                f"Found {len(df)} page similarities"
            )
            
            self.log_result(
                "Page similarities have valid scores",
                (df["hybrid_score"] >= 0).all() and (df["hybrid_score"] <= 1).all(),
                f"Score range: {df['hybrid_score'].min():.4f} - {df['hybrid_score'].max():.4f}"
            )
        else:
            self.log_result("Hybrid page similarities exist", False, "File not found")
    
    def test_admin_recommendations(self):
        """Test admin recommendation generation."""
        print("\n" + "="*60)
        print("Testing Admin Recommendations")
        print("="*60)
        
        admin_path = RECOMMENDATIONS_DIR / "admin_recommendations.parquet"
        if admin_path.exists():
            df = pd.read_parquet(admin_path)
            
            self.log_result(
                "Admin recommendations exist",
                True,
                f"Found {len(df)} recommendations"
            )
            
            required_cols = ["category", "message"]
            has_cols = all(col in df.columns for col in required_cols)
            self.log_result(
                "Admin recommendations have required fields",
                has_cols,
                f"Columns: {list(df.columns)}"
            )
            
            self.log_result(
                "Admin messages are non-empty",
                (df["message"].str.len() > 0).all() if "message" in df.columns else False,
                f"All {len(df)} messages are populated"
            )
        else:
            self.log_result("Admin recommendations exist", False, "File not found")
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics validity."""
        print("\n" + "="*60)
        print("Testing Evaluation Metrics")
        print("="*60)
        
        metrics_path = RECOMMENDATIONS_DIR / "evaluation_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            
            self.log_result(
                "Evaluation metrics file exists",
                True,
                f"Metrics: {list(metrics.keys())}"
            )
            
            # Check required metrics
            required_metrics = ["hit_rate", "ndcg", "users_evaluated"]
            has_metrics = all(m in metrics for m in required_metrics)
            self.log_result(
                "Required metrics present",
                has_metrics,
                f"Found: {list(metrics.keys())}"
            )
            
            # Validate metric ranges
            if "hit_rate" in metrics:
                hr = metrics["hit_rate"]
                self.log_result(
                    "Hit rate in valid range",
                    0 <= hr <= 1,
                    f"Hit rate: {hr:.4f}"
                )
            
            if "ndcg" in metrics:
                ndcg = metrics["ndcg"]
                self.log_result(
                    "NDCG in valid range",
                    0 <= ndcg <= 1,
                    f"NDCG: {ndcg:.4f}"
                )
            
            if "users_evaluated" in metrics:
                users = metrics["users_evaluated"]
                self.log_result(
                    "Users evaluated count reasonable",
                    users > 0,
                    f"{users} users evaluated"
                )
        else:
            self.log_result("Evaluation metrics file exists", False, "File not found")
    
    def test_specific_user_recommendations(self):
        """Test recommendations for specific known users."""
        print("\n" + "="*60)
        print("Testing Specific User Recommendations")
        print("="*60)
        
        hybrid_user_path = RECOMMENDATIONS_DIR / "hybrid_user_recommendations.parquet"
        if hybrid_user_path.exists():
            df = pd.read_parquet(hybrid_user_path)
            
            # Test U00001 (should have recommendations)
            user_recs = df[df["user_id"] == "U00001"]
            self.log_result(
                "User U00001 has recommendations",
                len(user_recs) > 0,
                f"Found {len(user_recs)} recommendations"
            )
            
            if len(user_recs) > 0:
                self.log_result(
                    "U00001 recommendations are ranked",
                    user_recs["rank"].min() == 1,
                    f"Top rank: {user_recs['rank'].min()}, Count: {len(user_recs)}"
                )
            
            # Test U00004 (should have fallback if no collaborative)
            user_recs_4 = df[df["user_id"] == "U00004"]
            self.log_result(
                "User U00004 has recommendations (collaborative or fallback)",
                len(user_recs_4) > 0,
                f"Found {len(user_recs_4)} recommendations"
            )
        else:
            self.log_result("Can test specific users", False, "Hybrid recommendations file missing")
    
    def test_specific_page_recommendations(self):
        """Test page similarity recommendations for specific pages."""
        print("\n" + "="*60)
        print("Testing Specific Page Recommendations")
        print("="*60)
        
        hybrid_page_path = RECOMMENDATIONS_DIR / "hybrid_page_similarities.parquet"
        content_path = RECOMMENDATIONS_DIR / "content_recommendations.parquet"
        
        # Test specific pages
        test_pages = ["P001", "P003", "P005"]
        
        for page_id in test_pages:
            if hybrid_page_path.exists():
                df = pd.read_parquet(hybrid_page_path)
                page_recs = df[df["page_id"] == page_id]
                
                self.log_result(
                    f"Page {page_id} has similarity recommendations",
                    len(page_recs) > 0,
                    f"Found {len(page_recs)} similar pages"
                )
                
                if len(page_recs) > 0:
                    # Check top recommendation exists
                    top_rec = page_recs[page_recs["rank"] == 1]
                    if len(top_rec) > 0:
                        self.log_result(
                            f"Page {page_id} has top recommendation",
                            True,
                            f"Top match: {top_rec.iloc[0]['recommended_page_id']} (score: {top_rec.iloc[0]['hybrid_score']:.4f})"
                        )
                    
                    # Check scores are reasonable
                    self.log_result(
                        f"Page {page_id} recommendations have valid scores",
                        (page_recs["hybrid_score"] > 0).all(),
                        f"Score range: {page_recs['hybrid_score'].min():.4f} - {page_recs['hybrid_score'].max():.4f}"
                    )
                    
                    # Check no self-recommendations
                    self.log_result(
                        f"Page {page_id} doesn't recommend itself",
                        (page_recs["recommended_page_id"] != page_id).all(),
                        "No self-references found"
                    )
            
            # Also test content-based directly
            if content_path.exists():
                df_content = pd.read_parquet(content_path)
                content_recs = df_content[df_content["page_id"] == page_id]
                
                if len(content_recs) > 0:
                    self.log_result(
                        f"Page {page_id} has content-based similarities",
                        True,
                        f"Found {len(content_recs)} content-based recommendations with scores {content_recs['score'].min():.4f}-{content_recs['score'].max():.4f}"
                    )
        
        # Test that all pages have recommendations
        if hybrid_page_path.exists():
            df = pd.read_parquet(hybrid_page_path)
            unique_source_pages = df["page_id"].nunique()
            page_meta_path = RECOMMENDATIONS_DIR / "page_metadata.parquet"
            if page_meta_path.exists():
                total_pages = len(pd.read_parquet(page_meta_path))
                self.log_result(
                    "All pages have similarity recommendations",
                    unique_source_pages == total_pages,
                    f"{unique_source_pages}/{total_pages} pages have recommendations"
                )
    
    def test_data_consistency(self):
        """Test cross-file data consistency."""
        print("\n" + "="*60)
        print("Testing Data Consistency")
        print("="*60)
        
        # Check page IDs are consistent across files
        page_meta_path = RECOMMENDATIONS_DIR / "page_metadata.parquet"
        content_path = RECOMMENDATIONS_DIR / "content_recommendations.parquet"
        
        if page_meta_path.exists() and content_path.exists():
            meta_pages = set(pd.read_parquet(page_meta_path)["page_id"].unique())
            content_pages = set(pd.read_parquet(content_path)["page_id"].unique())
            
            self.log_result(
                "Page IDs consistent between metadata and content recs",
                content_pages.issubset(meta_pages),
                f"Content recs: {len(content_pages)} pages, Metadata: {len(meta_pages)} pages"
            )
        
        # Check user IDs consistency
        interactions_path = RECOMMENDATIONS_DIR / "user_page_interactions.parquet"
        hybrid_user_path = RECOMMENDATIONS_DIR / "hybrid_user_recommendations.parquet"
        
        if interactions_path.exists() and hybrid_user_path.exists():
            interaction_users = set(pd.read_parquet(interactions_path)["user_id"].unique())
            hybrid_users = set(pd.read_parquet(hybrid_user_path)["user_id"].unique())
            
            # Hybrid should have at least as many users (with fallbacks)
            self.log_result(
                "User coverage maintained (interactions to hybrid)",
                len(hybrid_users) >= len(interaction_users) * 0.8,  # Allow some filtering
                f"Interactions: {len(interaction_users)} users, Hybrid: {len(hybrid_users)} users"
            )
    
    def run_all_tests(self):
        """Run all test suites."""
        print("\n" + "="*60)
        print("RECOMMENDATION SYSTEM TEST SUITE")
        print("="*60)
        
        self.test_data_pipeline_outputs()
        self.test_content_recommendations()
        self.test_collaborative_recommendations()
        self.test_hybrid_recommendations()
        self.test_admin_recommendations()
        self.test_evaluation_metrics()
        self.test_specific_user_recommendations()
        self.test_specific_page_recommendations()
        self.test_data_consistency()
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        total = len(self.test_results)
        passed = len([r for r in self.test_results if "[PASS]" in r])
        failed = len(self.failed_tests)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if self.failed_tests:
            print(f"\nFailed Tests:")
            for test in self.failed_tests:
                print(f"  - {test}")
        else:
            print("\n[SUCCESS] All tests passed!")
        
        return failed == 0


if __name__ == "__main__":
    tester = RecommendationTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

