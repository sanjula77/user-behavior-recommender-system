"""
Validation tests for recommendation quality, semantic accuracy, and ranking optimization.
These tests go beyond functional correctness to assess recommendation usefulness.
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


class RecommendationQualityValidator:
    """Validates recommendation quality, semantics, and ranking."""
    
    def __init__(self):
        self.results = []
        self.warnings = []
        
    def log_result(self, category: str, test_name: str, passed: bool, 
                   score: float = None, details: str = ""):
        """Log validation result."""
        status = "[PASS]" if passed else "[FAIL]"
        result = f"{status} [{category}] {test_name}"
        if score is not None:
            result += f" (Score: {score:.3f})"
        if details:
            result += f" - {details}"
        self.results.append(result)
        print(result)
    
    def log_warning(self, message: str):
        """Log warning (not a failure but worth noting)."""
        self.warnings.append(message)
        print(f"[WARN] {message}")
    
    def validate_semantic_relationships(self):
        """Validate that recommendations reflect semantically meaningful relationships."""
        print("\n" + "="*60)
        print("Validating Semantic Accuracy")
        print("="*60)
        
        # Define expected semantic relationships
        expected_relationships = {
            "P001": {  # Homepage
                "strong": ["P002", "P003", "P005"],  # Category, Product, Pricing
                "weak": ["P004", "P006"],  # Blog, Checkout
            },
            "P003": {  # Product
                "strong": ["P005", "P002", "P001"],  # Pricing, Category, Home
                "weak": ["P006", "P004"],  # Checkout, Blog
            },
            "P005": {  # Pricing
                "strong": ["P003", "P006", "P001"],  # Product, Checkout, Home
                "weak": ["P002", "P004"],  # Category, Blog
            },
            "P006": {  # Checkout
                "strong": ["P005", "P003"],  # Pricing, Product
                "weak": ["P001", "P002"],  # Home, Category
            },
        }
        
        hybrid_page_path = RECOMMENDATIONS_DIR / "hybrid_page_similarities.parquet"
        if not hybrid_page_path.exists():
            self.log_result("Semantic", "Page similarities file exists", False)
            return
        
        df = pd.read_parquet(hybrid_page_path)
        
        total_tests = 0
        passed_tests = 0
        
        for source_page, relationships in expected_relationships.items():
            page_recs = df[df["page_id"] == source_page].sort_values("rank")
            
            if len(page_recs) == 0:
                self.log_result("Semantic", f"{source_page} has recommendations", False)
                continue
            
            # Check top 3 recommendations include strong relationships
            top_3 = page_recs.head(3)["recommended_page_id"].tolist()
            strong_matches = sum(1 for rec in top_3 if rec in relationships["strong"])
            weak_matches = sum(1 for rec in top_3 if rec in relationships["weak"])
            
            total_tests += 1
            semantic_score = strong_matches / 3.0  # How many of top 3 are "strong" relationships
            
            passed = semantic_score >= 0.33  # At least 1 of top 3 should be strong
            passed_tests += passed
            
            self.log_result(
                "Semantic",
                f"{source_page} top recs include expected relationships",
                passed,
                semantic_score,
                f"Top 3: {top_3}, Expected strong: {relationships['strong']}, Found {strong_matches} strong matches"
            )
        
        overall_score = passed_tests / total_tests if total_tests > 0 else 0
        self.log_result(
            "Semantic",
            "Overall semantic accuracy",
            overall_score >= 0.5,
            overall_score,
            f"{passed_tests}/{total_tests} pages have semantically plausible recommendations"
        )
    
    def validate_recommendation_diversity(self):
        """Check that recommendations aren't too repetitive or homogenous."""
        print("\n" + "="*60)
        print("Validating Recommendation Diversity")
        print("="*60)
        
        hybrid_user_path = RECOMMENDATIONS_DIR / "hybrid_user_recommendations.parquet"
        if not hybrid_user_path.exists():
            self.log_result("Quality", "Hybrid user recommendations exist", False)
            return
        
        df = pd.read_parquet(hybrid_user_path)
        
        # Check page diversity across users
        page_counts = df.groupby("recommended_page_id").size()
        total_recs = len(df)
        page_distribution = page_counts / total_recs
        
        # Calculate entropy (higher = more diverse)
        import numpy as np
        entropy = -np.sum(page_distribution * np.log2(page_distribution + 1e-10))
        max_entropy = np.log2(len(page_distribution))
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0
        
        self.log_result(
            "Quality",
            "Recommendation diversity (entropy)",
            diversity_score >= 0.5,
            diversity_score,
            f"Entropy: {entropy:.3f} / {max_entropy:.3f}, All {len(page_distribution)} pages recommended"
        )
        
        # Check no single page dominates
        max_page_share = page_distribution.max()
        self.log_result(
            "Quality",
            "No page dominates recommendations",
            max_page_share <= 0.4,
            max_page_share,
            f"Most popular page: {page_counts.idxmax()} ({max_page_share*100:.1f}% of recs)"
        )
    
    def validate_score_distribution(self):
        """Validate that scores show meaningful variation (not all same)."""
        print("\n" + "="*60)
        print("Validating Score Distribution Quality")
        print("="*60)
        
        # Test content-based scores
        content_path = RECOMMENDATIONS_DIR / "content_recommendations.parquet"
        if content_path.exists():
            df = pd.read_parquet(content_path)
            
            score_std = df["score"].std()
            score_mean = df["score"].mean()
            cv = score_std / score_mean if score_mean > 0 else 0  # Coefficient of variation
            
            self.log_result(
                "Quality",
                "Content scores show variation",
                cv >= 0.1,  # At least 10% variation
                cv,
                f"Mean: {score_mean:.4f}, Std: {score_std:.4f}, CV: {cv:.3f}"
            )
        
        # Test hybrid scores
        hybrid_page_path = RECOMMENDATIONS_DIR / "hybrid_page_similarities.parquet"
        if hybrid_page_path.exists():
            df = pd.read_parquet(hybrid_page_path)
            
            score_std = df["hybrid_score"].std()
            score_mean = df["hybrid_score"].mean()
            cv = score_std / score_mean if score_mean > 0 else 0
            
            self.log_result(
                "Quality",
                "Hybrid scores show variation",
                cv >= 0.1,
                cv,
                f"Mean: {score_mean:.4f}, Std: {score_std:.4f}, CV: {cv:.3f}"
            )
    
    def validate_ranking_quality(self):
        """Validate that rankings make sense (higher scores = better ranks)."""
        print("\n" + "="*60)
        print("Validating Ranking Optimization")
        print("="*60)
        
        # Test page similarities ranking
        hybrid_page_path = RECOMMENDATIONS_DIR / "hybrid_page_similarities.parquet"
        if hybrid_page_path.exists():
            df = pd.read_parquet(hybrid_page_path)
            
            ranking_correct = True
            issues = []
            
            for page_id, group in df.groupby("page_id"):
                sorted_by_score = group.sort_values("hybrid_score", ascending=False)
                sorted_by_rank = group.sort_values("rank")
                
                # Check if ranking matches score ordering
                if not sorted_by_score["recommended_page_id"].equals(sorted_by_rank["recommended_page_id"]):
                    ranking_correct = False
                    issues.append(f"{page_id}: ranks don't match scores")
            
            self.log_result(
                "Ranking",
                "Ranks match score ordering",
                ranking_correct,
                details=f"{len(issues)} pages with ranking issues" if issues else "All pages ranked correctly"
            )
        
        # Test user recommendations ranking
        hybrid_user_path = RECOMMENDATIONS_DIR / "hybrid_user_recommendations.parquet"
        if hybrid_user_path.exists():
            df = pd.read_parquet(hybrid_user_path)
            
            # Sample a few users to check ranking quality
            sample_users = df["user_id"].unique()[:10]
            ranking_issues = 0
            
            for user_id in sample_users:
                user_recs = df[df["user_id"] == user_id]
                if len(user_recs) < 2:
                    continue
                    
                sorted_by_score = user_recs.sort_values("hybrid_score", ascending=False)
                sorted_by_rank = user_recs.sort_values("rank")
                
                if not sorted_by_score["recommended_page_id"].equals(sorted_by_rank["recommended_page_id"]):
                    ranking_issues += 1
            
            ranking_score = 1.0 - (ranking_issues / len(sample_users))
            self.log_result(
                "Ranking",
                "User recommendation ranking quality",
                ranking_score >= 0.8,
                ranking_score,
                f"{ranking_issues}/{len(sample_users)} users with ranking issues"
            )
    
    def validate_coverage_vs_quality_tradeoff(self):
        """Check that we have good coverage without sacrificing quality."""
        print("\n" + "="*60)
        print("Validating Coverage vs Quality Tradeoff")
        print("="*60)
        
        hybrid_user_path = RECOMMENDATIONS_DIR / "hybrid_user_recommendations.parquet"
        interactions_path = RECOMMENDATIONS_DIR / "user_page_interactions.parquet"
        
        if not hybrid_user_path.exists() or not interactions_path.exists():
            self.log_result("Quality", "Files exist for coverage check", False)
            return
        
        hybrid_df = pd.read_parquet(hybrid_user_path)
        interactions_df = pd.read_parquet(interactions_path)
        
        # Coverage: % of users with interactions who get recommendations
        users_with_interactions = set(interactions_df["user_id"].unique())
        users_with_recs = set(hybrid_df["user_id"].unique())
        coverage = len(users_with_recs & users_with_interactions) / len(users_with_interactions) if len(users_with_interactions) > 0 else 0
        
        self.log_result(
            "Quality",
            "User coverage (interactions -> recommendations)",
            coverage >= 0.9,
            coverage,
            f"{len(users_with_recs & users_with_interactions)}/{len(users_with_interactions)} users covered"
        )
        
        # Quality: Average score for collaborative vs fallback
        collaborative_scores = hybrid_df[hybrid_df["method"].str.contains("collaborative", na=False)]["hybrid_score"]
        fallback_scores = hybrid_df[hybrid_df["method"].str.contains("segment", na=False)]["hybrid_score"]
        
        if len(collaborative_scores) > 0:
            avg_collab = collaborative_scores.mean()
            self.log_result(
                "Quality",
                "Average collaborative recommendation score",
                avg_collab > 0.1,
                avg_collab,
                f"Mean score: {avg_collab:.4f}"
            )
        
        if len(fallback_scores) > 0:
            avg_fallback = fallback_scores.mean()
            self.log_result(
                "Quality",
                "Average fallback recommendation score",
                avg_fallback > 0.05,
                avg_fallback,
                f"Mean score: {avg_fallback:.4f} (expected lower than collaborative)"
            )
            
            if len(collaborative_scores) > 0:
                avg_collab = collaborative_scores.mean()
                fallback_quality_gap = avg_fallback / avg_collab if avg_collab > 0 else 0
                self.log_result(
                    "Quality",
                    "Fallback vs collaborative quality gap",
                    fallback_quality_gap >= 0.3,  # Fallback should be at least 30% of collaborative
                    fallback_quality_gap,
                    f"Fallback is {fallback_quality_gap*100:.1f}% of collaborative quality"
                )
    
    def validate_evaluation_metrics_quality(self):
        """Check that evaluation metrics indicate reasonable performance."""
        print("\n" + "="*60)
        print("Validating Evaluation Metrics Quality")
        print("="*60)
        
        metrics_path = RECOMMENDATIONS_DIR / "evaluation_metrics.json"
        if not metrics_path.exists():
            self.log_result("Quality", "Evaluation metrics file exists", False)
            return
        
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        # Check hit rate
        hit_rate = metrics.get("hit_rate", 0)
        self.log_result(
            "Quality",
            "Hit rate indicates useful predictions",
            hit_rate >= 0.1,  # At least 10% hit rate
            hit_rate,
            f"Hit rate: {hit_rate:.1%} (higher is better)"
        )
        
        # Check NDCG
        ndcg = metrics.get("ndcg", 0)
        self.log_result(
            "Quality",
            "NDCG indicates good ranking quality",
            ndcg >= 0.1,
            ndcg,
            f"NDCG: {ndcg:.3f} (1.0 = perfect, 0.0 = random)"
        )
        
        # Check hybrid vs collaborative comparison
        if "hybrid" in metrics and "collaborative" in metrics:
            hybrid_hr = metrics["hybrid"].get("hit_rate", 0)
            collab_hr = metrics["collaborative"].get("hit_rate", 0)
            
            hybrid_better = hybrid_hr >= collab_hr
            improvement = (hybrid_hr - collab_hr) / collab_hr if collab_hr > 0 else 0
            
            self.log_result(
                "Quality",
                "Hybrid outperforms collaborative (or matches)",
                hybrid_better or abs(improvement) < 0.1,  # Allow small difference
                improvement,
                f"Hybrid HR: {hybrid_hr:.1%}, Collaborative HR: {collab_hr:.1%}, Improvement: {improvement*100:+.1f}%"
            )
    
    def run_all_validations(self):
        """Run all quality validation checks."""
        print("\n" + "="*70)
        print("RECOMMENDATION QUALITY VALIDATION SUITE")
        print("="*70)
        
        self.validate_semantic_relationships()
        self.validate_recommendation_diversity()
        self.validate_score_distribution()
        self.validate_ranking_quality()
        self.validate_coverage_vs_quality_tradeoff()
        self.validate_evaluation_metrics_quality()
        
        # Summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        total = len(self.results)
        passed = len([r for r in self.results if "[PASS]" in r])
        failed = total - passed
        
        print(f"Total Validations: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if self.warnings:
            print(f"\nWarnings: {len(self.warnings)}")
            for warn in self.warnings[:5]:  # Show first 5
                print(f"  - {warn}")
        
        if failed == 0:
            print("\n[SUCCESS] All quality validations passed!")
        else:
            print(f"\n[INFO] {failed} validations failed - review recommendations for quality improvements")
        
        return failed == 0


if __name__ == "__main__":
    validator = RecommendationQualityValidator()
    success = validator.run_all_validations()
    sys.exit(0 if success else 1)

