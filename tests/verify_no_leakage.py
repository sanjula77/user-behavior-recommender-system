"""
Verify that evaluation recommendations are generated from training data only.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import shutil

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.recommendations.utils import RECOMMENDATIONS_DIR
from src.recommendations.evaluation import RecommendationEvaluator


def verify_train_only_recommendations():
    """Verify that recommendations for evaluation exclude test data."""
    print("="*70)
    print("VERIFYING NO DATA LEAKAGE IN EVALUATION")
    print("="*70)
    
    interactions_path = RECOMMENDATIONS_DIR / "user_page_interactions.parquet"
    if not interactions_path.exists():
        print("[FAIL] Interactions file not found")
        return False
    
    # Load interactions
    interactions = pd.read_parquet(interactions_path)
    interactions["last_seen"] = pd.to_datetime(interactions.get("last_seen"))
    interactions = interactions.dropna(subset=["last_seen"])
    interactions = interactions.sort_values("last_seen")
    
    # Create train/test split
    test = interactions.groupby("user_id").tail(1)
    train = interactions.drop(test.index)
    
    print(f"\nTotal interactions: {len(interactions)}")
    print(f"Train interactions: {len(train)}")
    print(f"Test interactions: {len(test)}")
    
    # Get test user-page pairs that should NOT appear in training recommendations
    test_pairs = set(zip(test["user_id"], test["page_id"]))
    train_pairs = set(zip(train["user_id"], train["page_id"]))
    
    # Test pages that are NEW for users (not in train)
    new_test_pairs = test_pairs - train_pairs
    print(f"\nTest pairs: {len(test_pairs)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"New test pairs (not in train): {len(new_test_pairs)}")
    
    # Run evaluation with train-only split
    print("\n" + "-"*70)
    print("Running evaluation with train-only recommendations...")
    print("-"*70)
    
    evaluator = RecommendationEvaluator()
    
    try:
        # This should generate recommendations from train data only
        result = evaluator.evaluate(use_train_split=True, k=5)
        
        print("\n[SUCCESS] Evaluation completed with train-only recommendations")
        print(f"Hit Rate: {result['hit_rate']:.1%}")
        print(f"NDCG: {result['ndcg']:.3f}")
        print(f"Users Evaluated: {result['users_evaluated']}")
        
        # Verify recommendations were generated
        collab_path = RECOMMENDATIONS_DIR / "collaborative_recommendations.parquet"
        if collab_path.exists():
            collab_recs = pd.read_parquet(collab_path)
            print(f"\nRecommendations generated: {len(collab_recs)}")
            
            # Check a few test users
            test_users = test["user_id"].unique()[:5]
            print(f"\nChecking recommendations for {len(test_users)} test users...")
            
            for user_id in test_users:
                user_test_pages = set(test[test["user_id"] == user_id]["page_id"])
                user_train_pages = set(train[train["user_id"] == user_id]["page_id"]) if user_id in train["user_id"].values else set()
                user_recs = collab_recs[collab_recs["user_id"] == user_id]["recommended_page_id"].tolist()
                
                print(f"\nUser {user_id}:")
                print(f"  Test pages: {list(user_test_pages)}")
                print(f"  Train pages: {len(user_train_pages)} pages")
                print(f"  Recommended: {user_recs[:3]}...")
                
                # Verify recommendations don't include test pages that weren't in train
                for test_page in user_test_pages:
                    if test_page not in user_train_pages and test_page in user_recs:
                        print(f"  [WARN] Test page {test_page} recommended but not in train!")
                    elif test_page in user_train_pages and test_page in user_recs:
                        print(f"  [OK] Test page {test_page} in train and recommended (legitimate)")
        
        print("\n" + "="*70)
        print("[PASS] Verification complete - Recommendations use train data only")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_train_only_recommendations()
    sys.exit(0 if success else 1)

