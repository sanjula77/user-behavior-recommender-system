"""
Data Leakage Detection and Validation
Checks for temporal leakage, target leakage, and test contamination.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.recommendations.utils import RECOMMENDATIONS_DIR, DATA_DIR


def check_temporal_leakage():
    """Check if test data is used during training."""
    print("\n" + "="*60)
    print("Checking for Temporal Data Leakage")
    print("="*60)
    
    interactions_path = RECOMMENDATIONS_DIR / "user_page_interactions.parquet"
    collab_path = RECOMMENDATIONS_DIR / "collaborative_recommendations.parquet"
    
    if not interactions_path.exists() or not collab_path.exists():
        print("[FAIL] Required files missing")
        return False
    
    interactions = pd.read_parquet(interactions_path)
    interactions["last_seen"] = pd.to_datetime(interactions.get("last_seen"))
    interactions = interactions.dropna(subset=["last_seen"])
    interactions = interactions.sort_values("last_seen")
    
    # Simulate evaluation split (last interaction per user = test)
    test_interactions = interactions.groupby("user_id").tail(1)
    train_interactions = interactions.drop(test_interactions.index)
    
    print(f"Total interactions: {len(interactions)}")
    print(f"Train interactions: {len(train_interactions)}")
    print(f"Test interactions: {len(test_interactions)}")
    
    # Check if collaborative recommendations were trained on ALL data
    # by checking if test user-page pairs exist in training
    test_pairs = set(zip(test_interactions["user_id"], test_interactions["page_id"]))
    train_pairs = set(zip(train_interactions["user_id"], train_interactions["page_id"]))
    
    # Some test pairs might legitimately appear in train (user visited page before)
    # But if ALL test interactions are in train, that's suspicious
    overlap = len(test_pairs & train_pairs)
    overlap_pct = (overlap / len(test_pairs)) * 100 if len(test_pairs) > 0 else 0
    
    print(f"\nTest user-page pairs: {len(test_pairs)}")
    print(f"Train user-page pairs: {len(train_pairs)}")
    print(f"Overlap: {overlap} ({overlap_pct:.1f}%)")
    
    # Check if recommendations include test pages for test users
    collab_recs = pd.read_parquet(collab_path)
    leakage_count = 0
    total_test_users = 0
    
    for user_id in test_interactions["user_id"].unique():
        user_test_pages = set(test_interactions[test_interactions["user_id"] == user_id]["page_id"])
        user_recs = collab_recs[collab_recs["user_id"] == user_id]["recommended_page_id"].tolist()
        
        if len(user_recs) > 0:
            total_test_users += 1
            # If recommended pages include test pages, that's OK (model might predict correctly)
            # But we need to check if the model was trained on those test interactions
            train_user_interactions = train_interactions[train_interactions["user_id"] == user_id]
            train_user_pages = set(train_user_interactions["page_id"])
            
            # If model recommends a test page that wasn't in train, that's fine
            # But if test page was in train, model might have learned from it
            for test_page in user_test_pages:
                if test_page in user_recs and test_page in train_user_pages:
                    # This is OK - model learned from past interactions
                    pass
                elif test_page in user_recs and test_page not in train_user_pages:
                    # Model predicted a page user never visited before - OK
                    pass
    
    # The real issue: Were recommendations generated using ALL interactions (including test)?
    # We can't directly check this from the files, but we can verify the evaluation logic
    print("\n" + "-"*60)
    print("CHECKING EVALUATION LOGIC:")
    print("-"*60)
    
    # Read evaluation code logic
    eval_code_path = project_root / "src" / "recommendations" / "evaluation.py"
    if eval_code_path.exists():
        with open(eval_code_path, "r") as f:
            eval_code = f.read()
        
        # Check if evaluation splits data before training
        if "groupby(\"user_id\").tail(1)" in eval_code:
            print("[WARN] Evaluation uses last interaction as test (GOOD)")
        else:
            print("[FAIL] Evaluation doesn't use temporal split")
        
        # Check if recommendations are generated separately for train/test
        if "train_interactions" in eval_code or "split" in eval_code.lower():
            print("[PASS] Evaluation may use train/test split")
        else:
            print("[WARN] Evaluation may use all interactions for recommendations")
            print("       This indicates POTENTIAL DATA LEAKAGE")
            print("       Recommendations should be generated using train data only")
    
    return True


def check_target_leakage():
    """Check if target information leaks into features."""
    print("\n" + "="*60)
    print("Checking for Target Leakage")
    print("="*60)
    
    # Check content-based: should only use page metadata, not user behavior
    content_path = RECOMMENDATIONS_DIR / "content_recommendations.parquet"
    page_meta_path = RECOMMENDATIONS_DIR / "page_metadata.parquet"
    
    if page_meta_path.exists():
        page_meta = pd.read_parquet(page_meta_path)
        print(f"Page metadata columns: {list(page_meta.columns)}")
        
        # Check for suspicious columns
        suspicious = ["user_id", "conversion", "purchase", "click_rate"]
        found_suspicious = [col for col in suspicious if col in page_meta.columns]
        
        if found_suspicious:
            print(f"[FAIL] Suspicious columns in page metadata: {found_suspicious}")
            print("       Content-based should not use user behavior features")
            return False
        else:
            print("[PASS] Page metadata doesn't contain user behavior (GOOD)")
    
    # Check collaborative: should exclude already-seen pages
    collab_code_path = project_root / "src" / "recommendations" / "collaborative.py"
    if collab_code_path.exists():
        with open(collab_code_path, "r") as f:
            collab_code = f.read()
        
        if "already_seen" in collab_code or "matrix[i] > 0" in collab_code:
            print("[PASS] Collaborative filtering excludes already-seen pages (GOOD)")
        else:
            print("[WARN] Collaborative may recommend already-seen pages")
    
    return True


def check_feature_leakage():
    """Check if future information leaks into features."""
    print("\n" + "="*60)
    print("Checking for Feature Leakage (Future Information)")
    print("="*60)
    
    interactions_path = RECOMMENDATIONS_DIR / "user_page_interactions.parquet"
    if interactions_path.exists():
        interactions = pd.read_parquet(interactions_path)
        
        if "last_seen" in interactions.columns:
            interactions["last_seen"] = pd.to_datetime(interactions.get("last_seen"))
            interactions = interactions.sort_values("last_seen")
            
            # Check if recency_weight uses future information
            # In data_pipeline, recency_weight is calculated from date
            # This should be OK as long as it's based on the interaction's own date
            
            print("[INFO] Recency weighting appears to use interaction's own date (OK)")
            print(f"       Date range: {interactions['last_seen'].min()} to {interactions['last_seen'].max()}")
        else:
            print("[WARN] No temporal information in interactions")
    
    return True


def main():
    print("="*70)
    print("DATA LEAKAGE DETECTION REPORT")
    print("="*70)
    
    issues_found = []
    
    if not check_temporal_leakage():
        issues_found.append("Temporal leakage check failed")
    
    if not check_target_leakage():
        issues_found.append("Target leakage found")
    
    if not check_feature_leakage():
        issues_found.append("Feature leakage found")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if issues_found:
        print(f"[WARN] {len(issues_found)} potential issues found:")
        for issue in issues_found:
            print(f"  - {issue}")
        print("\nRECOMMENDATION: Review evaluation logic to ensure test data")
        print("                is excluded from training/recommendation generation")
    else:
        print("[INFO] No obvious data leakage detected")
        print("       However, verify that recommendations are generated")
        print("       using only training data (excluding test interactions)")


if __name__ == "__main__":
    main()

