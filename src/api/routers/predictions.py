"""
ML Predictions API endpoints
"""

from fastapi import APIRouter, HTTPException, Path, Query
from typing import Optional, List
from src.api.models.schemas import MLSegmentSummary, MLInsightsResponse, UserPrediction
from src.api.utils.data_loader import data_loader
import pandas as pd

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.get("/segments", response_model=MLInsightsResponse)
async def get_ml_segment_summary():
    """
    Get ML model insights by segment
    
    Returns segment-level summary including:
    - High conversion probability users
    - At-risk bounce users
    - Model performance by segment
    """
    try:
        segment_data = data_loader.get_ml_segment_summary()
        
        if not segment_data:
            raise HTTPException(
                status_code=404,
                detail="No ML segment summary found. Please run ml_insights.py first."
            )
        
        # Get user lists
        high_value_rf = data_loader.get_high_value_users("rf")
        high_value_xgb = data_loader.get_high_value_users("xgb")
        at_risk = data_loader.get_at_risk_users()
        
        # Convert to response models
        segment_summary = [MLSegmentSummary(**item) for item in segment_data]
        
        return MLInsightsResponse(
            segment_summary=segment_summary,
            high_value_users_rf_count=len(high_value_rf),
            high_value_users_xgb_count=len(high_value_xgb),
            at_risk_users_count=len(at_risk)
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading ML insights: {str(e)}")


@router.get("/high-value/{model}", response_model=List[UserPrediction])
async def get_high_value_users(
    model: str = Path(..., description="Model type: 'rf', 'xgb', or 'lr'"),
    limit: Optional[int] = Query(100, description="Limit number of records"),
    min_probability: Optional[float] = Query(0.7, description="Minimum conversion probability")
):
    """
    Get high-value users (high conversion probability)
    
    Returns users with high conversion probability from specified model
    """
    try:
        if model not in ["rf", "xgb", "lr"]:
            raise HTTPException(
                status_code=400,
                detail="Model must be 'rf', 'xgb', or 'lr'"
            )
        
        users = data_loader.get_high_value_users(model)
        
        if not users:
            return []
        
        # Filter by minimum probability
        filtered_users = []
        prob_col = f"conversion_prob_{model}" if model != "lr" else "conversion_prob_lr"
        
        for user in users:
            prob = user.get(prob_col) or user.get("conversion_probability")
            if prob and prob >= min_probability:
                filtered_users.append(user)
        
        # Apply limit
        if limit:
            filtered_users = filtered_users[:limit]
        
        # Convert to response models
        predictions = [UserPrediction(**user) for user in filtered_users]
        
        return predictions
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading high-value users: {str(e)}")


@router.get("/at-risk", response_model=List[UserPrediction])
async def get_at_risk_users(
    limit: Optional[int] = Query(100, description="Limit number of records"),
    min_probability: Optional[float] = Query(0.5327, description="Minimum bounce probability")
):
    """
    Get at-risk users (high bounce probability)
    
    Returns users with high bounce probability
    """
    try:
        users = data_loader.get_at_risk_users()
        
        if not users:
            return []
        
        # Filter by minimum probability
        filtered_users = []
        for user in users:
            prob = user.get("bounce_prob") or user.get("bounce_probability")
            if prob and prob >= min_probability:
                filtered_users.append(user)
        
        # Apply limit
        if limit:
            filtered_users = filtered_users[:limit]
        
        # Convert to response models
        predictions = [UserPrediction(**user) for user in filtered_users]
        
        return predictions
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading at-risk users: {str(e)}")


@router.get("/summary")
async def get_predictions_summary():
    """
    Get summary statistics for predictions
    """
    try:
        segment_data = data_loader.get_ml_segment_summary()
        high_value_rf = data_loader.get_high_value_users("rf")
        high_value_xgb = data_loader.get_high_value_users("xgb")
        at_risk = data_loader.get_at_risk_users()
        
        summary = {
            "total_segments": len(segment_data),
            "high_value_users": {
                "rf": len(high_value_rf),
                "xgb": len(high_value_xgb)
            },
            "at_risk_users": len(at_risk),
            "segments": [item.get("behavior_segment") for item in segment_data]
        }
        
        return summary
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

