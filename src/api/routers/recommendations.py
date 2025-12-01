"""
Recommendation system API endpoints
"""

from fastapi import APIRouter, HTTPException, Path, Query
from typing import List, Optional

from src.api.models.schemas import (
    UserRecommendationResponse,
    PageRecommendationResponse,
    UserRecommendation,
    PageRecommendation,
    AdminRecommendation,
    RecommendationMetrics,
)
from src.api.utils.data_loader import data_loader

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


@router.get("/users/{user_id}", response_model=UserRecommendationResponse)
async def get_user_recommendations(
    user_id: str = Path(..., description="User ID to fetch recommendations for"),
    limit: int = Query(5, ge=1, le=50, description="Maximum recommendations to return"),
):
    """
    Get personalized recommendations for a user
    """
    data = data_loader.get_user_recommendations(user_id=user_id, limit=limit)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for user {user_id}. Run the recommendation pipeline first.",
        )

    recommendations = [
        UserRecommendation(
            user_id=item.get("user_id", user_id),
            recommended_page_id=item["recommended_page_id"],
            score=item.get("score"),
            hybrid_score=item.get("hybrid_score"),
            rank=item.get("rank"),
            method=item.get("method"),
        )
        for item in data
    ]

    return UserRecommendationResponse(
        user_id=user_id,
        total=len(recommendations),
        recommendations=recommendations,
    )


@router.get("/pages/{page_id}", response_model=PageRecommendationResponse)
async def get_page_recommendations(
    page_id: str = Path(..., description="Page ID to find similar content for"),
    limit: int = Query(5, ge=1, le=50, description="Maximum similar pages to return"),
):
    """
    Get similar pages/items using the content-based model
    """
    data = data_loader.get_page_recommendations(page_id=page_id, limit=limit)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"No similar pages found for {page_id}. Run the recommendation pipeline first.",
        )

    recommendations = [
        PageRecommendation(
            page_id=item.get("page_id", page_id),
            recommended_page_id=item["recommended_page_id"],
            score=item.get("score"),
            hybrid_score=item.get("hybrid_score"),
            rank=item.get("rank"),
            method=item.get("method"),
        )
        for item in data
    ]

    return PageRecommendationResponse(
        page_id=page_id,
        total=len(recommendations),
        recommendations=recommendations,
    )


@router.get("/admin", response_model=List[AdminRecommendation])
async def get_admin_recommendations():
    """
    Get admin-focused recommendations
    """
    data = data_loader.get_admin_recommendations()
    if not data:
        raise HTTPException(
            status_code=404,
            detail="No admin recommendations available. Run the recommendation pipeline first.",
        )

    return [
        AdminRecommendation(
            category=item.get("category", "General"),
            message=item.get("message", ""),
            score=item.get("score"),
        )
        for item in data
    ]


@router.get("/metrics", response_model=RecommendationMetrics)
async def get_recommendation_metrics():
    """
    Get evaluation metrics for the recommendation models
    """
    metrics = data_loader.get_recommendation_metrics()
    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="Recommendation metrics not found. Run the evaluation pipeline first.",
        )

    return RecommendationMetrics(
        hit_rate=metrics.get("hit_rate", 0.0),
        ndcg=metrics.get("ndcg", 0.0),
        mrr=metrics.get("mrr", 0.0),
        users_evaluated=metrics.get("users_evaluated", 0),
    )



