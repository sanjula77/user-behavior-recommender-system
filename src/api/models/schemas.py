"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import date, datetime


# ============================
# Common Models
# ============================
class HealthCheck(BaseModel):
    """Health check response"""
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================
# Trends Models
# ============================
class DailyTrend(BaseModel):
    """Daily trend data point"""
    date: date
    sessions: int
    events: int
    avg_conversion_probability: float = Field(..., alias="avg_conversion_probability")
    avg_bounce_probability: float = Field(..., alias="avg_bounce_probability")

    class Config:
        populate_by_name = True


class WeeklyTrend(BaseModel):
    """Weekly trend data point"""
    week: date
    total_sessions: int
    total_events: int
    avg_conversion_prob: float
    avg_bounce_prob: float
    conversion_rate: float
    bounce_rate: float
    conversion_rate_anomaly: int
    bounce_rate_anomaly: int
    has_anomaly: int


class TrendsResponse(BaseModel):
    """Trends response with metadata"""
    data: List[DailyTrend]
    total_records: int
    date_range: Dict[str, date]


# ============================
# Segment Models
# ============================
class SegmentWeeklyTrend(BaseModel):
    """Segment weekly trend data point"""
    behavior_segment: str
    week: date
    total_users: int
    total_conversions: Optional[int] = None
    avg_conversion_prob: float
    total_bounces: Optional[int] = None
    avg_bounce_prob: float
    conversion_rate: float
    bounce_rate: float
    conversion_rate_anomaly: int
    bounce_rate_anomaly: int
    has_anomaly: int


class SegmentInsightsResponse(BaseModel):
    """Segment insights response"""
    data: List[SegmentWeeklyTrend]
    total_records: int
    segments: List[str]
    date_range: Dict[str, date]


# ============================
# ML Predictions Models
# ============================
class UserPrediction(BaseModel):
    """User prediction data"""
    user_id: str
    behavior_segment: Optional[str] = None
    conversion_prob_rf: Optional[float] = None
    conversion_prob_xgb: Optional[float] = None
    conversion_prob_lr: Optional[float] = None
    bounce_prob: Optional[float] = None
    y_convert_actual: Optional[int] = None
    y_bounce_actual: Optional[int] = None


class MLSegmentSummary(BaseModel):
    """ML segment summary"""
    behavior_segment: str
    total_users: int
    rf_high_prob: Optional[int] = None
    xgb_high_prob: Optional[int] = None
    lr_high_prob: Optional[int] = None
    bounce_risk: Optional[int] = None
    rf_high_prob_pct: Optional[float] = None
    xgb_high_prob_pct: Optional[float] = None
    lr_high_prob_pct: Optional[float] = None
    bounce_risk_pct: Optional[float] = None


class MLInsightsResponse(BaseModel):
    """ML insights response"""
    segment_summary: List[MLSegmentSummary]
    high_value_users_rf_count: int
    high_value_users_xgb_count: int
    at_risk_users_count: int


# ============================
# Anomaly Models
# ============================
class AnomalyData(BaseModel):
    """Anomaly data point"""
    date: date
    avg_conversion_rf: Optional[float] = None
    avg_conversion_xgb: Optional[float] = None
    avg_conversion_lr: Optional[float] = None
    avg_bounce: Optional[float] = None
    high_value_users_rf: Optional[int] = None
    high_bounce_users: Optional[int] = None
    avg_conversion_rf_anomaly: Optional[bool] = None
    avg_conversion_xgb_anomaly: Optional[bool] = None
    avg_conversion_lr_anomaly: Optional[bool] = None
    avg_bounce_anomaly: Optional[bool] = None
    avg_conversion_rf_anomaly_type: Optional[str] = None
    avg_conversion_xgb_anomaly_type: Optional[str] = None
    avg_bounce_anomaly_type: Optional[str] = None


class AnomaliesResponse(BaseModel):
    """Anomalies response"""
    data: List[AnomalyData]
    total_records: int
    total_anomalies: int
    date_range: Dict[str, date]


# ============================
# Summary Models
# ============================
class LLMSummaryResponse(BaseModel):
    """LLM summary response"""
    summary: str
    metadata: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================
# Recommendation Models
# ============================
class UserRecommendation(BaseModel):
    """Recommendation for a specific user"""
    user_id: str
    recommended_page_id: str
    score: Optional[float] = None
    hybrid_score: Optional[float] = None
    rank: Optional[int] = None
    method: Optional[str] = None


class UserRecommendationResponse(BaseModel):
    """User recommendation response"""
    user_id: str
    total: int
    recommendations: List[UserRecommendation]


class PageRecommendation(BaseModel):
    """Similar page recommendation"""
    page_id: str
    recommended_page_id: str
    score: Optional[float] = None
    hybrid_score: Optional[float] = None
    rank: Optional[int] = None
    method: Optional[str] = None


class PageRecommendationResponse(BaseModel):
    """Page recommendation response"""
    page_id: str
    total: int
    recommendations: List[PageRecommendation]


class AdminRecommendation(BaseModel):
    """Admin-focused recommendation"""
    category: str
    message: str
    score: Optional[float] = None


class RecommendationMetrics(BaseModel):
    """Evaluation metrics for recommenders"""
    hit_rate: float
    ndcg: float
    mrr: Optional[float] = None
    users_evaluated: int


# ============================
# Query Parameters
# ============================
class DateRangeParams(BaseModel):
    """Date range query parameters"""
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class SegmentFilterParams(BaseModel):
    """Segment filter parameters"""
    segment: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None

