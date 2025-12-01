"""
Trends API endpoints
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from datetime import date
from src.api.models.schemas import DailyTrend, WeeklyTrend, TrendsResponse
from src.api.utils.data_loader import data_loader
import pandas as pd

router = APIRouter(prefix="/trends", tags=["Trends"])


@router.get("/daily", response_model=TrendsResponse)
async def get_daily_trends(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: Optional[int] = Query(None, description="Limit number of records")
):
    """
    Get daily trends data
    
    Returns aggregated daily metrics including:
    - Sessions count
    - Events count
    - Average conversion probability
    - Average bounce probability
    """
    try:
        data = data_loader.get_daily_trends(start_date, end_date)
        
        if not data:
            raise HTTPException(
                status_code=404,
                detail="No daily trends data found. Please run daily_trends.py first."
            )
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Apply limit if provided
        if limit:
            df = df.tail(limit)
            data = df.to_dict(orient='records')
        
        # Get date range
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            date_range = {
                "start": df['date'].min().date(),
                "end": df['date'].max().date()
            }
        else:
            date_range = {"start": None, "end": None}
        
        # Convert to response models
        trends = [DailyTrend(**item) for item in data]
        
        return TrendsResponse(
            data=trends,
            total_records=len(trends),
            date_range=date_range
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading daily trends: {str(e)}")


@router.get("/weekly", response_model=List[WeeklyTrend])
async def get_weekly_trends(
    limit: Optional[int] = Query(None, description="Limit number of records")
):
    """
    Get weekly trends data
    
    Returns aggregated weekly metrics including:
    - Total sessions and events
    - Average conversion and bounce probabilities
    - Conversion and bounce rates
    - Anomaly flags
    """
    try:
        data = data_loader.get_weekly_trends()
        
        if not data:
            raise HTTPException(
                status_code=404,
                detail="No weekly trends data found. Please run weekly_trends.py first."
            )
        
        # Apply limit if provided
        if limit:
            data = data[-limit:]
        
        # Convert to response models
        trends = [WeeklyTrend(**item) for item in data]
        
        return trends
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading weekly trends: {str(e)}")


@router.get("/daily/summary")
async def get_daily_trends_summary():
    """
    Get summary statistics for daily trends
    """
    try:
        data = data_loader.get_daily_trends()
        
        if not data:
            raise HTTPException(
                status_code=404,
                detail="No daily trends data found."
            )
        
        df = pd.DataFrame(data)
        
        summary = {
            "total_days": len(df),
            "avg_sessions_per_day": float(df['sessions'].mean()) if 'sessions' in df.columns else None,
            "avg_events_per_day": float(df['events'].mean()) if 'events' in df.columns else None,
            "avg_conversion_probability": float(df['avg_conversion_probability'].mean()) if 'avg_conversion_probability' in df.columns else None,
            "avg_bounce_probability": float(df['avg_bounce_probability'].mean()) if 'avg_bounce_probability' in df.columns else None,
            "date_range": {
                "start": pd.to_datetime(df['date']).min().date().isoformat() if 'date' in df.columns else None,
                "end": pd.to_datetime(df['date']).max().date().isoformat() if 'date' in df.columns else None
            }
        }
        
        return summary
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

