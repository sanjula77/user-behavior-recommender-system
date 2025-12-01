"""
Segment insights API endpoints
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from datetime import date
from src.api.models.schemas import SegmentWeeklyTrend, SegmentInsightsResponse
from src.api.utils.data_loader import data_loader
import pandas as pd

router = APIRouter(prefix="/segments", tags=["Segments"])


@router.get("/trends", response_model=SegmentInsightsResponse)
async def get_segment_trends(
    segment: Optional[str] = Query(None, description="Filter by behavior segment"),
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: Optional[int] = Query(None, description="Limit number of records")
):
    """
    Get segment weekly trends data
    
    Returns segment performance metrics over time including:
    - User counts per segment
    - Conversion and bounce rates
    - Anomaly detection flags
    """
    try:
        data = data_loader.get_segment_trends(segment, start_date, end_date)
        
        if not data:
            raise HTTPException(
                status_code=404,
                detail="No segment trends data found. Please run segment_insights.py first."
            )
        
        # Apply limit if provided
        if limit:
            data = data[-limit:]
        
        # Get unique segments
        df = pd.DataFrame(data)
        segments = df['behavior_segment'].unique().tolist() if 'behavior_segment' in df.columns else []
        
        # Get date range
        if 'week' in df.columns:
            df['week'] = pd.to_datetime(df['week'])
            date_range = {
                "start": df['week'].min().date(),
                "end": df['week'].max().date()
            }
        else:
            date_range = {"start": None, "end": None}
        
        # Convert to response models
        trends = [SegmentWeeklyTrend(**item) for item in data]
        
        return SegmentInsightsResponse(
            data=trends,
            total_records=len(trends),
            segments=segments,
            date_range=date_range
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading segment trends: {str(e)}")


@router.get("/list")
async def get_segments_list():
    """
    Get list of available behavior segments
    """
    try:
        data = data_loader.get_segment_trends()
        
        if not data:
            raise HTTPException(
                status_code=404,
                detail="No segment data found."
            )
        
        df = pd.DataFrame(data)
        
        if 'behavior_segment' in df.columns:
            segments = df['behavior_segment'].unique().tolist()
            segment_counts = df['behavior_segment'].value_counts().to_dict()
            
            return {
                "segments": segments,
                "counts": segment_counts,
                "total_segments": len(segments)
            }
        else:
            return {"segments": [], "counts": {}, "total_segments": 0}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading segments: {str(e)}")


@router.get("/summary")
async def get_segment_summary(
    segment: Optional[str] = Query(None, description="Filter by behavior segment")
):
    """
    Get summary statistics for a specific segment or all segments
    """
    try:
        data = data_loader.get_segment_trends(segment=segment)
        
        if not data:
            raise HTTPException(
                status_code=404,
                detail="No segment data found."
            )
        
        df = pd.DataFrame(data)
        
        if segment:
            # Summary for specific segment
            summary = {
                "segment": segment,
                "total_records": len(df),
                "avg_conversion_rate": float(df['conversion_rate'].mean()) if 'conversion_rate' in df.columns else None,
                "avg_bounce_rate": float(df['bounce_rate'].mean()) if 'bounce_rate' in df.columns else None,
                "total_anomalies": int(df['has_anomaly'].sum()) if 'has_anomaly' in df.columns else 0
            }
        else:
            # Summary for all segments
            if 'behavior_segment' in df.columns:
                segment_summaries = []
                for seg in df['behavior_segment'].unique():
                    seg_df = df[df['behavior_segment'] == seg]
                    segment_summaries.append({
                        "segment": seg,
                        "total_records": len(seg_df),
                        "avg_conversion_rate": float(seg_df['conversion_rate'].mean()) if 'conversion_rate' in seg_df.columns else None,
                        "avg_bounce_rate": float(seg_df['bounce_rate'].mean()) if 'bounce_rate' in seg_df.columns else None,
                        "total_anomalies": int(seg_df['has_anomaly'].sum()) if 'has_anomaly' in seg_df.columns else 0
                    })
                summary = {"segments": segment_summaries}
            else:
                summary = {"segments": []}
        
        return summary
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

