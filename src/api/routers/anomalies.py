"""
Anomaly detection API endpoints
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from datetime import date
from src.api.models.schemas import AnomalyData, AnomaliesResponse
from src.api.utils.data_loader import data_loader
import pandas as pd

router = APIRouter(prefix="/anomalies", tags=["Anomalies"])


@router.get("", response_model=AnomaliesResponse)
async def get_anomalies(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: Optional[int] = Query(None, description="Limit number of records"),
    anomalies_only: bool = Query(False, description="Return only anomaly days")
):
    """
    Get anomaly detection data
    
    Returns daily metrics with anomaly flags including:
    - Conversion probabilities
    - Bounce probabilities
    - Anomaly detection flags
    - Anomaly types (spike/drop)
    """
    try:
        data = data_loader.get_anomaly_insights(start_date, end_date)
        
        if not data:
            raise HTTPException(
                status_code=404,
                detail="No anomaly data found. Please run anomaly_detection.py first."
            )
        
        # Filter anomalies only if requested
        if anomalies_only:
            filtered_data = []
            for item in data:
                # Check if any anomaly flag is True/1
                has_anomaly = False
                for key in item.keys():
                    if 'anomaly' in key.lower() and key != 'has_anomaly':
                        if item.get(key) in [True, 1, "True", "1"]:
                            has_anomaly = True
                            break
                if has_anomaly:
                    filtered_data.append(item)
            data = filtered_data
        
        # Apply limit if provided
        if limit:
            data = data[-limit:]
        
        # Count total anomalies
        df = pd.DataFrame(data)
        total_anomalies = 0
        for col in df.columns:
            if 'anomaly' in col.lower() and col != 'has_anomaly':
                total_anomalies += int(df[col].sum()) if df[col].dtype in ['bool', 'int64'] else 0
        
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
        anomalies = [AnomalyData(**item) for item in data]
        
        return AnomaliesResponse(
            data=anomalies,
            total_records=len(anomalies),
            total_anomalies=total_anomalies,
            date_range=date_range
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading anomalies: {str(e)}")


@router.get("/summary")
async def get_anomalies_summary():
    """
    Get summary statistics for anomalies
    """
    try:
        data = data_loader.get_anomaly_insights()
        
        if not data:
            raise HTTPException(
                status_code=404,
                detail="No anomaly data found."
            )
        
        df = pd.DataFrame(data)
        
        # Count anomalies by type
        anomaly_counts = {}
        anomaly_types = {}
        
        for col in df.columns:
            if 'anomaly' in col.lower() and 'type' in col.lower():
                anomaly_type_col = col
                metric_name = col.replace('_anomaly_type', '')
                
                # Count by type
                type_counts = df[anomaly_type_col].value_counts().to_dict()
                anomaly_types[metric_name] = {
                    "spike": type_counts.get("spike", 0),
                    "drop": type_counts.get("drop", 0),
                    "total": type_counts.get("spike", 0) + type_counts.get("drop", 0)
                }
        
        # Count total anomaly days
        total_anomaly_days = 0
        if 'has_anomaly' in df.columns:
            total_anomaly_days = int(df['has_anomaly'].sum())
        
        summary = {
            "total_days": len(df),
            "total_anomaly_days": total_anomaly_days,
            "anomaly_rate": round(total_anomaly_days / len(df) * 100, 2) if len(df) > 0 else 0,
            "anomaly_types": anomaly_types,
            "date_range": {
                "start": pd.to_datetime(df['date']).min().date().isoformat() if 'date' in df.columns else None,
                "end": pd.to_datetime(df['date']).max().date().isoformat() if 'date' in df.columns else None
            }
        }
        
        return summary
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


@router.get("/recent")
async def get_recent_anomalies(
    limit: int = Query(10, description="Number of recent anomalies to return")
):
    """
    Get recent anomalies only
    """
    try:
        data = data_loader.get_anomaly_insights()
        
        if not data:
            raise HTTPException(
                status_code=404,
                detail="No anomaly data found."
            )
        
        # Filter to anomalies only
        filtered_data = []
        for item in data:
            has_anomaly = False
            for key in item.keys():
                if 'anomaly' in key.lower() and key != 'has_anomaly':
                    if item.get(key) in [True, 1, "True", "1"]:
                        has_anomaly = True
                        break
            if has_anomaly:
                filtered_data.append(item)
        
        # Sort by date (most recent first) and limit
        if filtered_data:
            df = pd.DataFrame(filtered_data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date', ascending=False)
                filtered_data = df.head(limit).to_dict(orient='records')
        
        anomalies = [AnomalyData(**item) for item in filtered_data]
        
        return {
            "anomalies": anomalies,
            "count": len(anomalies)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading recent anomalies: {str(e)}")

