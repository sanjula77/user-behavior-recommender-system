"""
Data loading utilities for API endpoints
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import date, datetime


def get_project_root() -> Path:
    """Detect project root directory"""
    current_dir = Path(__file__).resolve().parent
    # Go up from src/api/utils to project root
    if current_dir.name == "utils" and current_dir.parent.name == "api":
        return current_dir.parent.parent.parent
    if (current_dir / "data").exists():
        return current_dir
    return Path.cwd()


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
INSIGHT_DIR = DATA_DIR / "insights"
PREDICTIONS_DIR = DATA_DIR / "predictions"
RECOMMENDATIONS_DIR = DATA_DIR / "recommendations"


class DataLoader:
    """Centralized data loading with caching"""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self.cache_ttl_seconds = 300  # 5 minutes cache
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid"""
        if key not in self._cache or key not in self._cache_timestamps:
            return False
        age = (datetime.now() - self._cache_timestamps[key]).total_seconds()
        return age < self.cache_ttl_seconds
    
    def _load_json(self, file_path: Path) -> List[Dict]:
        """Load JSON file"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return pd.read_csv(file_path, parse_dates=True)
    
    def _load_parquet(self, file_path: Path) -> pd.DataFrame:
        """Load Parquet file"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_parquet(file_path)
    
    def get_daily_trends(self, start_date: Optional[date] = None, 
                        end_date: Optional[date] = None) -> List[Dict]:
        """Load daily trends data"""
        cache_key = f"daily_trends_{start_date}_{end_date}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        file_path = INSIGHT_DIR / "daily_trends.json"
        data = self._load_json(file_path)
        
        # Filter by date range if provided
        if start_date or end_date:
            filtered_data = []
            for item in data:
                item_date = pd.to_datetime(item.get('date')).date()
                if start_date and item_date < start_date:
                    continue
                if end_date and item_date > end_date:
                    continue
                filtered_data.append(item)
            data = filtered_data
        
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()
        return data
    
    def get_weekly_trends(self) -> List[Dict]:
        """Load weekly trends data"""
        cache_key = "weekly_trends"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        file_path = INSIGHT_DIR / "weekly_trends.json"
        data = self._load_json(file_path)
        
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()
        return data
    
    def get_segment_trends(self, segment: Optional[str] = None,
                           start_date: Optional[date] = None,
                           end_date: Optional[date] = None) -> List[Dict]:
        """Load segment weekly trends data"""
        cache_key = f"segment_trends_{segment}_{start_date}_{end_date}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        file_path = INSIGHT_DIR / "segment_weekly_trends.json"
        data = self._load_json(file_path)
        
        # Filter by segment and date range
        filtered_data = []
        for item in data:
            if segment and item.get('behavior_segment') != segment:
                continue
            
            if start_date or end_date:
                item_date = pd.to_datetime(item.get('week')).date()
                if start_date and item_date < start_date:
                    continue
                if end_date and item_date > end_date:
                    continue
            
            filtered_data.append(item)
        
        data = filtered_data
        
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()
        return data
    
    def get_ml_segment_summary(self) -> List[Dict]:
        """Load ML segment summary"""
        cache_key = "ml_segment_summary"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        file_path = INSIGHT_DIR / "ml_segment_summary.json"
        data = self._load_json(file_path)
        
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()
        return data
    
    def get_high_value_users(self, model: str = "rf") -> List[Dict]:
        """Load high-value users"""
        cache_key = f"high_value_users_{model}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        file_path = INSIGHT_DIR / f"high_value_users_{model}.csv"
        if not file_path.exists():
            return []
        
        df = self._load_csv(file_path)
        data = df.to_dict(orient='records')
        
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()
        return data
    
    def get_at_risk_users(self) -> List[Dict]:
        """Load at-risk users"""
        cache_key = "at_risk_users"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        file_path = INSIGHT_DIR / "at_risk_users_bounce.csv"
        if not file_path.exists():
            return []
        
        df = self._load_csv(file_path)
        data = df.to_dict(orient='records')
        
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()
        return data
    
    def get_anomaly_insights(self, start_date: Optional[date] = None,
                            end_date: Optional[date] = None) -> List[Dict]:
        """Load anomaly insights"""
        cache_key = f"anomaly_insights_{start_date}_{end_date}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        file_path = INSIGHT_DIR / "anomaly_insights.json"
        data = self._load_json(file_path)
        
        # Filter by date range if provided
        if start_date or end_date:
            filtered_data = []
            for item in data:
                item_date = pd.to_datetime(item.get('date')).date()
                if start_date and item_date < start_date:
                    continue
                if end_date and item_date > end_date:
                    continue
                filtered_data.append(item)
            data = filtered_data
        
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()
        return data
    
    def get_llm_summary(self) -> Dict:
        """Load LLM summary"""
        cache_key = "llm_summary"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        file_path = INSIGHT_DIR / "llm_summary.json"
        if not file_path.exists():
            raise FileNotFoundError(f"LLM summary not found: {file_path}")
        
        data = self._load_json(file_path)
        
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()
        return data
    
    def get_user_recommendations(
        self, user_id: Optional[str] = None, limit: int = 5
    ) -> List[Dict]:
        """Load hybrid user recommendations"""
        cache_key = f"user_recs_{user_id}_{limit}"

        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        file_path = RECOMMENDATIONS_DIR / "hybrid_user_recommendations.parquet"
        if not file_path.exists():
            return []

        df = self._load_parquet(file_path)
        # Ensure all fields are JSON-serializable (e.g., convert datetimes to strings)
        datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
        for col in datetime_cols:
            df[col] = df[col].astype(str)

        if user_id:
            df = df[df["user_id"] == user_id]
        df = df.sort_values(["user_id", "rank"]).groupby("user_id").head(limit)
        data = df.to_dict(orient="records")

        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()
        return data
    
    def get_page_recommendations(
        self, page_id: Optional[str] = None, limit: int = 5
    ) -> List[Dict]:
        """Load content-based/hybrid page recommendations"""
        cache_key = f"page_recs_{page_id}_{limit}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        file_path = RECOMMENDATIONS_DIR / "hybrid_page_similarities.parquet"
        if not file_path.exists():
            return []
        
        df = self._load_parquet(file_path)
        if page_id:
            df = df[df["page_id"] == page_id]
        df = df.sort_values(["page_id", "rank"]).groupby("page_id").head(limit)
        data = df.to_dict(orient="records")
        
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()
        return data
    
    def get_admin_recommendations(self) -> List[Dict]:
        """Load admin-focused recommendations"""
        cache_key = "admin_recs"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        file_path = RECOMMENDATIONS_DIR / "admin_recommendations.parquet"
        if not file_path.exists():
            return []
        
        df = self._load_parquet(file_path)
        data = df.to_dict(orient="records")
        
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()
        return data
    
    def get_recommendation_metrics(self) -> Dict[str, Any]:
        """Load evaluation metrics"""
        cache_key = "recommendation_metrics"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        file_path = RECOMMENDATIONS_DIR / "evaluation_metrics.json"
        if not file_path.exists():
            return {}
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()
        return data
    
    def clear_cache(self):
        """Clear all cached data"""
        self._cache.clear()
        self._cache_timestamps.clear()


# Global data loader instance
data_loader = DataLoader()

