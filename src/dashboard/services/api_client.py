import requests
import streamlit as st
from typing import Dict, List, Optional, Any
from src.dashboard.config import API_BASE_URL

class APIClient:
    """Client for fetching data from the User Behavior API"""

    @staticmethod
    def _get(endpoint: str, params: Optional[Dict] = None) -> Any:
        """Helper to perform GET requests with error handling"""
        url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            if st:
                st.error(f"❌ Cannot connect to API at {API_BASE_URL}. Make sure the FastAPI server is running.")
            return None
        except requests.exceptions.Timeout:
            if st:
                st.error("⏱️ API request timed out. Please try again.")
            return None
        except requests.exceptions.HTTPError as e:
            if st:
                st.warning(f"⚠️ API Error ({e.response.status_code}): {e.response.text[:100]}")
            return None
        except requests.exceptions.RequestException as e:
            if st:
                st.error(f"❌ API Error: {str(e)}")
            return None
        except Exception as e:
            if st:
                st.error(f"❌ Unexpected error: {str(e)}")
            return None

    @staticmethod
    @st.cache_data(ttl=300)
    def get_daily_trends(start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        params = {}
        if start_date: params['start_date'] = start_date
        if end_date: params['end_date'] = end_date
        response = APIClient._get("trends/daily", params)
        if response and isinstance(response, dict) and 'data' in response:
            return response['data']
        return response or []

    @staticmethod
    @st.cache_data(ttl=300)
    def get_weekly_trends() -> List[Dict]:
        return APIClient._get("trends/weekly") or []

    @staticmethod
    @st.cache_data(ttl=300)
    def get_segment_trends(segment: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        params = {}
        if segment: params['segment'] = segment
        if start_date: params['start_date'] = start_date
        if end_date: params['end_date'] = end_date
        response = APIClient._get("segments/trends", params)
        if response and isinstance(response, dict) and 'data' in response:
            return response['data']
        return response or []

    @staticmethod
    @st.cache_data(ttl=300)
    def get_segment_list() -> List[str]:
        response = APIClient._get("segments/list")
        if response and isinstance(response, dict) and 'segments' in response:
            return response['segments']
        return []

    @staticmethod
    @st.cache_data(ttl=300)
    def get_segment_summary(segment: Optional[str] = None) -> List[Dict]:
        params = {}
        if segment: params['segment'] = segment
        response = APIClient._get("segments/summary", params)
        if response and isinstance(response, dict):
            if 'segments' in response:
                # Extract segments list
                segments_list = response['segments']
                if isinstance(segments_list, list):
                    return segments_list
                else:
                    return []
            elif 'segment' in response:
                return [response]  # Single segment summary
        elif isinstance(response, list):
            # Already a list
            return response
        return []

    @staticmethod
    @st.cache_data(ttl=300)
    def get_high_value_users(model: str = "rf") -> List[Dict]:
        return APIClient._get(f"predictions/high-value/{model}") or []

    @staticmethod
    @st.cache_data(ttl=300)
    def get_at_risk_users() -> List[Dict]:
        return APIClient._get("predictions/at-risk") or []

    @staticmethod
    @st.cache_data(ttl=300)
    def get_anomalies(start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        params = {}
        if start_date: params['start_date'] = start_date
        if end_date: params['end_date'] = end_date
        response = APIClient._get("anomalies", params)
        if response and isinstance(response, dict) and 'data' in response:
            return response['data']
        return response or []
    
    @staticmethod
    @st.cache_data(ttl=300)
    def get_anomaly_summary() -> Dict:
        return APIClient._get("anomalies/summary") or {}

    @staticmethod
    @st.cache_data(ttl=3600)  # Cache longer for LLM summary
    def get_llm_summary() -> Dict:
        response = APIClient._get("summary/llm")
        if response and isinstance(response, dict):
            # Extract recommendations if available in metadata
            if 'metadata' in response and isinstance(response['metadata'], dict):
                recommendations = response['metadata'].get('recommendations', [])
                if recommendations:
                    response['recommendations'] = recommendations
        return response or {}
    
    @staticmethod
    @st.cache_data(ttl=300)
    def get_user_recommendations(user_id: str, limit: int = 5) -> Dict:
        if not user_id:
            return {}
        params = {"limit": limit}
        return APIClient._get(f"recommendations/users/{user_id}", params) or {}
    
    @staticmethod
    @st.cache_data(ttl=300)
    def get_page_recommendations(page_id: str, limit: int = 5) -> Dict:
        if not page_id:
            return {}
        params = {"limit": limit}
        return APIClient._get(f"recommendations/pages/{page_id}", params) or {}
    
    @staticmethod
    @st.cache_data(ttl=300)
    def get_admin_recommendations() -> List[Dict]:
        return APIClient._get("recommendations/admin") or []
    
    @staticmethod
    @st.cache_data(ttl=300)
    def get_recommendation_metrics() -> Dict:
        return APIClient._get("recommendations/metrics") or {}

    @staticmethod
    def get_health() -> bool:
        try:
            response = requests.get(f"{API_BASE_URL.replace('/api', '')}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
