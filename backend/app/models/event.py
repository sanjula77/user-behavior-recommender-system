from pydantic import BaseModel
from typing import Optional, Dict, Any

class Event(BaseModel):
    event_id: Optional[str] = None
    user_id: Optional[str]
    session_id: Optional[str]
    event_type: str
    page_url: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None
