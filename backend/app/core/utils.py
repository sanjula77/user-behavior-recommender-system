from datetime import datetime
import uuid

def generate_event_id() -> str:
    return str(uuid.uuid4())

def current_timestamp() -> str:
    return datetime.utcnow().isoformat()
