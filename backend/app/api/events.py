from fastapi import APIRouter
from app.models.event import Event
from app.services.event_service import EventService

router = APIRouter(prefix="/api/events", tags=["Events"])
service = EventService()

@router.post("/track")
def track_event(event: Event):
    processed = service.process_event(event)
    return {"status": "success", "event": processed}
