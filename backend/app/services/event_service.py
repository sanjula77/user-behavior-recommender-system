from app.core.utils import generate_event_id, current_timestamp
from app.database.event_repository import EventRepository
from app.models.event import Event
from copy import deepcopy

class EventService:

    def __init__(self):
        self.repository = EventRepository()

    def process_event(self, event: Event):
        enriched_event = event.dict()
        enriched_event["event_id"] = generate_event_id()
        
        if enriched_event.get("timestamp") is None:
            enriched_event["timestamp"] = current_timestamp()

        # Make a copy to avoid MongoDB modifying the original dict
        event_to_insert = deepcopy(enriched_event)
        mongo_id = self.repository.insert_event(event_to_insert)
        enriched_event["_id"] = mongo_id
        return enriched_event
