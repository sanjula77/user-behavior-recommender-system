from app.database.mongo import MongoDB

class EventRepository:

    def __init__(self):
        self.db = MongoDB.get_db()
        self.collection = self.db["events"]

    def insert_event(self, event_data: dict):
        result = self.collection.insert_one(event_data)
        return str(result.inserted_id)
