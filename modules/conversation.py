import uuid
from datetime import datetime

class ConversationManager:
    def __init__(self, db, gap_threshold_seconds=300):  # 5 minutes default
        self.db = db
        self.gap_threshold = gap_threshold_seconds
        self.current_conversation_id = None
        self.last_utterance_time = None

    def get_conversation_for_utterance(self, timestamp_str):
        """Determine if utterance belongs to current conversation or requires a new one."""
        current_time = datetime.fromisoformat(timestamp_str.replace(' ', 'T'))

        # If no current conversation or last utterance time, create a new conversation
        if not self.current_conversation_id or not self.last_utterance_time:
            self.create_new_conversation(current_time)
            return self.current_conversation_id

        # Calculate time gap
        time_gap = (current_time - self.last_utterance_time).total_seconds()

        # If gap exceeds threshold, create a new conversation
        if time_gap > self.gap_threshold:
            self.create_new_conversation(current_time)

        # Update last utterance time
        self.last_utterance_time = current_time
        return self.current_conversation_id

    def create_new_conversation(self, timestamp):
        """Create a new conversation with the given timestamp."""
        conversation_id = str(uuid.uuid4())
        start_time = timestamp.isoformat()

        self.db.create(
            "conversations",
            id=conversation_id,
            document=f"Conversation started at {start_time}",
            metadata={"start_time": start_time, "updated_at": start_time}
        )

        self.current_conversation_id = conversation_id
        self.last_utterance_time = timestamp
        return conversation_id

    def update_conversation_metadata(self, conversation_id, timestamp):
        """Update conversation metadata with latest timestamp."""
        if not conversation_id:
            return

        metadata = {"updated_at": timestamp.isoformat()}
        self.db.update("conversations", conversation_id, metadata=metadata)
