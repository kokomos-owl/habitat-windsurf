from datetime import datetime

class TimestampService:
    """Service for managing timestamps."""
    
    def __init__(self):
        pass
    
    def get_timestamp(self) -> datetime:
        """Get current timestamp."""
        return datetime.now()
