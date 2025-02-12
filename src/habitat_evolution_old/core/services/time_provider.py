"""Time provider service."""

from datetime import datetime

class TimeProvider:
    """Provides time-related functionality."""
    
    @staticmethod
    def now() -> datetime:
        """Get current time."""
        return datetime.now()
    
    @staticmethod
    def timestamp() -> float:
        """Get current timestamp."""
        return datetime.now().timestamp()
