from enum import Enum, auto
from typing import Dict, Any, Callable, List

class EventType(Enum):
    """Types of events that can be emitted."""
    PATTERN_OBSERVED = auto()
    WINDOW_UPDATED = auto()
    PATTERN_EVOLVED = auto()

class EventManager:
    """Manages event emission and subscription."""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        
    def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def emit_event(self, event_type: EventType, data: Dict[str, Any]):
        """Emit an event to all subscribers."""
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                callback(data)
