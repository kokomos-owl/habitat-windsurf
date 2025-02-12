"""Event bus for pattern evolution system."""

from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Event:
    """Event in the pattern evolution system."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = datetime.now()

class LocalEventBus:
    """Local event bus implementation."""
    
    def __init__(self):
        """Initialize event bus."""
        self._subscribers: Dict[str, List[Callable]] = {}
        
    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        
    def publish(self, event: Event) -> None:
        """Publish an event."""
        if event.type in self._subscribers:
            for callback in self._subscribers[event.type]:
                callback(event)
