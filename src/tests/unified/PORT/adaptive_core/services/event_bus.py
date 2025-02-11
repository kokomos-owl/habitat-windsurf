"""
Local event bus for pattern evolution and coherence tracking.

This module provides a simple, in-process event system that replaces the 
more complex EventManager. It maintains the ability to track pattern evolution
and coherence changes while eliminating external dependencies.
"""

from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional
from datetime import datetime
from .time_provider import TimeProvider

@dataclass
class Event:
    """Represents a system event."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    source: Optional[str] = None
    
    @classmethod
    def create(cls, type: str, data: Dict[str, Any], source: Optional[str] = None) -> 'Event':
        """Create a new event with current timestamp."""
        return cls(
            type=type,
            data=data,
            timestamp=TimeProvider.now(),
            source=source
        )

class EventFilter:
    """Filter for subscribing to specific event patterns."""
    def __init__(self, 
                 type_pattern: Optional[str] = None,
                 source_pattern: Optional[str] = None,
                 data_matcher: Optional[Callable[[Dict[str, Any]], bool]] = None):
        self.type_pattern = type_pattern
        self.source_pattern = source_pattern
        self.data_matcher = data_matcher
    
    def matches(self, event: Event) -> bool:
        """Check if event matches this filter."""
        if self.type_pattern and self.type_pattern != event.type:
            return False
            
        if self.source_pattern and self.source_pattern != event.source:
            return False
            
        if self.data_matcher and not self.data_matcher(event.data):
            return False
            
        return True

class LocalEventBus:
    """Local event handling for pattern evolution."""
    
    def __init__(self):
        """Initialize empty handler registry."""
        self._handlers: Dict[str, List[Callable[[Event], None]]] = {}
        self._filtered_handlers: List[tuple[EventFilter, Callable[[Event], None]]] = []
        self._event_history: List[Event] = []
        self._max_history: int = 1000  # Prevent unbounded growth
    
    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        """Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Callback function to handle events
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def subscribe_filtered(self, 
                         filter: EventFilter,
                         handler: Callable[[Event], None]) -> None:
        """Subscribe to events matching a filter.
        
        Args:
            filter: EventFilter defining match criteria
            handler: Callback function to handle matching events
        """
        self._filtered_handlers.append((filter, handler))
    
    def publish(self, event: Event) -> None:
        """Publish an event to all interested subscribers.
        
        Args:
            event: Event to publish
        """
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Notify type-based handlers
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Log error but continue processing
                print(f"Error in event handler: {e}")
        
        # Notify filter-based handlers
        for filter, handler in self._filtered_handlers:
            if filter.matches(event):
                try:
                    handler(event)
                except Exception as e:
                    print(f"Error in filtered event handler: {e}")
    
    def get_history(self, 
                   filter: Optional[EventFilter] = None,
                   limit: Optional[int] = None) -> List[Event]:
        """Get event history, optionally filtered.
        
        Args:
            filter: Optional filter to apply
            limit: Optional limit on number of events to return
            
        Returns:
            List of matching events, newest first
        """
        events = self._event_history.copy()
        if filter:
            events = [e for e in events if filter.matches(e)]
        if limit:
            events = events[-limit:]
        return events
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
    
    def unsubscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        """Unsubscribe a handler from an event type.
        
        Args:
            event_type: Event type to unsubscribe from
            handler: Handler to remove
        """
        if event_type in self._handlers:
            self._handlers[event_type] = [h for h in self._handlers[event_type] if h != handler]
            if not self._handlers[event_type]:
                del self._handlers[event_type]
    
    def unsubscribe_filtered(self, 
                           filter: EventFilter,
                           handler: Callable[[Event], None]) -> None:
        """Unsubscribe a filtered handler.
        
        Args:
            filter: Filter to unsubscribe
            handler: Handler to remove
        """
        self._filtered_handlers = [(f, h) for f, h in self._filtered_handlers 
                                 if f != filter or h != handler]
