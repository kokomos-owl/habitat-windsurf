"""
Event bus integration for dynamic pattern detection.

This module provides adapters and publishers to integrate the dynamic pattern
detection components with the event bus architecture of the pattern-aware RAG system.
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging
from ..id.adaptive_id import AdaptiveID

# Import the Event class
try:
    from habitat_evolution.core.services.event_bus import Event
except ImportError:
    # Define a simple Event class for testing
    class Event:
        def __init__(self, type, data, source=None):
            self.type = type
            self.data = data
            self.source = source
            self.timestamp = datetime.now().isoformat()

logger = logging.getLogger(__name__)

class AdaptiveIDEventAdapter:
    """
    Adapter that converts AdaptiveID state changes to event bus events.
    
    This adapter observes an AdaptiveID instance and publishes events
    to the event bus when state changes occur.
    """
    
    def __init__(self, adaptive_id: AdaptiveID, event_bus):
        """
        Initialize the adapter.
        
        Args:
            adaptive_id: The AdaptiveID instance to observe
            event_bus: The event bus to publish events to
        """
        self.adaptive_id = adaptive_id
        self.event_bus = event_bus
        self.entity_id = adaptive_id.id
        
        # Register as observer for state changes
        self._register_as_observer()
        
    def _register_as_observer(self):
        """Register as an observer for AdaptiveID state changes."""
        # Create a tracker that will call our record_state_change method
        class StateChangeTracker:
            def __init__(self, adapter):
                self.adapter = adapter
                
            def record_state_change(self, entity_id, change_type, old_value, new_value, origin):
                self.adapter.record_state_change(entity_id, change_type, old_value, new_value, origin)
        
        # Store the tracker as an instance variable to prevent garbage collection
        self.tracker = StateChangeTracker(self)
        
        # Register the tracker with the AdaptiveID
        # This is a bit of a hack, but it's the simplest way to observe state changes
        # without modifying the AdaptiveID class
        if hasattr(self.adaptive_id, 'learning_windows'):
            self.adaptive_id.learning_windows.append(self.tracker)
        else:
            self.adaptive_id.learning_windows = [self.tracker]
    
    def record_state_change(self, entity_id: str, change_type: str, old_value: Any, new_value: Any, origin: str):
        """
        Record a state change from the AdaptiveID and publish it as an event.
        
        Args:
            entity_id: ID of the entity that changed
            change_type: Type of change (e.g., 'temporal_context', 'relationship')
            old_value: Previous value (can be None)
            new_value: New value
            origin: Origin of the change
        """
        try:
            # Map change types to event types
            event_type_map = {
                "temporal_context": "pattern.context.updated",
                "spatial_context": "pattern.context.updated",
                "relationship_added": "pattern.relationship.added",
                "relationship_removed": "pattern.relationship.removed"
            }
            
            # Create event data
            event_data = {
                "entity_id": entity_id,
                "old_value": old_value,
                "new_value": new_value,
                "origin": origin,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add specific fields based on change type
            if change_type == "temporal_context" and isinstance(new_value, dict):
                # Check if this is a relationship observation
                if "relationship" in new_value:
                    event_type = "pattern.relationship.observed"
                    event_data["relationship"] = new_value["relationship"]
                # Check if this is a pattern detection
                elif "pattern_id" in new_value:
                    event_type = "pattern.detected"
                    event_data["pattern_id"] = new_value["pattern_id"]
                    event_data["pattern_data"] = new_value
                else:
                    event_type = event_type_map.get(change_type, "pattern.state.changed")
            else:
                event_type = event_type_map.get(change_type, "pattern.state.changed")
            
            # Create the event
            event = Event(event_type, event_data, source=f"adaptive_id:{entity_id}")
            
            # Publish the event
            logger.debug(f"Publishing event: {event_type}")
            self.event_bus.publish(event)
            
        except Exception as e:
            logger.error(f"Error publishing event: {e}")


class PatternEventPublisher:
    """
    Publisher for pattern-related events.
    
    This class provides methods to publish standardized pattern events
    to the event bus, ensuring consistent event format.
    """
    
    def __init__(self, event_bus):
        """
        Initialize the publisher.
        
        Args:
            event_bus: The event bus to publish events to
        """
        self.event_bus = event_bus
        
    def publish_pattern_detected(self, pattern_id: str, pattern_data: Dict[str, Any], confidence: float, source: str = "pattern_detector"):
        """
        Publish a pattern detected event.
        
        Args:
            pattern_id: ID of the detected pattern
            pattern_data: Data about the pattern
            confidence: Confidence level (0.0-1.0)
            source: Source of the detection
        """
        event_data = {
            "pattern_id": pattern_id,
            "pattern_data": pattern_data,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        event = Event("pattern.detected", event_data, source=source)
        self.event_bus.publish(event)
        
    def publish_pattern_evolved(self, pattern_id: str, old_state: Dict[str, Any], new_state: Dict[str, Any], source: str = "pattern_detector"):
        """
        Publish a pattern evolved event.
        
        Args:
            pattern_id: ID of the evolved pattern
            old_state: Previous state of the pattern
            new_state: New state of the pattern
            source: Source of the evolution
        """
        event_data = {
            "pattern_id": pattern_id,
            "old_state": old_state,
            "new_state": new_state,
            "timestamp": datetime.now().isoformat()
        }
        
        event = Event("pattern.evolved", event_data, source=source)
        self.event_bus.publish(event)
        
    def publish_pattern_resonance(self, pattern_id: str, position: Dict[str, float], strength: float, source: str = "resonance_observer"):
        """
        Publish a pattern resonance event.
        
        Args:
            pattern_id: ID of the resonating pattern
            position: Position in semantic space
            strength: Strength of the resonance
            source: Source of the resonance observation
        """
        event_data = {
            "pattern_id": pattern_id,
            "position": position,
            "strength": strength,
            "timestamp": datetime.now().isoformat()
        }
        
        event = Event("pattern.resonance", event_data, source=source)
        self.event_bus.publish(event)
        
    def publish_learning_window_state(self, window_id: str, state: str, metrics: Dict[str, Any], source: str = "learning_window"):
        """
        Publish a learning window state event.
        
        Args:
            window_id: ID of the learning window
            state: State of the window (OPEN, CLOSED, OPENING)
            metrics: Metrics about the window state
            source: Source of the state change
        """
        event_data = {
            "window_id": window_id,
            "state": state,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        event = Event("learning.window.state", event_data, source=source)
        self.event_bus.publish(event)
