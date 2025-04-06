"""Bidirectional Flow Manager for PatternAwareRAG and Vector-Tonic System.

This module implements the bidirectional communication between PatternAwareRAG and
the vector-tonic system, ensuring proper field-aware event handling, coherence-based
filtering, and prevention of feedback loops.
"""

import logging
import uuid
from typing import Dict, Any, Set, Optional, List
from datetime import datetime
from collections import deque

from src.habitat_evolution.adaptive_core.emergence.interfaces.learning_window_observer import LearningWindowState as WindowState
from src.habitat_evolution.core.services.event_bus import Event

logger = logging.getLogger(__name__)

class StateChangeBuffer:
    """Buffer for managing state changes to prevent feedback loops.
    
    This class implements a buffer mechanism that tracks recent state changes
    to prevent oscillations in the bidirectional communication between
    PatternAwareRAG and the vector-tonic system.
    """
    
    def __init__(self, buffer_size: int = 50, cooldown_period: float = 0.5):
        """Initialize the state change buffer.
        
        Args:
            buffer_size: Maximum number of recent changes to track
            cooldown_period: Minimum time (in seconds) between similar changes
        """
        self.recent_changes = deque(maxlen=buffer_size)
        self.cooldown_period = cooldown_period
        
    def should_propagate(self, entity_id: str, change_type: str) -> bool:
        """Determine if a change should be propagated.
        
        Args:
            entity_id: ID of the entity being changed
            change_type: Type of change being made
            
        Returns:
            True if the change should be propagated, False otherwise
        """
        # Check if similar change was recently processed
        now = datetime.now()
        for change in self.recent_changes:
            if (change["entity_id"] == entity_id and 
                change["change_type"] == change_type and
                (now - change["timestamp"]).total_seconds() < self.cooldown_period):
                return False
        
        # Record this change
        self.recent_changes.append({
            "entity_id": entity_id,
            "change_type": change_type,
            "timestamp": now
        })
        return True


class DirectionalEventBus:
    """Event bus with direction awareness for bidirectional communication.
    
    This class extends the standard event bus with direction awareness,
    allowing events to be published and subscribed to with specific
    directional context (ingestion, retrieval, or bidirectional).
    """
    
    def __init__(self, event_bus):
        """Initialize the directional event bus.
        
        Args:
            event_bus: The underlying event bus to wrap
        """
        self.event_bus = event_bus
        self._processed_event_ids: Set[str] = set()
        
    def publish(self, event_name: str, data: Dict[str, Any], direction: str = "bidirectional"):
        """Publish an event with direction awareness.
        
        Args:
            event_name: Name of the event to publish
            data: Event data
            direction: Direction of the event (ingestion, retrieval, or bidirectional)
        """
        # Add direction to data if not already present
        if "direction" not in data:
            data["direction"] = direction
            
        # Add event ID if not already present
        if "event_id" not in data:
            data["event_id"] = str(uuid.uuid4())
            
        # Add timestamp if not already present
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()
            
        # Publish the event
        self.event_bus.publish(event_name, data)
        
    def subscribe(self, event_name: str, handler, metadata: Optional[Dict[str, Any]] = None):
        """Subscribe to an event with direction awareness.
        
        Args:
            event_name: Name of the event to subscribe to
            handler: Handler function to call when the event is published
            metadata: Optional metadata for the subscription, including direction_filter
        """
        # Get direction filter if provided
        direction_filter = metadata.get("direction") if metadata else None
        
        # If direction filter is provided, wrap handler to filter by direction
        if direction_filter:
            original_handler = handler
            
            async def direction_filtered_handler(event_data):
                # Check direction
                direction = event_data.get("direction", "bidirectional")
                if direction_filter == "bidirectional" or direction == direction_filter:
                    # Check for event ID to prevent processing the same event multiple times
                    event_id = event_data.get("event_id")
                    if event_id and event_id in self._processed_event_ids:
                        return
                        
                    # Record event ID
                    if event_id:
                        self._processed_event_ids.add(event_id)
                        # Limit cache size
                        if len(self._processed_event_ids) > 1000:
                            self._processed_event_ids = set(list(self._processed_event_ids)[-500:])
                            
                    # Call original handler
                    await original_handler(event_data)
                    
            # Subscribe with filtered handler
            self.event_bus.subscribe(event_name, direction_filtered_handler)
        else:
            # Subscribe with original handler
            self.event_bus.subscribe(event_name, handler)


class BidirectionalFlowManager:
    """Manager for bidirectional flow between PatternAwareRAG and vector-tonic system.
    
    This class implements the bidirectional communication between PatternAwareRAG and
    the vector-tonic system, ensuring proper field-aware event handling, coherence-based
    filtering, and prevention of feedback loops.
    """
    
    def __init__(self, event_bus, semantic_potential_calculator=None):
        """Initialize the bidirectional flow manager.
        
        Args:
            event_bus: The event bus to use for communication
            semantic_potential_calculator: Optional calculator for coherence-based filtering
        """
        self.event_bus = DirectionalEventBus(event_bus)
        self.state_change_buffer = StateChangeBuffer()
        self.semantic_potential_calculator = semantic_potential_calculator
        self.coherence_threshold = 0.3
        self.constructive_dissonance_allowance = 0.1
        
    def should_process_field_change(self, event_data: Dict[str, Any], current_coherence: float) -> bool:
        """Determine if a field change should be processed based on coherence.
        
        This implements coherence-based filtering to prevent processing changes
        that would reduce system coherence below an acceptable threshold.
        
        Args:
            event_data: Event data containing the change
            current_coherence: Current coherence level of the system
            
        Returns:
            True if the change should be processed, False otherwise
        """
        # Extract entity ID and change type
        entity_id = event_data.get("entity_id") or event_data.get("field_id") or event_data.get("pattern_id")
        change_type = event_data.get("change_type") or "field_update"
        
        if not entity_id:
            # If no entity ID, default to processing the change
            return True
            
        # Check buffer to prevent oscillations
        if not self.state_change_buffer.should_propagate(entity_id, change_type):
            return False
            
        # If no semantic potential calculator, default to processing all changes
        if not self.semantic_potential_calculator:
            return True
            
        # Calculate projected coherence after change
        try:
            projected_coherence = self.semantic_potential_calculator.project_coherence_after_change(
                entity_id, change_type, event_data
            )
        except Exception as e:
            logger.warning(f"Error calculating projected coherence: {e}")
            return True  # Default to processing on error
            
        # Allow changes that maintain coherence within acceptable bands
        # Some reduction in coherence is acceptable if it's within the constructive dissonance range
        if (projected_coherence >= self.coherence_threshold and 
            (projected_coherence >= current_coherence or 
             current_coherence - projected_coherence <= self.constructive_dissonance_allowance)):
            return True
            
        # Log rejection due to coherence filtering
        logger.info(f"Filtered change to {entity_id} (type: {change_type}) due to coherence drop: "
                   f"{current_coherence} -> {projected_coherence} (threshold: {self.coherence_threshold})")
        return False
        
    def map_window_state(self, window_state: str) -> str:
        """Map vector-tonic window state to learning window state.
        
        Args:
            window_state: Vector-tonic window state
            
        Returns:
            Mapped learning window state
        """
        try:
            # Convert string to WindowState enum
            state = WindowState(window_state)
            
            # Map to learning window state
            if state == WindowState.OPENING:
                return "opening"
            elif state == WindowState.OPEN:
                return "open"
            else:  # CLOSED
                return "closed"
        except (ValueError, KeyError):
            logger.warning(f"Invalid window state: {window_state}")
            return "closed"  # Default to closed
