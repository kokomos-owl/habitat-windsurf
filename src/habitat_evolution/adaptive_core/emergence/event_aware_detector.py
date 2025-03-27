"""
Event-aware pattern detector that integrates with the event bus.

This module extends the EmergentPatternDetector to work with the event bus,
allowing it to publish pattern detection events and subscribe to relevant events
from the pattern-aware RAG system.
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime
import uuid
import logging
import math
from collections import defaultdict

from ..id.adaptive_id import AdaptiveID
from .emergent_pattern_detector import EmergentPatternDetector
from .semantic_current_observer import SemanticCurrentObserver
from .event_bus_integration import PatternEventPublisher

class EventAwarePatternDetector(EmergentPatternDetector):
    """
    Pattern detector that integrates with the event bus.
    
    This class extends the EmergentPatternDetector to work with the event bus,
    allowing it to publish pattern detection events and subscribe to relevant
    events from the pattern-aware RAG system.
    """
    
    def __init__(
        self, 
        semantic_observer: SemanticCurrentObserver, 
        event_bus=None, 
        pattern_publisher=None,
        threshold: int = 3
    ):
        """
        Initialize an event-aware pattern detector.
        
        Args:
            semantic_observer: Observer for semantic currents
            event_bus: Event bus for publishing and subscribing to events
            pattern_publisher: Publisher for pattern events
            threshold: Minimum frequency threshold for pattern detection
        """
        super().__init__(semantic_observer, threshold)
        
        self.event_bus = event_bus
        self.pattern_publisher = pattern_publisher
        
        # Dynamic threshold based on learning window state
        self.dynamic_threshold = threshold
        self.learning_window_open = False
        
        # Subscribe to events if event bus is provided
        if self.event_bus:
            self._subscribe_to_events()
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events on the event bus."""
        # Subscribe to learning window state events
        self.event_bus.subscribe("learning.window.state", self._handle_learning_window_state)
        
        # Subscribe to field gradient events
        self.event_bus.subscribe("field.gradient.update", self._handle_field_gradient_update)
        
        # Subscribe to semantic observation events
        self.event_bus.subscribe("semantic.observation", self._handle_semantic_observation)
        
        self.logger.info("Subscribed to events on the event bus")
    
    def _handle_learning_window_state(self, event):
        """
        Handle learning window state change events.
        
        Adjusts pattern detection sensitivity based on window state.
        
        Args:
            event: Learning window state event
        """
        try:
            state = event.data.get("state")
            if state == "OPEN":
                # Lower threshold when window is open to increase sensitivity
                self.dynamic_threshold = max(2, self.threshold - 1)
                self.learning_window_open = True
                self.logger.debug(f"Learning window open, threshold lowered to {self.dynamic_threshold}")
            elif state == "CLOSED":
                # Reset threshold when window is closed
                self.dynamic_threshold = self.threshold
                self.learning_window_open = False
                self.logger.debug(f"Learning window closed, threshold reset to {self.dynamic_threshold}")
        except Exception as e:
            self.logger.error(f"Error handling learning window state event: {e}")
    
    def _handle_field_gradient_update(self, event):
        """
        Handle field gradient update events.
        
        Adjusts pattern detection parameters based on field gradients.
        
        Args:
            event: Field gradient update event
        """
        try:
            gradients = event.data.get("gradients", {})
            
            # Use coherence gradient to adjust pattern detection sensitivity
            if "coherence" in gradients:
                coherence = gradients["coherence"]
                # Higher coherence means lower threshold (more sensitive)
                # Lower coherence means higher threshold (less sensitive)
                self.dynamic_threshold = max(2, self.threshold - int(coherence * 2))
                self.logger.debug(f"Adjusted threshold to {self.dynamic_threshold} based on coherence {coherence}")
            
            # Use turbulence to adjust confidence calculation
            if "turbulence" in gradients:
                turbulence = gradients["turbulence"]
                # Store turbulence for use in confidence calculation
                try:
                    self.adaptive_id.update_spatial_context("field_turbulence", turbulence, "field_gradient")
                    self.logger.debug(f"Updated field turbulence to {turbulence}")
                except Exception as e:
                    # Handle the case where field_turbulence is not a valid spatial context key
                    # This is an architectural insight: AdaptiveID has predefined spatial context keys
                    # and field_turbulence is not one of them
                    self.logger.debug(f"Could not update field_turbulence: {e}")
                    # Store turbulence as a local attribute instead
                    self._field_turbulence = turbulence
        except Exception as e:
            self.logger.error(f"Error handling field gradient update event: {e}")
    
    def _handle_semantic_observation(self, event):
        """
        Handle semantic observation events.
        
        Processes observations from external sources.
        
        Args:
            event: Semantic observation event
        """
        try:
            observation = event.data.get("observation")
            if observation and "source" in observation and "predicate" in observation and "target" in observation:
                # Forward to semantic observer
                self.semantic_observer.observe_relationship(
                    observation["source"],
                    observation["predicate"],
                    observation["target"],
                    observation.get("context", {})
                )
                self.logger.debug(f"Processed external semantic observation: {observation}")
        except Exception as e:
            self.logger.error(f"Error handling semantic observation event: {e}")
    
    def detect_patterns(self) -> List[Dict[str, Any]]:
        """
        Detect potential patterns and publish events.
        
        This method extends the base implementation to publish events
        when patterns are detected.
        
        Returns:
            List of detected patterns
        """
        # Use dynamic threshold instead of fixed threshold
        original_threshold = self.threshold
        self.threshold = self.dynamic_threshold
        
        # Call the parent implementation to detect patterns
        patterns = super().detect_patterns()
        
        # Restore original threshold
        self.threshold = original_threshold
        
        # Publish events for detected patterns
        if patterns and self.pattern_publisher:
            for pattern in patterns:
                try:
                    self.pattern_publisher.publish_pattern_detected(
                        pattern_id=pattern["id"],
                        pattern_data=pattern,
                        confidence=pattern.get("confidence", 0.5),
                        source="emergent_pattern_detector"
                    )
                    self.logger.debug(f"Published pattern detection event for {pattern['id']}")
                except Exception as e:
                    self.logger.error(f"Error publishing pattern event: {e}")
        
        return patterns
    
    def _check_pattern_evolution(self, pattern: Dict[str, Any]) -> Optional[str]:
        """
        Check if this pattern is an evolution of an existing pattern.
        
        This method extends the base implementation to publish events
        when patterns evolve.
        
        Args:
            pattern: The new pattern to check
            
        Returns:
            ID of the pattern this evolved from, or None
        """
        # Call the parent implementation to check for evolution
        evolved_from = super()._check_pattern_evolution(pattern)
        
        # Publish event if this is an evolution and we have a publisher
        if evolved_from and self.pattern_publisher:
            try:
                # Get the original pattern
                original_pattern = next(
                    (p for p in self.pattern_history if p["id"] == evolved_from), 
                    None
                )
                
                if original_pattern:
                    self.pattern_publisher.publish_pattern_evolved(
                        pattern_id=pattern["id"],
                        old_state=original_pattern,
                        new_state=pattern,
                        source="emergent_pattern_detector"
                    )
                    self.logger.debug(f"Published pattern evolution event for {pattern['id']}")
            except Exception as e:
                self.logger.error(f"Error publishing pattern evolution event: {e}")
        
        return evolved_from
