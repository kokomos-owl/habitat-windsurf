"""
Propensity Integration Module

This module integrates the Meta-Pattern Propensity Calculator with the
EmergentPatternDetector to enable automatic registration of patterns and
meta-patterns, creating a feedback loop for pattern prediction.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from src.habitat_evolution.core.services.event_bus import Event, EventBus
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from src.habitat_evolution.adaptive_core.emergence.meta_pattern_propensity import MetaPatternPropensityCalculator
from src.habitat_evolution.adaptive_core.emergence.emergent_pattern_detector import EmergentPatternDetector


class PropensityIntegrationService:
    """
    Integrates the Meta-Pattern Propensity Calculator with pattern detection systems.
    
    This service creates a feedback loop between pattern detection and propensity
    calculation, enabling the system to predict future patterns based on meta-patterns
    and field metrics while adjusting to changing field conditions.
    """
    
    def __init__(
        self,
        propensity_calculator: MetaPatternPropensityCalculator,
        pattern_detector: Optional[EmergentPatternDetector] = None,
        event_bus: Optional[EventBus] = None,
        adaptive_id: Optional[AdaptiveID] = None
    ):
        """
        Initialize the propensity integration service.
        
        Args:
            propensity_calculator: The meta-pattern propensity calculator
            pattern_detector: Optional pattern detector to integrate with
            event_bus: Optional event bus for event-based integration
            adaptive_id: Optional adaptive ID for tracking
        """
        self.propensity_calculator = propensity_calculator
        self.pattern_detector = pattern_detector
        self.event_bus = event_bus
        self.adaptive_id = adaptive_id
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Subscribe to events if event bus is provided
        if self.event_bus:
            self._subscribe_to_events()
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events on the event bus."""
        self.event_bus.subscribe("pattern.detected", self._on_pattern_detected)
        self.event_bus.subscribe("meta_pattern.detected", self._on_meta_pattern_detected)
        self.event_bus.subscribe("field.gradient.update", self._on_field_gradient_update)
        self.event_bus.subscribe("field.state.updated", self._on_field_state_updated)
        
        self.logger.info("PropensityIntegrationService subscribed to events")
    
    def _on_pattern_detected(self, event: Event):
        """
        Handle pattern detection events.
        
        Args:
            event: Pattern detection event
        """
        pattern = event.data.get("pattern")
        if pattern:
            self.logger.debug(f"Registering detected pattern: {pattern.get('id', 'unknown')}")
            self.propensity_calculator.register_pattern(pattern)
            
            # Update AdaptiveID with pattern information if available
            if self.adaptive_id:
                self.adaptive_id.update_temporal_context(
                    "pattern_registered",
                    {
                        "pattern_id": pattern.get("id"),
                        "timestamp": datetime.now().isoformat()
                    },
                    "propensity_integration"
                )
    
    def _on_meta_pattern_detected(self, event: Event):
        """
        Handle meta-pattern detection events.
        
        Args:
            event: Meta-pattern detection event
        """
        meta_pattern = event.data.get("meta_pattern")
        if meta_pattern:
            self.logger.info(f"Registering detected meta-pattern: {meta_pattern.get('id', 'unknown')}")
            self.propensity_calculator.register_meta_pattern(meta_pattern)
            
            # Update AdaptiveID with meta-pattern information if available
            if self.adaptive_id:
                self.adaptive_id.update_temporal_context(
                    "meta_pattern_registered",
                    {
                        "meta_pattern_id": meta_pattern.get("id"),
                        "evolution_type": meta_pattern.get("evolution_type"),
                        "timestamp": datetime.now().isoformat()
                    },
                    "propensity_integration"
                )
    
    def _on_field_gradient_update(self, event: Event):
        """
        Handle field gradient update events.
        
        Args:
            event: Field gradient update event
        """
        gradient = event.data.get("gradient", {})
        metrics = gradient.get("metrics", {})
        
        if metrics:
            self.logger.debug(f"Updating field metrics from gradient: {metrics}")
            self.propensity_calculator.update_field_metrics(metrics)
    
    def _on_field_state_updated(self, event: Event):
        """
        Handle field state update events.
        
        Args:
            event: Field state update event
        """
        field_state = event.data.get("field_state", {})
        field_properties = field_state.get("field_properties", {})
        
        if field_properties:
            metrics = {
                "coherence": field_properties.get("coherence", 0.5),
                "stability": field_properties.get("stability", 0.5),
                "navigability": field_properties.get("navigability_score", 0.5)
            }
            
            self.logger.debug(f"Updating field metrics from field state: {metrics}")
            self.propensity_calculator.update_field_metrics(metrics)
    
    def integrate_with_detector(self, detector: EmergentPatternDetector):
        """
        Integrate with a pattern detector.
        
        Args:
            detector: Pattern detector to integrate with
        """
        self.pattern_detector = detector
        
        # Hook into detector's pattern detection method to register patterns
        original_detect_patterns = detector.detect_patterns
        
        def detect_patterns_hook(*args, **kwargs):
            patterns = original_detect_patterns(*args, **kwargs)
            
            # Register detected patterns with propensity calculator
            for pattern in patterns:
                self.propensity_calculator.register_pattern(pattern)
            
            return patterns
        
        # Replace the method with our hooked version
        detector.detect_patterns = detect_patterns_hook
        
        # Hook into detector's meta-pattern detection method
        original_detect_meta_patterns = detector._detect_meta_patterns
        
        def detect_meta_patterns_hook(*args, **kwargs):
            meta_patterns = original_detect_meta_patterns(*args, **kwargs)
            
            # Register detected meta-patterns with propensity calculator
            for meta_pattern in meta_patterns:
                self.propensity_calculator.register_meta_pattern(meta_pattern)
            
            return meta_patterns
        
        # Replace the method with our hooked version
        detector._detect_meta_patterns = detect_meta_patterns_hook
        
        self.logger.info(f"Integrated with pattern detector: {detector.__class__.__name__}")
    
    def get_top_propensities(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top N pattern propensities.
        
        Args:
            n: Number of top propensities to return
            
        Returns:
            List of dictionaries with pattern and propensity
        """
        return self.propensity_calculator.get_top_propensities(n)
    
    def publish_propensity_update(self):
        """Publish a propensity update event with the latest propensities."""
        if not self.event_bus:
            return
        
        # Get top propensities
        propensities = self.get_top_propensities(10)
        
        # Publish event
        self.event_bus.publish(Event.create(
            type="pattern.propensity.update",
            source=self.adaptive_id.id if self.adaptive_id else "propensity_integration",
            data={
                "propensities": propensities,
                "timestamp": datetime.now().isoformat()
            }
        ))
        
        self.logger.info(f"Published propensity update with {len(propensities)} propensities")
