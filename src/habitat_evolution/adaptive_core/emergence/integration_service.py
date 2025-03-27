"""
Integration service for connecting dynamic pattern detection with the event bus.

This module provides a service to manage the integration between dynamic pattern
detection components and the event bus architecture of the pattern-aware RAG system.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from ..id.adaptive_id import AdaptiveID
from .semantic_current_observer import SemanticCurrentObserver
from .event_aware_detector import EventAwarePatternDetector
from .resonance_trail_observer import ResonanceTrailObserver
from .event_bus_integration import AdaptiveIDEventAdapter, PatternEventPublisher

logger = logging.getLogger(__name__)

class EventBusIntegrationService:
    """
    Service to manage integration between dynamic pattern detection and event bus.
    
    This service creates and manages the necessary adapters and publishers
    to connect our components with the event bus, enabling them to participate
    in the sophisticated regulatory relationships of the pattern-aware RAG system.
    """
    
    def __init__(self, event_bus):
        """
        Initialize the integration service.
        
        Args:
            event_bus: The event bus to integrate with
        """
        self.event_bus = event_bus
        self.adapters = {}  # entity_id -> AdaptiveIDEventAdapter
        self.publishers = {}  # entity_id -> PatternEventPublisher
        self.integrated_components = {}  # entity_id -> component
        
        # Create a default pattern publisher
        self.default_publisher = PatternEventPublisher(event_bus)
        
        logger.info("Initialized event bus integration service")
    
    def integrate_adaptive_id(self, adaptive_id: AdaptiveID, entity_id: str = None) -> AdaptiveIDEventAdapter:
        """
        Integrate an AdaptiveID instance with the event bus.
        
        Creates an adapter that converts AdaptiveID state changes to event bus events.
        
        Args:
            adaptive_id: The AdaptiveID instance to integrate
            entity_id: Optional entity ID (defaults to adaptive_id.id)
            
        Returns:
            The created adapter
        """
        entity_id = entity_id or adaptive_id.id
        
        # Create adapter if it doesn't exist
        if entity_id not in self.adapters:
            adapter = AdaptiveIDEventAdapter(adaptive_id, self.event_bus)
            self.adapters[entity_id] = adapter
            logger.info(f"Integrated AdaptiveID {entity_id} with event bus")
        
        return self.adapters[entity_id]
    
    def create_pattern_publisher(self, entity_id: str = None) -> PatternEventPublisher:
        """
        Create a pattern event publisher.
        
        Args:
            entity_id: Optional entity ID for the publisher
            
        Returns:
            The created publisher
        """
        if not entity_id:
            return self.default_publisher
        
        # Create publisher if it doesn't exist
        if entity_id not in self.publishers:
            publisher = PatternEventPublisher(self.event_bus)
            self.publishers[entity_id] = publisher
            logger.info(f"Created pattern publisher for {entity_id}")
        
        return self.publishers[entity_id]
    
    def integrate_pattern_detector(
        self, 
        semantic_observer: SemanticCurrentObserver,
        entity_id: str = None,
        threshold: int = 3
    ) -> EventAwarePatternDetector:
        """
        Create and integrate an event-aware pattern detector.
        
        Args:
            semantic_observer: The semantic observer to use
            entity_id: Optional entity ID for the detector
            threshold: Pattern detection threshold
            
        Returns:
            The created detector
        """
        entity_id = entity_id or f"pattern_detector_{len(self.integrated_components)}"
        
        # Create publisher
        publisher = self.create_pattern_publisher(entity_id)
        
        # Create detector
        detector = EventAwarePatternDetector(
            semantic_observer=semantic_observer,
            event_bus=self.event_bus,
            pattern_publisher=publisher,
            threshold=threshold
        )
        
        # Integrate detector's AdaptiveID
        self.integrate_adaptive_id(detector.adaptive_id, entity_id)
        
        # Store in integrated components
        self.integrated_components[entity_id] = detector
        logger.info(f"Integrated pattern detector {entity_id} with event bus")
        
        return detector
    
    def integrate_resonance_observer(
        self,
        adaptive_id: AdaptiveID,
        entity_id: str = None
    ) -> ResonanceTrailObserver:
        """
        Integrate a resonance trail observer with the event bus.
        
        Args:
            adaptive_id: The AdaptiveID to use
            entity_id: Optional entity ID for the observer
            
        Returns:
            The integrated observer
        """
        entity_id = entity_id or f"resonance_observer_{len(self.integrated_components)}"
        
        # Create observer
        observer = ResonanceTrailObserver(adaptive_id=adaptive_id)
        
        # Integrate observer's AdaptiveID
        self.integrate_adaptive_id(adaptive_id, entity_id)
        
        # Create publisher
        publisher = self.create_pattern_publisher(entity_id)
        
        # Add publisher to observer (monkey patch)
        observer.pattern_publisher = publisher
        
        # Patch the observe_pattern_movement method to publish events
        original_observe = observer.observe_pattern_movement
        
        def observe_with_events(pattern_id, old_position, new_position, timestamp):
            # Call original method
            result = original_observe(pattern_id, old_position, new_position, timestamp)
            
            # Publish event
            try:
                publisher.publish_pattern_resonance(
                    pattern_id=pattern_id,
                    position={"old": old_position, "new": new_position},
                    strength=1.0,  # Default strength
                    source=f"resonance_observer:{entity_id}"
                )
                logger.debug(f"Published resonance event for {pattern_id}")
            except Exception as e:
                logger.error(f"Error publishing resonance event: {e}")
            
            return result
        
        # Replace method
        observer.observe_pattern_movement = observe_with_events
        
        # Store in integrated components
        self.integrated_components[entity_id] = observer
        logger.info(f"Integrated resonance observer {entity_id} with event bus")
        
        return observer
    
    def integrate_semantic_observer(
        self,
        adaptive_id: AdaptiveID,
        entity_id: str = None
    ) -> SemanticCurrentObserver:
        """
        Integrate a semantic current observer with the event bus.
        
        Args:
            adaptive_id: The AdaptiveID to use
            entity_id: Optional entity ID for the observer
            
        Returns:
            The integrated observer
        """
        entity_id = entity_id or f"semantic_observer_{len(self.integrated_components)}"
        
        # Create observer
        observer = SemanticCurrentObserver(adaptive_id=adaptive_id)
        
        # Integrate observer's AdaptiveID
        self.integrate_adaptive_id(adaptive_id, entity_id)
        
        # Store in integrated components
        self.integrated_components[entity_id] = observer
        logger.info(f"Integrated semantic observer {entity_id} with event bus")
        
        return observer
