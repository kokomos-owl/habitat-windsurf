"""
Test for event bus integration with dynamic pattern detection.

This module demonstrates how to integrate the dynamic pattern detection
components with the event bus architecture of the pattern-aware RAG system.
"""

import logging
import uuid
from datetime import datetime

from habitat_evolution.core.services.event_bus import LocalEventBus, Event
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from habitat_evolution.adaptive_core.emergence.semantic_current_observer import SemanticCurrentObserver
from habitat_evolution.adaptive_core.emergence.event_bus_integration import AdaptiveIDEventAdapter, PatternEventPublisher
from habitat_evolution.adaptive_core.emergence.event_aware_detector import EventAwarePatternDetector
from habitat_evolution.adaptive_core.emergence.integration_service import EventBusIntegrationService

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_event_integration():
    """Test the integration of dynamic pattern detection with event bus."""
    logger.info("Starting event integration test")
    
    # Create event bus
    event_bus = LocalEventBus()
    
    # Create integration service
    integration_service = EventBusIntegrationService(event_bus)
    
    # Create AdaptiveID instances
    semantic_id = AdaptiveID(base_concept="semantic_observer", creator_id="test")
    pattern_id = AdaptiveID(base_concept="pattern_detector", creator_id="test")
    
    # Integrate components using the service
    semantic_observer = integration_service.integrate_semantic_observer(
        adaptive_id=semantic_id,
        entity_id="test_semantic_observer"
    )
    
    pattern_detector = integration_service.integrate_pattern_detector(
        semantic_observer=semantic_observer,
        entity_id="test_pattern_detector",
        threshold=2  # Lower threshold for testing
    )
    
    # Subscribe to pattern detection events
    pattern_detected_count = 0
    
    def on_pattern_detected(event):
        nonlocal pattern_detected_count
        pattern_detected_count += 1
        logger.info(f"Pattern detected: {event.data.get('pattern_id')}")
        logger.info(f"Pattern data: {event.data.get('pattern_data')}")
    
    event_bus.subscribe("pattern.detected", on_pattern_detected)
    
    # Create a learning window state event handler to log window state changes
    def on_learning_window_state(event):
        logger.info(f"Learning window state changed: {event.data.get('state')}")
        logger.info(f"Window metrics: {event.data.get('metrics')}")
    
    event_bus.subscribe("learning.window.state", on_learning_window_state)
    
    # Publish a learning window open event
    publisher = integration_service.create_pattern_publisher()
    publisher.publish_learning_window_state(
        window_id="test_window",
        state="OPEN",
        metrics={"turbulence": 0.3, "coherence": 0.7}
    )
    
    # Observe some relationships
    for _ in range(3):
        semantic_observer.observe_relationship(
            source="climate_risk",
            predicate="affects",
            target="coastal_infrastructure",
            context={"location": "Boston Harbor", "severity": "high"}
        )
    
    for _ in range(3):
        semantic_observer.observe_relationship(
            source="sea_level_rise",
            predicate="threatens",
            target="Martha's Vineyard",
            context={"timeframe": "2050", "confidence": "medium"}
        )
    
    # Detect patterns
    patterns = pattern_detector.detect_patterns()
    
    # Verify results
    logger.info(f"Detected {len(patterns)} patterns")
    logger.info(f"Pattern detected events: {pattern_detected_count}")
    
    # Should be equal since we're publishing events for each pattern
    assert len(patterns) == pattern_detected_count, \
        f"Expected {len(patterns)} pattern events, got {pattern_detected_count}"
    
    # Publish a learning window close event
    publisher.publish_learning_window_state(
        window_id="test_window",
        state="CLOSED",
        metrics={"turbulence": 0.1, "coherence": 0.9}
    )
    
    logger.info("Event integration test completed successfully")
    return patterns

if __name__ == "__main__":
    patterns = test_event_integration()
    
    # Print pattern details
    for i, pattern in enumerate(patterns):
        print(f"\nPattern {i+1}:")
        print(f"  ID: {pattern['id']}")
        print(f"  Relationship: {pattern['source']} {pattern['predicate']} {pattern['target']}")
        print(f"  Confidence: {pattern['confidence']:.2f}")
        print(f"  Frequency: {pattern['frequency']}")
