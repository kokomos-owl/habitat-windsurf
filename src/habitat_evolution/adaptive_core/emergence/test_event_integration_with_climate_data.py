"""
Test for event bus integration with dynamic pattern detection using real climate data.

This module demonstrates how to integrate the dynamic pattern detection
components with the event bus architecture of the pattern-aware RAG system,
using real climate risk data from the data/climate_risk directory.
"""

import logging
import uuid
import os
from datetime import datetime
from unittest.mock import MagicMock

from ...core.services.event_bus import LocalEventBus, Event
from ..id.adaptive_id import AdaptiveID
from .semantic_current_observer import SemanticCurrentObserver
from .enhanced_semantic_observer import EnhancedSemanticObserver
from .event_bus_integration import AdaptiveIDEventAdapter, PatternEventPublisher
from .event_aware_detector import EventAwarePatternDetector
from .integration_service import EventBusIntegrationService
from .climate_data_loader import ClimateDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_field_navigator():
    """Create a mock field navigator for testing."""
    mock = MagicMock()
    mock.get_position.return_value = {"x": 0.5, "y": 0.5, "z": 0.5}
    mock.register_observer = MagicMock()
    return mock

def create_mock_journey_tracker():
    """Create a mock journey tracker for testing."""
    mock = MagicMock()
    mock.record_state_change = MagicMock()
    return mock

def test_event_integration_with_climate_data():
    """Test the integration of dynamic pattern detection with event bus using climate data."""
    logger.info("Starting event integration test with climate data")
    
    # Create event bus
    event_bus = LocalEventBus()
    
    # Create integration service
    integration_service = EventBusIntegrationService(event_bus)
    
    # Create AdaptiveID instance for pattern detector
    pattern_id = AdaptiveID(base_concept="pattern_detector", creator_id="test")
    
    # Create mock field navigator and journey tracker
    mock_field_navigator = create_mock_field_navigator()
    mock_journey_tracker = create_mock_journey_tracker()
    
    # Integrate components using the service
    semantic_observer = integration_service.integrate_semantic_observer(
        field_navigator=mock_field_navigator,
        journey_tracker=mock_journey_tracker,
        entity_id="test_semantic_observer"
    )
    
    pattern_detector = integration_service.integrate_pattern_detector(
        semantic_observer=semantic_observer,
        entity_id="test_pattern_detector",
        threshold=2  # Lower threshold for testing
    )
    
    # Subscribe to pattern detection events
    pattern_detected_count = 0
    detected_patterns = []
    
    def on_pattern_detected(event):
        nonlocal pattern_detected_count, detected_patterns
        pattern_detected_count += 1
        pattern_data = event.data.get('pattern_data', {})
        detected_patterns.append(pattern_data)
        logger.info(f"Pattern detected: {event.data.get('pattern_id')}")
        logger.info(f"Pattern relationship: {pattern_data.get('source')} {pattern_data.get('predicate')} {pattern_data.get('target')}")
    
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
    
    # Load climate risk data
    climate_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 'data', 'climate_risk')
    climate_loader = ClimateDataLoader(climate_data_dir)
    
    # Load all climate risk files
    climate_loader.load_all_files()
    
    # Get extracted relationships
    relationships = climate_loader.relationships
    logger.info(f"Loaded {len(relationships)} relationships from climate risk data")
    
    # Feed relationships into the semantic observer
    for rel in relationships[:30]:  # Use a subset for testing
        semantic_observer.observe_relationship(
            source=rel["source"],
            predicate=rel["predicate"],
            target=rel["target"],
            context=rel.get("context", {})
        )
        logger.info(f"Observed relationship: {rel['source']} {rel['predicate']} {rel['target']}")
    
    # Generate and feed synthetic relationships to ensure pattern detection
    synthetic_relationships = climate_loader.generate_synthetic_relationships(count=15)
    logger.info(f"Generated {len(synthetic_relationships)} synthetic relationships")
    
    for rel in synthetic_relationships:
        for _ in range(3):  # Observe each synthetic relationship multiple times to ensure pattern detection
            semantic_observer.observe_relationship(
                source=rel["source"],
                predicate=rel["predicate"],
                target=rel["target"],
                context=rel.get("context", {})
            )
        logger.info(f"Observed synthetic relationship: {rel['source']} {rel['predicate']} {rel['target']}")
    
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
    
    logger.info("Event integration test with climate data completed successfully")
    return patterns, detected_patterns

if __name__ == "__main__":
    patterns, detected_patterns = test_event_integration_with_climate_data()
    
    # Print pattern details
    print("\nDetected Patterns:")
    for i, pattern in enumerate(patterns):
        print(f"\nPattern {i+1}:")
        print(f"  ID: {pattern['id']}")
        print(f"  Relationship: {pattern['source']} {pattern['predicate']} {pattern['target']}")
        print(f"  Confidence: {pattern['confidence']:.2f}")
        print(f"  Frequency: {pattern['frequency']}")
    
    # Print pattern statistics
    predicates = {}
    sources = {}
    targets = {}
    
    for pattern in patterns:
        pred = pattern['predicate']
        src = pattern['source']
        tgt = pattern['target']
        
        predicates[pred] = predicates.get(pred, 0) + 1
        sources[src] = sources.get(src, 0) + 1
        targets[tgt] = targets.get(tgt, 0) + 1
    
    print("\nPattern Statistics:")
    print(f"  Total patterns: {len(patterns)}")
    print(f"  Unique predicates: {len(predicates)}")
    print(f"  Unique sources: {len(sources)}")
    print(f"  Unique targets: {len(targets)}")
    
    print("\nTop Predicates:")
    for pred, count in sorted(predicates.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {pred}: {count}")
    
    print("\nTop Sources:")
    for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {src}: {count}")
    
    print("\nTop Targets:")
    for tgt, count in sorted(targets.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {tgt}: {count}")
