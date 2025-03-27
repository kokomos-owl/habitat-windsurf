"""
Test for learning window integration with dynamic pattern detection.

This module demonstrates how to integrate the dynamic pattern detection
components with the learning window system, enabling field-aware state
transitions and back pressure control for pattern evolution.
"""

import logging
import uuid
import os
from datetime import datetime, timedelta
import time
from unittest.mock import MagicMock

# Use absolute imports to avoid module path issues
from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from src.habitat_evolution.adaptive_core.emergence.enhanced_semantic_observer import EnhancedSemanticObserver
from src.habitat_evolution.adaptive_core.emergence.event_bus_integration import PatternEventPublisher
from src.habitat_evolution.adaptive_core.emergence.event_aware_detector import EventAwarePatternDetector
from src.habitat_evolution.adaptive_core.emergence.integration_service import EventBusIntegrationService
from src.habitat_evolution.adaptive_core.emergence.climate_data_loader import ClimateDataLoader
from src.habitat_evolution.adaptive_core.emergence.learning_window_integration import LearningWindowAwareDetector, FieldAwarePatternController
from src.habitat_evolution.pattern_aware_rag.learning.learning_control import WindowState, BackPressureController

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

def test_learning_window_integration():
    """Test the integration of dynamic pattern detection with learning windows."""
    logger.info("Starting learning window integration test")
    
    # Create event bus
    event_bus = LocalEventBus()
    
    # Create integration service
    integration_service = EventBusIntegrationService(event_bus)
    
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
    
    # Create pattern publisher
    pattern_publisher = integration_service.create_pattern_publisher("test_publisher")
    
    # Create back pressure controller
    back_pressure = BackPressureController(
        base_delay=0.1,
        max_delay=2.0,
        stability_threshold=0.7
    )
    
    # Create learning window aware detector
    learning_window_detector = LearningWindowAwareDetector(
        detector=pattern_detector,
        pattern_publisher=pattern_publisher,
        back_pressure_controller=back_pressure
    )
    
    # Create field-aware pattern controller
    field_controller = FieldAwarePatternController(
        detector=learning_window_detector,
        event_bus=event_bus
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
    
    # Subscribe to learning window state events
    def on_learning_window_state(event):
        logger.info(f"Learning window state changed: {event.data.get('state')}")
        logger.info(f"Window metrics: {event.data.get('metrics')}")
    
    event_bus.subscribe("learning.window.state", on_learning_window_state)
    
    # Load climate risk data
    climate_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 'data', 'climate_risk')
    climate_loader = ClimateDataLoader(climate_data_dir)
    
    # Load all climate risk files
    climate_loader.load_all_files()
    
    # Get extracted relationships
    relationships = climate_loader.relationships
    logger.info(f"Loaded {len(relationships)} relationships from climate risk data")
    
    # Generate synthetic relationships to ensure pattern detection
    synthetic_relationships = climate_loader.generate_synthetic_relationships(count=15)
    logger.info(f"Generated {len(synthetic_relationships)} synthetic relationships")
    
    # Test pattern detection with window in CLOSED state
    logger.info("Testing pattern detection with window CLOSED")
    learning_window_detector.update_window_state(WindowState.CLOSED)
    
    # Feed some relationships - these should not generate patterns since window is closed
    for rel in synthetic_relationships[:5]:
        for _ in range(3):
            semantic_observer.observe_relationship(
                source=rel["source"],
                predicate=rel["predicate"],
                target=rel["target"],
                context=rel.get("context", {})
            )
        logger.info(f"Observed relationship: {rel['source']} {rel['predicate']} {rel['target']}")
    
    # Detect patterns - should return empty list since window is closed
    patterns_closed = field_controller.detect_patterns()
    logger.info(f"Patterns detected with window CLOSED: {len(patterns_closed)}")
    
    # Test pattern detection with window in OPEN state
    logger.info("Testing pattern detection with window OPEN")
    learning_window_detector.update_window_state(WindowState.OPEN)
    
    # Feed more relationships
    for rel in synthetic_relationships[5:10]:
        for _ in range(3):
            semantic_observer.observe_relationship(
                source=rel["source"],
                predicate=rel["predicate"],
                target=rel["target"],
                context=rel.get("context", {})
            )
        logger.info(f"Observed relationship: {rel['source']} {rel['predicate']} {rel['target']}")
    
    # Detect patterns - should detect patterns since window is open
    patterns_open = field_controller.detect_patterns()
    logger.info(f"Patterns detected with window OPEN: {len(patterns_open)}")
    
    # Test field-aware transitions
    logger.info("Testing field-aware transitions")
    
    # Create a new learning window with field-aware transitions
    field_controller.create_new_learning_window(
        duration_minutes=5,
        field_aware=True
    )
    
    # Publish field gradient updates
    logger.info("Publishing field gradient with low coherence")
    event_bus.publish(Event.create(
        "field.gradient.update",
        {
            "gradients": {
                "coherence": 0.3,  # Low coherence should close the window
                "turbulence": 0.7,
                "stability": 0.5
            }
        },
        source="test"
    ))
    
    # Short delay to allow event processing
    time.sleep(0.1)
    
    # Check window state - should be CLOSED due to low coherence
    logger.info(f"Window state after low coherence: {learning_window_detector.window_state}")
    
    # Publish field gradient with high coherence
    logger.info("Publishing field gradient with high coherence")
    event_bus.publish(Event.create(
        "field.gradient.update",
        {
            "gradients": {
                "coherence": 0.8,  # High coherence should open the window
                "turbulence": 0.2,
                "stability": 0.9
            }
        },
        source="test"
    ))
    
    # Short delay to allow event processing
    time.sleep(0.1)
    
    # Check window state - should be OPENING due to high coherence
    logger.info(f"Window state after high coherence: {learning_window_detector.window_state}")
    
    # Manually force window to OPEN state to test back pressure
    # In a real scenario, this would happen after the 30-second delay
    logger.info("Manually transitioning window to OPEN state to test back pressure")
    learning_window_detector.update_window_state(WindowState.OPEN)
    logger.info(f"Window state after manual transition: {learning_window_detector.window_state}")
    
    # Reset detection delay to ensure back pressure starts from baseline
    learning_window_detector.detection_delay = learning_window_detector.back_pressure_controller.base_delay
    learning_window_detector.last_detection_time = datetime.now() - timedelta(seconds=10)  # Ensure first detection passes
    
    # Test back pressure control with window in OPEN state
    logger.info("Testing back pressure control with OPEN window")
    
    # Feed relationships in rapid succession to trigger back pressure
    detected_pattern_counts = []
    for i, rel in enumerate(synthetic_relationships[10:]):
        # Feed relationship multiple times to ensure pattern detection
        for _ in range(3):
            semantic_observer.observe_relationship(
                source=rel["source"],
                predicate=rel["predicate"],
                target=rel["target"],
                context=rel.get("context", {})
            )
        
        # Try to detect patterns - first should work, subsequent ones should be limited by back pressure
        patterns = field_controller.detect_patterns()
        detected_pattern_counts.append(len(patterns))
        logger.info(f"Iteration {i+1}: Patterns detected: {len(patterns)}, Delay: {learning_window_detector.detection_delay:.2f}s")
        
        # Short delay but not enough to bypass back pressure
        time.sleep(0.05)
    
    # Verify back pressure is working by checking pattern counts
    non_zero_detections = sum(1 for count in detected_pattern_counts if count > 0)
    logger.info(f"Detected patterns in {non_zero_detections} out of {len(detected_pattern_counts)} iterations")
    assert non_zero_detections > 0, "No patterns were detected even with OPEN window"
    assert non_zero_detections < len(detected_pattern_counts), "Back pressure did not limit pattern detection"
    
    logger.info("Learning window integration test completed")
    return detected_patterns

if __name__ == "__main__":
    detected_patterns = test_learning_window_integration()
    
    # Print pattern statistics
    print("\nDetected Patterns Summary:")
    print(f"Total patterns detected: {len(detected_patterns)}")
    
    if detected_patterns:
        predicates = {}
        sources = {}
        targets = {}
        
        for pattern in detected_patterns:
            pred = pattern.get('predicate', 'unknown')
            src = pattern.get('source', 'unknown')
            tgt = pattern.get('target', 'unknown')
            
            predicates[pred] = predicates.get(pred, 0) + 1
            sources[src] = sources.get(src, 0) + 1
            targets[tgt] = targets.get(tgt, 0) + 1
        
        print("\nTop Predicates:")
        for pred, count in sorted(predicates.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {pred}: {count}")
        
        print("\nTop Sources:")
        for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {src}: {count}")
        
        print("\nTop Targets:")
        for tgt, count in sorted(targets.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {tgt}: {count}")
    else:
        print("No patterns were detected during the test.")
