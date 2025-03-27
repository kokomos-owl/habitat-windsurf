"""
Test for tonic-harmonic integration with pattern detection.

This module demonstrates how to integrate the vector+ tonic-harmonics system
with pattern detection, enabling semantic boundary detection and enhanced
pattern identification through harmonic resonance analysis.
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
from src.habitat_evolution.field.field_state import TonicHarmonicFieldState
from src.habitat_evolution.field.harmonic_field_io_bridge import HarmonicFieldIOBridge
from src.habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService, OperationType
from src.habitat_evolution.adaptive_core.resonance.tonic_harmonic_metrics import TonicHarmonicMetrics
from src.habitat_evolution.adaptive_core.emergence.enhanced_semantic_observer import EnhancedSemanticObserver
from src.habitat_evolution.adaptive_core.emergence.event_bus_integration import PatternEventPublisher
from src.habitat_evolution.adaptive_core.emergence.event_aware_detector import EventAwarePatternDetector
from src.habitat_evolution.adaptive_core.emergence.integration_service import EventBusIntegrationService
from src.habitat_evolution.adaptive_core.emergence.learning_window_integration import LearningWindowAwareDetector, FieldAwarePatternController
from src.habitat_evolution.adaptive_core.emergence.tonic_harmonic_integration import TonicHarmonicPatternDetector, VectorPlusFieldBridge
from src.habitat_evolution.adaptive_core.emergence.climate_data_loader import ClimateDataLoader
from src.habitat_evolution.pattern_aware_rag.learning.learning_control import WindowState, BackPressureController

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_field_state():
    """Create a mock field state for testing."""
    # Create a simplified field analysis result
    field_analysis = {
        "topology": {
            "effective_dimensionality": 3,
            "principal_dimensions": [0, 1, 2],
            "eigenvalues": [0.8, 0.5, 0.2],
            "eigenvectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        },
        "metrics": {
            "coherence": 0.75,
            "turbulence": 0.3,
            "stability": 0.8
        }
    }
    
    # Create field state
    field_state = TonicHarmonicFieldState(field_analysis)
    
    # Add methods for metrics if they don't exist
    if not hasattr(field_state, 'get_coherence_metric'):
        field_state.get_coherence_metric = lambda: field_analysis["metrics"]["coherence"]
    
    if not hasattr(field_state, 'get_turbulence_metric'):
        field_state.get_turbulence_metric = lambda: field_analysis["metrics"]["turbulence"]
    
    if not hasattr(field_state, 'get_stability_metric'):
        field_state.get_stability_metric = lambda: field_analysis["metrics"]["stability"]
    
    return field_state


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


def test_tonic_harmonic_integration():
    """Test the integration of vector+ tonic-harmonics with pattern detection."""
    logger.info("Starting tonic-harmonic integration test")
    
    # Create event bus
    event_bus = LocalEventBus()
    
    # Create harmonic I/O service
    harmonic_io_service = HarmonicIOService(base_frequency=0.2, harmonics=3)
    
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
    
    # Create tonic-harmonic detector
    tonic_harmonic_detector = TonicHarmonicPatternDetector(
        base_detector=learning_window_detector,
        harmonic_io_service=harmonic_io_service,
        event_bus=event_bus
    )
    
    # Create vector+ field bridge
    vector_bridge = VectorPlusFieldBridge(
        event_bus=event_bus,
        harmonic_io_service=harmonic_io_service
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
    
    # Subscribe to semantic boundary events
    semantic_boundary_count = 0
    
    def on_semantic_boundary(event):
        nonlocal semantic_boundary_count
        semantic_boundary_count += 1
        boundary_type = event.data.get('boundary_type', 'unknown')
        pattern_id = event.data.get('pattern_id', 'unknown')
        logger.info(f"Semantic boundary detected: {boundary_type} for pattern {pattern_id}")
    
    event_bus.subscribe("pattern.semantic_boundary", on_semantic_boundary)
    
    # Subscribe to vector representation events
    vector_representation_count = 0
    
    def on_vector_representation(event):
        nonlocal vector_representation_count
        vector_representation_count += 1
        pattern_id = event.data.get('pattern_id', 'unknown')
        representation = event.data.get('vector_representation', {})
        logger.info(f"Vector representation received for pattern {pattern_id}")
        if representation:
            logger.info(f"Combined vector: {representation.get('combined_vector', [])}")
    
    event_bus.subscribe("pattern.vector_representation", on_vector_representation)
    
    # Subscribe to harmonic analysis events
    harmonic_analysis_count = 0
    
    def on_harmonic_analysis(event):
        nonlocal harmonic_analysis_count
        harmonic_analysis_count += 1
        pattern_id = event.data.get('pattern_id', 'unknown')
        analysis = event.data.get('analysis', {})
        logger.info(f"Harmonic analysis received for pattern {pattern_id}")
        if analysis:
            coherence = analysis.get('harmonic_coherence', [])
            if coherence:
                logger.info(f"Harmonic coherence: {len(coherence)} domains analyzed")
    
    event_bus.subscribe("pattern.harmonic_analysis", on_harmonic_analysis)
    
    # Create and publish field state
    field_state = create_mock_field_state()
    
    # Publish field state update
    event_bus.publish(Event(
        "field.state.updated",
        {
            "field_state": {
                "topology": {
                    "effective_dimensionality": field_state.effective_dimensionality,
                    "principal_dimensions": field_state.principal_dimensions,
                    "eigenvalues": field_state.eigenvalues,
                    "eigenvectors": field_state.eigenvectors
                },
                "metrics": {
                    "coherence": field_state.get_coherence_metric(),
                    "turbulence": field_state.get_turbulence_metric(),
                    "stability": field_state.get_stability_metric()
                }
            }
        },
        source="test"
    ))
    
    # Short delay to allow event processing
    time.sleep(0.1)
    
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
    
    # Set learning window to OPEN
    learning_window_detector.update_window_state(WindowState.OPEN)
    
    # Feed relationships to observe
    for rel in synthetic_relationships[:10]:
        for _ in range(3):  # Repeat to ensure pattern detection
            semantic_observer.observe_relationship(
                source=rel["source"],
                predicate=rel["predicate"],
                target=rel["target"],
                context=rel.get("context", {})
            )
        logger.info(f"Observed relationship: {rel['source']} {rel['predicate']} {rel['target']}")
    
    # Detect patterns with tonic-harmonic awareness
    patterns = tonic_harmonic_detector.detect_patterns()
    logger.info(f"Detected {len(patterns)} patterns with tonic-harmonic awareness")
    
    # Test semantic boundary detection
    if patterns:
        # Create a pattern evolution that crosses a semantic boundary
        original_pattern = patterns[0]
        evolved_pattern = original_pattern.copy()
        evolved_pattern["predicate"] = "impacts" if original_pattern["predicate"] != "impacts" else "affects"
        
        # Publish pattern evolution event
        event_bus.publish(Event(
            "pattern.evolved",
            {
                "pattern_id": str(uuid.uuid4()),
                "from_state": original_pattern,
                "to_state": evolved_pattern
            },
            source="test"
        ))
        
        # Short delay to allow event processing
        time.sleep(0.1)
        
        logger.info(f"Semantic boundaries detected: {semantic_boundary_count}")
    
    # Test field gradient updates
    event_bus.publish(Event(
        "vector.gradient.updated",
        {
            "gradient": {
                "coherence": 0.85,
                "turbulence": 0.2,
                "stability": 0.9
            }
        },
        source="test"
    ))
    
    # Short delay to allow event processing
    time.sleep(0.1)
    
    # Test changing field state
    event_bus.publish(Event(
        "field.state.updated",
        {
            "field_state": {
                "topology": {
                    "effective_dimensionality": 2,
                    "principal_dimensions": [0, 1],
                    "eigenvalues": [0.9, 0.4],
                    "eigenvectors": [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]
                },
                "metrics": {
                    "coherence": 0.9,
                    "turbulence": 0.1,
                    "stability": 0.95
                }
            }
        },
        source="test"
    ))
    
    # Short delay to allow event processing
    time.sleep(0.1)
    
    # Feed more relationships with high coherence field state
    for rel in synthetic_relationships[10:]:
        for _ in range(3):
            semantic_observer.observe_relationship(
                source=rel["source"],
                predicate=rel["predicate"],
                target=rel["target"],
                context=rel.get("context", {})
            )
        logger.info(f"Observed relationship: {rel['source']} {rel['predicate']} {rel['target']}")
    
    # Detect patterns again with updated field state
    patterns_after_update = tonic_harmonic_detector.detect_patterns()
    logger.info(f"Detected {len(patterns_after_update)} patterns after field state update")
    
    # Print statistics
    logger.info("\nTest Statistics:")
    logger.info(f"Total patterns detected: {pattern_detected_count}")
    logger.info(f"Semantic boundaries detected: {semantic_boundary_count}")
    logger.info(f"Vector representations generated: {vector_representation_count}")
    logger.info(f"Harmonic analyses performed: {harmonic_analysis_count}")
    
    logger.info("Tonic-harmonic integration test completed")
    return detected_patterns


if __name__ == "__main__":
    detected_patterns = test_tonic_harmonic_integration()
    
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
