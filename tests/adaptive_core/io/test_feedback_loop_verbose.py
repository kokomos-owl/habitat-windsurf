"""
Verbose test script for HarmonicIOService feedback loop.

This script demonstrates the feedback loop functionality with detailed logging,
showing how the system responds to meta-pattern detection and topology metrics.
"""

import sys
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any

# Add src to path if not already there
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import core components
from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService
from src.habitat_evolution.field.harmonic_field_io_bridge import HarmonicFieldIOBridge
from src.habitat_evolution.field.field_state import TonicHarmonicFieldState
from src.habitat_evolution.adaptive_core.resonance.tonic_harmonic_metrics import TonicHarmonicMetrics
from src.habitat_evolution.adaptive_core.emergence.learning_window_integration import LearningWindowAwareDetector
from src.habitat_evolution.adaptive_core.emergence.tonic_harmonic_integration import TonicHarmonicPatternDetector
from src.habitat_evolution.pattern_aware_rag.learning.learning_control import WindowState, BackPressureController

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("feedback_loop_test")

class MockLearningWindowDetector(LearningWindowAwareDetector):
    """Mock learning window aware detector for testing."""
    
    def __init__(self, event_bus):
        """Initialize the mock detector."""
        self.event_bus = event_bus
        self.detected_patterns = []
        self.window_state = WindowState.OPENING
        self.back_pressure = BackPressureController()
        self.logger = logging.getLogger("MockDetector")
    
    def detect_patterns(self, data):
        """Detect patterns in data."""
        pattern = {"id": f"pattern_{len(self.detected_patterns)}", "data": data}
        self.detected_patterns.append(pattern)
        self.logger.info(f"Detected pattern: {pattern['id']}")
        return pattern
    
    def get_window_state(self):
        """Get the current window state."""
        return self.window_state
    
    def set_window_state(self, state):
        """Set the window state."""
        self.window_state = state
        self.logger.info(f"Window state changed to: {state}")
    
    def get_back_pressure(self):
        """Get the back pressure controller."""
        return self.back_pressure

def event_tracker(event):
    """Track events for debugging."""
    logger.debug(f"Event received: {event.type if hasattr(event, 'type') else event.get('type')}")

def run_verbose_test():
    """Run a verbose test of the feedback loop functionality."""
    logger.info("Starting verbose feedback loop test")
    
    # Create event bus
    event_bus = LocalEventBus()
    event_bus.subscribe("*", event_tracker)
    
    # Create HarmonicIOService with event bus
    io_service = HarmonicIOService(0.2, event_bus=event_bus)
    io_service.start()
    
    # Create mock components
    base_detector = MockLearningWindowDetector(event_bus)
    field_bridge = HarmonicFieldIOBridge()
    metrics = TonicHarmonicMetrics()
    
    # Create field analysis for field state
    field_analysis = {
        "density": 0.7,
        "topology": {
            "effective_dimensionality": 4,
            "principal_dimensions": [0.9, 0.7, 0.5, 0.3]
        },
        "field_properties": {
            "coherence": 0.6,
            "stability": 0.5
        }
    }
    
    # Create field state
    field_state = TonicHarmonicFieldState(field_analysis)
    
    # Create TonicHarmonicPatternDetector
    tonic_detector = TonicHarmonicPatternDetector(
        base_detector=base_detector,
        harmonic_io_service=io_service,
        event_bus=event_bus,
        field_bridge=field_bridge,
        metrics=metrics
    )
    
    logger.info("Components initialized")
    
    # Log initial state
    logger.info(f"Initial parameters:")
    logger.info(f"  Base frequency: {io_service.base_frequency}")
    logger.info(f"  Eigenspace stability: {io_service.eigenspace_stability}")
    logger.info(f"  Pattern coherence: {io_service.pattern_coherence}")
    
    # Step 1: Publish field state update
    logger.info("\n=== Step 1: Field State Update ===")
    event_bus.publish(Event.create(
        type="field.state.updated",
        source="test_script",
        data={
            "stability": 0.65,
            "coherence": 0.75,
            "resonance": 0.6
        }
    ))
    
    # Allow time for processing
    time.sleep(0.5)
    
    # Step 2: Publish field gradient with topology data
    logger.info("\n=== Step 2: Field Gradient Update with Topology Data ===")
    
    # Create detailed topology data
    topology_data = {
        "resonance_centers": {
            "center_1": [0.1, 0.2, 0.3],
            "center_2": [0.4, 0.5, 0.6],
            "center_3": [0.7, 0.8, 0.9]
        },
        "interference_patterns": {
            "pattern_1": [0.2, 0.3, 0.4],
            "pattern_2": [0.5, 0.6, 0.7]
        },
        "field_density_centers": {
            "density_1": [0.3, 0.4, 0.5],
            "density_2": [0.6, 0.7, 0.8],
            "density_3": [0.9, 1.0, 1.1],
            "density_4": [1.2, 1.3, 1.4]
        },
        "flow_vectors": {
            "vector_1": [0.1, 0.1, 0.1],
            "vector_2": [0.2, 0.2, 0.2],
            "vector_3": [0.3, 0.3, 0.3]
        },
        "effective_dimensionality": 3,
        "principal_dimensions": [0.8, 0.6, 0.4]
    }
    
    # Create vectors data
    vectors_data = {
        "gradient_1": [0.1, 0.2, 0.3],
        "gradient_2": [0.4, 0.5, 0.6],
        "gradient_3": [0.7, 0.8, 0.9]
    }
    
    # Create metrics data
    metrics_data = {
        "field_energy": 0.75,
        "pattern_density": 0.65,
        "coherence_index": 0.85
    }
    
    # Publish field gradient update
    event_bus.publish(Event.create(
        type="field.gradient.update",
        source="test_field_service",
        data={
            "gradients": metrics_data,
            "gradient": {
                "metrics": metrics_data,
                "vectors": vectors_data,
                "topology": topology_data
            },
            "topology": topology_data,
            "vectors": vectors_data,
            "timestamp": datetime.now().isoformat()
        }
    ))
    
    # Allow time for processing
    time.sleep(1.0)
    
    # Step 3: Publish meta-pattern detection event for object_evolution
    logger.info("\n=== Step 3: Meta-Pattern Detection (object_evolution) ===")
    
    # Store parameters before adjustment
    pre_adjustment_frequency = io_service.base_frequency
    pre_adjustment_stability = io_service.eigenspace_stability
    pre_adjustment_coherence = io_service.pattern_coherence
    
    # Create meta-pattern event
    event_bus.publish(Event.create(
        type="pattern.meta.detected",
        source="test_meta_pattern_detector",
        data={
            "id": "meta_pattern_1",
            "type": "object_evolution",
            "confidence": 0.85,
            "frequency": 7,
            "examples": [f"example_{i}" for i in range(5)],
            "timestamp": datetime.now().isoformat()
        }
    ))
    
    # Allow time for processing
    time.sleep(1.0)
    
    # Log parameter changes
    logger.info("Parameter changes after object_evolution pattern:")
    logger.info(f"  Base frequency: {pre_adjustment_frequency} → {io_service.base_frequency}")
    logger.info(f"  Eigenspace stability: {pre_adjustment_stability} → {io_service.eigenspace_stability}")
    logger.info(f"  Pattern coherence: {pre_adjustment_coherence} → {io_service.pattern_coherence}")
    
    # Step 4: Publish meta-pattern detection event for causal_cascade
    logger.info("\n=== Step 4: Meta-Pattern Detection (causal_cascade) ===")
    
    # Store parameters before adjustment
    pre_adjustment_frequency = io_service.base_frequency
    pre_adjustment_stability = io_service.eigenspace_stability
    pre_adjustment_coherence = io_service.pattern_coherence
    
    # Create meta-pattern event
    event_bus.publish(Event.create(
        type="pattern.meta.detected",
        source="test_meta_pattern_detector",
        data={
            "id": "meta_pattern_2",
            "type": "causal_cascade",
            "confidence": 0.75,
            "frequency": 5,
            "examples": [f"example_{i}" for i in range(3)],
            "timestamp": datetime.now().isoformat()
        }
    ))
    
    # Allow time for processing
    time.sleep(1.0)
    
    # Log parameter changes
    logger.info("Parameter changes after causal_cascade pattern:")
    logger.info(f"  Base frequency: {pre_adjustment_frequency} → {io_service.base_frequency}")
    logger.info(f"  Eigenspace stability: {pre_adjustment_stability} → {io_service.eigenspace_stability}")
    logger.info(f"  Pattern coherence: {pre_adjustment_coherence} → {io_service.pattern_coherence}")
    
    # Step 5: Publish field gradient with updated topology data
    logger.info("\n=== Step 5: Updated Field Gradient with Changed Topology ===")
    
    # Create updated topology data with more resonance centers and different dimensionality
    updated_topology = {
        "resonance_centers": {
            "center_1": [0.1, 0.2, 0.3, 0.4],
            "center_2": [0.4, 0.5, 0.6, 0.7],
            "center_3": [0.7, 0.8, 0.9, 1.0],
            "center_4": [1.1, 1.2, 1.3, 1.4],
            "center_5": [1.5, 1.6, 1.7, 1.8]
        },
        "interference_patterns": {
            "pattern_1": [0.2, 0.3, 0.4, 0.5],
            "pattern_2": [0.5, 0.6, 0.7, 0.8],
            "pattern_3": [0.8, 0.9, 1.0, 1.1]
        },
        "field_density_centers": {
            "density_1": [0.3, 0.4, 0.5, 0.6],
            "density_2": [0.6, 0.7, 0.8, 0.9]
        },
        "flow_vectors": {
            "vector_1": [0.1, 0.1, 0.1, 0.1],
            "vector_2": [0.2, 0.2, 0.2, 0.2],
            "vector_3": [0.3, 0.3, 0.3, 0.3],
            "vector_4": [0.4, 0.4, 0.4, 0.4],
            "vector_5": [0.5, 0.5, 0.5, 0.5]
        },
        "effective_dimensionality": 4,
        "principal_dimensions": [0.9, 0.7, 0.5, 0.3]
    }
    
    # Publish updated field gradient
    event_bus.publish(Event.create(
        type="field.gradient.update",
        source="test_field_service",
        data={
            "topology": updated_topology,
            "vectors": vectors_data,
            "timestamp": datetime.now().isoformat()
        }
    ))
    
    # Allow time for processing
    time.sleep(1.0)
    
    # Step 6: Get final metrics
    logger.info("\n=== Step 6: Final Metrics ===")
    
    # Get and log metrics
    metrics = io_service.get_metrics()
    
    # Log system state metrics
    logger.info("System State Metrics:")
    for key, value in metrics["system_state"].items():
        logger.info(f"  {key}: {value}")
    
    # Log topology metrics
    if "topology" in metrics:
        logger.info("Topology Metrics:")
        for key, value in metrics["topology"].items():
            logger.info(f"  {key}: {value}")
    
    # Clean up
    io_service.stop()
    logger.info("Test completed")

if __name__ == "__main__":
    run_verbose_test()
