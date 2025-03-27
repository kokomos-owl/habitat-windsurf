"""
Combined test for feedback loop and topology metrics extraction.

This test demonstrates how the HarmonicIOService adapts to both 
meta-pattern detection and topology changes.
"""

import unittest
import time
from datetime import datetime
import logging
import sys
import os

# Add src to path if not already there
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import core components
from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService
from src.habitat_evolution.adaptive_core.emergence.tonic_harmonic_integration import TonicHarmonicPatternDetector
from src.habitat_evolution.adaptive_core.emergence.learning_window_integration import LearningWindowAwareDetector
from src.habitat_evolution.field.harmonic_field_io_bridge import HarmonicFieldIOBridge
from src.habitat_evolution.field.field_state import TonicHarmonicFieldState
from src.habitat_evolution.adaptive_core.resonance.tonic_harmonic_metrics import TonicHarmonicMetrics
from src.habitat_evolution.pattern_aware_rag.learning.learning_control import WindowState, BackPressureController

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("combined_test")

class MockLearningWindowDetector(LearningWindowAwareDetector):
    """Mock learning window aware detector for testing."""
    
    def __init__(self, event_bus):
        """Initialize the mock detector."""
        self.event_bus = event_bus
        self.detected_patterns = []
        self.window_state = WindowState.OPENING
        self.back_pressure = BackPressureController()
        
    def detect_patterns(self, data):
        """Detect patterns in data."""
        pattern = {"id": f"pattern_{len(self.detected_patterns)}", "data": data}
        self.detected_patterns.append(pattern)
        return pattern
    
    def get_window_state(self):
        """Get the current window state."""
        return self.window_state
    
    def set_window_state(self, state):
        """Set the window state."""
        self.window_state = state
    
    def get_back_pressure(self):
        """Get the back pressure controller."""
        return self.back_pressure

class TestCombinedFeedbackLoop(unittest.TestCase):
    """Test the combined feedback loop and topology metrics extraction."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create event bus
        self.event_bus = LocalEventBus()
        
        # Create HarmonicIOService
        self.io_service = HarmonicIOService(0.1, event_bus=self.event_bus)
        self.io_service.start()
        
        # Create mock detector
        self.base_detector = MockLearningWindowDetector(self.event_bus)
        
        # Create field bridge
        self.field_bridge = HarmonicFieldIOBridge(self.io_service)
        
        # Create metrics
        self.metrics = TonicHarmonicMetrics()
        
        # Create field analysis for field state
        self.field_analysis = {
            "topology": {
                "effective_dimensionality": 3,
                "principal_dimensions": [0, 1, 2],
                "eigenvalues": [0.8, 0.5, 0.3],
                "eigenvectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                "resonance_centers": [{"x": 0.1, "y": 0.2, "z": 0.3}],
                "interference_patterns": [{"source": "test", "intensity": 0.5}],
                "field_density_centers": [{"x": 0.4, "y": 0.5, "z": 0.6}],
                "flow_vectors": [{"x": 0.7, "y": 0.8, "z": 0.9}]
            },
            "density": {
                "density_centers": [{"x": 0.4, "y": 0.5, "z": 0.6, "intensity": 0.7}],
                "density_map": [[0.1, 0.2], [0.3, 0.4]]
            },
            "field_properties": {
                "coherence": 0.7,
                "navigability_score": 0.8,
                "stability": 0.6
            },
            "patterns": {
                "pattern_1": {"confidence": 0.8, "vectors": [[0.1, 0.2, 0.3]]},
                "pattern_2": {"confidence": 0.7, "vectors": [[0.4, 0.5, 0.6]]}
            },
            "pattern_projections": {
                "pattern_1": {"0": 0.8, "1": 0.6, "2": 0.3},
                "pattern_2": {"0": 0.3, "1": 0.7, "2": 0.8}
            },
            "resonance_relationships": {
                "pattern_1_pattern_2": 0.8
            },
            "metrics": {
                "stability": 0.7,
                "coherence": 0.6,
                "resonance": 0.8
            }
        }
        
        # Create field state
        self.field_state = TonicHarmonicFieldState(self.field_analysis)
        
        # Create TonicHarmonicPatternDetector
        self.tonic_detector = TonicHarmonicPatternDetector(
            base_detector=self.base_detector,
            harmonic_io_service=self.io_service,
            event_bus=self.event_bus,
            field_bridge=self.field_bridge,
            metrics=self.metrics
        )
        
        # Allow time for initialization
        time.sleep(0.2)
        
    def tearDown(self):
        """Clean up after the test."""
        self.io_service.stop()
        time.sleep(0.2)
    
    def test_combined_feedback_and_topology(self):
        """Test the combined feedback loop and topology metrics extraction."""
        logger.info("Starting combined feedback loop and topology test")
        
        # Store initial parameters
        initial_frequency = self.io_service.base_frequency
        initial_stability = self.io_service.eigenspace_stability
        initial_coherence = self.io_service.pattern_coherence
        
        logger.info(f"Initial parameters:")
        logger.info(f"  Base frequency: {initial_frequency}")
        logger.info(f"  Eigenspace stability: {initial_stability}")
        logger.info(f"  Pattern coherence: {initial_coherence}")
        
        # Step 1: Send field gradient with topology data
        logger.info("\n=== Step 1: Field Gradient with Topology Data ===")
        
        # Create topology data
        topology_data = {
            "resonance_centers": {"center_1": [0.1, 0.2], "center_2": [0.3, 0.4]},
            "interference_patterns": {"pattern_1": [0.5, 0.6], "pattern_2": [0.7, 0.8]},
            "field_density_centers": {"density_1": [0.9, 1.0], "density_2": [1.1, 1.2]},
            "flow_vectors": {"vector_1": [0.1, 0.1], "vector_2": [0.2, 0.2]},
            "effective_dimensionality": 3,
            "principal_dimensions": [0.8, 0.6, 0.4]
        }
        
        # Publish field gradient update
        self.event_bus.publish(Event.create(
            type="field.gradient.update",
            source="test_field_service",
            data={
                "topology": topology_data,
                "vectors": {"v1": [1, 2], "v2": [3, 4]},
                "timestamp": datetime.now().isoformat()
            }
        ))
        
        # Allow time for processing
        time.sleep(0.5)
        
        # Get and log metrics after topology update
        metrics_after_topology = self.io_service.get_metrics()
        
        logger.info("Metrics after topology update:")
        if "topology" in metrics_after_topology:
            for key, value in metrics_after_topology["topology"].items():
                logger.info(f"  {key}: {value}")
        
        # Step 2: Send meta-pattern detection event
        logger.info("\n=== Step 2: Meta-Pattern Detection ===")
        
        # Store parameters before meta-pattern
        pre_pattern_frequency = self.io_service.base_frequency
        pre_pattern_stability = self.io_service.eigenspace_stability
        pre_pattern_coherence = self.io_service.pattern_coherence
        
        # Create meta-pattern event
        self.event_bus.publish(Event.create(
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
        time.sleep(0.5)
        
        # Get parameters after meta-pattern
        post_pattern_frequency = self.io_service.base_frequency
        post_pattern_stability = self.io_service.eigenspace_stability
        post_pattern_coherence = self.io_service.pattern_coherence
        
        # Log parameter changes
        logger.info("Parameter changes after meta-pattern:")
        logger.info(f"  Base frequency: {pre_pattern_frequency} → {post_pattern_frequency}")
        logger.info(f"  Eigenspace stability: {pre_pattern_stability} → {post_pattern_stability}")
        logger.info(f"  Pattern coherence: {pre_pattern_coherence} → {post_pattern_coherence}")
        
        # Step 3: Send updated topology with more complex structure
        logger.info("\n=== Step 3: Updated Topology with More Complex Structure ===")
        
        # Create updated topology data
        updated_topology = {
            "resonance_centers": {
                "center_1": [0.1, 0.2, 0.3],
                "center_2": [0.4, 0.5, 0.6],
                "center_3": [0.7, 0.8, 0.9],
                "center_4": [1.0, 1.1, 1.2]
            },
            "interference_patterns": {
                "pattern_1": [0.2, 0.3, 0.4],
                "pattern_2": [0.5, 0.6, 0.7],
                "pattern_3": [0.8, 0.9, 1.0]
            },
            "field_density_centers": {
                "density_1": [0.3, 0.4, 0.5],
                "density_2": [0.6, 0.7, 0.8]
            },
            "flow_vectors": {
                "vector_1": [0.1, 0.1, 0.1],
                "vector_2": [0.2, 0.2, 0.2],
                "vector_3": [0.3, 0.3, 0.3],
                "vector_4": [0.4, 0.4, 0.4]
            },
            "effective_dimensionality": 4,
            "principal_dimensions": [0.9, 0.7, 0.5, 0.3]
        }
        
        # Store parameters before updated topology
        pre_update_frequency = self.io_service.base_frequency
        
        # Publish updated field gradient
        self.event_bus.publish(Event.create(
            type="field.gradient.update",
            source="test_field_service",
            data={
                "topology": updated_topology,
                "vectors": {"v1": [1, 2, 3], "v2": [4, 5, 6]},
                "timestamp": datetime.now().isoformat()
            }
        ))
        
        # Allow time for processing
        time.sleep(0.5)
        
        # Get parameters after updated topology
        post_update_frequency = self.io_service.base_frequency
        
        # Log parameter changes
        logger.info("Parameter changes after updated topology:")
        logger.info(f"  Base frequency: {pre_update_frequency} → {post_update_frequency}")
        
        # Step 4: Send a different type of meta-pattern
        logger.info("\n=== Step 4: Different Meta-Pattern Type ===")
        
        # Store parameters before second meta-pattern
        pre_pattern2_frequency = self.io_service.base_frequency
        pre_pattern2_stability = self.io_service.eigenspace_stability
        pre_pattern2_coherence = self.io_service.pattern_coherence
        
        # Create second meta-pattern event
        self.event_bus.publish(Event.create(
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
        time.sleep(0.5)
        
        # Get parameters after second meta-pattern
        post_pattern2_frequency = self.io_service.base_frequency
        post_pattern2_stability = self.io_service.eigenspace_stability
        post_pattern2_coherence = self.io_service.pattern_coherence
        
        # Log parameter changes
        logger.info("Parameter changes after causal_cascade meta-pattern:")
        logger.info(f"  Base frequency: {pre_pattern2_frequency} → {post_pattern2_frequency}")
        logger.info(f"  Eigenspace stability: {pre_pattern2_stability} → {post_pattern2_stability}")
        logger.info(f"  Pattern coherence: {pre_pattern2_coherence} → {post_pattern2_coherence}")
        
        # Step 5: Get final metrics
        logger.info("\n=== Step 5: Final Metrics ===")
        
        # Get and log final metrics
        final_metrics = self.io_service.get_metrics()
        
        logger.info("Final System State Metrics:")
        for key, value in final_metrics["system_state"].items():
            logger.info(f"  {key}: {value}")
        
        logger.info("Final Topology Metrics:")
        if "topology" in final_metrics:
            for key, value in final_metrics["topology"].items():
                logger.info(f"  {key}: {value}")
        
        # Verify that parameters were adjusted
        self.assertNotEqual(initial_frequency, final_metrics["system_state"]["eigenspace_stability"],
                          "Parameters should be adjusted by feedback loop")
        
        # Verify that topology metrics were extracted
        self.assertIn("topology", final_metrics, "Topology metrics should be available")
        self.assertGreaterEqual(final_metrics["topology"]["meta_pattern_count"], 2,
                              "Should have at least 2 meta-patterns recorded")
        
        logger.info("Test completed successfully")

if __name__ == "__main__":
    unittest.main()
