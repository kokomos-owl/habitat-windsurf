"""
Test script for Gradient Controller integration with dynamic pattern detection.
"""
import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src')))

from habitat_evolution.core.field.gradient.controller import (
    GradientFlowController, FieldGradients, FlowMetrics
)
from habitat_evolution.adaptive_core.emergence.emergent_pattern_detector import EmergentPatternDetector
from habitat_evolution.adaptive_core.emergence.semantic_current_observer import SemanticCurrentObserver
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

def main():
    """Run the integration test."""
    logger.info("Starting Gradient Controller integration test")
    
    try:
        # Initialize Gradient Controller
        logger.info("Initializing Gradient Controller")
        controller = GradientFlowController()
        logger.info("Gradient Controller initialized successfully")
        
        # Initialize dynamic pattern detection components
        logger.info("Initializing dynamic pattern detection components")
        adaptive_id = AdaptiveID(entity_id="test_entity")
        pattern_detector = EmergentPatternDetector(adaptive_id=adaptive_id)
        logger.info("Dynamic pattern detection components initialized successfully")
        
        # Create sample field gradients
        gradients = FieldGradients(
            coherence=0.8,
            energy=0.6,
            density=0.7,
            turbulence=0.2
        )
        
        # Create sample pattern
        pattern = {
            "id": "pattern_1",
            "coherence": 0.75,
            "energy": 0.65,
            "density": 0.7,
            "components": ["sea_level_rise", "coastal_flooding", "infrastructure_damage"]
        }
        
        # Create related patterns
        related_patterns = [
            {
                "id": "pattern_2",
                "coherence": 0.7,
                "energy": 0.6,
                "density": 0.65,
                "components": ["community_displacement", "economic_impact", "adaptation_measures"]
            }
        ]
        
        # Calculate flow metrics
        logger.info("Calculating flow metrics")
        flow_metrics = controller.calculate_flow(gradients, pattern, related_patterns)
        logger.info(f"Flow metrics: {flow_metrics}")
        
        # Simulate pattern detection
        logger.info("Simulating pattern detection")
        # Add observations to pattern detector
        for component in pattern["components"]:
            pattern_detector.observe_frequency(component, 5)
        
        # Detect patterns
        detected_patterns = pattern_detector.detect_patterns()
        logger.info(f"Detected patterns: {detected_patterns}")
        
        # Integration point: Use detected patterns with gradient controller
        logger.info("Integration point: Use detected patterns with gradient controller")
        # This would require converting detected_patterns to the format expected by calculate_flow
        
        logger.info("Integration test completed")
        
    except Exception as e:
        logger.error(f"Error in integration test: {e}")
        raise

if __name__ == "__main__":
    main()
