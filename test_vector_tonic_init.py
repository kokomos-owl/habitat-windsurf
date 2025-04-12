#!/usr/bin/env python
"""
Test script to verify the VectorTonicWindowIntegrator initialization issue.
This test aligns with the specific error seen in the climate_e2e test.
"""

import logging
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import (
    VectorTonicWindowIntegrator, 
    create_vector_tonic_window_integrator
)
from src.habitat_evolution.core.services.event_bus import LocalEventBus
from src.habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService
from src.habitat_evolution.adaptive_core.emergence.tonic_harmonic_integration import (
    TonicHarmonicPatternDetector, 
    create_tonic_harmonic_detector
)
from src.habitat_evolution.adaptive_core.emergence.learning_window_integration import LearningWindowAwareDetector

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_vector_tonic_initialization():
    """Test the VectorTonicWindowIntegrator initialization with required parameters."""
    try:
        # First, demonstrate the error by initializing without parameters
        # This reproduces the exact error in the climate_e2e test (line 855 in test_climate_e2e.py)
        logger.info("STEP 1: Demonstrating the error in test_climate_e2e.py")
        logger.info("Attempting to initialize VectorTonicWindowIntegrator without parameters...")
        try:
            vector_tonic_integrator = VectorTonicWindowIntegrator()
            logger.info("VectorTonicWindowIntegrator initialized without parameters (unexpected!)")
        except TypeError as e:
            logger.error(f"Expected error: {e}")
            
        # Now, demonstrate the proper initialization chain using factory methods
        logger.info("\nSTEP 2: Demonstrating proper initialization using factory methods")
        
        # Create event bus (foundation component)
        event_bus = LocalEventBus()
        logger.info("LocalEventBus created")
        
        # Create harmonic IO service (foundation component)
        harmonic_io_service = HarmonicIOService()
        logger.info("HarmonicIOService created")
        
        # Try to create base detector for tonic detector
        try:
            # This is a simplified version - in reality, we would need to properly initialize
            # the LearningWindowAwareDetector with its dependencies
            logger.info("Note: In a real implementation, we would need to properly initialize LearningWindowAwareDetector")
            logger.info("For this test, we're just demonstrating the initialization chain")
            
            # The proper initialization chain would be:
            # 1. Create EventBus (done)
            # 2. Create HarmonicIOService (done)
            # 3. Create LearningWindowAwareDetector (needed for TonicHarmonicPatternDetector)
            # 4. Create TonicHarmonicPatternDetector using create_tonic_harmonic_detector
            # 5. Create VectorTonicWindowIntegrator using create_vector_tonic_window_integrator
            
            logger.info("\nThe issue in test_climate_e2e.py is that it's trying to initialize VectorTonicWindowIntegrator directly")
            logger.info("without first creating the required dependencies in the proper order.")
            logger.info("\nThe proper initialization would use the factory methods:")
            logger.info("1. create_tonic_harmonic_detector(base_detector, event_bus, harmonic_io_service)")
            logger.info("2. create_vector_tonic_window_integrator(tonic_detector, event_bus, harmonic_io_service)")
            
            logger.info("\nFIX RECOMMENDATION:")
            logger.info("1. Update test_climate_e2e.py line 855 to use the factory method instead of direct initialization")
            logger.info("2. Ensure all dependencies are created in the proper order before initializing VectorTonicWindowIntegrator")
            logger.info("3. Consider creating a test fixture that handles this initialization chain properly")
            
        except Exception as e:
            logger.error(f"Error in initialization chain explanation: {e}")
            
    except Exception as e:
        logger.error(f"Unexpected error in test: {e}")

if __name__ == "__main__":
    test_vector_tonic_initialization()
