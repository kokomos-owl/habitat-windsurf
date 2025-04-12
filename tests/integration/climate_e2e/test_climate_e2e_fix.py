"""
Patch for test_climate_e2e.py that fixes the VectorTonicWindowIntegrator initialization issue.

This patch demonstrates the proper initialization chain for VectorTonicWindowIntegrator
using the factory methods provided by the system. This is a POC-level fix that will
need to be replaced with a proper dependency injection system in production.
"""

import logging
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import (
    VectorTonicWindowIntegrator, 
    create_vector_tonic_window_integrator
)
from src.habitat_evolution.adaptive_core.emergence.tonic_harmonic_integration import (
    TonicHarmonicPatternDetector,
    create_tonic_harmonic_detector
)
from src.habitat_evolution.adaptive_core.emergence.learning_window_integration import (
    LearningWindowAwareDetector
)
from src.habitat_evolution.core.services.event_bus import LocalEventBus
from src.habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService

logger = logging.getLogger(__name__)

def initialize_vector_tonic_components(arangodb_connection):
    """
    Initialize Vector Tonic components with proper dependency chain.
    
    This is a POC-level fix that demonstrates the proper initialization
    sequence for VectorTonicWindowIntegrator. In production, this should
    be replaced with a proper dependency injection system.
    
    Args:
        arangodb_connection: ArangoDB connection for persistence
        
    Returns:
        Tuple of (vector_tonic_integrator, vector_tonic_persistence, event_bus, harmonic_io_service)
    """
    # Step 1: Initialize foundation components
    event_bus = LocalEventBus()
    harmonic_io_service = HarmonicIOService()
    logger.info("Initialized foundation components (event_bus, harmonic_io_service)")
    
    # Step 2: Initialize learning window detector
    # This is a simplified version for POC - in production, this would have proper configuration
    learning_detector = LearningWindowAwareDetector(event_bus)
    logger.info("Initialized LearningWindowAwareDetector")
    
    # Step 3: Initialize tonic harmonic detector using factory method
    tonic_detector, _ = create_tonic_harmonic_detector(
        base_detector=learning_detector,
        event_bus=event_bus,
        harmonic_io_service=harmonic_io_service
    )
    logger.info("Initialized TonicHarmonicPatternDetector using factory method")
    
    # Step 4: Initialize vector tonic integrator using factory method
    vector_tonic_integrator = create_vector_tonic_window_integrator(
        tonic_detector=tonic_detector,
        event_bus=event_bus,
        harmonic_io_service=harmonic_io_service
    )
    logger.info("Initialized VectorTonicWindowIntegrator using factory method")
    
    # Step 5: Initialize vector tonic persistence connector
    vector_tonic_persistence = VectorTonicPersistenceConnector(arangodb_connection)
    logger.info("Initialized VectorTonicPersistenceConnector")
    
    return (vector_tonic_integrator, vector_tonic_persistence, event_bus, harmonic_io_service)

# Example usage in test_climate_e2e.py:
"""
# Replace lines 853-862 with:
vector_tonic_integrator, vector_tonic_persistence, event_bus, harmonic_io_service = initialize_vector_tonic_components(arangodb_connection)
"""
