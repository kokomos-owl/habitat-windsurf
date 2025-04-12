"""
Vector Tonic Initialization Module for Habitat Evolution.

This module provides simplified initialization functions for Vector Tonic components,
making it easier to create and configure these components with proper dependencies.
This is a POC-level implementation that will be replaced with a proper dependency
injection system in production.
"""

import logging
from typing import Tuple, Optional, Any, Dict

from src.habitat_evolution.core.services.event_bus import LocalEventBus
from src.habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService
from src.habitat_evolution.adaptive_core.emergence.learning_window_integration import (
    LearningWindowAwareDetector,
    PatternEventPublisher,
    BackPressureController,
    EventAwarePatternDetector
)
from src.habitat_evolution.adaptive_core.emergence.tonic_harmonic_integration import (
    TonicHarmonicPatternDetector,
    create_tonic_harmonic_detector,
    VectorPlusFieldBridge
)
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import (
    VectorTonicWindowIntegrator,
    create_vector_tonic_window_integrator
)
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector import (
    VectorTonicPersistenceConnector
)
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.field.field_state import TonicHarmonicFieldState

logger = logging.getLogger(__name__)

def initialize_vector_tonic_system(
    arangodb_connection: Optional[ArangoDBConnection] = None,
    event_bus: Optional[LocalEventBus] = None,
    harmonic_io_service: Optional[HarmonicIOService] = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[VectorTonicWindowIntegrator, VectorTonicPersistenceConnector, LocalEventBus, HarmonicIOService]:
    """
    Initialize the complete Vector Tonic system with all required dependencies.
    
    This function handles the complex initialization chain for Vector Tonic components,
    creating any missing dependencies and ensuring they are properly configured.
    
    Args:
        arangodb_connection: Optional ArangoDB connection for persistence
        event_bus: Optional event bus for event distribution
        harmonic_io_service: Optional harmonic I/O service
        config: Optional configuration parameters
        
    Returns:
        Tuple of (vector_tonic_integrator, vector_tonic_persistence, event_bus, harmonic_io_service)
    """
    # Create default config if not provided
    config = config or {}
    
    # Step 1: Initialize foundation components if not provided
    if not event_bus:
        event_bus = LocalEventBus()
        logger.info("Created new LocalEventBus")
    
    if not harmonic_io_service:
        harmonic_io_service = HarmonicIOService()
        logger.info("Created new HarmonicIOService")
    
    # Step 2: Create a simple pattern event publisher using the event bus
    pattern_publisher = PatternEventPublisher(event_bus)
    logger.info("Created PatternEventPublisher")
    
    # Step 3: Create a base event-aware pattern detector (simple implementation for POC)
    class SimpleEventAwareDetector(EventAwarePatternDetector):
        def detect_patterns(self, data, context=None):
            # Simple implementation for POC
            return [], {}
            
        def process_event(self, event):
            # Simple implementation for POC
            pass
    
    base_detector = SimpleEventAwareDetector()
    logger.info("Created base EventAwarePatternDetector")
    
    # Step 4: Initialize learning window detector with required dependencies
    learning_detector = LearningWindowAwareDetector(
        detector=base_detector,
        pattern_publisher=pattern_publisher,
        back_pressure_controller=BackPressureController()
    )
    logger.info("Initialized LearningWindowAwareDetector with all required dependencies")
    
    # Step 5: Create field bridge for tonic harmonic detector
    field_bridge = VectorPlusFieldBridge()
    logger.info("Created VectorPlusFieldBridge")
    
    # Step 6: Initialize tonic harmonic detector using factory method
    tonic_detector, _ = create_tonic_harmonic_detector(
        base_detector=learning_detector,
        event_bus=event_bus,
        harmonic_io_service=harmonic_io_service,
        field_bridge=field_bridge
    )
    logger.info("Initialized TonicHarmonicPatternDetector")
    
    # Step 7: Initialize vector tonic integrator using factory method
    vector_tonic_integrator = create_vector_tonic_window_integrator(
        tonic_detector=tonic_detector,
        event_bus=event_bus,
        harmonic_io_service=harmonic_io_service
    )
    logger.info("Initialized VectorTonicWindowIntegrator")
    
    # Step 8: Initialize vector tonic persistence connector if ArangoDB connection provided
    if arangodb_connection:
        vector_tonic_persistence = VectorTonicPersistenceConnector(arangodb_connection)
        logger.info("Initialized VectorTonicPersistenceConnector with ArangoDB connection")
    else:
        # Create a dummy persistence connector for POC testing
        vector_tonic_persistence = VectorTonicPersistenceConnector(None)
        logger.warning("Created VectorTonicPersistenceConnector without ArangoDB connection (limited functionality)")
    
    logger.info("Vector Tonic system initialization complete")
    return (vector_tonic_integrator, vector_tonic_persistence, event_bus, harmonic_io_service)
