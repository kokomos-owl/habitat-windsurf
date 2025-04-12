#!/usr/bin/env python
"""
Test script for component initialization utilities in Habitat Evolution.

This script verifies that the component initialization utilities properly
create and initialize components with all required dependencies.
"""

import logging
import sys
import os
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import component initializer
from src.habitat_evolution.infrastructure.initialization.component_initializer import (
    create_event_service,
    create_vector_tonic_components,
    initialize_component
)

def test_event_service_initialization():
    """Test EventService initialization."""
    logger.info("Testing EventService initialization...")
    
    # Create event service
    event_service = create_event_service()
    
    # Verify initialization
    if not event_service._initialized:
        logger.error("EventService not properly initialized")
        return False
    
    logger.info("EventService successfully initialized")
    return True

def test_vector_tonic_initialization():
    """Test VectorTonicWindowIntegrator initialization."""
    logger.info("Testing VectorTonicWindowIntegrator initialization...")
    
    try:
        # Create vector-tonic components
        integrator, event_bus, harmonic_io_service = create_vector_tonic_components()
        
        # Verify initialization
        if not integrator:
            logger.error("VectorTonicWindowIntegrator not properly created")
            return False
        
        if not event_bus:
            logger.error("LocalEventBus not properly created")
            return False
        
        if not harmonic_io_service:
            logger.error("HarmonicIOService not properly created")
            return False
        
        logger.info("VectorTonicWindowIntegrator successfully initialized with all dependencies")
        return True
    except Exception as e:
        logger.error(f"Error initializing VectorTonicWindowIntegrator: {e}")
        return False

def test_general_component_initializer():
    """Test the general component initializer."""
    logger.info("Testing general component initializer...")
    
    try:
        # Initialize event service
        event_service = initialize_component("event_service")
        
        # Verify initialization
        if not event_service._initialized:
            logger.error("EventService not properly initialized via general initializer")
            return False
        
        # Initialize vector-tonic components with the event service
        components = initialize_component(
            "vector_tonic",
            dependencies={"event_service": event_service}
        )
        
        # Verify initialization
        integrator, event_bus, harmonic_io_service = components
        
        if not integrator:
            logger.error("VectorTonicWindowIntegrator not properly created via general initializer")
            return False
        
        logger.info("General component initializer successfully initialized components")
        return True
    except Exception as e:
        logger.error(f"Error using general component initializer: {e}")
        return False

def main():
    """Run the component initialization tests."""
    logger.info("Starting component initialization tests")
    
    # Test event service initialization
    event_service_success = test_event_service_initialization()
    
    # Test vector-tonic initialization
    vector_tonic_success = test_vector_tonic_initialization()
    
    # Test general component initializer
    general_initializer_success = test_general_component_initializer()
    
    # Print summary
    logger.info("\n=== Component Initialization Test Summary ===")
    logger.info(f"EventService initialization: {'SUCCESS' if event_service_success else 'FAILURE'}")
    logger.info(f"VectorTonicWindowIntegrator initialization: {'SUCCESS' if vector_tonic_success else 'FAILURE'}")
    logger.info(f"General component initializer: {'SUCCESS' if general_initializer_success else 'FAILURE'}")
    
    # Exit with appropriate code
    if event_service_success and vector_tonic_success and general_initializer_success:
        logger.info("All component initialization tests passed")
        sys.exit(0)
    else:
        logger.error("Some component initialization tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
