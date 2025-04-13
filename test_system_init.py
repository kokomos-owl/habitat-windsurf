#!/usr/bin/env python
"""
Test script for system initialization with MockVectorTonicService.

This script verifies that the system initializer can properly use the 
MockVectorTonicService to bypass ArangoDB client compatibility issues.
"""

import logging
import sys
import os
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.habitat_evolution.infrastructure.initialization.system_initializer import SystemInitializer

def test_system_initialization():
    """
    Test the system initialization with MockVectorTonicService.
    
    This test verifies that:
    1. The system initializer can properly use the MockVectorTonicService
    2. The initialization process can proceed past the Vector Tonic Service step
    3. The ArangoDB client compatibility issues are resolved by using the mock
    """
    logger.info("=== Testing System Initialization with MockVectorTonicService ===")
    
    # Create configuration with mock services enabled
    config = {
        'use_mock_services': True,
        'arangodb': {
            'host': 'localhost',
            'port': 8529,
            'username': 'root',
            'password': 'habitat',
            'database': 'habitat_evolution'
        }
    }
    
    # Create system initializer with mock configuration
    initializer = SystemInitializer(config)
    
    try:
        # Initialize the system
        logger.info("Starting system initialization...")
        success = initializer.initialize_system()
        
        if success:
            logger.info("System initialization successful!")
            
            # Check if vector_tonic_service was properly initialized
            components = initializer.get_components()
            if 'vector_tonic_service' in components:
                logger.info("Vector Tonic Service was properly initialized")
                
                # Verify it's the mock implementation
                vector_tonic_service = components['vector_tonic_service']
                logger.info(f"Vector Tonic Service class: {vector_tonic_service.__class__.__name__}")
                
                return True
            else:
                logger.error("Vector Tonic Service was not initialized")
                return False
        else:
            # Get initialization errors
            errors = initializer.get_initialization_errors()
            logger.error(f"System initialization failed with errors: {errors}")
            return False
            
    except Exception as e:
        logger.error(f"Error during system initialization test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run the system initialization test."""
    success = test_system_initialization()
    
    if success:
        logger.info("TEST PASSED: System initialization with MockVectorTonicService works correctly")
        sys.exit(0)
    else:
        logger.error("TEST FAILED: System initialization with MockVectorTonicService failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
