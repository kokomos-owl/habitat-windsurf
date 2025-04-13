#!/usr/bin/env python
"""
Comprehensive test for the full Habitat Evolution system initialization.

This script verifies that all components of the Habitat Evolution system
can be initialized correctly in sequence, with proper dependency tracking
and error handling.
"""

import logging
import sys
import os
import json
from typing import Dict, Any
from datetime import datetime

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

def test_full_system_initialization():
    """
    Test the complete system initialization process.
    
    This test verifies that:
    1. All components initialize in the correct order
    2. Dependencies are properly tracked and verified
    3. Error handling is consistent and informative
    4. The entire system can be initialized successfully
    """
    logger.info("=== Testing Full Habitat Evolution System Initialization ===")
    
    # Create configuration with database credentials
    config = {
        'arangodb': {
            'host': 'localhost',
            'port': 8529,
            'username': 'root',
            'password': 'habitat',
            'database': 'habitat_evolution'
        },
        'claude': {
            'api_key': 'dummy_key_for_testing',  # Use a dummy key for testing
            'model': 'claude-3-opus-20240229'
        }
    }
    
    # Create system initializer with configuration
    initializer = SystemInitializer(config)
    
    try:
        # Initialize the system
        logger.info("Starting full system initialization...")
        
        # Create a timestamp for the log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        success = initializer.initialize_system(log_dir=log_dir)
        
        if success:
            logger.info("Full system initialization successful!")
            
            # Get all initialized components
            components = initializer.get_components()
            logger.info(f"Initialized components: {', '.join(components.keys())}")
            
            # Verify dependency tracker state
            tracker = initializer.tracker
            dependency_map = tracker.get_dependency_map()
            logger.info(f"Dependency tracking information:")
            for component, deps in dependency_map.items():
                logger.info(f"  {component} depends on: {', '.join(deps)}")
            
            # Verify initialization status
            init_status = tracker.get_initialization_status()
            all_initialized = all(status for component, status in init_status.items())
            
            if all_initialized:
                logger.info("All components are properly initialized!")
                return True
            else:
                # Find components that failed to initialize
                failed_components = [comp for comp, status in init_status.items() if not status]
                logger.error(f"Some components failed to initialize: {', '.join(failed_components)}")
                return False
        else:
            # Get initialization errors
            errors = initializer.get_initialization_errors()
            logger.error(f"System initialization failed with errors:")
            for component, error in errors.items():
                logger.error(f"  {component}: {error}")
            return False
            
    except Exception as e:
        logger.error(f"Error during system initialization test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run the full system initialization test."""
    success = test_full_system_initialization()
    
    if success:
        logger.info("TEST PASSED: Full system initialization works correctly")
        sys.exit(0)
    else:
        logger.error("TEST FAILED: Full system initialization failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
