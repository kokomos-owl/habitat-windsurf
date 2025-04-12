#!/usr/bin/env python
"""
Script to fix the EventService integration in the Habitat Evolution system.

This script creates a global EventService instance and patches the necessary
components to use this instance instead of creating their own.
"""

import logging
import sys
import os
import importlib
import types
from unittest.mock import patch
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def create_global_event_service():
    """Create a global EventService instance."""
    try:
        from src.habitat_evolution.infrastructure.services.event_service import EventService
        
        # Create and initialize EventService
        event_service = EventService()
        event_service.initialize({
            "buffer_size": 100,
            "flush_interval": 5,
            "auto_flush": True,
            "persistence_enabled": True
        })
        
        # Test event publication
        event_service.publish("test.event", {
            "message": "Global EventService initialization test",
            "timestamp": "2025-04-12T11:55:00"
        })
        
        logger.info("Created and initialized global EventService")
        return event_service
    
    except Exception as e:
        logger.error(f"Error creating global EventService: {e}")
        return None

def patch_event_service_access():
    """
    Patch the EventService access in various components.
    
    This function modifies how components access the EventService to ensure
    they all use the same global instance.
    """
    try:
        from src.habitat_evolution.infrastructure.services.event_service import EventService
        
        # Create global EventService instance
        global_event_service = create_global_event_service()
        if not global_event_service:
            logger.error("Failed to create global EventService")
            return False
        
        # Add a global instance accessor to EventService class
        EventService._global_instance = global_event_service
        
        # Add get_instance method if it doesn't exist
        if not hasattr(EventService, 'get_instance'):
            @classmethod
            def get_instance(cls):
                if not hasattr(cls, '_global_instance') or cls._global_instance is None:
                    cls._global_instance = cls()
                    cls._global_instance.initialize()
                return cls._global_instance
            
            EventService.get_instance = get_instance
        
        # Patch the EventService constructor to return the global instance
        original_init = EventService.__init__
        
        def patched_init(self, *args, **kwargs):
            if hasattr(EventService, '_global_instance') and EventService._global_instance is not None:
                # Copy attributes from global instance
                for attr, value in vars(EventService._global_instance).items():
                    setattr(self, attr, value)
                logger.debug("Using global EventService instance")
            else:
                # Call original init
                original_init(self, *args, **kwargs)
                # Initialize if not already initialized
                if not hasattr(self, '_initialized') or not self._initialized:
                    self.initialize()
        
        EventService.__init__ = patched_init
        
        # Set environment variable to indicate EventService is initialized
        os.environ["EVENT_SERVICE_INITIALIZED"] = "true"
        
        logger.info("Successfully patched EventService access")
        return True
    
    except Exception as e:
        logger.error(f"Error patching EventService access: {e}")
        return False

def run_integrated_test():
    """Run the integrated climate e2e test with patched EventService."""
    try:
        # Patch EventService access
        success = patch_event_service_access()
        if not success:
            logger.error("Failed to patch EventService access")
            return False
        
        # Run the integrated test
        logger.info("Running integrated climate e2e test...")
        result = pytest.main(["-xvs", "tests/integration/climate_e2e/test_climate_e2e.py::test_integrated_climate_e2e"])
        
        if result != 0:
            logger.error(f"Test failed with exit code: {result}")
            return False
        
        logger.info("Test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error running integrated test: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting EventService integration fix")
    success = run_integrated_test()
    
    if success:
        logger.info("EventService integration fix completed successfully")
        sys.exit(0)
    else:
        logger.error("EventService integration fix failed")
        sys.exit(1)
