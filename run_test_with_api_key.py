#!/usr/bin/env python
"""
Script to run the integrated climate e2e test with the provided Claude API key.
"""

import logging
import sys
import os
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def run_test_with_api_key():
    """Run the integrated climate e2e test with the provided Claude API key."""
    try:
        # Set the Claude API key in the environment
        os.environ["CLAUDE_API_KEY"] = "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"
        logger.info("Claude API key set in environment")
        
        # Ensure the EventService is properly initialized
        from src.habitat_evolution.infrastructure.services.event_service import EventService
        
        # Create and initialize EventService
        event_service = EventService()
        event_service.initialize({
            "buffer_size": 100,
            "flush_interval": 5,
            "auto_flush": True,
            "persistence_enabled": True
        })
        
        # Set global instance if available
        if hasattr(EventService, 'set_global_instance'):
            EventService.set_global_instance(event_service)
            logger.info("Set EventService as global instance")
        
        # Set environment variable to indicate EventService is initialized
        os.environ["EVENT_SERVICE_INITIALIZED"] = "true"
        
        # Run the integrated test
        logger.info("Running integrated climate e2e test...")
        result = pytest.main(["-xvs", "tests/integration/climate_e2e/test_climate_e2e.py::test_integrated_climate_e2e"])
        
        if result != 0:
            logger.error(f"Test failed with exit code: {result}")
            return False
        
        logger.info("Test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error running test: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting test with Claude API key")
    success = run_test_with_api_key()
    
    if success:
        logger.info("Test with Claude API key completed successfully")
        sys.exit(0)
    else:
        logger.error("Test with Claude API key failed")
        sys.exit(1)
