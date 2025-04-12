#!/usr/bin/env python
"""
Script to ensure the EventService is properly initialized for the Habitat Evolution system.
"""

import logging
import sys
from src.habitat_evolution.infrastructure.services.event_service import EventService
from src.habitat_evolution.infrastructure.initialization.component_initializer import initialize_component

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def ensure_event_service():
    """Ensure the EventService is properly initialized."""
    try:
        # Initialize EventService with explicit configuration
        logger.info("Initializing EventService...")
        
        event_service_config = {
            "buffer_size": 100,
            "flush_interval": 5,
            "auto_flush": True,
            "persistence_enabled": True,
            "event_handlers": {
                "pattern.created": True,
                "pattern.updated": True,
                "pattern.deleted": True,
                "relationship.created": True,
                "window.state_changed": True,
                "field.state_updated": True
            }
        }
        
        # Use our component initializer with strict error handling
        event_service = initialize_component(
            "event_service", 
            event_service_config,
            fallback_on_error=False
        )
        
        if not event_service:
            logger.error("Failed to initialize EventService")
            return None
        
        # Verify initialization
        if not hasattr(event_service, '_initialized') or not event_service._initialized:
            logger.error("EventService not properly initialized")
            return None
        
        # Test event publication
        test_event_name = "test.event"
        test_event_data = {
            "message": "EventService initialization test",
            "timestamp": "2025-04-12T11:47:00"
        }
        
        event_service.publish(test_event_name, test_event_data)
        logger.info("Published test event successfully")
        
        # Set global instance if available
        if hasattr(EventService, 'set_global_instance'):
            EventService.set_global_instance(event_service)
            logger.info("Set EventService as global instance")
        
        logger.info("EventService initialized successfully")
        return event_service
    
    except Exception as e:
        logger.error(f"Error ensuring EventService: {e}")
        return None

if __name__ == "__main__":
    logger.info("Starting EventService initialization")
    event_service = ensure_event_service()
    
    if event_service:
        logger.info("EventService initialization completed successfully")
        sys.exit(0)
    else:
        logger.error("EventService initialization failed")
        sys.exit(1)
