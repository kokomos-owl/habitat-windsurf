#!/usr/bin/env python
"""
Test script to verify the EventService initialization issue.
"""

import logging
from src.habitat_evolution.infrastructure.services.event_service import EventService
from src.habitat_evolution.core.services.event_bus import LocalEventBus

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_event_service_initialization():
    """Test the EventService initialization."""
    try:
        # Create event bus (for reference, not used by EventService)
        event_bus = LocalEventBus()
        logger.info("LocalEventBus created")
        
        # Create event service (doesn't take event_bus parameter)
        event_service = EventService()
        logger.info("EventService created")
        
        # Check if event service is initialized
        if hasattr(event_service, '_initialized'):
            initialized = event_service._initialized
            logger.info(f"EventService._initialized = {initialized}")
        else:
            logger.error("EventService does not have _initialized attribute")
            
        # Try to initialize the event service
        try:
            event_service.initialize()
            logger.info("EventService initialized successfully")
            
            # Check initialization status again
            if hasattr(event_service, '_initialized'):
                initialized = event_service._initialized
                logger.info(f"EventService._initialized after initialize() = {initialized}")
        except Exception as e:
            logger.error(f"Error initializing EventService: {e}")
            
        # Test event subscription
        def test_handler(event):
            logger.info(f"Test handler received event: {event}")
            
        try:
            event_service.subscribe("test_event", test_handler)
            logger.info("Successfully subscribed to test_event")
            
            # Try to publish an event
            event_service.publish("test_event", {"data": "test"})
            logger.info("Successfully published test_event")
        except Exception as e:
            logger.error(f"Error with event subscription/publishing: {e}")
            
        # Check problem: EventService doesn't use an event bus
        logger.info("ISSUE: EventService doesn't use an event bus for event distribution")
        logger.info("This is likely why components depending on EventService for event distribution aren't working")
            
    except Exception as e:
        logger.error(f"Error in EventService test: {e}")

if __name__ == "__main__":
    test_event_service_initialization()
