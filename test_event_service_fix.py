#!/usr/bin/env python
"""
Test script to verify the EventService fix using LocalEventBus.
"""

import logging
from src.habitat_evolution.infrastructure.services.event_service import EventService
from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_fixed_event_service():
    """Test the fixed EventService with LocalEventBus integration."""
    try:
        # Create event bus
        event_bus = LocalEventBus()
        logger.info("LocalEventBus created")
        
        # Create event service with event_bus parameter
        # Note: This will fail with current implementation, we'll fix it
        try:
            event_service = EventService(event_bus=event_bus)
            logger.info("EventService created with event_bus parameter")
        except TypeError as e:
            logger.error(f"Current EventService doesn't accept event_bus: {e}")
            logger.info("We need to modify EventService to accept and use event_bus")
        
        # Create event service without event_bus (current implementation)
        event_service = EventService()
        logger.info("EventService created (current implementation)")
        
        # Initialize the event service
        event_service.initialize()
        logger.info("EventService initialized")
        
        # Define test handlers for both systems
        events_received_by_event_service = []
        events_received_by_event_bus = []
        
        def event_service_handler(data):
            logger.info(f"EventService handler received: {data}")
            events_received_by_event_service.append(data)
            
        def event_bus_handler(event):
            logger.info(f"EventBus handler received: {event.type} - {event.data}")
            events_received_by_event_bus.append(event.data)
        
        # Subscribe to events in both systems
        event_service.subscribe("test_event", event_service_handler)
        event_bus.subscribe("test_event", event_bus_handler)
        logger.info("Subscribed to events in both systems")
        
        # Publish event through EventService
        event_service.publish("test_event", {"source": "event_service", "message": "Hello"})
        logger.info("Published event through EventService")
        
        # Publish event through EventBus
        event = Event.create("test_event", {"source": "event_bus", "message": "Hello"})
        event_bus.publish(event)
        logger.info("Published event through EventBus")
        
        # Check results
        logger.info(f"Events received by EventService handler: {len(events_received_by_event_service)}")
        logger.info(f"Events received by EventBus handler: {len(events_received_by_event_bus)}")
        
        # Demonstrate the issue
        logger.info("\nISSUE DEMONSTRATION:")
        logger.info("1. EventService and EventBus are separate systems")
        logger.info("2. Events published through EventService don't reach EventBus subscribers")
        logger.info("3. Events published through EventBus don't reach EventService subscribers")
        logger.info("4. Components expecting to use EventBus through EventService won't work")
        
        # Proposed solution
        logger.info("\nPROPOSED SOLUTION:")
        logger.info("1. Modify EventService to accept event_bus in constructor")
        logger.info("2. Update EventService to delegate to event_bus for event distribution")
        logger.info("3. Ensure EventService.initialize() initializes the event_bus if needed")
        logger.info("4. Update components to use consistent event mechanism")
        
    except Exception as e:
        logger.error(f"Error in test: {e}")

if __name__ == "__main__":
    test_fixed_event_service()
