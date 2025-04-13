"""
Unit tests for the EventService.

This module contains tests that verify the EventService properly handles initialization
and throws exceptions when methods are called without initialization.
"""

import pytest
import logging
from typing import Dict, Any

from src.habitat_evolution.infrastructure.services.event_service import EventService

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_event_service_initialization():
    """Test that EventService initializes properly."""
    # Create a new EventService
    event_service = EventService()
    
    # Verify it's not initialized by default
    assert hasattr(event_service, "_initialized")
    assert not event_service._initialized
    
    # Initialize it
    event_service.initialize()
    
    # Verify it's now initialized
    assert event_service._initialized
    
    # Test that initializing again logs a warning but doesn't error
    event_service.initialize()
    assert event_service._initialized


def test_event_service_methods_require_initialization():
    """Test that EventService methods throw exceptions when not initialized."""
    # Create a new EventService without initializing it
    event_service = EventService()
    
    # Define a simple handler for testing
    def test_handler(data: Dict[str, Any]):
        pass
    
    # Test publish method
    with pytest.raises(RuntimeError, match="EventService not initialized"):
        event_service.publish("test_event", {"data": "value"})
    
    # Test subscribe method
    with pytest.raises(RuntimeError, match="EventService not initialized"):
        event_service.subscribe("test_event", test_handler)
    
    # Test unsubscribe method
    with pytest.raises(RuntimeError, match="EventService not initialized"):
        event_service.unsubscribe("test_event", test_handler)
    
    # Test clear_subscriptions method
    with pytest.raises(RuntimeError, match="EventService not initialized"):
        event_service.clear_subscriptions()
    
    # Test shutdown method
    with pytest.raises(RuntimeError, match="EventService not initialized"):
        event_service.shutdown()


def test_event_service_end_to_end():
    """Test the complete lifecycle of the EventService."""
    # Create and initialize the EventService
    event_service = EventService()
    event_service.initialize()
    
    # Track received events
    received_events = []
    
    # Define a handler that appends to received_events
    def test_handler(data: Dict[str, Any]):
        received_events.append(data)
    
    # Subscribe to an event
    event_service.subscribe("test_event", test_handler)
    
    # Publish an event
    test_data = {"message": "Hello, world!", "timestamp": "2025-04-12T21:53:17"}
    event_service.publish("test_event", test_data)
    
    # Verify the event was received
    assert len(received_events) == 1
    assert received_events[0] == test_data
    
    # Unsubscribe from the event
    event_service.unsubscribe("test_event", test_handler)
    
    # Publish another event (should not be received)
    event_service.publish("test_event", {"message": "This should not be received"})
    
    # Verify no new event was received
    assert len(received_events) == 1
    
    # Clean up
    event_service.clear_subscriptions()
    event_service.shutdown()
    
    # Verify the EventService is no longer initialized
    assert not event_service._initialized


def test_event_service_error_handling():
    """Test that EventService properly handles errors in event handlers."""
    # Create and initialize the EventService
    event_service = EventService()
    event_service.initialize()
    
    # Define a handler that raises an exception
    def error_handler(data: Dict[str, Any]):
        raise ValueError("Test error")
    
    # Subscribe to an event
    event_service.subscribe("error_event", error_handler)
    
    # Publish an event (should not raise an exception)
    event_service.publish("error_event", {"message": "This should trigger an error"})
    
    # The test passes if we get here without an exception
    
    # Clean up
    event_service.shutdown()
