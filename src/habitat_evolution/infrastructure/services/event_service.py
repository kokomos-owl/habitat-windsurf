"""
Event service implementation for Habitat Evolution.

This module provides a concrete implementation of the EventServiceInterface,
enabling event-based communication between components in the system.
"""

import logging
from typing import Dict, Any, Callable, Optional, List

from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface

logger = logging.getLogger(__name__)


class EventService(EventServiceInterface):
    """
    Implementation of the EventServiceInterface for Habitat Evolution.
    
    This service enables components to communicate through events in a decoupled manner,
    supporting the pattern evolution and co-evolution principles of Habitat Evolution.
    """
    
    def __init__(self):
        """Initialize a new event service."""
        self._subscribers: Dict[str, List[Dict[str, Any]]] = {}
        self._initialized = False
        logger.debug("EventService created")
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the event service with the specified configuration.
        
        Args:
            config: Optional configuration for the event service
        """
        if self._initialized:
            logger.warning("EventService already initialized")
            return
            
        logger.info("Initializing EventService")
        self._initialized = True
        logger.info("EventService initialized")
    
    def shutdown(self) -> None:
        """
        Release resources when shutting down the event service.
        """
        if not self._initialized:
            logger.warning("EventService not initialized")
            return
            
        logger.info("Shutting down EventService")
        self.clear_subscriptions()
        self._initialized = False
        logger.info("EventService shut down")
    
    def publish(self, event_name: str, data: Dict[str, Any]) -> None:
        """
        Publish an event with the specified name and data.
        
        Args:
            event_name: The name of the event to publish
            data: The data associated with the event
        """
        if not self._initialized:
            logger.warning("EventService not initialized")
            return
            
        if event_name not in self._subscribers:
            logger.debug(f"No subscribers for event: {event_name}")
            return
            
        logger.debug(f"Publishing event: {event_name}")
        for subscriber in self._subscribers[event_name]:
            try:
                subscriber["handler"](data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_name}: {str(e)}")
    
    def subscribe(self, event_name: str, handler: Callable[[Dict[str, Any]], None], 
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Subscribe to an event with the specified name.
        
        Args:
            event_name: The name of the event to subscribe to
            handler: The function to call when the event is published
            metadata: Optional metadata for the subscription
        """
        if not self._initialized:
            logger.warning("EventService not initialized")
            return
            
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
            
        subscription = {
            "handler": handler,
            "metadata": metadata or {}
        }
        
        self._subscribers[event_name].append(subscription)
        logger.debug(f"Subscribed to event: {event_name}")
    
    def unsubscribe(self, event_name: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Unsubscribe from an event with the specified name.
        
        Args:
            event_name: The name of the event to unsubscribe from
            handler: The handler function to unsubscribe
        """
        if not self._initialized:
            logger.warning("EventService not initialized")
            return
            
        if event_name not in self._subscribers:
            logger.debug(f"No subscribers for event: {event_name}")
            return
            
        self._subscribers[event_name] = [
            s for s in self._subscribers[event_name] if s["handler"] != handler
        ]
        
        if not self._subscribers[event_name]:
            del self._subscribers[event_name]
            
        logger.debug(f"Unsubscribed from event: {event_name}")
    
    def clear_subscriptions(self) -> None:
        """
        Clear all event subscriptions.
        """
        if not self._initialized:
            logger.warning("EventService not initialized")
            return
            
        self._subscribers.clear()
        logger.debug("Cleared all subscriptions")
