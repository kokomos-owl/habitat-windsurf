"""
Local Event Bus for the Habitat Evolution system.

This module provides a local implementation of an event bus for the vector-tonic
subsystem, enabling event-driven communication between components.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import uuid

logger = logging.getLogger(__name__)

class LocalEventBus:
    """
    Local implementation of an event bus for the vector-tonic subsystem.
    
    This implementation provides the minimal required functionality to support
    event-driven communication between components in the Habitat Evolution system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LocalEventBus.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._initialized = False
        self._subscribers = {}
        self._event_history = []
        self._max_history = self.config.get("max_history", 100)
        logger.info("LocalEventBus created")
    
    def initialize(self) -> None:
        """Initialize the event bus."""
        self._initialized = True
        logger.info("LocalEventBus initialized")
    
    def subscribe(self, event_type: str, callback: Callable) -> str:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
            
        Returns:
            Subscription ID
        """
        subscription_id = str(uuid.uuid4())
        
        if event_type not in self._subscribers:
            self._subscribers[event_type] = {}
        
        self._subscribers[event_type][subscription_id] = callback
        logger.debug(f"Added subscription {subscription_id} for event type: {event_type}")
        
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            subscription_id: ID of the subscription to remove
            
        Returns:
            True if subscription was removed, False otherwise
        """
        for event_type, subscribers in self._subscribers.items():
            if subscription_id in subscribers:
                del subscribers[subscription_id]
                logger.debug(f"Removed subscription {subscription_id} for event type: {event_type}")
                return True
        
        logger.warning(f"Subscription {subscription_id} not found")
        return False
    
    async def publish(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Publish an event asynchronously.
        
        Args:
            event_type: Type of event to publish
            event_data: Data associated with the event
        """
        event = {
            "id": str(uuid.uuid4()),
            "type": event_type,
            "data": event_data,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]
        
        # Notify subscribers
        if event_type in self._subscribers:
            for subscription_id, callback in self._subscribers[event_type].items():
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Error in event handler {subscription_id}: {e}")
        
        logger.debug(f"Published event {event['id']} of type: {event_type}")
    
    def publish_sync(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Publish an event synchronously.
        
        Args:
            event_type: Type of event to publish
            event_data: Data associated with the event
        """
        # Create a new event loop for the async operation
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.publish(event_type, event_data))
        finally:
            loop.close()
    
    def get_event_history(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the event history.
        
        Args:
            event_type: Optional type of events to filter by
            
        Returns:
            List of events
        """
        if event_type is None:
            return self._event_history.copy()
        else:
            return [event for event in self._event_history if event["type"] == event_type]
    
    @property
    def is_initialized(self) -> bool:
        """Check if the event bus is initialized."""
        return self._initialized
