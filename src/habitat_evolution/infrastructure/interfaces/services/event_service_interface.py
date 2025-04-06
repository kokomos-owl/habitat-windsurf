"""
Event service interface for Habitat Evolution.

This module defines the interface for the event service, which is responsible
for managing event publication and subscription in the Habitat Evolution system.
"""

from typing import Protocol, Any, Dict, Callable, Optional
from abc import abstractmethod

from src.habitat_evolution.infrastructure.interfaces.service_interface import ServiceInterface


class EventServiceInterface(ServiceInterface, Protocol):
    """
    Interface for the event service in Habitat Evolution.
    
    The event service is responsible for managing event publication and subscription,
    enabling components to communicate through events in a decoupled manner.
    This supports the pattern evolution and co-evolution principles of Habitat
    by allowing components to evolve independently while maintaining communication.
    """
    
    @abstractmethod
    def publish(self, event_name: str, data: Dict[str, Any]) -> None:
        """
        Publish an event with the specified name and data.
        
        Args:
            event_name: The name of the event to publish
            data: The data associated with the event
        """
        ...
        
    @abstractmethod
    def subscribe(self, event_name: str, handler: Callable[[Dict[str, Any]], None], 
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Subscribe to an event with the specified name.
        
        Args:
            event_name: The name of the event to subscribe to
            handler: The function to call when the event is published
            metadata: Optional metadata for the subscription
        """
        ...
        
    @abstractmethod
    def unsubscribe(self, event_name: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Unsubscribe from an event with the specified name.
        
        Args:
            event_name: The name of the event to unsubscribe from
            handler: The handler function to unsubscribe
        """
        ...
        
    @abstractmethod
    def clear_subscriptions(self) -> None:
        """
        Clear all event subscriptions.
        """
        ...
