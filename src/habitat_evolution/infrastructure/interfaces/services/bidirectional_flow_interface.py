"""
Bidirectional flow interface for Habitat Evolution.

This module defines the interface for bidirectional flow management in the Habitat Evolution system,
providing a consistent approach to bidirectional communication between components.
"""

from typing import Protocol, Dict, List, Any, Optional, Callable
from abc import abstractmethod

from src.habitat_evolution.infrastructure.interfaces.service_interface import ServiceInterface


class BidirectionalFlowInterface(ServiceInterface, Protocol):
    """
    Interface for bidirectional flow management in Habitat Evolution.
    
    Bidirectional flow management provides a consistent approach to communication
    between the Pattern-Aware RAG system and other components, enabling the exchange
    of patterns, field states, and other information. This supports the pattern
    evolution and co-evolution principles of Habitat by enabling components to
    influence each other's evolution in a coordinated manner.
    """
    
    @abstractmethod
    def register_pattern_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a handler for pattern events.
        
        Args:
            handler: The handler function to call when a pattern event occurs
        """
        ...
        
    @abstractmethod
    def register_field_state_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a handler for field state events.
        
        Args:
            handler: The handler function to call when a field state event occurs
        """
        ...
        
    @abstractmethod
    def register_relationship_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a handler for relationship events.
        
        Args:
            handler: The handler function to call when a relationship event occurs
        """
        ...
        
    @abstractmethod
    def publish_pattern(self, pattern: Dict[str, Any]) -> None:
        """
        Publish a pattern event.
        
        Args:
            pattern: The pattern data to publish
        """
        ...
        
    @abstractmethod
    def publish_field_state(self, field_state: Dict[str, Any]) -> None:
        """
        Publish a field state event.
        
        Args:
            field_state: The field state data to publish
        """
        ...
        
    @abstractmethod
    def publish_relationship(self, relationship: Dict[str, Any]) -> None:
        """
        Publish a relationship event.
        
        Args:
            relationship: The relationship data to publish
        """
        ...
        
    @abstractmethod
    def start(self) -> None:
        """
        Start the bidirectional flow manager.
        """
        ...
        
    @abstractmethod
    def stop(self) -> None:
        """
        Stop the bidirectional flow manager.
        """
        ...
        
    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the bidirectional flow manager is running.
        
        Returns:
            True if running, False otherwise
        """
        ...
        
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the bidirectional flow manager.
        
        Returns:
            The current status
        """
        ...
