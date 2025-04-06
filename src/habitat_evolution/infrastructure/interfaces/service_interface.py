"""
Base interface for services in Habitat Evolution.

This module defines the base interface for services in the Habitat Evolution
system, providing a consistent approach to service definition and lifecycle.
"""

from typing import Protocol, Any, Dict, Optional
from abc import abstractmethod


class ServiceInterface(Protocol):
    """
    Base interface for services in Habitat Evolution.
    
    This interface defines the common methods that all services should implement,
    providing a consistent approach to service initialization and lifecycle.
    """
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the service with the provided configuration.
        
        Args:
            config: Optional configuration dictionary for the service
        """
        ...
        
    @abstractmethod
    def shutdown(self) -> None:
        """
        Shutdown the service, releasing any resources.
        """
        ...
