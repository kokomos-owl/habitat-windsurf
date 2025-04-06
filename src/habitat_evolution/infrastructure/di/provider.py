"""
ServiceProvider: Provides a standardized way to access services from the DI container.

This provider acts as a facade for the DI container, making it easier to access
services while maintaining the principles of pattern evolution and co-evolution.
"""

from typing import Dict, Any, Type, TypeVar, cast
import logging

from .container import DIContainer

T = TypeVar('T')
logger = logging.getLogger(__name__)

class ServiceProvider:
    """
    Service provider for Habitat Evolution.
    
    This provider acts as a facade for the DI container, making it easier to access
    services while maintaining the principles of pattern evolution and co-evolution.
    
    The provider supports:
    - Typed service resolution
    - Service existence checking
    - Scoped service access
    """
    
    def __init__(self, container: DIContainer):
        """
        Initialize a new service provider.
        
        Args:
            container: The DI container to use for service resolution
        """
        self._container = container
        
    def get(self, service_type: Type[T]) -> T:
        """
        Get a service of the specified type.
        
        Args:
            service_type: The type of service to get
            
        Returns:
            An instance of the requested service
            
        Raises:
            KeyError: If the service is not registered
        """
        try:
            return self._container.resolve(service_type)
        except KeyError:
            logger.error(f"Service {service_type.__name__} not registered")
            raise
            
    def has(self, service_type: Type[T]) -> bool:
        """
        Check if a service of the specified type is registered.
        
        Args:
            service_type: The type of service to check
            
        Returns:
            True if the service is registered, False otherwise
        """
        try:
            self._container.resolve(service_type)
            return True
        except KeyError:
            return False
            
    @property
    def container(self) -> DIContainer:
        """Get the underlying DI container."""
        return self._container
