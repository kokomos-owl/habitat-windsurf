"""
ServiceFactory: Factory for creating and configuring services in the DI container.

This factory provides a standardized way to create and register services,
supporting the pattern evolution principles of Habitat by enabling flexible
service creation and configuration.
"""

from typing import Dict, Any, Type, TypeVar, Callable, Optional, cast
import logging

from .container import DIContainer

T = TypeVar('T')
logger = logging.getLogger(__name__)

class ServiceFactory:
    """
    Factory for creating and configuring services in the DI container.
    
    This factory provides a standardized way to create and register services,
    supporting the pattern evolution principles of Habitat by enabling flexible
    service creation and configuration.
    
    The factory supports:
    - Registration of services with interfaces
    - Creation of services with dependencies
    - Configuration of services after creation
    """
    
    def __init__(self, container: DIContainer):
        """
        Initialize a new service factory.
        
        Args:
            container: The DI container to use for service registration and resolution
        """
        self._container = container
        
    def register_singleton(self, interface: Type[T], implementation: Type[T] = None,
                           factory: Callable[..., T] = None) -> None:
        """
        Register a singleton service with the container.
        
        Args:
            interface: The interface type to register
            implementation: The implementation type (optional if factory is provided)
            factory: A factory function that creates the implementation (optional)
        """
        self._container.register(interface, implementation, factory, singleton=True)
        logger.debug(f"Registered singleton {interface.__name__}")
        
    def register_transient(self, interface: Type[T], implementation: Type[T] = None,
                          factory: Callable[..., T] = None) -> None:
        """
        Register a transient service with the container.
        
        Args:
            interface: The interface type to register
            implementation: The implementation type (optional if factory is provided)
            factory: A factory function that creates the implementation (optional)
        """
        self._container.register(interface, implementation, factory, singleton=False)
        logger.debug(f"Registered transient {interface.__name__}")
        
    def create_factory(self, implementation: Type[T], **kwargs) -> Callable[..., T]:
        """
        Create a factory function for a service.
        
        Args:
            implementation: The implementation type to create
            **kwargs: Additional keyword arguments to pass to the constructor
            
        Returns:
            A factory function that creates instances of the implementation
        """
        def factory(container: DIContainer) -> T:
            return implementation(**kwargs)
            
        return factory
        
    def create_configured_factory(self, implementation: Type[T], 
                                 configurator: Callable[[T], None]) -> Callable[..., T]:
        """
        Create a factory function that configures the service after creation.
        
        Args:
            implementation: The implementation type to create
            configurator: A function that configures the service after creation
            
        Returns:
            A factory function that creates and configures instances of the implementation
        """
        def factory(container: DIContainer) -> T:
            instance = implementation()
            configurator(instance)
            return instance
            
        return factory
