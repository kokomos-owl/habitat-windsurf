"""
DIContainer: The core dependency injection container for Habitat Evolution.

This container manages service registration, resolution, and lifecycle,
providing a clean, consistent approach to dependency management.
"""

from typing import Dict, Any, Type, Optional, Callable, TypeVar, cast
import inspect
import logging

from .registry import ServiceRegistry

T = TypeVar('T')
logger = logging.getLogger(__name__)

class DIContainer:
    """
    Dependency Injection Container for Habitat Evolution.
    
    This container manages service registration, resolution, and lifecycle,
    providing a clean, consistent approach to dependency management that
    aligns with Habitat's principles of pattern evolution and co-evolution.
    
    The container supports:
    - Interface-based registration
    - Factory-based registration
    - Singleton and transient lifecycles
    - Lazy initialization
    - Automatic dependency resolution
    """
    
    def __init__(self):
        """Initialize a new DI container."""
        self._registry = ServiceRegistry()
        self._instances: Dict[str, Any] = {}
        
    def register(self, interface: Type[T], implementation: Type[T] = None, 
                factory: Callable[..., T] = None, singleton: bool = True) -> None:
        """
        Register a service with the container.
        
        Args:
            interface: The interface type to register
            implementation: The implementation type (optional if factory is provided)
            factory: A factory function that creates the implementation (optional)
            singleton: Whether the service should be a singleton (default: True)
        """
        if implementation is None and factory is None:
            raise ValueError("Either implementation or factory must be provided")
            
        self._registry.register(interface, implementation, factory, singleton)
        logger.debug(f"Registered {interface.__name__}")
        
    def resolve(self, interface: Type[T]) -> T:
        """
        Resolve a service from the container.
        
        Args:
            interface: The interface type to resolve
            
        Returns:
            An instance of the requested service
            
        Raises:
            KeyError: If the service is not registered
        """
        # Check if we already have an instance (for singletons)
        interface_name = interface.__name__
        if interface_name in self._instances:
            return cast(T, self._instances[interface_name])
            
        # Get registration info
        registration = self._registry.get(interface)
        if registration is None:
            raise KeyError(f"Service {interface_name} not registered")
            
        # Create the instance
        instance = self._create_instance(registration)
        
        # Store singleton instances
        if registration.singleton:
            self._instances[interface_name] = instance
            
        return instance
        
    def _create_instance(self, registration):
        """Create an instance based on the registration."""
        if registration.factory:
            return registration.factory(self)
            
        if registration.implementation:
            # Get constructor parameters
            constructor = registration.implementation.__init__
            if constructor is object.__init__:
                # No constructor parameters
                return registration.implementation()
                
            # Inspect constructor parameters
            sig = inspect.signature(constructor)
            params = {}
            
            for name, param in sig.parameters.items():
                if name == 'self':
                    continue
                    
                # Try to resolve the parameter
                if param.annotation != inspect.Parameter.empty:
                    try:
                        params[name] = self.resolve(param.annotation)
                    except KeyError:
                        if param.default != inspect.Parameter.empty:
                            params[name] = param.default
                        else:
                            raise ValueError(
                                f"Cannot resolve parameter {name} of type {param.annotation}"
                            )
                elif param.default != inspect.Parameter.empty:
                    params[name] = param.default
                    
            return registration.implementation(**params)
            
        raise ValueError("Invalid registration: no implementation or factory")
        
    def reset(self) -> None:
        """Reset the container, clearing all instances but keeping registrations."""
        self._instances.clear()
        
    def clear(self) -> None:
        """Clear the container, removing all registrations and instances."""
        self._registry.clear()
        self._instances.clear()
