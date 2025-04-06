"""
ServiceRegistry: Manages service registrations for the DI container.

This registry tracks interface-to-implementation mappings and their lifecycle
configuration, supporting the pattern evolution principles of Habitat.
"""

from typing import Dict, Any, Type, Optional, Callable, TypeVar, NamedTuple
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class ServiceRegistration:
    """Registration information for a service."""
    interface: Type
    implementation: Optional[Type] = None
    factory: Optional[Callable] = None
    singleton: bool = True


class ServiceRegistry:
    """
    Registry for service registrations in the DI container.
    
    This registry tracks interface-to-implementation mappings and their lifecycle
    configuration, supporting the pattern evolution principles of Habitat.
    """
    
    def __init__(self):
        """Initialize a new service registry."""
        self._registrations: Dict[str, ServiceRegistration] = {}
        
    def register(self, interface: Type[T], implementation: Optional[Type[T]] = None, 
                factory: Optional[Callable[..., T]] = None, singleton: bool = True) -> None:
        """
        Register a service with the registry.
        
        Args:
            interface: The interface type to register
            implementation: The implementation type (optional if factory is provided)
            factory: A factory function that creates the implementation (optional)
            singleton: Whether the service should be a singleton (default: True)
        """
        interface_name = interface.__name__
        self._registrations[interface_name] = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            factory=factory,
            singleton=singleton
        )
        
    def get(self, interface: Type[T]) -> Optional[ServiceRegistration]:
        """
        Get registration information for an interface.
        
        Args:
            interface: The interface type to look up
            
        Returns:
            The registration information, or None if not registered
        """
        interface_name = interface.__name__
        return self._registrations.get(interface_name)
        
    def clear(self) -> None:
        """Clear all registrations from the registry."""
        self._registrations.clear()
        
    def is_registered(self, interface: Type[T]) -> bool:
        """
        Check if an interface is registered.
        
        Args:
            interface: The interface type to check
            
        Returns:
            True if the interface is registered, False otherwise
        """
        interface_name = interface.__name__
        return interface_name in self._registrations
