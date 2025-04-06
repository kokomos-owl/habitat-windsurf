"""
Module registration system for Habitat Evolution DI framework.

This module provides a clean, consistent approach to registering components
with the DI container, organizing dependencies by functional modules.
"""

from typing import Dict, Any, Type, Optional, Callable, List
import logging

from .container import DIContainer
from .factory import ServiceFactory

logger = logging.getLogger(__name__)


class Module:
    """
    Module for registering related components with the DI container.
    
    A module represents a cohesive group of components that are registered
    together, providing a clean, organized approach to dependency management
    that aligns with Habitat's principles of pattern evolution and co-evolution.
    """
    
    def __init__(self, name: str):
        """
        Initialize a new module.
        
        Args:
            name: The name of the module
        """
        self.name = name
        self.registrations: List[Callable[[ServiceFactory], None]] = []
        
    def register_singleton(self, interface: Type, implementation: Optional[Type] = None,
                          factory: Optional[Callable] = None):
        """
        Register a singleton component with the module.
        
        Args:
            interface: The interface type to register
            implementation: The implementation type (optional if factory is provided)
            factory: A factory function that creates the implementation (optional)
            
        Returns:
            The module instance for chaining
        """
        def register(service_factory: ServiceFactory):
            service_factory.register_singleton(interface, implementation, factory)
            
        self.registrations.append(register)
        return self
        
    def register_transient(self, interface: Type, implementation: Optional[Type] = None,
                          factory: Optional[Callable] = None):
        """
        Register a transient component with the module.
        
        Args:
            interface: The interface type to register
            implementation: The implementation type (optional if factory is provided)
            factory: A factory function that creates the implementation (optional)
            
        Returns:
            The module instance for chaining
        """
        def register(service_factory: ServiceFactory):
            service_factory.register_transient(interface, implementation, factory)
            
        self.registrations.append(register)
        return self
        
    def register_with(self, container: DIContainer):
        """
        Register all components with the DI container.
        
        Args:
            container: The DI container to register with
        """
        factory = ServiceFactory(container)
        logger.info(f"Registering module: {self.name}")
        
        for registration in self.registrations:
            registration(factory)
            
        logger.info(f"Module registered: {self.name}")


class ModuleRegistry:
    """
    Registry for modules in the DI framework.
    
    The module registry manages the registration of modules with the DI container,
    providing a clean, organized approach to dependency management that aligns
    with Habitat's principles of pattern evolution and co-evolution.
    """
    
    def __init__(self):
        """Initialize a new module registry."""
        self.modules: Dict[str, Module] = {}
        
    def register_module(self, module: Module):
        """
        Register a module with the registry.
        
        Args:
            module: The module to register
        """
        self.modules[module.name] = module
        logger.info(f"Module added to registry: {module.name}")
        
    def register_all_with(self, container: DIContainer):
        """
        Register all modules with the DI container.
        
        Args:
            container: The DI container to register with
        """
        logger.info(f"Registering {len(self.modules)} modules with container")
        
        for module in self.modules.values():
            module.register_with(container)
            
        logger.info("All modules registered with container")
