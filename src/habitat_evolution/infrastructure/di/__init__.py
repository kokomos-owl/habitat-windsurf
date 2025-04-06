"""
Dependency Injection framework for Habitat Evolution.

This package provides a clean, consistent dependency injection framework
that aligns with Habitat Evolution's principles of pattern evolution and
co-evolution, enabling flexible and maintainable component management.
"""

# Core DI components
from .container import DIContainer
from .registry import ServiceRegistry, ServiceRegistration
from .provider import ServiceProvider
from .factory import ServiceFactory
from .module import Module, ModuleRegistry
from .service_locator import ServiceLocator

# Import modules
from .modules import create_core_services_module

__all__ = [
    # Core DI components
    'DIContainer',
    'ServiceRegistry',
    'ServiceRegistration',
    'ServiceProvider',
    'ServiceFactory',
    'Module',
    'ModuleRegistry',
    'ServiceLocator',
    
    # Module factories
    'create_core_services_module'
]
