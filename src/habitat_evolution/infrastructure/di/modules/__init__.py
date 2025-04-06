"""
Module registrations for Habitat Evolution DI framework.

This package contains modules for registering components with the DI container,
organizing dependencies by functional areas in a clean, consistent way that
aligns with the principles of pattern evolution and co-evolution.
"""

from .core_services_module import create_core_services_module
from .infrastructure_module import create_infrastructure_module

__all__ = [
    'create_core_services_module',
    'create_infrastructure_module'
]
