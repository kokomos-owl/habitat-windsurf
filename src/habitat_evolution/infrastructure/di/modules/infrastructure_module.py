"""
Infrastructure module for Habitat Evolution DI framework.

This module provides registrations for the core infrastructure services in Habitat Evolution,
organizing them in a clean, consistent way that aligns with the principles
of pattern evolution and co-evolution.
"""

from typing import Dict, Any, Optional

from src.habitat_evolution.infrastructure.di.module import Module
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.services.event_service import EventService


def create_infrastructure_module() -> Module:
    """
    Create a module for registering infrastructure services.
    
    Returns:
        A module configured with infrastructure service registrations
    """
    module = Module("Infrastructure")
    
    # Register the EventService as a singleton
    module.register_singleton(
        EventServiceInterface,
        EventService
    )
    
    # Additional infrastructure services will be registered here as they are implemented
    
    return module
