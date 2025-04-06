"""
Adapter module for Habitat Evolution DI framework.

This module provides registrations for adapters in Habitat Evolution,
organizing them in a clean, consistent way that aligns with the principles
of pattern evolution and co-evolution.
"""

from typing import Dict, Any, Optional

from src.habitat_evolution.infrastructure.di.module import Module
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.adapters.pattern_bridge import PatternBridge
from src.habitat_evolution.adaptive_core.models.pattern import Pattern as AdaptiveCorePattern


def create_adapter_module() -> Module:
    """
    Create a module for registering adapters.
    
    Returns:
        A module configured with adapter registrations
    """
    module = Module("Adapters")
    
    # Register the PatternBridge for AdaptiveCorePattern as a singleton
    def create_adaptive_core_pattern_bridge(container):
        event_service = container.resolve(EventServiceInterface)
        return PatternBridge(event_service, AdaptiveCorePattern)
    
    module.register_singleton(
        PatternBridge[AdaptiveCorePattern],
        factory=create_adaptive_core_pattern_bridge
    )
    
    # Register additional adapters as they are implemented
    
    return module
