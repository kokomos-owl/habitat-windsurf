"""
Core services module for Habitat Evolution DI framework.

This module provides registrations for the core services in Habitat Evolution,
organizing them in a clean, consistent way that aligns with the principles
of pattern evolution and co-evolution.
"""

from typing import Dict, Any, Optional, Type

from src.habitat_evolution.infrastructure.di.module import Module
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.pattern_evolution_service_interface import PatternEvolutionServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.field_state_service_interface import FieldStateServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.gradient_service_interface import GradientServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.flow_dynamics_service_interface import FlowDynamicsServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.metrics_service_interface import MetricsServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.quality_metrics_service_interface import QualityMetricsServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.graph_service_interface import GraphServiceInterface


def create_core_services_module() -> Module:
    """
    Create a module for registering core services.
    
    Returns:
        A module configured with core service registrations
    """
    module = Module("CoreServices")
    
    # The actual implementation classes will be registered here once they're refactored
    # For now, we're just setting up the module structure
    
    # Example registration pattern (commented out until implementations are available):
    # module.register_singleton(
    #     EventServiceInterface,
    #     EventService
    # )
    
    # module.register_singleton(
    #     PatternEvolutionServiceInterface,
    #     PatternEvolutionService
    # )
    
    # module.register_singleton(
    #     FieldStateServiceInterface,
    #     FieldStateService
    # )
    
    # module.register_singleton(
    #     GradientServiceInterface,
    #     GradientService
    # )
    
    # module.register_singleton(
    #     FlowDynamicsServiceInterface,
    #     FlowDynamicsService
    # )
    
    # module.register_singleton(
    #     MetricsServiceInterface,
    #     MetricsService
    # )
    
    # module.register_singleton(
    #     QualityMetricsServiceInterface,
    #     QualityMetricsService
    # )
    
    # module.register_singleton(
    #     GraphServiceInterface,
    #     GraphService
    # )
    
    return module
