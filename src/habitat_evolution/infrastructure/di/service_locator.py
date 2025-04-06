"""
Service locator for Habitat Evolution DI framework.

This module provides a central service locator that makes it easy to access
all services through a single interface, simplifying the refactoring process.
"""

from typing import TypeVar, Type, Optional, Dict, Any
import logging

from .container import DIContainer
from .module import ModuleRegistry
from .modules import create_core_services_module, create_infrastructure_module
# Core service interfaces
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.pattern_evolution_service_interface import PatternEvolutionServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.field_state_service_interface import FieldStateServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.gradient_service_interface import GradientServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.flow_dynamics_service_interface import FlowDynamicsServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.metrics_service_interface import MetricsServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.quality_metrics_service_interface import QualityMetricsServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.graph_service_interface import GraphServiceInterface

# New service interfaces
from src.habitat_evolution.infrastructure.interfaces.services.document_service_interface import DocumentServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.api_service_interface import APIServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.bidirectional_flow_interface import BidirectionalFlowInterface
from src.habitat_evolution.infrastructure.interfaces.services.unified_graph_service_interface import UnifiedGraphServiceInterface

T = TypeVar('T')
logger = logging.getLogger(__name__)


class ServiceLocator:
    """
    Service locator for Habitat Evolution.
    
    The service locator provides a central point for accessing services,
    simplifying the refactoring process by providing a consistent interface
    that aligns with Habitat's principles of pattern evolution and co-evolution.
    """
    
    _instance: Optional['ServiceLocator'] = None
    
    @classmethod
    def instance(cls) -> 'ServiceLocator':
        """
        Get the singleton instance of the service locator.
        
        Returns:
            The service locator instance
        """
        if cls._instance is None:
            cls._instance = ServiceLocator()
        return cls._instance
    
    def __init__(self):
        """Initialize a new service locator."""
        self.container = DIContainer()
        self.registry = ModuleRegistry()
        self._initialized = False
    
    def initialize(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the service locator with the specified configuration.
        
        Args:
            config: Optional configuration for services
        """
        if self._initialized:
            logger.warning("ServiceLocator already initialized")
            return
        
        # Register core modules
        self.registry.register_module(create_core_services_module())
        self.registry.register_module(create_infrastructure_module())
        
        # Register modules with container
        self.registry.register_all_with(self.container)
        
        # Initialize services
        self._initialized = True
        logger.info("ServiceLocator initialized")
    
    def get_service(self, service_type: Type[T]) -> T:
        """
        Get a service of the specified type.
        
        Args:
            service_type: The type of service to get
            
        Returns:
            The service instance
            
        Raises:
            ValueError: If the service is not registered
        """
        if not self._initialized:
            self.initialize()
            
        return self.container.resolve(service_type)
    
    @property
    def event_service(self) -> EventServiceInterface:
        """Get the event service."""
        return self.get_service(EventServiceInterface)
    
    @property
    def pattern_evolution_service(self) -> PatternEvolutionServiceInterface:
        """Get the pattern evolution service."""
        return self.get_service(PatternEvolutionServiceInterface)
    
    @property
    def field_state_service(self) -> FieldStateServiceInterface:
        """Get the field state service."""
        return self.get_service(FieldStateServiceInterface)
    
    @property
    def gradient_service(self) -> GradientServiceInterface:
        """Get the gradient service."""
        return self.get_service(GradientServiceInterface)
    
    @property
    def flow_dynamics_service(self) -> FlowDynamicsServiceInterface:
        """Get the flow dynamics service."""
        return self.get_service(FlowDynamicsServiceInterface)
    
    @property
    def metrics_service(self) -> MetricsServiceInterface:
        """Get the metrics service."""
        return self.get_service(MetricsServiceInterface)
    
    @property
    def quality_metrics_service(self) -> QualityMetricsServiceInterface:
        """Get the quality metrics service."""
        return self.get_service(QualityMetricsServiceInterface)
    
    @property
    def graph_service(self) -> GraphServiceInterface:
        """Get the graph service."""
        return self.get_service(GraphServiceInterface)
    
    @property
    def unified_graph_service(self) -> UnifiedGraphServiceInterface:
        """Get the unified graph service."""
        return self.get_service(UnifiedGraphServiceInterface)
    
    @property
    def document_service(self) -> DocumentServiceInterface:
        """Get the document service."""
        return self.get_service(DocumentServiceInterface)
    
    @property
    def api_service(self) -> APIServiceInterface:
        """Get the API service."""
        return self.get_service(APIServiceInterface)
    
    @property
    def bidirectional_flow(self) -> BidirectionalFlowInterface:
        """Get the bidirectional flow manager."""
        return self.get_service(BidirectionalFlowInterface)
    
    def shutdown(self):
        """Shut down all services."""
        if not self._initialized:
            return
            
        # Shutdown logic will be implemented as services are refactored
        logger.info("ServiceLocator shutdown")
