"""
Infrastructure module for Habitat Evolution DI framework.

This module provides registrations for the core infrastructure services in Habitat Evolution,
organizing them in a clean, consistent way that aligns with the principles
of pattern evolution and co-evolution.
"""

from typing import Dict, Any, Optional

from src.habitat_evolution.infrastructure.di.module import Module
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.document_service_interface import DocumentServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.unified_graph_service_interface import UnifiedGraphServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.vector_tonic_service_interface import VectorTonicServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.pattern_aware_rag_interface import PatternAwareRAGInterface
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.infrastructure.interfaces.persistence.arangodb_connection_interface import ArangoDBConnectionInterface
from src.habitat_evolution.infrastructure.services.arangodb_document_service import ArangoDBDocumentService
from src.habitat_evolution.infrastructure.services.arangodb_graph_service import ArangoDBGraphService
from src.habitat_evolution.infrastructure.services.vector_tonic_service import VectorTonicService
from src.habitat_evolution.infrastructure.adapters.pattern_bridge import PatternBridge
from src.habitat_evolution.adaptive_core.models.pattern import Pattern as AdaptiveCorePattern
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_pattern_repository import ArangoDBPatternRepository
from src.habitat_evolution.infrastructure.services.event_service import EventService
from src.habitat_evolution.infrastructure.services.pattern_aware_rag_service import PatternAwareRAGService


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
    
    # Register the ArangoDBConnection as a singleton
    module.register_singleton(
        ArangoDBConnectionInterface,
        ArangoDBConnection
    )
    
    # Register ArangoDBGraphService as the implementation of UnifiedGraphServiceInterface
    def create_graph_service(container):
        db_connection = container.resolve(ArangoDBConnectionInterface)
        event_service = container.resolve(EventServiceInterface)
        return ArangoDBGraphService(db_connection, event_service)
    
    module.register_singleton(
        UnifiedGraphServiceInterface,
        factory=create_graph_service
    )
    
    # Register VectorTonicService
    def create_vector_tonic_service(container):
        db_connection = container.resolve(ArangoDBConnectionInterface)
        event_service = container.resolve(EventServiceInterface)
        pattern_repository = container.resolve(ArangoDBPatternRepository)
        return VectorTonicService(db_connection, event_service, pattern_repository)
    
    module.register_singleton(
        VectorTonicServiceInterface,
        factory=create_vector_tonic_service
    )
    
    # Register PatternAwareRAGService
    def create_pattern_aware_rag_service(container):
        db_connection = container.resolve(ArangoDBConnectionInterface)
        event_service = container.resolve(EventServiceInterface)
        vector_tonic_service = container.resolve(VectorTonicServiceInterface)
        pattern_repository = container.resolve(ArangoDBPatternRepository)
        pattern_bridge = container.resolve(PatternBridge[AdaptiveCorePattern])
        return PatternAwareRAGService(db_connection, event_service, vector_tonic_service, pattern_repository, pattern_bridge)
    
    module.register_singleton(
        PatternAwareRAGInterface,
        factory=create_pattern_aware_rag_service
    )
    
    # Register the ArangoDBDocumentService as a singleton
    module.register_singleton(
        DocumentServiceInterface,
        ArangoDBDocumentService
    )
    
    return module
