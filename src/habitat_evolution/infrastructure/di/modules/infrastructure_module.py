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
from src.habitat_evolution.infrastructure.interfaces.services.bidirectional_flow_interface import BidirectionalFlowInterface
from src.habitat_evolution.infrastructure.interfaces.services.user_interaction_interface import UserInteractionInterface
from src.habitat_evolution.infrastructure.interfaces.services.pattern_evolution_interface import PatternEvolutionInterface
from src.habitat_evolution.infrastructure.interfaces.persistence.arangodb_connection_interface import ArangoDBConnectionInterface
from src.habitat_evolution.infrastructure.services.arangodb_document_service import ArangoDBDocumentService
from src.habitat_evolution.infrastructure.services.arangodb_graph_service import ArangoDBGraphService
from src.habitat_evolution.infrastructure.services.vector_tonic_service import VectorTonicService
from src.habitat_evolution.infrastructure.adapters.pattern_bridge import PatternBridge
from src.habitat_evolution.adaptive_core.models.pattern import Pattern as AdaptiveCorePattern
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_pattern_repository import ArangoDBPatternRepository
from src.habitat_evolution.infrastructure.services.event_service import EventService
from src.habitat_evolution.infrastructure.services.pattern_aware_rag_service import PatternAwareRAGService
from src.habitat_evolution.infrastructure.services.bidirectional_flow_service import BidirectionalFlowService
from src.habitat_evolution.infrastructure.services.user_interaction_service import UserInteractionService
from src.habitat_evolution.infrastructure.services.pattern_evolution_service import PatternEvolutionService
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection


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
    module.register_singleton(
        VectorTonicServiceInterface,
        factory=lambda container: VectorTonicService()
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
    
    # Register BidirectionalFlowService
    def create_bidirectional_flow_service(container):
        event_service = container.resolve(EventServiceInterface)
        pattern_aware_rag_service = container.resolve(PatternAwareRAGInterface)
        db_connection = container.resolve(ArangoDBConnectionInterface)
        return BidirectionalFlowService(event_service, pattern_aware_rag_service, db_connection)
    
    module.register_singleton(
        BidirectionalFlowInterface,
        factory=create_bidirectional_flow_service
    )
    
    # Register PatternEvolutionService
    def create_pattern_evolution_service(container):
        event_service = container.resolve(EventServiceInterface)
        bidirectional_flow_service = container.resolve(BidirectionalFlowInterface)
        db_connection = container.resolve(ArangoDBConnectionInterface)
        return PatternEvolutionService(event_service, bidirectional_flow_service, db_connection)
    
    module.register_singleton(
        PatternEvolutionInterface,
        factory=create_pattern_evolution_service
    )
    
    # Register UserInteractionService
    def create_user_interaction_service(container):
        event_service = container.resolve(EventServiceInterface)
        pattern_aware_rag_service = container.resolve(PatternAwareRAGInterface)
        bidirectional_flow_service = container.resolve(BidirectionalFlowInterface)
        db_connection = container.resolve(ArangoDBConnectionInterface)
        return UserInteractionService(event_service, pattern_aware_rag_service, bidirectional_flow_service, db_connection)
    
    module.register_singleton(
        UserInteractionInterface,
        factory=create_user_interaction_service
    )
    
    # Register the ArangoDBDocumentService as a singleton
    module.register_singleton(
        DocumentServiceInterface,
        ArangoDBDocumentService
    )
    
    return module
