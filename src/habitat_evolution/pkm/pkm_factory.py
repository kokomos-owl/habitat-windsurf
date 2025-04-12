"""
Factory for PKM components in Habitat Evolution.

This module provides factory methods for creating PKM components with their
dependencies properly initialized, addressing the complex dependency chain
in the Habitat Evolution system.
"""

import logging
from typing import Dict, Any, Optional

from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.infrastructure.services.event_service import EventService
from src.habitat_evolution.infrastructure.services.bidirectional_flow_service import BidirectionalFlowService
from src.habitat_evolution.infrastructure.services.pattern_aware_rag_service import PatternAwareRAGService
from src.habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter
from src.habitat_evolution.pkm.pkm_repository import PKMRepository
from src.habitat_evolution.pkm.pkm_bidirectional_integration import PKMBidirectionalIntegration

logger = logging.getLogger(__name__)

def create_pkm_repository(
    arangodb_connection: Optional[ArangoDBConnection] = None,
    db_config: Optional[Dict[str, Any]] = None
) -> PKMRepository:
    """
    Create a PKM repository with its dependencies properly initialized.
    
    Args:
        arangodb_connection: Optional pre-initialized ArangoDB connection
        db_config: Optional database configuration if arangodb_connection is not provided
        
    Returns:
        Initialized PKM repository
    """
    # Create ArangoDB connection if not provided
    if arangodb_connection is None:
        if db_config is None:
            db_config = {
                "host": "localhost",
                "port": 8529,
                "username": "root",
                "password": "habitat",
                "database_name": "habitat_evolution"
            }
        
        arangodb_connection = ArangoDBConnection(
            host=db_config.get("host", "localhost"),
            port=db_config.get("port", 8529),
            username=db_config.get("username", "root"),
            password=db_config.get("password", ""),
            database_name=db_config.get("database_name", "habitat_evolution")
        )
        
        # Initialize ArangoDB connection
        arangodb_connection.initialize()
        logger.info("Initialized ArangoDB connection for PKM repository")
    
    # Create PKM repository
    pkm_repository = PKMRepository(arangodb_connection)
    logger.info("Created PKM repository")
    
    return pkm_repository

def create_pkm_bidirectional_integration(
    pkm_repository: Optional[PKMRepository] = None,
    bidirectional_flow_service: Optional[BidirectionalFlowService] = None,
    event_service: Optional[EventService] = None,
    pattern_aware_rag: Optional[PatternAwareRAGService] = None,
    claude_adapter: Optional[ClaudeAdapter] = None,
    arangodb_connection: Optional[ArangoDBConnection] = None,
    db_config: Optional[Dict[str, Any]] = None,
    creator_id: Optional[str] = None
) -> PKMBidirectionalIntegration:
    """
    Create a PKM bidirectional integration with its dependencies properly initialized.
    
    This factory method handles the complex dependency chain required for the
    PKM bidirectional integration, creating and initializing components as needed.
    
    Args:
        pkm_repository: Optional pre-initialized PKM repository
        bidirectional_flow_service: Optional pre-initialized bidirectional flow service
        event_service: Optional pre-initialized event service
        pattern_aware_rag: Optional pre-initialized pattern-aware RAG service
        claude_adapter: Optional pre-initialized Claude adapter
        arangodb_connection: Optional pre-initialized ArangoDB connection
        db_config: Optional database configuration if arangodb_connection is not provided
        creator_id: Optional creator ID for PKM files
        
    Returns:
        Initialized PKM bidirectional integration
    """
    # Create ArangoDB connection if not provided
    if arangodb_connection is None:
        if db_config is None:
            db_config = {
                "host": "localhost",
                "port": 8529,
                "username": "root",
                "password": "habitat",
                "database_name": "habitat_evolution"
            }
        
        arangodb_connection = ArangoDBConnection(
            host=db_config.get("host", "localhost"),
            port=db_config.get("port", 8529),
            username=db_config.get("username", "root"),
            password=db_config.get("password", ""),
            database_name=db_config.get("database_name", "habitat_evolution")
        )
        
        # Initialize ArangoDB connection
        arangodb_connection.initialize()
        logger.info("Initialized ArangoDB connection for PKM bidirectional integration")
    
    # Create PKM repository if not provided
    if pkm_repository is None:
        pkm_repository = create_pkm_repository(arangodb_connection)
        logger.info("Created PKM repository for PKM bidirectional integration")
    
    # Create event service if not provided
    if event_service is None:
        event_service = EventService()
        event_service.initialize()
        logger.info("Created event service for PKM bidirectional integration")
    
    # Create Claude adapter if not provided
    if claude_adapter is None:
        claude_adapter = ClaudeAdapter()
        logger.info("Created Claude adapter for PKM bidirectional integration")
    
    # Create pattern-aware RAG if not provided
    if pattern_aware_rag is None:
        # Note: We're using a mock PatternAwareRAGService for simplicity in this factory
        # In a real implementation, you would need to provide all required dependencies
        from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_pattern_repository import ArangoDBPatternRepository
        from src.habitat_evolution.infrastructure.adapters.pattern_bridge import PatternBridge
        from src.habitat_evolution.adaptive_core.models.pattern import Pattern as AdaptiveCorePattern
        from src.habitat_evolution.infrastructure.interfaces.services.vector_tonic_service_interface import VectorTonicServiceInterface
        
        # Create a minimal pattern repository
        pattern_repository = ArangoDBPatternRepository(
            db_connection=arangodb_connection,
            event_service=event_service
        )
        
        # Create a minimal pattern bridge
        pattern_bridge = PatternBridge[
            AdaptiveCorePattern
        ](
            event_service=event_service,
            pattern_class=AdaptiveCorePattern
        )
        
        # For the vector tonic service, we'd need a mock implementation
        # This is a simplified approach for demonstration purposes
        class MockVectorTonicService(VectorTonicServiceInterface):
            def initialize(self, config=None): pass
            def shutdown(self): pass
        
        vector_tonic_service = MockVectorTonicService()
        
        pattern_aware_rag = PatternAwareRAGService(
            db_connection=arangodb_connection,
            event_service=event_service,
            vector_tonic_service=vector_tonic_service,
            pattern_repository=pattern_repository,
            pattern_bridge=pattern_bridge
        )
        
        # Initialize the service
        try:
            pattern_aware_rag.initialize()
            logger.info("Created pattern-aware RAG service for PKM bidirectional integration")
        except Exception as e:
            logger.warning(f"Error initializing pattern-aware RAG service: {e}")
            logger.info("Using uninitialized pattern-aware RAG service")
    
    # Create bidirectional flow service if not provided
    if bidirectional_flow_service is None:
        bidirectional_flow_service = BidirectionalFlowService(
            event_service=event_service,
            pattern_aware_rag_service=pattern_aware_rag,
            arangodb_connection=arangodb_connection
        )
        bidirectional_flow_service.initialize()
        logger.info("Created bidirectional flow service for PKM bidirectional integration")
    
    # Create PKM bidirectional integration
    pkm_bidirectional = PKMBidirectionalIntegration(
        pkm_repository=pkm_repository,
        bidirectional_flow_service=bidirectional_flow_service,
        event_service=event_service,
        claude_adapter=claude_adapter,
        creator_id=creator_id
    )
    logger.info("Created PKM bidirectional integration")
    
    return pkm_bidirectional
