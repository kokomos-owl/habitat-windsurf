"""
Factory for PKM components in Habitat Evolution.

This module provides factory methods for creating PKM components with their
dependencies properly initialized, addressing the complex dependency chain
in the Habitat Evolution system.
"""

import logging
import os
import sys
import uuid
from typing import Dict, Any, Optional, Tuple, Union, List
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
        
        # Create a mock implementation of VectorTonicServiceInterface
        class MockVectorTonicService(VectorTonicServiceInterface):
            """Mock implementation of VectorTonicServiceInterface for testing."""
            
            def __init__(self):
                self.vector_spaces = {}
                self.vectors = {}
                self.patterns = {}
            
            def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
                """Initialize the vector tonic service."""
                pass
            
            def shutdown(self) -> None:
                """Shutdown the vector tonic service."""
                pass
            
            def register_vector_space(self, name: str, dimensions: int, 
                                     metadata: Optional[Dict[str, Any]] = None) -> str:
                """Register a new vector space."""
                space_id = str(uuid.uuid4())
                self.vector_spaces[space_id] = {
                    "name": name,
                    "dimensions": dimensions,
                    "metadata": metadata or {}
                }
                return space_id
            
            def store_vector(self, vector_space_id: str, vector: List[float], 
                           entity_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
                """Store a vector in the specified vector space."""
                vector_id = str(uuid.uuid4())
                self.vectors[vector_id] = {
                    "vector_space_id": vector_space_id,
                    "vector": vector,
                    "entity_id": entity_id,
                    "metadata": metadata or {}
                }
                return vector_id
            
            def find_similar_vectors(self, vector_space_id: str, 
                                   query_vector: List[float],
                                   limit: int = 10,
                                   threshold: float = 0.0) -> List[Dict[str, Any]]:
                """Find vectors similar to the query vector."""
                # Mock implementation returns empty list
                return []
            
            def detect_tonic_patterns(self, vector_space_id: str, 
                                     vectors: List[List[float]],
                                     threshold: float = 0.7) -> List[Dict[str, Any]]:
                """Detect tonic patterns in the specified vectors."""
                # Mock implementation returns a single pattern
                pattern_id = str(uuid.uuid4())
                self.patterns[pattern_id] = {
                    "id": pattern_id,
                    "vector_space_id": vector_space_id,
                    "vectors": vectors,
                    "centroid": [0.0] * len(vectors[0]) if vectors else [],
                    "coherence": 0.8
                }
                return [self.patterns[pattern_id]]
            
            def validate_harmonic_coherence(self, pattern_id: str, 
                                          new_vector: List[float]) -> Dict[str, Any]:
                """Validate the harmonic coherence of a new vector with an existing pattern."""
                # Mock implementation always returns high coherence
                return {
                    "coherence": 0.9,
                    "recommendation": "include",
                    "pattern_id": pattern_id
                }
            
            def update_pattern_with_vector(self, pattern_id: str, 
                                         vector: List[float],
                                         weight: float = 1.0) -> Dict[str, Any]:
                """Update a pattern with a new vector."""
                if pattern_id in self.patterns:
                    self.patterns[pattern_id]["vectors"].append(vector)
                    return self.patterns[pattern_id]
                return {}
            
            def get_pattern_centroid(self, pattern_id: str) -> List[float]:
                """Get the centroid vector of a pattern."""
                if pattern_id in self.patterns:
                    return self.patterns[pattern_id]["centroid"]
                return []
            
            def get_pattern_vectors(self, pattern_id: str) -> List[Dict[str, Any]]:
                """Get all vectors associated with a pattern."""
                if pattern_id in self.patterns:
                    return [{
                        "vector": v,
                        "metadata": {}
                    } for v in self.patterns[pattern_id]["vectors"]]
                return []
            
            def calculate_vector_gradient(self, vector_space_id: str,
                                         start_vector: List[float],
                                         end_vector: List[float],
                                         steps: int = 10) -> List[List[float]]:
                """Calculate a gradient between two vectors."""
                # Mock implementation returns a simple linear interpolation
                result = []
                for i in range(steps):
                    t = i / (steps - 1) if steps > 1 else 0
                    gradient_vector = [
                        start_vector[j] * (1 - t) + end_vector[j] * t
                        for j in range(len(start_vector))
                    ]
                    result.append(gradient_vector)
                return result
        
        vector_tonic_service = MockVectorTonicService()
        
        pattern_aware_rag = PatternAwareRAGService(
            db_connection=arangodb_connection,
            pattern_repository=pattern_repository,
            vector_tonic_service=vector_tonic_service,
            claude_adapter=claude_adapter,
            event_service=event_service,
            config={"creator_id": creator_id}
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
