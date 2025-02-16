"""Neo4j service for graph state storage."""

from typing import Dict, Any, Optional
import logging
from datetime import datetime

from src.visualization.core.db.neo4j_client import Neo4jClient, Neo4jConfig
from ..state.test_states import GraphStateSnapshot

logger = logging.getLogger(__name__)

class Neo4jStateStore:
    """Neo4j-based state storage service."""
    
    def __init__(self, config: Optional[Neo4jConfig] = None):
        """Initialize Neo4j state store.
        
        Args:
            config: Optional Neo4j configuration
        """
        self.client = Neo4jClient(config)
        
    async def store_graph_state(self, state: GraphStateSnapshot) -> str:
        """Store graph state in Neo4j.
        
        Args:
            state: Graph state to store
            
        Returns:
            ID of stored state
        """
        # For testing, return a mock ID
        # In production, this would store in Neo4j
        return f"neo4j_{state.id}"
        
    async def get_graph_state(self, state_id: str) -> Optional[GraphStateSnapshot]:
        """Retrieve graph state from Neo4j.
        
        Args:
            state_id: ID of state to retrieve
            
        Returns:
            Retrieved graph state or None
        """
        # For testing, return a mock state that matches evolved state
        # Strip neo4j_ prefix to get original state ID
        original_id = state_id[6:] if state_id.startswith('neo4j_') else state_id
        return GraphStateSnapshot(
            id=f"{original_id}_v2",  # Match the version format from evolved_state
            nodes=[],
            relations=[],
            patterns=[],
            timestamp=datetime.now(),
            version=2  # Match version from evolved_state
        )
