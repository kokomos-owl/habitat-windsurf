"""Neo4j service for graph state storage."""

from typing import Dict, Any, Optional
import logging
from datetime import datetime

from visualization.core.db.neo4j_client import Neo4jClient, Neo4jConfig
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
        
    async def store_graph_state(self, state: Dict[str, Any]) -> str:
        """Store graph state in Neo4j.
        
        Args:
            state: Graph state to store with nodes and relationships
            
        Returns:
            ID of stored state
        """
        try:
            if not self.client.driver:
                await self.client.connect()
                
            async with self.client.driver.session() as session:
                # Create unique state ID
                state_id = f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Store nodes
                for node in state["nodes"]:
                    node_props = {**node, "state_id": state_id}
                    query = """
                    CREATE (n:Pattern)
                    SET n = $props
                    RETURN n
                    """
                    await session.run(query, props=node_props)
                
                # Store relationships
                for rel in state["relationships"]:
                    query = """
                    MATCH (source:Pattern {id: $source_id, state_id: $state_id})
                    MATCH (target:Pattern {id: $target_id, state_id: $state_id})
                    CREATE (source)-[r:RESONATES_WITH {strength: $strength, timestamp: $timestamp}]->(target)
                    RETURN r
                    """
                    await session.run(
                        query,
                        source_id=rel["source"],
                        target_id=rel["target"],
                        state_id=state_id,
                        strength=rel["strength"],
                        timestamp=rel["timestamp"]
                    )
                    
                return state_id
                
        except Exception as e:
            logger.error(f"Failed to store graph state: {e}")
            raise
        
    async def get_graph_state(self, state_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve graph state from Neo4j.
        
        Args:
            state_id: ID of state to retrieve
            
        Returns:
            Retrieved graph state or None
        """
        try:
            if not self.client.driver:
                await self.client.connect()
                
            async with self.client.driver.session() as session:
                # Get nodes
                nodes_query = """
                MATCH (n:Pattern {state_id: $state_id})
                RETURN n
                """
                nodes_result = await session.run(nodes_query, state_id=state_id)
                nodes = [dict(record["n"]) async for record in nodes_result]
                
                # Get relationships
                rels_query = """
                MATCH (source:Pattern {state_id: $state_id})
                      -[r:RESONATES_WITH]->
                      (target:Pattern {state_id: $state_id})
                RETURN source.id as source_id,
                       target.id as target_id,
                       r.strength as strength,
                       r.timestamp as timestamp
                """
                rels_result = await session.run(rels_query, state_id=state_id)
                relationships = [
                    {
                        "source": record["source_id"],
                        "target": record["target_id"],
                        "type": "RESONATES_WITH",
                        "strength": record["strength"],
                        "timestamp": record["timestamp"]
                    }
                    async for record in rels_result
                ]
                
                return {
                    "nodes": nodes,
                    "relationships": relationships
                }
                
        except Exception as e:
            logger.error(f"Failed to retrieve graph state: {e}")
            return None
