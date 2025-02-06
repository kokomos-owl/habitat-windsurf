"""Neo4j client for graph data."""

from typing import Dict, Any, List, Optional
import logging
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class Neo4jConfig(BaseModel):
    """Neo4j connection configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=7687)
    username: str = Field(default="neo4j")
    password: str = Field(default="password")
    database: str = Field(default="neo4j")

class Neo4jClient:
    """Async Neo4j client for graph operations."""
    
    def __init__(self, config: Optional[Neo4jConfig] = None):
        """Initialize Neo4j client.
        
        Args:
            config: Neo4j configuration
        """
        self.config = config or Neo4jConfig()
        self.driver = None
        
    async def connect(self):
        """Establish database connection."""
        try:
            uri = f"neo4j://{self.config.host}:{self.config.port}"
            self.driver = AsyncGraphDatabase.driver(
                uri,
                auth=(self.config.username, self.config.password)
            )
            await self.driver.verify_connectivity()
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
            
    async def disconnect(self):
        """Close database connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Disconnected from Neo4j")
            
    async def get_graph_data(
        self,
        doc_id: str
    ) -> Dict[str, Any]:
        """Retrieve graph data for visualization.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Graph data including nodes and relationships
        """
        async with self.driver.session(database=self.config.database) as session:
            # Get nodes
            nodes_query = """
            MATCH (n:Node {doc_id: $doc_id})
            RETURN n
            """
            nodes_result = await session.run(nodes_query, doc_id=doc_id)
            nodes = [dict(record["n"]) async for record in nodes_result]
            
            # Get relationships
            rels_query = """
            MATCH (n:Node {doc_id: $doc_id})-[r]->(m:Node {doc_id: $doc_id})
            RETURN r, type(r) as type, startNode(r) as source, endNode(r) as target
            """
            rels_result = await session.run(rels_query, doc_id=doc_id)
            relationships = [
                {
                    "type": record["type"],
                    "source": dict(record["source"])["id"],
                    "target": dict(record["target"])["id"],
                    **dict(record["r"])
                }
                async for record in rels_result
            ]
            
            return {
                "nodes": nodes,
                "relationships": relationships
            }
            
    async def store_graph_layout(
        self,
        doc_id: str,
        layout_data: Dict[str, Dict[str, float]]
    ):
        """Store graph layout positions.
        
        Args:
            doc_id: Document identifier
            layout_data: Node positions
        """
        async with self.driver.session(database=self.config.database) as session:
            query = """
            UNWIND $positions as pos
            MATCH (n:Node {id: pos.id, doc_id: $doc_id})
            SET n.x = pos.x, n.y = pos.y
            """
            positions = [
                {"id": node_id, **coords}
                for node_id, coords in layout_data.items()
            ]
            await session.run(query, doc_id=doc_id, positions=positions)
            
    async def get_graph_metrics(
        self,
        doc_id: str
    ) -> Dict[str, Any]:
        """Calculate graph metrics.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dictionary of graph metrics
        """
        async with self.driver.session(database=self.config.database) as session:
            query = """
            MATCH (n:Node {doc_id: $doc_id})
            OPTIONAL MATCH (n)-[r]->()
            WITH count(DISTINCT n) as nodes, count(r) as edges
            RETURN {
                nodes: nodes,
                edges: edges,
                density: CASE nodes 
                    WHEN 0 THEN 0 
                    ELSE toFloat(edges) / (nodes * (nodes-1))
                END
            } as metrics
            """
            result = await session.run(query, doc_id=doc_id)
            record = await result.single()
            return record["metrics"]
