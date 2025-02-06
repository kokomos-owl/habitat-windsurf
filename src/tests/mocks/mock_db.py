"""Mock database clients for testing."""

from typing import Dict, Any, Optional, List
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class MockMongoClient:
    """Mock MongoDB client for testing."""
    
    def __init__(self):
        """Initialize mock client."""
        self.visualizations = {}
        self.client = None
        self.db = self
        
    async def connect(self):
        """Mock connection."""
        logger.info("Connected to mock MongoDB")
        
    async def disconnect(self):
        """Mock disconnection."""
        logger.info("Disconnected from mock MongoDB")
        
    async def store_visualization(
        self,
        doc_id: str,
        visualization_data: Dict[str, Any]
    ) -> str:
        """Store visualization in mock database."""
        self.visualizations[doc_id] = visualization_data
        return doc_id
        
    async def get_visualization(
        self,
        doc_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve visualization from mock database."""
        return self.visualizations.get(doc_id)
        
    async def list_visualizations(
        self,
        limit: int = 10,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """List visualizations from mock database."""
        items = list(self.visualizations.values())
        return items[skip:skip + limit]

class MockNeo4jClient:
    """Mock Neo4j client for testing."""
    
    def __init__(self):
        """Initialize mock client."""
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.relationships: List[Dict[str, Any]] = []
        self.layouts: Dict[str, Dict[str, Dict[str, float]]] = {}
        
    async def connect(self):
        """Mock connection."""
        logger.info("Connected to mock Neo4j")
        
    async def disconnect(self):
        """Mock disconnection."""
        logger.info("Disconnected from mock Neo4j")
        
    async def get_graph_data(
        self,
        doc_id: str
    ) -> Dict[str, Any]:
        """Get mock graph data."""
        return {
            "nodes": [
                node for node in self.nodes.values()
                if node.get("doc_id") == doc_id
            ],
            "relationships": [
                rel for rel in self.relationships
                if rel.get("doc_id") == doc_id
            ]
        }
        
    async def store_graph_layout(
        self,
        doc_id: str,
        layout_data: Dict[str, Dict[str, float]]
    ):
        """Store mock graph layout."""
        self.layouts[doc_id] = layout_data
        
    async def get_graph_metrics(
        self,
        doc_id: str
    ) -> Dict[str, Any]:
        """Get mock graph metrics."""
        nodes = len([
            node for node in self.nodes.values()
            if node.get("doc_id") == doc_id
        ])
        edges = len([
            rel for rel in self.relationships
            if rel.get("doc_id") == doc_id
        ])
        
        return {
            "nodes": nodes,
            "edges": edges,
            "density": edges / (nodes * (nodes - 1)) if nodes > 1 else 0
        }
