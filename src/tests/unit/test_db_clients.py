"""Tests for database clients."""

import pytest
from typing import Dict, Any

from src.visualization.core.db.mongo_client import MongoConfig
from src.visualization.core.db.neo4j_client import Neo4jConfig
from src.tests.mocks.mock_db import MockMongoClient, MockNeo4jClient

pytestmark = pytest.mark.asyncio

async def test_mongo_client(mock_mongo: MockMongoClient):
    """Test MongoDB client operations."""
    # Test storing visualization
    doc_id = "test_doc"
    test_data = {
        "file_paths": {
            "timeline": "path/to/timeline.json",
            "network": "path/to/network.json"
        },
        "metadata": {"test": "data"}
    }
    
    stored_id = await mock_mongo.store_visualization(doc_id, test_data)
    assert stored_id == doc_id
    
    # Test retrieving visualization
    retrieved = await mock_mongo.get_visualization(doc_id)
    assert retrieved == test_data
    
    # Test listing visualizations
    visualizations = await mock_mongo.list_visualizations(limit=10)
    assert len(visualizations) == 1
    assert visualizations[0] == test_data
    
    # Test non-existent document
    missing = await mock_mongo.get_visualization("missing_doc")
    assert missing is None

async def test_neo4j_client(
    mock_neo4j: MockNeo4jClient,
    sample_graph_data: Dict[str, Any]
):
    """Test Neo4j client operations."""
    doc_id = sample_graph_data["doc_id"]
    
    # Add test nodes
    mock_neo4j.nodes = {
        "1": {"id": "1", "doc_id": doc_id, "label": "test1"},
        "2": {"id": "2", "doc_id": doc_id, "label": "test2"}
    }
    
    # Add test relationships
    mock_neo4j.relationships = [
        {
            "doc_id": doc_id,
            "from": "1",
            "to": "2",
            "type": "TEST"
        }
    ]
    
    # Test getting graph data
    graph_data = await mock_neo4j.get_graph_data(doc_id)
    assert len(graph_data["nodes"]) == 2
    assert len(graph_data["relationships"]) == 1
    
    # Test storing layout
    layout_data = {
        "1": {"x": 0.0, "y": 0.0},
        "2": {"x": 1.0, "y": 1.0}
    }
    await mock_neo4j.store_graph_layout(doc_id, layout_data)
    assert mock_neo4j.layouts[doc_id] == layout_data
    
    # Test getting metrics
    metrics = await mock_neo4j.get_graph_metrics(doc_id)
    assert metrics["nodes"] == 2
    assert metrics["edges"] == 1
    assert metrics["density"] == 0.5  # 1 edge / (2 * 1)

def test_mongo_config():
    """Test MongoDB configuration."""
    config = MongoConfig(
        host="testhost",
        port=27018,
        username="testuser",
        password="testpass",
        database="testdb"
    )
    
    assert config.host == "testhost"
    assert config.port == 27018
    assert config.username == "testuser"
    assert config.password == "testpass"
    assert config.database == "testdb"

def test_neo4j_config():
    """Test Neo4j configuration."""
    config = Neo4jConfig(
        host="testhost",
        port=7688,
        username="testuser",
        password="testpass",
        database="testdb"
    )
    
    assert config.host == "testhost"
    assert config.port == 7688
    assert config.username == "testuser"
    assert config.password == "testpass"
    assert config.database == "testdb"
