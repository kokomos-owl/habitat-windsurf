"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from fastapi import WebSocketDisconnect
from typing import Dict, Any

from src.visualization.api.app import app
from src.visualization.core.db.mongo_client import MongoClient
from src.visualization.core.db.neo4j_client import Neo4jClient
from src.visualization.websocket.manager import ConnectionManager
from src.tests.mocks.mock_db import MockMongoClient, MockNeo4jClient
from src.tests.mocks.mock_websocket import MockWebSocket, MockConnectionManager

@pytest.fixture
def test_client():
    """Create test client."""
    return TestClient(app)

@pytest.fixture
async def mock_deps(
    mock_mongo: MockMongoClient,
    mock_neo4j: MockNeo4jClient,
    mock_manager: MockConnectionManager
):
    """Setup mock dependencies."""
    # Initialize mock clients
    await mock_mongo.connect()
    await mock_neo4j.connect()
    
    # Override dependency injection
    app.dependency_overrides = {
        MongoClient: lambda: mock_mongo,
        Neo4jClient: lambda: mock_neo4j,
        ConnectionManager: lambda: mock_manager
    }
    yield
    
    # Cleanup
    await mock_mongo.disconnect()
    await mock_neo4j.disconnect()
    app.dependency_overrides = {}

@pytest.mark.asyncio
async def test_create_visualization(
    test_client: TestClient,
    mock_deps,
    sample_graph_data: Dict[str, Any]
):
    """Test visualization creation endpoint."""
    response = test_client.post(
        "/api/v1/visualize",
        json={
            "doc_id": "test_doc",
            "temporal_stages": ["stage1", "stage2"],
            "concept_evolution": {"concept1": ["stage1", "stage2"]},
            "relationship_changes": [{"from": "concept1", "to": "concept2"}],
            "coherence_metrics": {"metric1": 0.8}
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "visualization_id" in data
    assert "file_paths" in data
    assert all(
        path in data["file_paths"]
        for path in ["timeline", "network", "coherence"]
    )

@pytest.mark.asyncio
async def test_get_visualization(
    test_client: TestClient,
    mock_deps,
    mock_mongo: MockMongoClient,
    sample_graph_data: Dict[str, Any]
):
    """Test visualization retrieval endpoint."""
    # Store test data
    doc_id = "test_doc"
    test_data = {
        "file_paths": {
            "timeline": "path/to/timeline.json",
            "network": "path/to/network.json",
            "coherence": "path/to/coherence.json"
        },
        "metadata": {
            "doc_id": doc_id,
            "temporal_stages": ["stage1", "stage2"],
            "concept_evolution": {"concept1": ["stage1", "stage2"]},
            "relationship_changes": [{"from": "concept1", "to": "concept2"}],
            "coherence_metrics": {"metric1": 0.8}
        }
    }
    await mock_mongo.store_visualization(doc_id, test_data)
    
    # Test retrieval
    response = test_client.get(f"/api/v1/visualize/{doc_id}")
    assert response.status_code == 200
    data = response.json()
    assert data == test_data
    
    # Test non-existent visualization
    response = test_client.get("/api/v1/visualize/missing")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_websocket_endpoint(
    mock_deps,
    mock_manager: MockConnectionManager
):
    """Test WebSocket endpoint."""
    client_id = "test_client"
    websocket = MockWebSocket()
    
    # Test connection
    with pytest.raises(WebSocketDisconnect):
        await app.router.routes[-1].endpoint(websocket, client_id)
    
    # Test message handling
    test_message = {
        "type": "update",
        "data": {"test": "data"}
    }
    await mock_manager.broadcast(test_message)
    
    # Verify message broadcast
    assert len(mock_manager.broadcast_messages) > 0
    assert mock_manager.broadcast_messages[-1]["message"]["type"] == "update"

def test_error_handling(
    test_client: TestClient,
    mock_deps
):
    """Test API error handling."""
    # Test invalid request body
    response = test_client.post(
        "/api/v1/visualize",
        json={"invalid": "data"}
    )
    assert response.status_code == 422
    
    # Test missing visualization
    response = test_client.get("/api/v1/visualize/nonexistent")
    assert response.status_code == 404
