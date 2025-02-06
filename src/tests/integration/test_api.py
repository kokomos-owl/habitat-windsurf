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
async def mock_deps():
    """Setup mock dependencies."""
    # Create mock clients
    mock_mongo = MockMongoClient()
    mock_neo4j = MockNeo4jClient()
    mock_manager = MockConnectionManager()
    
    # Initialize mock clients
    await mock_mongo.connect()
    await mock_neo4j.connect()
    
    # Override dependency injection
    app.dependency_overrides = {
        MongoClient: lambda: mock_mongo,
        Neo4jClient: lambda: mock_neo4j,
        ConnectionManager: lambda: mock_manager
    }
    
    try:
        yield (mock_mongo, mock_neo4j, mock_manager)
    finally:
        # Cleanup
        await mock_mongo.disconnect()
        await mock_neo4j.disconnect()
        app.dependency_overrides = {}

@pytest.mark.asyncio
async def test_create_visualization(
    test_client: TestClient,
    mock_deps
):
    """Test visualization creation endpoint."""
    async with mock_deps as (mock_mongo, mock_neo4j, _):
        # Setup mock data
        doc_id = "test_doc"
        test_data = {
            "doc_id": doc_id,
            "temporal_stages": ["stage1", "stage2"],
            "concept_evolution": {"concept1": ["stage1", "stage2"]},
            "relationship_changes": [{"from": "concept1", "to": "concept2"}],
            "coherence_metrics": {"metric1": 0.8}
        }
        
        # Setup mock Neo4j data
        mock_neo4j.nodes = {
            "concept1": {"id": "concept1", "doc_id": doc_id, "label": "Concept 1"},
            "concept2": {"id": "concept2", "doc_id": doc_id, "label": "Concept 2"}
        }
        mock_neo4j.relationships = [
            {
                "doc_id": doc_id,
                "from": "concept1",
                "to": "concept2",
                "type": "EVOLVES_TO"
            }
        ]
        
        response = test_client.post(
            "/api/v1/visualize",
            json=test_data
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
    mock_deps
):
    """Test visualization retrieval endpoint."""
    async with mock_deps as (mock_mongo, _, _):
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
    mock_deps
):
    """Test WebSocket endpoint."""
    async with mock_deps as (_, _, mock_manager):
        client_id = "test_client"
        websocket = MockWebSocket()
        
        # Inject test message
        test_message = {
            "type": "update",
            "data": {"test": "data"}
        }
        websocket.inject_message(test_message)
        
        # Test connection and message handling
        await mock_manager.connect(websocket, client_id)
        
        # Verify connection
        assert websocket.connected
        assert client_id in mock_manager.active_connections
        
        # Test message broadcast
        await mock_manager.broadcast(test_message)
        
        # Verify message broadcast
        assert len(mock_manager.broadcast_messages) > 0
        assert mock_manager.broadcast_messages[-1]["message"]["type"] == "update"
        assert len(websocket.sent_messages) > 0
        assert websocket.sent_messages[-1]["type"] == "update"

@pytest.mark.asyncio
async def test_error_handling(
    test_client: TestClient,
    mock_deps
):
    """Test API error handling."""
    async with mock_deps as (mock_mongo, _, _):
        # Test invalid request body
        response = test_client.post(
            "/api/v1/visualize",
            json={"invalid": "data"}
        )
        assert response.status_code == 422
        
        # Test missing visualization
        response = test_client.get("/api/v1/visualize/nonexistent")
        assert response.status_code == 404
