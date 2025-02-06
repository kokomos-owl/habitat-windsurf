"""Test fixtures and configuration."""

import pytest
import asyncio
from typing import Dict, Any, Generator

from src.tests.mocks.mock_db import MockMongoClient, MockNeo4jClient
from src.tests.mocks.mock_websocket import MockWebSocket, MockConnectionManager
from src.visualization.core.graph_visualizer import GraphVisualizer, VisualizationConfig

@pytest.fixture
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def mock_mongo() -> MockMongoClient:
    """Create mock MongoDB client."""
    client = MockMongoClient()
    await client.connect()
    yield client
    await client.disconnect()

@pytest.fixture
async def mock_neo4j() -> MockNeo4jClient:
    """Create mock Neo4j client."""
    client = MockNeo4jClient()
    await client.connect()
    yield client
    await client.disconnect()

@pytest.fixture
def mock_websocket() -> MockWebSocket:
    """Create mock WebSocket."""
    return MockWebSocket()

@pytest.fixture
def mock_manager() -> MockConnectionManager:
    """Create mock connection manager."""
    return MockConnectionManager()

@pytest.fixture
def sample_graph_data() -> Dict[str, Any]:
    """Create sample graph data for testing."""
    return {
        "doc_id": "test_doc",
        "temporal_stages": ["stage1", "stage2", "stage3"],
        "concept_evolution": {
            "concept1": [
                {"stage": "stage1", "confidence": 0.8},
                {"stage": "stage2", "confidence": 0.9}
            ],
            "concept2": [
                {"stage": "stage2", "confidence": 0.7},
                {"stage": "stage3", "confidence": 0.85}
            ]
        },
        "relationship_changes": [
            {
                "from": "concept1",
                "to": "concept2",
                "type": "related",
                "weight": 0.75
            }
        ],
        "coherence_metrics": {
            "stage1": 0.8,
            "stage2": 0.85,
            "stage3": 0.9
        }
    }

@pytest.fixture
def visualizer() -> GraphVisualizer:
    """Create graph visualizer instance."""
    config = VisualizationConfig(
        output_dir="test_visualizations",
        max_nodes=10
    )
    return GraphVisualizer(config)
