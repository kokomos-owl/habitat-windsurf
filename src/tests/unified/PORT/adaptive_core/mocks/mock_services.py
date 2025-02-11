"""Mock services for testing."""

from typing import Dict, Any, Optional
from unittest.mock import MagicMock

class MockConfig:
    """Mock configuration for testing."""
    def __init__(self, config_data: Dict[str, Any] = None):
        self._config = config_data or {
            'PATTERN_CORE': {'threshold': 0.5},
            'KNOWLEDGE_COHERENCE': {'min_score': 0.7},
            'EVENT_HANDLING': {
                'LISTENER_THREADS': 2,
                'POLLING_INTERVAL': 1,
                'QUEUE_SIZE': 100,
                'TIMEOUT': 5
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def __getattr__(self, name: str) -> Any:
        return self._config.get(name)

class MockEventManager:
    """Mock event manager for testing."""
    def __init__(self):
        self.events = []
        self.publish = MagicMock()
        self.subscribe = MagicMock()

class MockMongoDBClient:
    """Mock MongoDB client for testing."""
    def __init__(self):
        self.db = {}
        self.insert_one = MagicMock()
        self.find_one = MagicMock()
        self.update_one = MagicMock()

class MockNeo4jClient:
    """Mock Neo4j client for testing."""
    def __init__(self):
        self.nodes = {}
        self.relationships = {}
        self.create_node = MagicMock()
        self.create_relationship = MagicMock()
        self.get_node = MagicMock()

class MockPatternCore:
    """Mock pattern core for testing."""
    def __init__(self):
        self.patterns = {}
        self.observe_pattern = MagicMock()
        self.get_pattern = MagicMock()

class MockKnowledgeCoherence:
    """Mock knowledge coherence for testing."""
    def __init__(self):
        self.coherence_scores = {}
        self.calculate_coherence = MagicMock(return_value=0.85)
        self.update_coherence = MagicMock()

class MockTimestampService:
    """Mock timestamp service for testing."""
    def __init__(self):
        self.get_timestamp = MagicMock(return_value="2025-01-13T20:00:00Z")
        self.format_timestamp = MagicMock(return_value="2025-01-13T20:00:00Z")

class MockRelationshipRepository:
    """Mock relationship repository for testing."""
    def __init__(self):
        self.relationships = {}
        self.create_relationship = MagicMock()
        self.get_relationship = MagicMock()
        self.update_relationship = MagicMock()
        self.delete_relationship = MagicMock()

class MockServiceConnector:
    """Mock service connector for testing."""
    def __init__(self):
        self.config = MockConfig()
        self.event_manager = MockEventManager()
        self.mongodb_client = MockMongoDBClient()
        self.neo4j_client = MockNeo4jClient()
        self.pattern_core = MockPatternCore()
        self.knowledge_coherence = MockKnowledgeCoherence()
        self.timestamp_service = MockTimestampService()
        self.relationship_repository = MockRelationshipRepository()

class MockGraphVisualizer:
    """Mock graph visualizer for testing."""
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir
        self.create_evolution_view = MagicMock(return_value={
            'timeline': f"{output_dir}/timeline.html",
            'network': f"{output_dir}/network.html",
            'metrics': f"{output_dir}/metrics.html"
        })
