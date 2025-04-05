"""
Tests for the VectorTonicPersistenceConnector.

This test suite validates the functionality of the VectorTonicPersistenceConnector,
which integrates the vector-tonic window system with the persistence layer.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.habitat_evolution.field.persistence.vector_tonic_persistence_connector import VectorTonicPersistenceConnector
from src.habitat_evolution.adaptive_core.persistence.services.graph_service import GraphService
from src.tests.adaptive_core.persistence.arangodb.test_state_models import PatternState


class MockEventBus:
    """Mock event bus for testing."""
    
    def __init__(self):
        self.subscribers = {}
        
    def subscribe(self, event_type, callback):
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def publish(self, event_type, event_data):
        """Publish an event."""
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                callback(event_data)


@pytest.fixture
def mock_graph_service():
    """Create a mock graph service for testing."""
    service = MagicMock(spec=GraphService)
    
    # Set up repository
    service.repository = MagicMock()
    service.repository.patterns_collection = "patterns"
    service.repository.nodes_collection = "concept_nodes"
    
    # Mock db for AQL queries
    service.repository.db = MagicMock()
    service.repository.db.aql = MagicMock()
    
    return service


@pytest.fixture
def event_bus():
    """Create an event bus for testing."""
    return MockEventBus()


@pytest.fixture
def connector(mock_graph_service, event_bus):
    """Create a VectorTonicPersistenceConnector with mock dependencies."""
    return VectorTonicPersistenceConnector(mock_graph_service, event_bus)


class TestVectorTonicPersistenceConnector:
    """Test suite for the VectorTonicPersistenceConnector."""
    
    @pytest.mark.asyncio
    async def test_register_event_handlers(self, connector, event_bus):
        """Test registering event handlers."""
        # Call the method
        await connector.register_event_handlers()
        
        # Check that event handlers were registered
        assert "field.state_changed" in event_bus.subscribers
        assert "window.transition" in event_bus.subscribers
        assert "metrics.coherence_updated" in event_bus.subscribers
        assert "pattern.statistical_detected" in event_bus.subscribers
        
    @pytest.mark.asyncio
    async def test_handle_field_state_changed(self, connector, mock_graph_service):
        """Test handling a field state change event."""
        # Create event data
        event_data = {
            "field_id": "test-field",
            "state": {
                "energy": 0.5,
                "stability": 0.8
            },
            "metrics": {
                "coherence": 0.7,
                "density": 0.6
            }
        }
        
        # Call the method
        await connector.handle_field_state_changed(event_data)
        
        # Check that save_node was called
        mock_graph_service.repository.save_node.assert_called_once()
        args, kwargs = mock_graph_service.repository.save_node.call_args
        
        # Check the node properties
        node = args[0]
        assert node.name.startswith("Field State")
        assert node.attributes["type"] == "field_state"
        assert node.attributes["field_id"] == "test-field"
        assert "state_energy" in node.attributes
        assert "state_stability" in node.attributes
        
        # Check the quality state
        assert kwargs["quality_state"] == "uncertain"
        
    @pytest.mark.asyncio
    async def test_handle_window_transition_closed(self, connector, mock_graph_service):
        """Test handling a window transition to CLOSED state."""
        # Create event data
        event_data = {
            "window_id": "test-window",
            "from_state": "OPEN",
            "to_state": "CLOSED",
            "context": {
                "duration": 3600,
                "patterns_detected": 5
            }
        }
        
        # Mock create_graph_snapshot
        mock_graph_service.create_graph_snapshot.return_value = "snapshot-123"
        
        # Call the method
        await connector.handle_window_transition(event_data)
        
        # Check that create_graph_snapshot was called
        mock_graph_service.create_graph_snapshot.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_handle_window_transition_open(self, connector, mock_graph_service):
        """Test handling a window transition to OPEN state."""
        # Create event data
        event_data = {
            "window_id": "test-window",
            "from_state": "CLOSED",
            "to_state": "OPEN",
            "context": {
                "trigger": "new_data",
                "threshold": 0.7
            }
        }
        
        # Call the method
        await connector.handle_window_transition(event_data)
        
        # Check that save_node was called
        mock_graph_service.repository.save_node.assert_called_once()
        args, kwargs = mock_graph_service.repository.save_node.call_args
        
        # Check the node properties
        node = args[0]
        assert node.name.startswith("Learning Window")
        assert node.attributes["type"] == "learning_window"
        assert node.attributes["window_id"] == "test-window"
        assert "opened_at" in node.attributes
        
        # Check the quality state
        assert kwargs["quality_state"] == "uncertain"
        
    @pytest.mark.asyncio
    async def test_handle_window_transition_opening(self, connector, mock_graph_service):
        """Test handling a window transition to OPENING state."""
        # Create event data
        event_data = {
            "window_id": "test-window",
            "from_state": "CLOSED",
            "to_state": "OPENING",
            "context": {
                "trigger": "pattern_emergence",
                "coherence_threshold": 0.4,
                "potential_patterns": 3
            }
        }
        
        # Reset mock to clear previous calls
        mock_graph_service.repository.save_node.reset_mock()
        
        # Call the method
        await connector.handle_window_transition(event_data)
        
        # Check that save_node was called
        mock_graph_service.repository.save_node.assert_called_once()
        args, kwargs = mock_graph_service.repository.save_node.call_args
        
        # Check the node properties
        node = args[0]
        assert node.name.startswith("Learning Window")
        assert node.attributes["type"] == "learning_window"
        assert node.attributes["window_id"] == "test-window"
        assert node.attributes["state"] == "OPENING"
        assert "potential_patterns" in node.attributes["context"]
        
        # Check the quality state - should be poor since patterns are just emerging
        assert kwargs["quality_state"] == "poor"
        
    @pytest.mark.asyncio
    async def test_handle_coherence_metrics_updated(self, connector, mock_graph_service):
        """Test handling a coherence metrics update event."""
        # Create event data
        event_data = {
            "pattern_id": "test-pattern",
            "coherence": 0.8,
            "stability": 0.6,
            "context": {
                "window_id": "test-window"
            }
        }
        
        # Call the method
        await connector.handle_coherence_metrics_updated(event_data)
        
        # Check that evolve_pattern_confidence was called
        mock_graph_service.evolve_pattern_confidence.assert_called_once()
        args, kwargs = mock_graph_service.evolve_pattern_confidence.call_args
        
        # Check the parameters
        assert kwargs["pattern_id"] == "test-pattern"
        assert kwargs["new_confidence"] == 0.7  # (0.8 + 0.6) / 2
        assert "coherence" in kwargs["context"]
        assert "stability" in kwargs["context"]
        assert kwargs["context"]["window_id"] == "test-window"
        
    @pytest.mark.asyncio
    async def test_handle_statistical_pattern_detected(self, connector, mock_graph_service):
        """Test handling a statistical pattern detection event."""
        # Create event data
        event_data = {
            "pattern_id": "test-pattern",
            "content": "Temperature correlates with energy consumption",
            "confidence": 0.75,
            "correlation": 0.85,
            "p_value": 0.01,
            "sample_size": 100,
            "metadata": {
                "domain": "climate"
            }
        }
        
        # Call the method
        await connector.handle_statistical_pattern_detected(event_data)
        
        # Check that create_pattern was called
        mock_graph_service.create_pattern.assert_called_once()
        args, kwargs = mock_graph_service.create_pattern.call_args
        
        # Check the parameters
        assert kwargs["content"] == "Temperature correlates with energy consumption"
        assert kwargs["confidence"] == 0.75
        assert kwargs["metadata"]["type"] == "statistical"
        assert kwargs["metadata"]["correlation"] == "0.85"
        assert kwargs["metadata"]["p_value"] == "0.01"
        assert kwargs["metadata"]["sample_size"] == "100"
        assert kwargs["metadata"]["domain"] == "climate"
        
    @pytest.mark.asyncio
    async def test_find_statistical_patterns(self, connector, mock_graph_service):
        """Test finding statistical patterns."""
        # Set up mock response for AQL query
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [
            {
                "_key": "pattern1",
                "content": "Pattern 1 content",
                "metadata": {"type": "statistical"},
                "timestamp": datetime.now().isoformat(),
                "confidence": "0.8"
            }
        ]
        mock_graph_service.repository.db.aql.execute.return_value = mock_cursor
        
        # Call the method
        patterns = await connector.find_statistical_patterns()
        
        # Check that AQL execute was called
        mock_graph_service.repository.db.aql.execute.assert_called_once()
        
        # Check the results
        assert len(patterns) == 1
        assert patterns[0].id == "pattern1"
        assert patterns[0].metadata["type"] == "statistical"
        
    @pytest.mark.asyncio
    async def test_correlate_semantic_and_statistical_patterns(self, connector):
        """Test correlating semantic and statistical patterns."""
        # Mock the methods used by correlate_semantic_and_statistical_patterns
        stat_pattern = PatternState(
            id="stat1",
            content="Statistical pattern",
            metadata={"type": "statistical"},
            timestamp=datetime.now(),
            confidence=0.8
        )
        
        sem_pattern = PatternState(
            id="sem1",
            content="Semantic pattern",
            metadata={"type": "semantic"},
            timestamp=datetime.now(),
            confidence=0.7
        )
        
        # Create async mock functions
        async def mock_find_statistical_patterns():
            return [stat_pattern]
            
        async def mock_find_semantic_patterns():
            return [sem_pattern]
            
        async def mock_calculate_similarity(p1, p2):
            return 0.9
        
        # Assign the mock functions
        connector.find_statistical_patterns = mock_find_statistical_patterns
        connector._find_semantic_patterns = mock_find_semantic_patterns
        connector._calculate_pattern_similarity = mock_calculate_similarity
        
        # Call the method
        correlations = await connector.correlate_semantic_and_statistical_patterns()
        
        # Check the results
        assert len(correlations) == 1
        assert correlations[0]["statistical_pattern_id"] == "stat1"
        assert correlations[0]["semantic_pattern_id"] == "sem1"
        assert correlations[0]["similarity"] == 0.9

    @pytest.mark.asyncio
    async def test_calculate_window_potential(self, connector, mock_graph_service):
        """Test calculating semantic potential for a window."""
        # Mock the potential calculator methods
        async def mock_calculate_field_potential(window_id):
            return {
                "avg_evolutionary_potential": 0.7,
                "avg_constructive_dissonance": 0.6,
                "gradient_field": {
                    "magnitude": 0.5,
                    "direction": "increasing",
                    "uniformity": 0.8
                },
                "pattern_count": 3,
                "window_id": "test-window",
                "timestamp": datetime.now().isoformat()
            }
            
        async def mock_calculate_topological_potential(window_id):
            return {
                "connectivity": {
                    "density": 0.6,
                    "clustering": 0.7,
                    "path_efficiency": 0.8
                },
                "centrality": {
                    "centralization": 0.4,
                    "heterogeneity": 0.3
                },
                "temporal_stability": {
                    "persistence": 0.8,
                    "evolution_rate": 0.3,
                    "temporal_coherence": 0.75
                },
                "manifold_curvature": {
                    "average_curvature": 0.5,
                    "curvature_variance": 0.2,
                    "topological_depth": 0.6
                },
                "topological_energy": 0.65,
                "window_id": "test-window",
                "timestamp": datetime.now().isoformat()
            }
        
        # Assign the mock functions
        connector.potential_calculator.calculate_field_potential = mock_calculate_field_potential
        connector.potential_calculator.calculate_topological_potential = mock_calculate_topological_potential
        
        # Reset save_node mock
        mock_graph_service.repository.save_node.reset_mock()
        
        # Call the method
        potential = await connector.calculate_window_potential("test-window")
        
        # Check the results
        assert potential["evolutionary_potential"] == 0.7
        assert potential["constructive_dissonance"] == 0.6
        assert potential["topological_energy"] == 0.65
        assert potential["temporal_stability"]["temporal_coherence"] == 0.75
        
        # Calculate expected balanced potential
        expected_balanced = 0.7 * 0.3 + 0.6 * 0.3 + 0.65 * 0.2 + 0.75 * 0.2
        assert abs(potential["balanced_potential"] - expected_balanced) < 0.001
        
        # Check that save_node was called to store the potential metrics
        mock_graph_service.repository.save_node.assert_called_once()
        args, kwargs = mock_graph_service.repository.save_node.call_args
        
        # Check the node properties
        node = args[0]
        assert node.name.startswith("Semantic Potential")
        assert node.attributes["type"] == "semantic_potential"
        assert node.attributes["window_id"] == "test-window"
        assert "evolutionary_potential" in node.attributes
        assert "constructive_dissonance" in node.attributes
        assert "topological_energy" in node.attributes
        assert "balanced_potential" in node.attributes
