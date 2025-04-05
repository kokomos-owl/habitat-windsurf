"""
Tests for the GraphService.

This test suite validates the functionality of the GraphService, which provides
higher-level operations on top of the GraphStateRepository.
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.habitat_evolution.adaptive_core.persistence.services.graph_service import GraphService
from src.habitat_evolution.adaptive_core.persistence.arangodb.graph_state_repository import ArangoDBGraphStateRepository
from src.tests.adaptive_core.persistence.arangodb.test_state_models import (
    ConceptNode, ConceptRelation, PatternState, GraphStateSnapshot
)


@pytest.fixture
def mock_repository():
    """Create a mock repository for testing."""
    repo = MagicMock(spec=ArangoDBGraphStateRepository)
    
    # Set up collections
    repo.nodes_collection = "concept_nodes"
    repo.relations_collection = "concept_relations"
    repo.patterns_collection = "patterns"
    repo.states_collection = "graph_states"
    repo.transitions_collection = "quality_transitions"
    
    # Set up mock db
    repo.db = MagicMock()
    repo.db.aql = MagicMock()
    
    return repo


@pytest.fixture
def graph_service(mock_repository):
    """Create a graph service with a mock repository."""
    return GraphService(mock_repository)


class TestGraphService:
    """Test suite for the GraphService."""
    
    def test_create_concept(self, graph_service, mock_repository):
        """Test creating a concept node."""
        # Call the method
        node = graph_service.create_concept(
            name="Test Concept",
            attributes={"key": "value"},
            quality_state="uncertain",
            confidence=0.5
        )
        
        # Check that save_node was called
        mock_repository.save_node.assert_called_once()
        
        # Check the node properties
        assert node.name == "Test Concept"
        assert node.attributes["key"] == "value"
        assert node.attributes["quality_state"] == "uncertain"
        assert node.attributes["confidence"] == "0.5"
        
    def test_find_concepts_by_name(self, graph_service, mock_repository):
        """Test finding concepts by name."""
        # Set up mock response
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [
            {
                "_key": "123",
                "name": "Test Concept",
                "attributes": {"quality_state": "uncertain"}
            }
        ]
        mock_repository.db.aql.execute.return_value = mock_cursor
        
        # Call the method
        nodes = graph_service.find_concepts_by_name("Test Concept")
        
        # Check that execute was called with the right query
        mock_repository.db.aql.execute.assert_called_once()
        args, kwargs = mock_repository.db.aql.execute.call_args
        assert "FILTER node.name ==" in args[0]
        assert kwargs["bind_vars"]["name"] == "Test Concept"
        
        # Check the result
        assert len(nodes) == 1
        assert nodes[0].name == "Test Concept"
        assert nodes[0].id == "123"
        
    def test_evolve_concept_quality(self, graph_service, mock_repository):
        """Test evolving a concept's quality state."""
        # Set up mock response for find_node_by_id
        mock_node = ConceptNode(
            id="123",
            name="Test Concept",
            attributes={"quality_state": "uncertain"},
            created_at=datetime.now()
        )
        mock_repository.find_node_by_id.return_value = mock_node
        
        # Call the method
        graph_service.evolve_concept_quality(
            node_id="123",
            to_quality="good",
            confidence=0.8,
            context={"source": "test"}
        )
        
        # Check that track_quality_transition was called
        mock_repository.track_quality_transition.assert_called_once_with(
            entity_id="123",
            from_quality="uncertain",
            to_quality="good",
            confidence=0.8,
            context={"source": "test"}
        )
        
    def test_create_relation(self, graph_service, mock_repository):
        """Test creating a relation between concepts."""
        # Call the method
        relation = graph_service.create_relation(
            source_id="123",
            target_id="456",
            relation_type="related_to",
            weight=0.7,
            quality_state="uncertain"
        )
        
        # Check that save_relation was called
        mock_repository.save_relation.assert_called_once()
        
        # Check the relation properties
        assert relation.source_id == "123"
        assert relation.target_id == "456"
        assert relation.relation_type == "related_to"
        assert relation.weight == 0.7
        
    def test_create_pattern(self, graph_service, mock_repository):
        """Test creating a pattern."""
        # Call the method
        pattern = graph_service.create_pattern(
            content="Test Pattern",
            metadata={"source": "test"},
            confidence=0.5
        )
        
        # Check that save_pattern was called
        mock_repository.save_pattern.assert_called_once()
        
        # Check the pattern properties
        assert pattern.content == "Test Pattern"
        assert pattern.metadata["source"] == "test"
        assert pattern.confidence == 0.5
        
    def test_evolve_pattern_confidence(self, graph_service, mock_repository):
        """Test evolving a pattern's confidence."""
        # Set up mock response for find_pattern_by_id
        mock_pattern = PatternState(
            id="123",
            content="Test Pattern",
            metadata={},
            timestamp=datetime.now(),
            confidence=0.3
        )
        mock_repository.find_pattern_by_id.return_value = mock_pattern
        
        # Call the method
        updated_pattern = graph_service.evolve_pattern_confidence(
            pattern_id="123",
            new_confidence=0.8,
            context={"source": "test"}
        )
        
        # Check that save_pattern was called
        mock_repository.save_pattern.assert_called_once()
        
        # Check that track_quality_transition was called (quality changed from poor to good)
        mock_repository.track_quality_transition.assert_called_once_with(
            entity_id="123",
            from_quality="poor",
            to_quality="good",
            confidence=0.8,
            context={"source": "test"}
        )
        
        # Check the updated pattern
        assert updated_pattern.confidence == 0.8
        
    def test_confidence_to_quality(self, graph_service):
        """Test converting confidence scores to quality states."""
        assert graph_service._confidence_to_quality(0.3) == "poor"
        assert graph_service._confidence_to_quality(0.5) == "uncertain"
        assert graph_service._confidence_to_quality(0.8) == "good"
        
    def test_create_graph_snapshot(self, graph_service, mock_repository):
        """Test creating a graph snapshot."""
        # Set up test data
        nodes = [
            ConceptNode(id="123", name="Concept 1", attributes={}, created_at=datetime.now()),
            ConceptNode(id="456", name="Concept 2", attributes={}, created_at=datetime.now())
        ]
        relations = [
            ConceptRelation(source_id="123", target_id="456", relation_type="related_to", weight=1.0)
        ]
        
        # Call the method
        snapshot_id = graph_service.create_graph_snapshot(nodes=nodes, relations=relations)
        
        # Check that save_state was called
        mock_repository.save_state.assert_called_once()
        
        # Check that the snapshot contains the right data
        args, kwargs = mock_repository.save_state.call_args
        snapshot = args[0]
        assert len(snapshot.nodes) == 2
        assert len(snapshot.relations) == 1
        assert snapshot.nodes[0].id == "123"
        assert snapshot.relations[0].source_id == "123"
        
    def test_get_concept_neighborhood(self, graph_service, mock_repository):
        """Test getting a concept's neighborhood."""
        # Set up mock responses
        center_node = ConceptNode(id="123", name="Center", attributes={}, created_at=datetime.now())
        neighbor_node = ConceptNode(id="456", name="Neighbor", attributes={}, created_at=datetime.now())
        relation = ConceptRelation(source_id="123", target_id="456", relation_type="related_to", weight=1.0)
        
        mock_repository.find_node_by_id.side_effect = lambda id: {
            "123": center_node,
            "456": neighbor_node
        }.get(id)
        
        # Mock the find_relations_by_concept method
        graph_service.find_relations_by_concept = MagicMock(return_value=[relation])
        
        # Call the method
        nodes, relations = graph_service.get_concept_neighborhood("123")
        
        # Check the results
        assert len(nodes) == 2
        assert len(relations) == 1
        assert nodes[0].id == "123"
        assert nodes[1].id == "456"
        assert relations[0].source_id == "123"
        assert relations[0].target_id == "456"
