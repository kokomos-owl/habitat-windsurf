"""
Tests for the GraphService implementation.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.habitat_evolution.pattern_aware_rag.services.graph_service import GraphService
from src.habitat_evolution.pattern_aware_rag.state.test_states import (
    GraphStateSnapshot, ConceptNode, ConceptRelation, PatternState
)
from src.habitat_evolution.adaptive_core.models.pattern import Pattern


@pytest.fixture
def mock_repository():
    """Create a mock repository for testing"""
    mock = MagicMock()
    
    # Set up mock behavior for find_node_by_id
    mock.find_node_by_id.side_effect = lambda id: ConceptNode(
        id=id,
        name=f"Node {id}",
        node_type="ENTITY",
        quality_state="uncertain",
        attributes={}
    ) if id else None
    
    return mock


@pytest.fixture
def graph_service(mock_repository):
    """Create a graph service instance with a mock repository"""
    return GraphService(repository=mock_repository)


@pytest.mark.asyncio
async def test_create_snapshot(graph_service, mock_repository):
    """Test creating a graph state snapshot"""
    # Prepare test data
    nodes = [
        ConceptNode(id="1", name="Node 1", node_type="ENTITY", quality_state="uncertain", attributes={}),
        ConceptNode(id="2", name="Node 2", node_type="ENTITY", quality_state="good", attributes={})
    ]
    
    relations = [
        ConceptRelation(
            source_id="1",
            target_id="2",
            relation_type="RELATED_TO",
            quality_state="uncertain",
            weight=0.75,
            attributes={}
        )
    ]
    
    patterns = [
        PatternState(
            id="p1",
            content="Test pattern",
            metadata={},
            timestamp=datetime.now(),
            confidence=0.8
        )
    ]
    
    # Set up mock behavior
    mock_repository.save_state.return_value = "snapshot1"
    
    # Call the method
    snapshot_id = await graph_service.create_snapshot(nodes, relations, patterns)
    
    # Verify the result
    assert snapshot_id == "snapshot1"
    mock_repository.save_state.assert_called_once()
    
    # Verify the snapshot passed to save_state
    call_args = mock_repository.save_state.call_args[0][0]
    assert isinstance(call_args, GraphStateSnapshot)
    assert len(call_args.nodes) == 2
    assert len(call_args.relations) == 1
    assert len(call_args.patterns) == 1


@pytest.mark.asyncio
async def test_add_node(graph_service, mock_repository):
    """Test adding a node to the graph"""
    # Set up mock behavior
    mock_repository.save_node.return_value = "node1"
    
    # Call the method
    node_id = await graph_service.add_node(
        name="Test Node",
        node_type="ENTITY",
        quality_state="uncertain",
        attributes={"category": "CLIMATE_HAZARD"}
    )
    
    # Verify the result
    assert node_id == "node1"
    mock_repository.save_node.assert_called_once()
    
    # Verify the node passed to save_node
    call_args = mock_repository.save_node.call_args[0][0]
    assert isinstance(call_args, ConceptNode)
    assert call_args.name == "Test Node"
    assert call_args.node_type == "ENTITY"
    assert call_args.quality_state == "uncertain"
    assert call_args.attributes == {"category": "CLIMATE_HAZARD"}


@pytest.mark.asyncio
async def test_add_relation(graph_service, mock_repository):
    """Test adding a relation between nodes"""
    # Set up mock behavior
    mock_repository.save_relation.return_value = "rel1"
    
    # Call the method
    relation_id = await graph_service.add_relation(
        source_id="1",
        target_id="2",
        relation_type="IMPACTS",
        quality_state="uncertain",
        weight=0.8,
        attributes={"confidence": 0.7}
    )
    
    # Verify the result
    assert relation_id == "rel1"
    mock_repository.save_relation.assert_called_once()
    
    # Verify the relation passed to save_relation
    call_args = mock_repository.save_relation.call_args[0][0]
    assert isinstance(call_args, ConceptRelation)
    assert call_args.source_id == "1"
    assert call_args.target_id == "2"
    assert call_args.relation_type == "IMPACTS"
    assert call_args.quality_state == "uncertain"
    assert call_args.weight == 0.8
    assert call_args.attributes == {"confidence": 0.7}


@pytest.mark.asyncio
async def test_add_pattern(graph_service, mock_repository):
    """Test adding a pattern to the graph"""
    # Set up mock behavior
    mock_repository.save_pattern.return_value = "pattern1"
    
    # Call the method
    pattern_id = await graph_service.add_pattern(
        content="Climate impact pattern",
        confidence=0.85,
        metadata={"domain": "climate_risk"}
    )
    
    # Verify the result
    assert pattern_id == "pattern1"
    mock_repository.save_pattern.assert_called_once()
    
    # Verify the pattern passed to save_pattern
    call_args = mock_repository.save_pattern.call_args[0][0]
    assert isinstance(call_args, PatternState)
    assert call_args.content == "Climate impact pattern"
    assert call_args.confidence == 0.85
    assert call_args.metadata == {"domain": "climate_risk"}


@pytest.mark.asyncio
async def test_store_pattern(graph_service, mock_repository):
    """Test storing a Pattern object in the graph"""
    # Create a Pattern object
    pattern = Pattern(
        id="p1",
        base_concept="climate_pattern",
        creator_id="test_user",
        weight=1.0,
        confidence=0.9,
        coherence=0.85,
        phase_stability=0.8,
        signal_strength=0.75
    )
    
    # Set up mock behavior
    mock_repository.save_pattern.return_value = "p1"
    
    # Call the method
    pattern_id = await graph_service.store_pattern(pattern)
    
    # Verify the result
    assert pattern_id == "p1"
    mock_repository.save_pattern.assert_called_once()
    
    # Verify the pattern state passed to save_pattern
    call_args = mock_repository.save_pattern.call_args[0][0]
    assert isinstance(call_args, PatternState)
    assert call_args.id == "p1"
    assert call_args.content == "climate_pattern"
    assert call_args.confidence == 0.9
    assert "coherence" in call_args.metadata
    assert call_args.metadata["coherence"] == 0.85


@pytest.mark.asyncio
async def test_update_node_quality(graph_service, mock_repository):
    """Test updating the quality state of a node"""
    # Set up mock behavior
    node = ConceptNode(
        id="node1",
        name="Test Node",
        node_type="ENTITY",
        quality_state="uncertain",
        attributes={}
    )
    mock_repository.find_node_by_id.return_value = node
    
    # Call the method
    result = await graph_service.update_node_quality("node1", "good")
    
    # Verify the result
    assert result is True
    mock_repository.track_quality_transition.assert_called_once_with("node1", "uncertain", "good")
    mock_repository.save_node.assert_called_once()
    
    # Verify the node was updated
    assert node.quality_state == "good"


@pytest.mark.asyncio
async def test_update_relation_quality(graph_service, mock_repository):
    """Test updating the quality state of a relation"""
    # Set up mock behavior
    relation = ConceptRelation(
        source_id="1",
        target_id="2",
        relation_type="IMPACTS",
        quality_state="uncertain",
        weight=0.8,
        attributes={}
    )
    mock_repository.find_relations_by_nodes.return_value = [relation]
    
    # Call the method
    result = await graph_service.update_relation_quality("1", "2", "IMPACTS", "good")
    
    # Verify the result
    assert result is True
    mock_repository.track_quality_transition.assert_called_once_with("1_2_IMPACTS", "uncertain", "good")
    mock_repository.save_relation.assert_called_once()
    
    # Verify the relation was updated
    assert relation.quality_state == "good"


@pytest.mark.asyncio
async def test_map_density_centers(graph_service, mock_repository):
    """Test mapping density centers in the graph"""
    # Set up mock behavior
    nodes = [
        ConceptNode(
            id="1",
            name="High Coherence Node",
            node_type="ENTITY",
            quality_state="good",
            attributes={"coherence": 0.9, "category": "CLIMATE_HAZARD"}
        ),
        ConceptNode(
            id="2",
            name="Medium Coherence Node",
            node_type="ENTITY",
            quality_state="good",
            attributes={"coherence": 0.7, "category": "INFRASTRUCTURE"}
        ),
        ConceptNode(
            id="3",
            name="Low Coherence Node",
            node_type="ENTITY",
            quality_state="good",
            attributes={"coherence": 0.5, "category": "ECOSYSTEM"}
        )
    ]
    mock_repository.find_nodes_by_quality.return_value = nodes
    
    # Call the method
    centers = await graph_service.map_density_centers(coherence_threshold=0.7)
    
    # Verify the result
    assert len(centers) == 2
    assert centers[0]["id"] == "1"
    assert centers[0]["coherence"] == 0.9
    assert centers[1]["id"] == "2"
    assert centers[1]["coherence"] == 0.7
    assert "3" not in [c["id"] for c in centers]


@pytest.mark.asyncio
async def test_get_quality_distribution(graph_service, mock_repository):
    """Test getting the quality distribution"""
    # Set up mock behavior
    mock_repository.find_nodes_by_quality.side_effect = lambda quality: [
        MagicMock() for _ in range({"poor": 5, "uncertain": 10, "good": 15}[quality])
    ]
    mock_repository.find_relations_by_quality.side_effect = lambda quality: [
        MagicMock() for _ in range({"poor": 2, "uncertain": 8, "good": 5}[quality])
    ]
    
    # Call the method
    distribution = await graph_service.get_quality_distribution()
    
    # Verify the result
    assert distribution["nodes"]["poor"] == 5
    assert distribution["nodes"]["uncertain"] == 10
    assert distribution["nodes"]["good"] == 15
    assert distribution["relations"]["poor"] == 2
    assert distribution["relations"]["uncertain"] == 8
    assert distribution["relations"]["good"] == 5
