"""
Tests for the ArangoDB graph state repository implementation.
"""

import pytest
from uuid import uuid4
from datetime import datetime
from typing import List, Dict, Any

# Use the local test state models to avoid import path issues
from src.tests.adaptive_core.persistence.arangodb.test_state_models import (
    GraphStateSnapshot, ConceptNode, ConceptRelation, PatternState
)
from src.habitat_evolution.adaptive_core.persistence.arangodb.graph_state_repository import ArangoDBGraphStateRepository
from src.habitat_evolution.adaptive_core.persistence.interfaces.graph_state_repository import GraphStateRepositoryInterface


@pytest.fixture
def graph_state_repository():
    """Create a graph state repository instance"""
    return ArangoDBGraphStateRepository()


@pytest.fixture
def sample_concept_node():
    """Create a sample concept node for testing"""
    return ConceptNode(
        id=str(uuid4()),
        name="test_entity",
        attributes={
            "category": "CLIMATE_HAZARD",
            "coherence": "0.7",
            "stability": "0.65",
            "type": "ENTITY",
            "quality_state": "uncertain"
        }
    )


@pytest.fixture
def sample_concept_relation(sample_concept_node):
    """Create a sample concept relation for testing"""
    target_node = ConceptNode(
        id=str(uuid4()),
        name="related_entity",
        attributes={
            "category": "INFRASTRUCTURE",
            "coherence": "0.6",
            "stability": "0.55",
            "type": "ENTITY",
            "quality_state": "uncertain"
        }
    )
    
    # In the actual implementation, we'll need to store quality_state in attributes
    # since ConceptRelation doesn't have this field directly
    return ConceptRelation(
        source_id=sample_concept_node.id,
        target_id=target_node.id,
        relation_type="IMPACTS",
        weight=0.75
    ), target_node


@pytest.fixture
def sample_pattern_state():
    """Create a sample pattern state for testing"""
    return PatternState(
        id=str(uuid4()),
        content="climate impact pattern",
        metadata={
            "domain": "climate_risk",
            "source": "test"
        },
        timestamp=datetime.now(),
        confidence=0.7
    )


@pytest.fixture
def sample_graph_state(sample_concept_node, sample_concept_relation, sample_pattern_state):
    """Create a sample graph state for testing"""
    relation, target_node = sample_concept_relation
    
    return GraphStateSnapshot(
        id=str(uuid4()),
        nodes=[sample_concept_node, target_node],
        relations=[relation],
        patterns=[sample_pattern_state],
        timestamp=datetime.now(),
        version=1
    )


def test_save_and_find_state(graph_state_repository, sample_graph_state):
    """Test saving and finding a graph state"""
    # Save graph state
    state_id = graph_state_repository.save_state(sample_graph_state)
    assert state_id is not None
    
    # Find graph state by ID
    found_state = graph_state_repository.find_by_id(state_id)
    assert found_state is not None
    assert len(found_state.nodes) == len(sample_graph_state.nodes)
    assert len(found_state.relations) == len(sample_graph_state.relations)
    assert len(found_state.patterns) == len(sample_graph_state.patterns)
    assert found_state.version == sample_graph_state.version


def test_save_and_find_node(graph_state_repository, sample_concept_node):
    """Test saving and finding a concept node"""
    # Save node
    node_id = graph_state_repository.save_node(sample_concept_node)
    assert node_id is not None
    assert node_id == sample_concept_node.id
    
    # Find node by ID
    found_node = graph_state_repository.find_node_by_id(node_id)
    assert found_node is not None
    assert found_node.name == sample_concept_node.name
    assert found_node.attributes["quality_state"] == sample_concept_node.attributes["quality_state"]
    assert found_node.attributes["category"] == sample_concept_node.attributes["category"]


def test_save_and_find_relation(graph_state_repository, sample_concept_relation):
    """Test saving and finding a concept relation"""
    relation, target_node = sample_concept_relation
    
    # Save nodes first
    source_node_id = graph_state_repository.save_node(ConceptNode(
        id=relation.source_id,
        name="source_node",
        attributes={
            "type": "ENTITY",
            "quality_state": "uncertain"
        }
    ))
    target_node_id = graph_state_repository.save_node(target_node)
    
    # Save relation
    relation_id = graph_state_repository.save_relation(relation)
    assert relation_id is not None
    
    # Find relation by source and target
    found_relations = graph_state_repository.find_relations_by_nodes(source_node_id, target_node_id)
    assert len(found_relations) > 0
    found_relation = found_relations[0]
    assert found_relation.relation_type == relation.relation_type
    assert found_relation.weight == relation.weight


def test_save_and_find_pattern(graph_state_repository, sample_pattern_state):
    """Test saving and finding a pattern state"""
    # Save pattern
    pattern_id = graph_state_repository.save_pattern(sample_pattern_state)
    assert pattern_id is not None
    assert pattern_id == sample_pattern_state.id
    
    # Find pattern by ID
    found_pattern = graph_state_repository.find_pattern_by_id(pattern_id)
    assert found_pattern is not None
    assert found_pattern.content == sample_pattern_state.content
    assert found_pattern.confidence == sample_pattern_state.confidence
    assert found_pattern.metadata["domain"] == sample_pattern_state.metadata["domain"]


def test_find_nodes_by_quality(graph_state_repository, sample_concept_node):
    """Test finding nodes by quality state"""
    # Save node
    graph_state_repository.save_node(sample_concept_node)
    
    # Create and save another node with different quality
    good_node = ConceptNode(
        id=str(uuid4()),
        name="good_entity",
        attributes={
            "category": "CLIMATE_HAZARD",
            "coherence": "0.85",
            "stability": "0.9",
            "type": "ENTITY",
            "quality_state": "good"
        }
    )
    graph_state_repository.save_node(good_node)
    
    # Find nodes by quality
    uncertain_nodes = graph_state_repository.find_nodes_by_quality("uncertain")
    assert len(uncertain_nodes) > 0
    assert any(node.id == sample_concept_node.id for node in uncertain_nodes)
    
    good_nodes = graph_state_repository.find_nodes_by_quality("good")
    assert len(good_nodes) > 0
    assert any(node.id == good_node.id for node in good_nodes)


def test_find_relations_by_quality(graph_state_repository, sample_concept_relation):
    """Test finding relations by quality state"""
    relation, target_node = sample_concept_relation
    
    # Save nodes first
    source_node = ConceptNode(
        id=relation.source_id,
        name="source_node",
        attributes={
            "type": "ENTITY",
            "quality_state": "uncertain"
        }
    )
    graph_state_repository.save_node(source_node)
    graph_state_repository.save_node(target_node)
    
    # Save relation with quality state in attributes
    relation_with_quality = relation
    # We need to add quality_state to the relation via the repository
    graph_state_repository.save_relation(relation_with_quality, quality_state="uncertain")
    
    # Create and save another relation with different quality
    good_relation = ConceptRelation(
        source_id=source_node.id,
        target_id=target_node.id,
        relation_type="RELATED_TO",
        weight=0.9
    )
    graph_state_repository.save_relation(good_relation, quality_state="good")
    
    # Find relations by quality
    uncertain_relations = graph_state_repository.find_relations_by_quality("uncertain")
    assert len(uncertain_relations) > 0
    assert any(rel.source_id == relation.source_id and rel.target_id == relation.target_id 
               and rel.relation_type == relation.relation_type for rel in uncertain_relations)
    
    good_relations = graph_state_repository.find_relations_by_quality("good")
    assert len(good_relations) > 0
    assert any(rel.source_id == good_relation.source_id and rel.target_id == good_relation.target_id 
               and rel.relation_type == good_relation.relation_type for rel in good_relations)


def test_find_nodes_by_category(graph_state_repository, sample_concept_node):
    """Test finding nodes by category"""
    # Save node
    graph_state_repository.save_node(sample_concept_node)
    
    # Create and save another node with different category
    infrastructure_node = ConceptNode(
        id=str(uuid4()),
        name="infrastructure_entity",
        attributes={
            "category": "INFRASTRUCTURE",
            "coherence": "0.7",
            "stability": "0.65",
            "type": "ENTITY",
            "quality_state": "uncertain"
        }
    )
    graph_state_repository.save_node(infrastructure_node)
    
    # Find nodes by category
    climate_hazard_nodes = graph_state_repository.find_nodes_by_category("CLIMATE_HAZARD")
    assert len(climate_hazard_nodes) > 0
    assert any(node.id == sample_concept_node.id for node in climate_hazard_nodes)
    
    infrastructure_nodes = graph_state_repository.find_nodes_by_category("INFRASTRUCTURE")
    assert len(infrastructure_nodes) > 0
    assert any(node.id == infrastructure_node.id for node in infrastructure_nodes)


def test_track_quality_transition(graph_state_repository, sample_concept_node):
    """Test tracking quality transitions"""
    # Save node
    graph_state_repository.save_node(sample_concept_node)
    
    # Get the initial quality state
    initial_node = graph_state_repository.find_node_by_id(sample_concept_node.id)
    assert initial_node is not None
    assert initial_node.attributes["quality_state"] == "uncertain"
    
    # Manually update the quality state in the database
    # This simulates what would happen in the pattern evolution process
    collection = graph_state_repository.db.collection(graph_state_repository.nodes_collection)
    collection.update(
        {"_key": sample_concept_node.id},
        {
            "quality_state": "good",
            "attributes": {
                "quality_state": "good",
                "category": "CLIMATE_HAZARD",
                "coherence": "0.7",
                "stability": "0.65",
                "type": "ENTITY"
            }
        }
    )
    
    # Track the quality transition
    graph_state_repository.track_quality_transition(
        entity_id=sample_concept_node.id,
        from_quality="uncertain",
        to_quality="good"
    )
    
    # Get the updated node
    updated_node = graph_state_repository.find_node_by_id(sample_concept_node.id)
    assert updated_node is not None
    assert updated_node.attributes["quality_state"] == "good"
    
    # Get quality transitions for entity
    transitions = graph_state_repository.get_quality_transitions(sample_concept_node.id)
    assert len(transitions) > 0
    assert transitions[0]["from_quality"] == "uncertain"
    assert transitions[0]["to_quality"] == "good"
