"""
Tests for the Neo4j pattern repository implementation.
"""

import pytest
from uuid import uuid4
from datetime import datetime

from habitat_evolution.adaptive_core.models.pattern import Pattern
from habitat_evolution.adaptive_core.persistence.neo4j.pattern_repository import Neo4jPatternRepository

@pytest.fixture
def pattern_repository():
    """Create a pattern repository instance"""
    return Neo4jPatternRepository()

@pytest.fixture
def sample_pattern():
    """Create a sample pattern for testing"""
    return Pattern(
        id=str(uuid4()),
        base_concept="test_pattern",
        creator_id="test_creator",
        weight=1.0,
        confidence=0.8,
        coherence=0.7,
        phase_stability=0.9,
        signal_strength=0.85
    )

def test_create_pattern(pattern_repository, sample_pattern):
    """Test creating a pattern"""
    # Create pattern
    pattern_id = pattern_repository.create(sample_pattern)
    assert pattern_id is not None
    
    # Verify pattern was created
    stored_pattern = pattern_repository.read(pattern_id)
    assert stored_pattern is not None
    assert stored_pattern.base_concept == sample_pattern.base_concept
    assert stored_pattern.creator_id == sample_pattern.creator_id
    assert stored_pattern.coherence == sample_pattern.coherence

def test_update_pattern(pattern_repository, sample_pattern):
    """Test updating a pattern"""
    # Create pattern
    pattern_id = pattern_repository.create(sample_pattern)
    
    # Update pattern
    sample_pattern.coherence = 0.9
    pattern_repository.update(sample_pattern)
    
    # Verify update
    updated_pattern = pattern_repository.read(pattern_id)
    assert updated_pattern.coherence == 0.9

def test_delete_pattern(pattern_repository, sample_pattern):
    """Test deleting a pattern"""
    # Create pattern
    pattern_id = pattern_repository.create(sample_pattern)
    
    # Delete pattern
    pattern_repository.delete(pattern_id)
    
    # Verify deletion
    deleted_pattern = pattern_repository.read(pattern_id)
    assert deleted_pattern is None

def test_get_by_concept(pattern_repository, sample_pattern):
    """Test getting patterns by concept"""
    # Create pattern
    pattern_repository.create(sample_pattern)
    
    # Get patterns by concept
    patterns = pattern_repository.get_by_concept(sample_pattern.base_concept)
    assert len(patterns) > 0
    assert patterns[0].base_concept == sample_pattern.base_concept

def test_get_by_creator(pattern_repository, sample_pattern):
    """Test getting patterns by creator"""
    # Create pattern
    pattern_repository.create(sample_pattern)
    
    # Get patterns by creator
    patterns = pattern_repository.get_by_creator(sample_pattern.creator_id)
    assert len(patterns) > 0
    assert patterns[0].creator_id == sample_pattern.creator_id

def test_get_by_coherence_range(pattern_repository, sample_pattern):
    """Test getting patterns by coherence range"""
    # Create pattern
    pattern_repository.create(sample_pattern)
    
    # Get patterns in coherence range
    patterns = pattern_repository.get_by_coherence_range(0.6, 0.8)
    assert len(patterns) > 0
    assert 0.6 <= patterns[0].coherence <= 0.8

def test_create_and_get_relationships(pattern_repository, sample_pattern):
    """Test creating and getting pattern relationships"""
    # Create two patterns
    pattern1_id = pattern_repository.create(sample_pattern)
    
    pattern2 = Pattern(
        id=str(uuid4()),
        base_concept="related_pattern",
        creator_id="test_creator",
        coherence=0.8
    )
    pattern2_id = pattern_repository.create(pattern2)
    
    # Create relationship
    rel_properties = {
        "strength": 0.9,
        "type": "phase_locked"
    }
    rel_id = pattern_repository.create_relationship(
        pattern1_id,
        pattern2_id,
        "PHASE_LOCKED",
        rel_properties
    )
    assert rel_id is not None
    
    # Get related patterns
    related_patterns = pattern_repository.get_related_patterns(pattern1_id, "PHASE_LOCKED")
    assert len(related_patterns) == 1
    assert related_patterns[0].id == pattern2_id

def test_update_pattern_metrics(pattern_repository, sample_pattern):
    """Test updating pattern metrics"""
    # Create pattern
    pattern_id = pattern_repository.create(sample_pattern)
    
    # Update metrics
    new_metrics = {
        "coherence": 0.9,
        "phase_stability": 0.85,
        "signal_strength": 0.95
    }
    pattern_repository.update_pattern_metrics(pattern_id, new_metrics)
    
    # Verify metrics update
    updated_pattern = pattern_repository.read(pattern_id)
    assert updated_pattern.metrics["coherence"] == 0.9
    assert updated_pattern.metrics["phase_stability"] == 0.85
    assert updated_pattern.metrics["signal_strength"] == 0.95

@pytest.mark.parametrize("coherence,turbulence", [
    (0.2, 0.8),  # Incoherent pattern with high turbulence
    (0.8, 0.4),  # Coherent pattern with moderate turbulence
])
def test_pattern_stability_conditions(pattern_repository, coherence, turbulence):
    """Test pattern stability under different conditions"""
    pattern = Pattern(
        id=str(uuid4()),
        base_concept="stability_test",
        creator_id="test_creator",
        coherence=coherence
    )
    
    # Create pattern
    pattern_id = pattern_repository.create(pattern)
    
    # Update with turbulence metrics
    metrics = {
        "turbulence": turbulence,
        "viscosity": 0.9 if coherence < 0.5 else 0.4,
        "volume": 0.3 if coherence < 0.5 else 0.6
    }
    pattern_repository.update_pattern_metrics(pattern_id, metrics)
    
    # Verify pattern state
    updated_pattern = pattern_repository.read(pattern_id)
    if coherence < 0.5:  # Incoherent pattern
        assert updated_pattern.metrics["viscosity"] > 0.8
        assert updated_pattern.metrics["volume"] < 0.3
    else:  # Coherent pattern
        assert updated_pattern.metrics["viscosity"] < 0.5
        assert updated_pattern.metrics["volume"] > 0.6
