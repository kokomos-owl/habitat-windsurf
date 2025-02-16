"""Tests for the Pattern-Aware RAG state handler.

These tests validate that the state handler properly maintains concept-relationship
coherence and pattern stability throughout state transitions.
"""

import pytest
from datetime import datetime
from typing import Dict

from ...pattern_aware_rag.state_handler import (
    GraphStateHandler,
    StateCoherenceMetrics
)
from ...pattern_aware_rag.test_states import (
    GraphStateSnapshot,
    ConceptNode,
    ConceptRelation,
    PatternState,
    create_test_graph_state
)

def test_validate_state_coherence():
    """Test that state coherence validation works correctly."""
    handler = GraphStateHandler()
    state = create_test_graph_state()
    
    metrics = handler.validate_state_coherence(state)
    assert isinstance(metrics, StateCoherenceMetrics)
    assert metrics.concept_confidence >= 0.8  # Test state has high confidence
    assert metrics.relationship_strength >= 0.8  # Test state has strong relationships
    assert metrics.pattern_stability >= 0.7  # Test state has stable patterns

def test_prepare_prompt_context():
    """Test that prompt context is properly formatted."""
    handler = GraphStateHandler()
    state = create_test_graph_state()
    
    context = handler.prepare_prompt_context(state)
    assert isinstance(context, dict)
    assert "graph_state" in context
    assert "concept_details" in context
    assert "concept_relations" in context
    assert "pattern_context" in context
    assert "temporal_context" in context
    
    # Verify content formatting
    assert state.id in context["graph_state"]
    assert str(state.timestamp) in context["graph_state"]
    assert "climate change" in context["concept_details"]
    assert "--[" in context["concept_relations"]  # Relationship formatting
    assert "Pattern" in context["pattern_context"]
    assert "Evolution Stage" in context["temporal_context"]

def test_validate_state_transition():
    """Test that state transitions maintain coherence."""
    handler = GraphStateHandler()
    state_1 = create_test_graph_state()
    
    # Create a slightly modified state
    state_2 = GraphStateSnapshot(
        id="test-state-2",
        timestamp=datetime.now(),
        concepts=state_1.concepts.copy(),
        relations=state_1.relations.copy(),
        patterns=state_1.patterns.copy(),
        metrics={
            "coherence": 0.89,  # Slight improvement
            "stability": 0.93,
            "relationship_strength": 0.88
        },
        temporal_context={
            "stage": "evolving",
            "stability": 0.93,
            "recent_changes": ["pattern_strengthened"]
        }
    )
    
    # Transition should be valid as metrics improved
    assert handler.validate_state_transition(state_1, state_2)
    
    # Create an invalid transition (coherence drop)
    state_3 = GraphStateSnapshot(
        id="test-state-3",
        timestamp=datetime.now(),
        concepts=state_1.concepts.copy(),
        relations=state_1.relations.copy(),
        patterns=state_1.patterns.copy(),
        metrics={
            "coherence": 0.70,  # Too much drop
            "stability": 0.92,
            "relationship_strength": 0.75
        },
        temporal_context={
            "stage": "destabilizing",
            "stability": 0.85,
            "recent_changes": ["coherence_drop"]
        }
    )
    
    # Transition should be invalid due to coherence drop
    assert not handler.validate_state_transition(state_1, state_3)

def test_edge_cases():
    """Test edge cases in state handling."""
    handler = GraphStateHandler()
    
    # Empty state should raise ValueError
    empty_state = GraphStateSnapshot(
        id="empty",
        timestamp=datetime.now(),
        concepts={},
        relations=[],
        patterns={},
        metrics={
            "coherence": 0.0,
            "stability": 0.0,
            "relationship_strength": 0.0
        },
        temporal_context={
            "stage": "initial",
            "stability": 0.0,
            "recent_changes": []
        }
    )
    
    with pytest.raises(ValueError):
        handler.validate_state_coherence(empty_state)
    
    # Single concept state should work but with low metrics
    single_concept_state = GraphStateSnapshot(
        id="single",
        timestamp=datetime.now(),
        concepts={
            "c1": ConceptNode(
                id="c1",
                content="test concept",
                type="test",
                confidence=0.8,
                emergence_time=datetime.now(),
                properties={}
            )
        },
        relations=[],
        patterns={},
        metrics={
            "coherence": 0.5,
            "stability": 0.5,
            "relationship_strength": 0.0
        },
        temporal_context={
            "stage": "emerging",
            "stability": 0.5,
            "recent_changes": ["concept_added"]
        }
    )
    
    metrics = handler.validate_state_coherence(single_concept_state)
    assert metrics.concept_confidence == 0.8
    assert metrics.relationship_strength == 0.0  # No relationships
    assert metrics.pattern_stability == 0.0  # No patterns
