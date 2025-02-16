"""
Test suite for Claude State Handler.

Tests the transformation and validation of graph states for Claude interaction.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List

from habitat_evolution.pattern_aware_rag.claude_state_handler import (
    ClaudeStateHandler,
    GraphStateContext,
    ConceptRelation
)

# Test Data
@pytest.fixture
def sample_concepts() -> Dict[str, Dict[str, any]]:
    """Provide sample concepts for testing."""
    return {
        "c1": {
            "name": "climate_change",
            "type": "environmental_concept",
            "confidence": 0.9
        },
        "c2": {
            "name": "renewable_energy",
            "type": "technology_concept",
            "confidence": 0.85
        },
        "c3": {
            "name": "policy_framework",
            "type": "governance_concept",
            "confidence": 0.75
        }
    }

@pytest.fixture
def sample_relations(sample_concepts) -> List[ConceptRelation]:
    """Provide sample concept relations."""
    now = datetime.now()
    return [
        ConceptRelation(
            source_id="c1",
            target_id="c2",
            relation_type="drives_adoption",
            strength=0.8,
            context={"evidence": "multiple_studies"},
            timestamp=now
        ),
        ConceptRelation(
            source_id="c2",
            target_id="c3",
            relation_type="requires_framework",
            strength=0.7,
            context={"policy_level": "national"},
            timestamp=now
        )
    ]

@pytest.fixture
def sample_state_context(
    sample_concepts,
    sample_relations
) -> GraphStateContext:
    """Provide sample graph state context."""
    return GraphStateContext(
        state_id="test_state_001",
        timestamp=datetime.now(),
        concepts=sample_concepts,
        relations=sample_relations,
        coherence_metrics={
            "overall": 0.85,
            "concept_coherence": 0.9,
            "relation_coherence": 0.8
        },
        temporal_context={
            "evolution_markers": ["concept_emergence", "relation_strengthening"],
            "stability_index": 0.75
        }
    )

class TestClaudeStateHandler:
    """Test suite for Claude state handling."""

    def test_state_preparation(self, sample_state_context):
        """Test preparation of state for Claude consumption."""
        handler = ClaudeStateHandler()
        document = handler.prepare_state_for_claude(sample_state_context)
        
        # Verify document content
        assert "Current Graph State Context" in document.page_content
        assert "Concept Network" in document.page_content
        assert "climate_change" in document.page_content
        assert "renewable_energy" in document.page_content
        
        # Verify metadata
        assert document.metadata["state_id"] == "test_state_001"
        assert document.metadata["concept_count"] == 3
        assert document.metadata["relation_count"] == 2
    
    def test_claude_requirements_validation(self, sample_state_context):
        """Test validation of Claude's state requirements."""
        handler = ClaudeStateHandler()
        
        # Test valid state
        is_valid, error = handler.validate_claude_requirements(sample_state_context)
        assert is_valid
        assert error is None
        
        # Test invalid state (no concepts)
        invalid_state = GraphStateContext(
            state_id="invalid_001",
            timestamp=datetime.now(),
            concepts={},
            relations=sample_state_context.relations,
            coherence_metrics=sample_state_context.coherence_metrics,
            temporal_context=sample_state_context.temporal_context
        )
        is_valid, error = handler.validate_claude_requirements(invalid_state)
        assert not is_valid
        assert "No concepts present" in error
    
    def test_temporal_context_handling(self, sample_state_context):
        """Test handling of temporal context in state preparation."""
        handler = ClaudeStateHandler()
        document = handler.prepare_state_for_claude(sample_state_context)
        
        # Verify temporal context inclusion
        assert "Temporal Context" in document.page_content
        assert "evolution_markers" in document.page_content
        assert "stability_index" in document.page_content
    
    def test_concept_network_representation(self, sample_state_context):
        """Test concept network representation in prepared state."""
        handler = ClaudeStateHandler()
        document = handler.prepare_state_for_claude(sample_state_context)
        
        # Verify network representation
        assert "drives_adoption" in document.page_content
        assert "requires_framework" in document.page_content
        assert "0.80" in document.page_content  # Relation strength
