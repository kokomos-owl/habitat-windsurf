"""Fixtures for coherence interface tests."""

import pytest
from habitat_evolution.pattern_aware_rag.core.pattern_processor import PatternProcessor
from habitat_evolution.pattern_aware_rag.state.test_states import GraphStateSnapshot
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

@pytest.fixture
def pattern_processor():
    """Initialize pattern processor for testing."""
    return PatternProcessor()

@pytest.fixture
def sample_document():
    """Sample document for testing pattern extraction."""
    return {
        "content": "Test pattern content",
        "metadata": {"source": "test", "timestamp": "2025-02-16T12:42:45-05:00"}
    }
