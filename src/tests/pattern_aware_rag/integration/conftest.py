"""Fixtures for integration tests."""

from typing import Dict, Any
import pytest
from datetime import datetime

from habitat_evolution.pattern_aware_rag.core.pattern_processor import PatternProcessor
from habitat_evolution.pattern_aware_rag.core.coherence_interface import CoherenceInterface
from habitat_evolution.pattern_aware_rag.state.test_states import (
    GraphStateSnapshot,
    PatternState,
    ConceptNode
)

@pytest.fixture
def pattern_processor():
    """Initialize pattern processor."""
    return PatternProcessor()

@pytest.fixture
def coherence_interface():
    """Initialize coherence interface."""
    return CoherenceInterface()

@pytest.fixture
def sample_document() -> Dict[str, Any]:
    """Create a sample document for testing."""
    return {
        "id": "test_doc_1",
        "content": "This is a test document for pattern extraction.",
        "metadata": {
            "source": "test",
            "timestamp": datetime.now().isoformat()
        }
    }
