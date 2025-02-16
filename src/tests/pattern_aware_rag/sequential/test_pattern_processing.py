"""
Sequential Foundation Tests for Pattern-Aware RAG.

These tests verify the critical sequential foundation required before
any concurrent operations can begin.
"""
import pytest
from habitat_evolution.pattern_aware_rag.core.pattern_processor import PatternProcessor
from habitat_evolution.pattern_aware_rag.state.graph_state import GraphState
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

class TestSequentialFoundation:
    """Test the sequential foundation requirements."""
    
    async def test_pattern_extraction(self, pattern_processor, sample_document):
        """Test pattern extraction from document."""
        pattern = await pattern_processor.extract_pattern(sample_document)
        assert pattern is not None
        assert pattern.content == sample_document["content"]
        assert pattern.metadata["source"] == "test"
    
    async def test_adaptive_id_assignment(self, pattern_processor, sample_document):
        """Test Adaptive ID assignment to pattern."""
        pattern = await pattern_processor.extract_pattern(sample_document)
        adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
        assert isinstance(adaptive_id, AdaptiveID)
        assert adaptive_id.base_concept is not None
    
    async def test_graph_ready_state(self, pattern_processor, sample_document):
        """Test pattern reaches graph-ready state."""
        pattern = await pattern_processor.extract_pattern(sample_document)
        adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
        graph_state = await pattern_processor.prepare_graph_state(pattern, adaptive_id)
        assert isinstance(graph_state, GraphState)
        assert graph_state.is_graph_ready()
    
    async def test_sequential_dependency(self, pattern_processor, sample_document):
        """Test that operations must occur in correct sequence."""
        # Should fail if trying to prepare graph state before ID assignment
        pattern = await pattern_processor.extract_pattern(sample_document)
        with pytest.raises(ValueError):
            await pattern_processor.prepare_graph_state(pattern, None)
    
    async def test_provenance_tracking(self, pattern_processor, sample_document):
        """Test provenance is established and tracked."""
        pattern = await pattern_processor.extract_pattern(sample_document)
        adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
        graph_state = await pattern_processor.prepare_graph_state(pattern, adaptive_id)
        
        provenance = graph_state.get_provenance()
        assert provenance.source == "test"
        assert provenance.timestamp is not None
        assert provenance.pattern_id == adaptive_id.id
