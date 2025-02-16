"""
Coherence Interface Tests for Pattern-Aware RAG.

These tests verify the coherence interface functionality after
the sequential foundation is established.
"""
import pytest
from habitat_evolution.pattern_aware_rag.core.coherence_interface import CoherenceInterface
from habitat_evolution.pattern_aware_rag.state.graph_state import GraphState
from habitat_evolution.pattern_aware_rag.learning.window_manager import LearningWindowManager

@pytest.fixture
def coherence_interface():
    """Initialize coherence interface for testing."""
    return CoherenceInterface()

@pytest.fixture
def window_manager():
    """Initialize learning window manager."""
    return LearningWindowManager()

@pytest.fixture
async def sample_graph_state(pattern_processor, sample_document):
    """Create a sample graph state for testing."""
    pattern = await pattern_processor.extract_pattern(sample_document)
    adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
    return await pattern_processor.prepare_graph_state(pattern, adaptive_id)

class TestCoherenceInterface:
    """Test the coherence interface functionality."""
    
    async def test_state_alignment(self, coherence_interface, sample_graph_state):
        """Test state alignment mechanism."""
        alignment = await coherence_interface.align_state(sample_graph_state)
        assert alignment.coherence_score > 0.0
        assert alignment.state_matches is True
    
    async def test_back_pressure(self, coherence_interface, window_manager, sample_graph_state):
        """Test back pressure controls."""
        # Create rapid state changes
        states = [sample_graph_state for _ in range(5)]
        
        # Back pressure should increase with rapid changes
        pressures = []
        for state in states:
            pressure = await coherence_interface.process_state_change(state)
            pressures.append(pressure)
        
        # Verify back pressure increases
        assert pressures[-1] > pressures[0]
    
    async def test_learning_window(self, window_manager, sample_graph_state):
        """Test learning window transitions."""
        window = window_manager.current_window
        
        # Add state changes
        for _ in range(3):
            await window_manager.record_state_change(sample_graph_state)
        
        # Verify window metrics
        metrics = window.get_metrics()
        assert metrics.total_changes == 3
        assert metrics.stability_score is not None
    
    async def test_state_agreement(self, coherence_interface, sample_graph_state):
        """Test state agreement verification."""
        agreement = await coherence_interface.verify_state_agreement(sample_graph_state)
        assert agreement.is_valid
        assert agreement.confidence_score > 0.0
