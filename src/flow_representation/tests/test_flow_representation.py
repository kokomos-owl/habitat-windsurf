"""Tests for flow representation package."""

import pytest
from datetime import datetime, timedelta
import numpy as np

from ..core.flow_state import FlowState, FlowDimensions
from ..visualization.flow_representation import FlowRepresentation

@pytest.fixture
def sample_dimensions():
    """Create sample flow dimensions."""
    return FlowDimensions(
        stability=0.8,
        coherence=0.7,
        emergence_rate=0.6,
        cross_pattern_flow=0.5,
        energy_state=0.4,
        adaptation_rate=0.3
    )

@pytest.fixture
def sample_state(sample_dimensions):
    """Create sample flow state."""
    return FlowState(
        dimensions=sample_dimensions,
        pattern_id="test_pattern",
        timestamp=datetime.now(),
        related_patterns=["related_1", "related_2"]
    )

class TestFlowRepresentation:
    """Test suite for flow representation visualization."""
    
    def test_state_addition(self, sample_state):
        """Test adding states to representation."""
        flow_rep = FlowRepresentation()
        flow_rep.add_state(sample_state)
        
        # Verify state was added to layers
        assert len(flow_rep.base_layer.states) == 1
        assert len(flow_rep.pattern_layer.states) == 1
        
    def test_insight_generation(self, sample_dimensions):
        """Test insight generation from state changes."""
        flow_rep = FlowRepresentation()
        
        # Add initial state
        initial_state = FlowState(
            dimensions=sample_dimensions,
            pattern_id="test_pattern",
            timestamp=datetime.now()
        )
        flow_rep.add_state(initial_state)
        
        # Verify initial insight
        assert len(flow_rep.insights) == 1
        assert flow_rep.insights[0]["type"] == "initial"
        
        # Add state with high coherence
        high_coherence_dims = FlowDimensions(
            stability=0.9,
            coherence=0.9,
            emergence_rate=0.6,
            cross_pattern_flow=0.5,
            energy_state=0.4,
            adaptation_rate=0.3
        )
        high_coherence_state = FlowState(
            dimensions=high_coherence_dims,
            pattern_id="test_pattern",
            timestamp=datetime.now() + timedelta(hours=1)
        )
        flow_rep.add_state(high_coherence_state)
        
        # Verify high coherence insight
        assert len(flow_rep.insights) == 2
        assert flow_rep.insights[1]["type"] == "high_coherence"
        
    def test_visualization_creation(self, sample_state):
        """Test creation of integrated visualization."""
        flow_rep = FlowRepresentation()
        flow_rep.add_state(sample_state)
        
        # Create visualization
        fig = flow_rep.create_visualization()
        
        # Verify figure structure
        assert len(fig.data) > 0  # Should have multiple traces
        assert fig.layout.title.text == "Integrated Flow Representation"
        
    def test_temporal_evolution(self, sample_dimensions):
        """Test visualization of temporal evolution."""
        flow_rep = FlowRepresentation()
        
        # Add sequence of states
        for i in range(3):
            # Modify dimensions slightly each time
            dims = FlowDimensions(
                stability=sample_dimensions.stability + i * 0.1,
                coherence=sample_dimensions.coherence + i * 0.1,
                emergence_rate=sample_dimensions.emergence_rate,
                cross_pattern_flow=sample_dimensions.cross_pattern_flow,
                energy_state=sample_dimensions.energy_state,
                adaptation_rate=sample_dimensions.adaptation_rate
            )
            
            state = FlowState(
                dimensions=dims,
                pattern_id="test_pattern",
                timestamp=datetime.now() + timedelta(hours=i)
            )
            flow_rep.add_state(state)
        
        # Verify timeline creation
        timeline = flow_rep.base_layer.create_timeline()
        assert len(timeline.data) > 0  # Should have multiple traces
        
        # Verify metrics reflect changes
        metrics = flow_rep.base_layer.create_metric_summary()
        assert "test_pattern" in metrics
        assert metrics["test_pattern"]["evolution_metrics"]["velocity"] > 0
