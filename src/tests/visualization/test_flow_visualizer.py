"""
Tests for flow visualization components.
Validates visualization of pattern evolution and structure-meaning relationships.
"""

import pytest
from typing import Dict, Any
from datetime import datetime

from src.core.flow.habitat_flow import FlowState
from src.visualization.core.flow_visualizer import FlowVisualizer
from src.tests.core.flow.mock_flow_patterns import MockFlowPatterns

@pytest.fixture
def mock_patterns():
    """Provide mock patterns for testing."""
    return MockFlowPatterns()

@pytest.fixture
def flow_visualizer():
    """Create flow visualizer instance."""
    return FlowVisualizer()

@pytest.fixture
def sample_flow_data(mock_patterns):
    """Generate sample flow data for testing."""
    return {
        "text": mock_patterns.patterns["temperature_rise"].content,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "source": "climate_data"
        }
    }

class TestFlowVisualization:
    """Test suite for flow visualization functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_flow_visualization(self, 
                                          flow_visualizer,
                                          sample_flow_data):
        """Test basic flow visualization generation."""
        result = await flow_visualizer.visualize_flow(sample_flow_data)
        
        assert isinstance(result, dict)
        assert "nodes" in result
        assert "edges" in result
        assert "metrics" in result
        assert "visualization" in result
        
        # Verify visualization structure
        vis = result["visualization"]
        assert "data" in vis
        assert "layout" in vis
        
        # Verify data traces
        data = vis["data"]
        assert len(data) == 2  # Edge and node traces
        
        # Verify node trace has coherence colorscale
        node_trace = data[1]
        assert "marker" in node_trace
        assert "colorscale" in node_trace["marker"]
        
        # Verify node structure
        for node in result["nodes"]:
            assert all(k in node for k in [
                "id", "label", "strength", "coherence"
            ])
        
        # Verify edge structure
        for edge in result["edges"]:
            assert all(k in edge for k in [
                "source", "target", "weight"
            ])
            
        # Verify metrics
        assert all(k in result["metrics"] for k in [
            "flow_velocity",
            "pattern_density",
            "coherence_score"
        ])

    @pytest.mark.asyncio
    async def test_structure_meaning_visualization(self,
                                                 flow_visualizer,
                                                 sample_flow_data):
        """Test visualization of structure-meaning relationships."""
        result = await flow_visualizer.visualize_flow(sample_flow_data)
        
        # Verify structure-meaning metrics
        assert "structure_meaning" in result
        metrics = result["structure_meaning"]
        
        assert all(k in metrics for k in [
            "structure_coherence",
            "meaning_coherence",
            "evolution_stage"
        ])
        
        # Validate metric ranges
        assert 0 <= metrics["structure_coherence"] <= 1
        assert 0 <= metrics["meaning_coherence"] <= 1
        
    @pytest.mark.asyncio
    async def test_temporal_flow_visualization(self,
                                             flow_visualizer,
                                             mock_patterns):
        """Test visualization of temporal flow patterns."""
        # Create sequence of patterns
        patterns = [
            mock_patterns.patterns["temperature_rise"],
            mock_patterns.patterns["coastal_impact"],
            mock_patterns.patterns["ecosystem_change"]
        ]
        
        # Process patterns in sequence
        results = []
        for pattern in patterns:
            data = {
                "text": pattern.content,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "source": "climate_data"
                }
            }
            result = await flow_visualizer.visualize_flow(data)
            results.append(result)
        
        # Verify temporal progression
        for i in range(1, len(results)):
            current = results[i]
            previous = results[i-1]
            
            # Verify flow evolution
            assert current["metrics"]["flow_velocity"] >= 0
            
            # Verify coherence progression
            current_coherence = current["metrics"]["coherence_score"]
            previous_coherence = previous["metrics"]["coherence_score"]
            assert abs(current_coherence - previous_coherence) <= 0.3  # Max allowed change
            
    @pytest.mark.asyncio
    async def test_pattern_density_calculation(self,
                                             flow_visualizer,
                                             mock_patterns):
        """Test pattern density visualization metrics."""
        # Get states for all patterns
        states = [
            mock_patterns.get_mock_flow_state(pattern_id)
            for pattern_id in mock_patterns.patterns.keys()
        ]
        
        # Calculate density
        density = flow_visualizer.calculate_pattern_density(states)
        
        assert isinstance(density, float)
        assert 0 <= density <= 1
        
        # Test with single pattern
        single_density = flow_visualizer.calculate_pattern_density([states[0]])
        # Single pattern should have maximum density (1.0)
        assert single_density == 1.0
        # Multiple patterns should have lower density due to spread
        assert density < 1.0
