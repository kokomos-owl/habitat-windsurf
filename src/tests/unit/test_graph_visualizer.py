"""Tests for graph visualization functionality."""

import pytest
from pathlib import Path
import json

from src.visualization.core.graph_visualizer import GraphVisualizer, VisualizationConfig

pytestmark = pytest.mark.asyncio

async def test_create_evolution_view(
    visualizer: GraphVisualizer,
    sample_graph_data: dict
):
    """Test creation of evolution visualizations."""
    result = await visualizer.create_evolution_view(
        sample_graph_data["temporal_stages"],
        sample_graph_data["concept_evolution"],
        sample_graph_data["relationship_changes"],
        sample_graph_data["coherence_metrics"]
    )
    
    # Verify output files
    assert "timeline" in result
    assert "network" in result
    assert "coherence" in result
    
    # Check timeline data
    timeline_path = Path(result["timeline"])
    assert timeline_path.exists()
    timeline_data = json.loads(timeline_path.read_text())
    assert "stages" in timeline_data
    assert "evolution" in timeline_data
    assert len(timeline_data["stages"]) == 3
    
    # Check network data
    network_path = Path(result["network"])
    assert network_path.exists()
    network_data = json.loads(network_path.read_text())
    assert "nodes" in network_data
    assert "links" in network_data
    
    # Check coherence data
    coherence_path = Path(result["coherence"])
    assert coherence_path.exists()
    coherence_data = json.loads(coherence_path.read_text())
    assert "metrics" in coherence_data
    assert len(coherence_data["metrics"]) == 3

async def test_create_timeline_data(
    visualizer: GraphVisualizer,
    sample_graph_data: dict
):
    """Test timeline data creation."""
    timeline_data = await visualizer._create_timeline_data(
        sample_graph_data["temporal_stages"],
        sample_graph_data["concept_evolution"]
    )
    
    assert "stages" in timeline_data
    assert "evolution" in timeline_data
    assert timeline_data["stages"] == sample_graph_data["temporal_stages"]
    assert timeline_data["evolution"] == sample_graph_data["concept_evolution"]

async def test_create_network_data(
    visualizer: GraphVisualizer,
    sample_graph_data: dict
):
    """Test network data creation."""
    network_data = await visualizer._create_network_data(
        sample_graph_data["relationship_changes"]
    )
    
    assert "nodes" in network_data
    assert "links" in network_data
    assert len(network_data["links"]) == len(
        sample_graph_data["relationship_changes"]
    )

def test_visualization_config():
    """Test visualization configuration."""
    config = VisualizationConfig(
        output_dir="custom_dir",
        max_nodes=50,
        layout_algorithm="custom",
        node_size=30,
        edge_width=2.0
    )
    
    assert config.output_dir == "custom_dir"
    assert config.max_nodes == 50
    assert config.layout_algorithm == "custom"
    assert config.node_size == 30
    assert config.edge_width == 2.0

async def test_error_handling(visualizer: GraphVisualizer):
    """Test error handling in visualization creation."""
    with pytest.raises(ValueError):
        await visualizer.create_evolution_view(
            [],  # Empty stages
            {},  # Empty evolution
            [],  # Empty relationships
            {}   # Empty metrics
        )
