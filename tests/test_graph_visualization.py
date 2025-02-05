"""Test suite for graph visualization components."""

import os
import pytest
import networkx as nx
from src.core.visualization.graph import GraphVisualizer
from src.core.visualization.layout import LayoutEngine

@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    G = nx.Graph()
    G.add_nodes_from([
        (1, {"label": "Node 1", "type": "concept"}),
        (2, {"label": "Node 2", "type": "concept"}),
        (3, {"label": "Node 3", "type": "relationship"})
    ])
    G.add_edges_from([(1, 2), (2, 3)])
    return G

@pytest.fixture
def layout_engine():
    """Create a layout engine instance."""
    return LayoutEngine()

@pytest.fixture
def graph_visualizer(layout_engine):
    """Create a graph visualizer instance."""
    return GraphVisualizer(layout_engine=layout_engine)

def test_layout_engine_initialization(layout_engine):
    """Test LayoutEngine initialization."""
    assert layout_engine is not None
    assert hasattr(layout_engine, 'calculate_layout')

def test_graph_visualizer_initialization(graph_visualizer):
    """Test GraphVisualizer initialization."""
    assert graph_visualizer is not None
    assert hasattr(graph_visualizer, 'create_visualization')

def test_basic_layout_calculation(layout_engine, sample_graph):
    """Test basic layout calculation."""
    layout = layout_engine.calculate_layout(sample_graph)
    assert layout is not None
    assert len(layout) == len(sample_graph.nodes)
    for node_id in sample_graph.nodes:
        assert node_id in layout
        pos = layout[node_id]
        assert len(pos) == 2  # x, y coordinates

def test_basic_visualization(graph_visualizer, sample_graph):
    """Test basic graph visualization."""
    fig = graph_visualizer.create_visualization(sample_graph)
    assert fig is not None
    # Basic plotly figure checks
    assert hasattr(fig, 'data')
    assert hasattr(fig, 'layout')
    assert len(fig.data) > 0  # Should have at least nodes or edges
