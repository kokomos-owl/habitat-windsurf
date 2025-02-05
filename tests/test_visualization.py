"""
Basic tests for visualization components.
More tests will be added as components are implemented.
"""

import pytest
from src.core.visualization.graph import GraphVisualizer
from src.core.visualization.layout import LayoutEngine

def test_graph_visualizer_initialization():
    """Test that GraphVisualizer can be initialized."""
    visualizer = GraphVisualizer()
    assert hasattr(visualizer, 'initialized')

def test_layout_engine_initialization():
    """Test that LayoutEngine can be initialized."""
    engine = LayoutEngine()
    assert hasattr(engine, 'initialized')
