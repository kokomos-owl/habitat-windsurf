"""
Dependency Injection container for Habitat Windsurf UI Course.
"""

from dependency_injector import containers, providers
from .visualization.graph import GraphVisualizer
from .visualization.layout import LayoutEngine

class Container(containers.DeclarativeContainer):
    """DI Container for core components."""
    
    config = providers.Configuration()
    
    # Core visualization components
    layout_engine = providers.Singleton(LayoutEngine)
    graph_visualizer = providers.Singleton(
        GraphVisualizer,
        layout_engine=layout_engine
    )
