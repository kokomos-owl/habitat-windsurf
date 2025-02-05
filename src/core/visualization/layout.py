"""Layout engine for graph visualization."""

from typing import Dict, Any, Optional, Tuple
import networkx as nx

class LayoutEngine:
    """Engine for calculating graph layouts using networkx algorithms."""

    def __init__(self, default_layout='spring'):
        """Initialize layout engine.
        
        Args:
            default_layout: Default layout algorithm to use
        """
        self.default_layout = default_layout
        self._layout_funcs = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'random': nx.random_layout,
            'shell': nx.shell_layout,
            'spectral': nx.spectral_layout
        }

    def calculate_layout(self, graph: nx.Graph, layout_type: Optional[str] = None) -> Dict[Any, Tuple[float, float]]:
        """Calculate positions for graph nodes.
        
        Args:
            graph: NetworkX graph to layout
            layout_type: Type of layout to use, must be one of:
                        'spring', 'circular', 'random', 'shell', 'spectral'
                        
        Returns:
            Dict mapping node ids to (x,y) positions
        """
        if not isinstance(graph, nx.Graph):
            raise ValueError("Input must be a NetworkX graph")

        layout_type = layout_type or self.default_layout
        if layout_type not in self._layout_funcs:
            raise ValueError(f"Unknown layout type: {layout_type}")

        layout_func = self._layout_funcs[layout_type]
        
        # Calculate layout
        try:
            pos = layout_func(graph)
        except Exception as e:
            # Fallback to spring layout if chosen layout fails
            pos = nx.spring_layout(graph)

        return pos
