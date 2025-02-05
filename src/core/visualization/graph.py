"""Graph visualization module adapted from habitat_poc."""

import os
from typing import Dict, Any, Optional
import networkx as nx
import plotly.graph_objects as go

class GraphVisualizer:
    """Visualizer for graph components using plotly."""

    def __init__(self, layout_engine=None):
        """Initialize visualizer.
        
        Args:
            layout_engine: LayoutEngine instance for calculating node positions
        """
        self.layout_engine = layout_engine
        self.default_node_color = '#1f77b4'  # Plotly default blue
        self.default_edge_color = '#7f7f7f'  # Plotly default gray
        
    def create_visualization(self, graph: nx.Graph) -> go.Figure:
        """Create an interactive graph visualization.
        
        Args:
            graph: NetworkX graph to visualize
            
        Returns:
            Plotly figure object
        """
        if not isinstance(graph, nx.Graph):
            raise ValueError("Input must be a NetworkX graph")
            
        # Get layout from engine or use spring layout
        if self.layout_engine:
            pos = self.layout_engine.calculate_layout(graph)
        else:
            pos = nx.spring_layout(graph)
            
        # Extract node positions
        node_x = []
        node_y = []
        node_text = []
        
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            # Get node label if available
            label = graph.nodes[node].get('label', str(node))
            node_text.append(label)
            
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        # Create figure
        fig = go.Figure(
            data=[
                # Edges
                go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color=self.default_edge_color),
                    hoverinfo='none',
                    mode='lines'
                ),
                # Nodes
                go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    marker=dict(
                        size=20,
                        color=self.default_node_color,
                    )
                )
            ],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig
