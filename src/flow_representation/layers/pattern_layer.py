"""Pattern relationship layer for flow representation."""

from typing import Dict, Any, List
import networkx as nx
import plotly.graph_objects as go
import numpy as np

from ..core.flow_state import FlowState

class PatternLayer:
    """Visualization layer for pattern relationships and interactions."""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.states: Dict[str, FlowState] = {}
        
    def add_state(self, state: FlowState):
        """Add or update a pattern state in the graph."""
        self.states[state.pattern_id] = state
        
        # Add or update node
        self.graph.add_node(
            state.pattern_id,
            dimensions=state.dimensions.__dict__,
            timestamp=state.timestamp
        )
        
        # Add edges for related patterns
        if state.related_patterns:
            for related_id in state.related_patterns:
                if related_id in self.states:
                    weight = self._calculate_interaction_strength(
                        state, self.states[related_id]
                    )
                    self.graph.add_edge(
                        state.pattern_id,
                        related_id,
                        weight=weight
                    )
    
    def _calculate_interaction_strength(self, state1: FlowState, state2: FlowState) -> float:
        """Calculate interaction strength between two patterns."""
        vec1 = state1.dimensions.to_vector()
        vec2 = state2.dimensions.to_vector()
        
        # Combine multiple metrics for interaction strength
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        temporal_factor = 1.0  # Could be based on timestamp difference
        
        return float(cosine_sim * temporal_factor)
    
    def create_network_visualization(self) -> go.Figure:
        """Create network visualization of pattern relationships."""
        # Use spring layout for node positioning
        pos = nx.spring_layout(self.graph)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in self.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Use coherence for color and energy_state for size
            state = self.states[node]
            node_colors.append(state.dimensions.coherence)
            node_sizes.append(state.dimensions.energy_state * 50)
            
            # Create hover text
            text = f"Pattern: {node}<br>"
            text += f"Coherence: {state.dimensions.coherence:.2f}<br>"
            text += f"Energy: {state.dimensions.energy_state:.2f}"
            node_text.append(text)
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                color=node_colors,
                size=node_sizes,
                line_width=2
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         title="Pattern Relationship Network",
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                     ))
        
        return fig
