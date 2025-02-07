"""
Flow visualization component for habitat-windsurf.

This module provides visualization capabilities for flow patterns,
structure-meaning relationships, and temporal evolution.
"""

from typing import Dict, Any, List
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import networkx as nx

from ...core.flow.habitat_flow import HabitatFlow, FlowState

class FlowVisualizer:
    """Visualizes flow patterns and their evolution."""
    
    def __init__(self):
        """Initialize flow visualizer."""
        self.flow_manager = HabitatFlow()
        self.previous_states: Dict[str, FlowState] = {}
        
    async def visualize_flow(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create flow visualization from pattern data."""
        # Process flow
        flow_result = await self.flow_manager.process_flow(data)
        flow_id = flow_result["flow_id"]
        current_state = flow_result["state"]
        
        # Create FlowState from current state
        flow_state = FlowState(
            strength=float(current_state["strength"]),
            coherence=float(current_state["coherence"]),
            emergence_potential=float(current_state["emergence_potential"]),
            temporal_context=current_state.get("temporal_context"),
            last_updated=datetime.now()
        )
        
        # Store state for temporal analysis
        self.previous_states[flow_id] = flow_state
        
        # Generate visualization components
        nodes = self._create_nodes(flow_result)
        edges = self._create_edges(flow_result)
        metrics = self._calculate_metrics(flow_result)
        
        # Calculate structure-meaning metrics
        structure_meaning = self._calculate_structure_meaning_metrics(flow_result)
        
        # Create graph visualization
        graph_fig = self._create_graph_visualization(nodes, edges)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metrics": metrics,
            "structure_meaning": structure_meaning,
            "visualization": graph_fig.to_dict()
        }
        
    def _create_nodes(self, flow_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create visualization nodes from flow result."""
        state = flow_result["state"]
        pattern_state = flow_result.get("pattern_state")
        
        nodes = [{
            "id": flow_result["flow_id"],
            "label": pattern_state.pattern if pattern_state else "Unknown",
            "strength": state["strength"],
            "coherence": state["coherence"]
        }]
        
        # Add related pattern nodes if available
        if pattern_state and hasattr(pattern_state, "related_patterns"):
            for related in pattern_state.related_patterns:
                nodes.append({
                    "id": f"{flow_result['flow_id']}_{related}",
                    "label": related,
                    "strength": 0.5,  # Default strength for related patterns
                    "coherence": 0.5  # Default coherence for related patterns
                })
                
        return nodes
        
    def _create_edges(self, flow_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create visualization edges from flow result."""
        edges = []
        pattern_state = flow_result.get("pattern_state")
        
        if pattern_state and hasattr(pattern_state, "related_patterns"):
            for related in pattern_state.related_patterns:
                edges.append({
                    "source": flow_result["flow_id"],
                    "target": f"{flow_result['flow_id']}_{related}",
                    "weight": 1.0  # Default weight
                })
                
        return edges
        
    def _calculate_metrics(self, flow_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate flow visualization metrics."""
        state = flow_result["state"]
        
        # Get all active states for density calculation
        active_states = list(self.previous_states.values())
        
        return {
            "flow_velocity": self._calculate_flow_velocity(state),
            "pattern_density": self.calculate_pattern_density(active_states),
            "coherence_score": state["coherence"]
        }
        
    def _calculate_structure_meaning_metrics(self, 
                                          flow_result: Dict[str, Any]
                                          ) -> Dict[str, Any]:
        """Calculate structure-meaning relationship metrics."""
        state = flow_result["state"]
        
        # Calculate structure and meaning coherence
        structure_coherence = state["coherence"] * 0.8  # Example weight
        meaning_coherence = state["coherence"] * 0.7    # Example weight
        
        # Determine evolution stage
        evolution_stage = self._determine_evolution_stage(
            structure_coherence,
            meaning_coherence
        )
        
        return {
            "structure_coherence": structure_coherence,
            "meaning_coherence": meaning_coherence,
            "evolution_stage": evolution_stage
        }
        
    def _calculate_flow_velocity(self, current_state: Dict[str, Any]) -> float:
        """Calculate flow velocity from state changes."""
        if not self.previous_states:
            return 0.0
            
        # Calculate average change in metrics
        changes = []
        curr_state = FlowState(
            strength=float(current_state["strength"]),
            coherence=float(current_state["coherence"]),
            emergence_potential=float(current_state["emergence_potential"])
        )
        
        for prev_state in self.previous_states.values():
            coherence_change = abs(curr_state.coherence - prev_state.coherence)
            strength_change = abs(curr_state.strength - prev_state.strength)
            changes.extend([coherence_change, strength_change])
            
        return np.mean(changes) if changes else 0.0
        
    def calculate_pattern_density(self, states: List[FlowState]) -> float:
        """Calculate pattern density in flow space."""
        if not states:
            return 0.0
            
        # Calculate average distance between states
        coherence_values = [state.coherence for state in states]
        strength_values = [state.strength for state in states]
        
        # Use standard deviation as a measure of spread
        coherence_std = np.std(coherence_values) if len(states) > 1 else 0
        strength_std = np.std(strength_values) if len(states) > 1 else 0
        
        # For single state, return maximum density
        if len(states) == 1:
            return 1.0
            
        # Normalize density (inverse of spread)
        density = 1.0 / (1.0 + coherence_std + strength_std)
        return min(1.0, density)
        
    def _determine_evolution_stage(self,
                                 structure_coherence: float,
                                 meaning_coherence: float) -> str:
        """Determine the evolution stage based on coherence metrics."""
        avg_coherence = (structure_coherence + meaning_coherence) / 2
        
        if avg_coherence < 0.3:
            return "emerging"
        elif avg_coherence < 0.6:
            return "evolving"
        elif avg_coherence < 0.8:
            return "stabilizing"
        else:
            return "stable"
            
    def _create_graph_visualization(self,
                                   nodes: List[Dict[str, Any]],
                                   edges: List[Dict[str, Any]]) -> go.Figure:
        """Create Plotly graph visualization."""
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes with attributes
        for node in nodes:
            G.add_node(
                node["id"],
                label=node["label"],
                strength=node["strength"],
                coherence=node["coherence"]
            )
        
        # Add edges with weights
        for edge in edges:
            G.add_edge(
                edge["source"],
                edge["target"],
                weight=edge["weight"]
            )
        
        # Calculate layout
        pos = nx.spring_layout(G)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(
                f"ID: {node}<br>"
                f"Label: {G.nodes[node]['label']}<br>"
                f"Strength: {G.nodes[node]['strength']:.2f}<br>"
                f"Coherence: {G.nodes[node]['coherence']:.2f}"
            )
            node_color.append(G.nodes[node]['coherence'])
            
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                color=node_color,
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Coherence',
                    xanchor='left',
                    titleside='right'
                )
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Flow Pattern Visualization',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig
