"""Main flow representation visualization component."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..core.flow_state import FlowState, FlowDimensions
from ..layers.base_layer import BaseLayer
from ..layers.pattern_layer import PatternLayer
from ..layers.choropleth_layer import ChoroplethLayer

class FlowRepresentation:
    """Integrated flow representation visualization."""
    
    def __init__(self):
        self.base_layer = BaseLayer()
        self.pattern_layer = PatternLayer()
        self.choropleth_layer = ChoroplethLayer()
        self.insights: List[Dict[str, Any]] = []
        self.active_dimension = "coherence"
        
    def add_state(self, state: FlowState):
        """Add a new state to the representation."""
        self.base_layer.add_state(state)
        self.pattern_layer.add_state(state)
        self.choropleth_layer.add_state(state)
        self._generate_insights(state)
        
    def set_active_dimension(self, dimension: str):
        """Set the active dimension for visualization."""
        self.active_dimension = dimension
        self.choropleth_layer.set_active_dimension(dimension)
        
    def toggle_choropleth_mode(self):
        """Toggle between abstract and geographic choropleth views."""
        self.choropleth_layer.toggle_view_mode()
        
    def _generate_insights(self, state: FlowState):
        """Generate textual insights from state changes."""
        if not self.insights:
            # First state
            self.insights.append({
                "timestamp": state.timestamp,
                "type": "initial",
                "content": f"Initial pattern {state.pattern_id} observed with "
                          f"coherence {state.dimensions.coherence:.2f}"
            })
            return
            
        # Get previous insight for comparison
        prev_insight = self.insights[-1]
        
        # Check for significant changes
        if state.dimensions.coherence > 0.8 and prev_insight["type"] != "high_coherence":
            self.insights.append({
                "timestamp": state.timestamp,
                "type": "high_coherence",
                "content": f"Pattern {state.pattern_id} achieved high coherence "
                          f"({state.dimensions.coherence:.2f})"
            })
            
        if state.dimensions.emergence_rate > 0.7 and prev_insight["type"] != "emerging":
            self.insights.append({
                "timestamp": state.timestamp,
                "type": "emerging",
                "content": f"Pattern {state.pattern_id} showing strong emergence "
                          f"(rate: {state.dimensions.emergence_rate:.2f})"
            })
    
    def create_visualization(self) -> go.Figure:
        """Create integrated visualization of all layers."""
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Timeline Evolution',
                'Pattern Network',
                'Topology View',
                'Current Metrics',
                'Key Insights',
                'Dimension Controls'
            ),
            column_widths=[0.3, 0.3, 0.4],
            specs=[
                [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
                [{'type': 'table'}, {'type': 'table'}, {'type': 'table'}]
            ]
        )
        
        # Add timeline plot
        timeline = self.base_layer.create_timeline()
        for trace in timeline.data:
            fig.add_trace(trace, row=1, col=1)
            
        # Add network plot
        network = self.pattern_layer.create_network_visualization()
        for trace in network.data:
            fig.add_trace(trace, row=1, col=2)
            
        # Add metrics summary
        metrics = self.base_layer.create_metric_summary()
        metrics_text = []
        for pattern_id, summary in metrics.items():
            metrics_text.append(f"<b>{pattern_id}</b>")
            for dim, value in summary["current_dimensions"].items():
                metrics_text.append(f"{dim}: {value:.2f}")
            metrics_text.append("")
            
        fig.add_trace(
            go.Table(
                header=dict(values=["Current Metrics"]),
                cells=dict(values=[metrics_text])
            ),
            row=2, col=1
        )
        
        # Add insights
        insights_text = [insight["content"] for insight in self.insights[-5:]]
        fig.add_trace(
            go.Table(
                header=dict(values=["Recent Insights"]),
                cells=dict(values=[insights_text])
            ),
            row=2, col=2
        )
        
        # Add choropleth
        choropleth = self.choropleth_layer.create_visualization()
        for trace in choropleth.data:
            fig.add_trace(trace, row=1, col=3)
            
        # Add dimension controls
        dimensions = [
            "stability", "coherence", "emergence_rate",
            "cross_pattern_flow", "energy_state", "adaptation_rate"
        ]
        
        buttons = []
        for dim in dimensions:
            buttons.append(dict(
                args=[{"visible": [True]}],
                label=dim,
                method="update"
            ))
            
        fig.add_trace(
            go.Table(
                header=dict(values=["Dimension Controls"]),
                cells=dict(
                    values=[[f"Active: {self.active_dimension}"] + 
                           [f"Click to view {dim}" for dim in dimensions if dim != self.active_dimension]]
                )
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1500,  # Increased width for choropleth
            title_text="Integrated Flow Representation",
            showlegend=True
        )
        
        return fig
