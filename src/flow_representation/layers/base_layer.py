"""Base layer for flow representation visualization."""

from typing import Dict, Any, List
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

from ..core.flow_state import FlowState

class BaseLayer:
    """Base visualization layer for temporal progression and metrics."""
    
    def __init__(self):
        self.time_points: List[datetime] = []
        self.states: Dict[str, List[FlowState]] = {}
        
    def add_state(self, state: FlowState):
        """Add a flow state to the layer."""
        if state.pattern_id not in self.states:
            self.states[state.pattern_id] = []
        self.states[state.pattern_id].append(state)
        if state.timestamp not in self.time_points:
            self.time_points.append(state.timestamp)
            self.time_points.sort()
    
    def create_timeline(self) -> go.Figure:
        """Create timeline visualization of flow evolution."""
        fig = go.Figure()
        
        for pattern_id, states in self.states.items():
            # Create traces for each dimension
            dimensions = states[0].dimensions.to_vector()
            for i, dim_name in enumerate([
                "stability", "coherence", "emergence_rate",
                "cross_pattern_flow", "energy_state", "adaptation_rate"
            ]):
                values = [s.dimensions.to_vector()[i] for s in states]
                times = [s.timestamp for s in states]
                
                fig.add_trace(go.Scatter(
                    x=times,
                    y=values,
                    name=f"{pattern_id} - {dim_name}",
                    mode='lines+markers',
                    line=dict(width=2),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title="Flow Evolution Timeline",
            xaxis_title="Time",
            yaxis_title="Dimension Value",
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def create_metric_summary(self) -> Dict[str, Any]:
        """Create summary of current metric states."""
        summaries = {}
        
        for pattern_id, states in self.states.items():
            if not states:
                continue
                
            current_state = states[-1]
            previous_state = states[-2] if len(states) > 1 else None
            
            evolution_metrics = current_state.calculate_evolution_metrics(previous_state)
            
            summaries[pattern_id] = {
                "current_dimensions": current_state.dimensions.__dict__,
                "evolution_metrics": evolution_metrics,
                "timestamp": current_state.timestamp.isoformat()
            }
            
        return summaries
