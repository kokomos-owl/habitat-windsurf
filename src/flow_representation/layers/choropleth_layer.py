"""Choropleth layer for abstract and geographic visualization."""

from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

from ..core.flow_state import FlowState, FlowDimensions

class ChoroplethLayer:
    """Layer for toggling between abstract and geographic choropleth views."""
    
    def __init__(self):
        self.states: Dict[str, FlowState] = {}
        self.abstract_regions = [
            "region_1", "region_2", "region_3", "region_4",  # Abstract space
            "region_5", "region_6", "region_7", "region_8"
        ]
        self.geographic_regions = [
            "Oak_Bluffs", "Tisbury", "West_Tisbury", "Edgartown",  # MV towns
            "Chilmark", "Aquinnah", "Vineyard_Haven"
        ]
        self.active_dimension = "coherence"  # Default dimension to display
        self.view_mode = "abstract"  # or "geographic"
        
    def add_state(self, state: FlowState):
        """Add a flow state to the layer."""
        self.states[state.pattern_id] = state
        
    def set_active_dimension(self, dimension: str):
        """Set the active dimension for visualization."""
        if hasattr(FlowDimensions, dimension):
            self.active_dimension = dimension
            
    def toggle_view_mode(self):
        """Toggle between abstract and geographic views."""
        self.view_mode = "geographic" if self.view_mode == "abstract" else "abstract"
        
    def _generate_abstract_coordinates(self) -> Dict[str, Dict[str, float]]:
        """Generate abstract grid coordinates for regions."""
        coords = {}
        grid_size = int(np.ceil(np.sqrt(len(self.abstract_regions))))
        
        for i, region in enumerate(self.abstract_regions):
            row = i // grid_size
            col = i % grid_size
            coords[region] = {
                "lat": row / (grid_size - 1),  # Normalize to 0-1
                "lon": col / (grid_size - 1)
            }
        return coords
        
    def _get_dimension_values(self) -> Dict[str, float]:
        """Get current values for active dimension across states."""
        values = {}
        for pattern_id, state in self.states.items():
            value = getattr(state.dimensions, self.active_dimension)
            if self.view_mode == "abstract":
                # Map patterns to abstract regions
                region = self.abstract_regions[hash(pattern_id) % len(self.abstract_regions)]
                values[region] = value
            else:
                # Map patterns to geographic regions based on relevance
                # This will be enhanced with actual geographic mapping
                region = self.geographic_regions[hash(pattern_id) % len(self.geographic_regions)]
                values[region] = value
        return values
        
    def create_visualization(self) -> go.Figure:
        """Create choropleth visualization based on current mode."""
        if self.view_mode == "abstract":
            return self._create_abstract_choropleth()
        else:
            return self._create_geographic_choropleth()
            
    def _create_abstract_choropleth(self) -> go.Figure:
        """Create abstract grid-based choropleth."""
        coords = self._generate_abstract_coordinates()
        values = self._get_dimension_values()
        
        # Create custom grid
        fig = go.Figure()
        
        # Add rectangles for each region
        for region, coord in coords.items():
            value = values.get(region, 0)
            
            fig.add_trace(go.Scatter(
                x=[coord["lon"]],
                y=[coord["lat"]],
                mode="markers+text",
                marker=dict(
                    size=50,
                    color=value,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=self.active_dimension)
                ),
                text=f"{region}<br>{value:.2f}",
                name=region,
                hoverinfo="text"
            ))
            
        fig.update_layout(
            title=f"Abstract Topology - {self.active_dimension}",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False
        )
        
        return fig
        
    def _create_geographic_choropleth(self) -> go.Figure:
        """Create Martha's Vineyard geographic choropleth."""
        values = self._get_dimension_values()
        
        # This is a placeholder for actual MV geojson
        # We'll enhance this with real geographic data
        fig = go.Figure()
        
        fig.add_trace(go.Choropleth(
            # This will be replaced with actual MV geometry
            locations=list(values.keys()),
            z=list(values.values()),
            colorscale="Viridis",
            colorbar_title=self.active_dimension
        ))
        
        fig.update_layout(
            title=f"Martha's Vineyard - {self.active_dimension}",
            geo=dict(
                scope='usa',
                projection=dict(type='mercator'),
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)'
            )
        )
        
        return fig
