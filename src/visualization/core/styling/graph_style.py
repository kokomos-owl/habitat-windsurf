"""Graph styling module."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import plotly.graph_objects as go

class NodeStyle(BaseModel):
    """Node styling configuration."""
    size: int = Field(default=20)
    color: str = Field(default="#1f77b4")
    symbol: str = Field(default="circle")
    border_width: int = Field(default=2)
    border_color: str = Field(default="#ffffff")

class EdgeStyle(BaseModel):
    """Edge styling configuration."""
    width: float = Field(default=1.0)
    color: str = Field(default="#888")
    style: str = Field(default="solid")
    arrow_size: int = Field(default=10)

class GraphStyle(BaseModel):
    """Overall graph styling configuration."""
    background_color: str = Field(default="#ffffff")
    font_size: int = Field(default=12)
    font_color: str = Field(default="#000000")
    
class StyleEngine:
    """Engine for applying styles to graph visualizations."""
    
    def __init__(
        self,
        node_styles: Optional[Dict[str, NodeStyle]] = None,
        edge_styles: Optional[Dict[str, EdgeStyle]] = None,
        graph_style: Optional[GraphStyle] = None
    ):
        """Initialize style engine.
        
        Args:
            node_styles: Dictionary mapping node types to styles
            edge_styles: Dictionary mapping edge types to styles
            graph_style: Overall graph styling
        """
        self.node_styles = node_styles or {"default": NodeStyle()}
        self.edge_styles = edge_styles or {"default": EdgeStyle()}
        self.graph_style = graph_style or GraphStyle()
        
    def apply_node_style(
        self,
        node_trace: go.Scatter,
        node_types: Dict[str, str]
    ) -> go.Scatter:
        """Apply styles to node trace.
        
        Args:
            node_trace: Plotly scatter trace for nodes
            node_types: Dictionary mapping node ids to their types
            
        Returns:
            Styled node trace
        """
        sizes = []
        colors = []
        symbols = []
        
        for node_id in node_trace.ids:
            node_type = node_types.get(node_id, "default")
            style = self.node_styles.get(node_type, self.node_styles["default"])
            
            sizes.append(style.size)
            colors.append(style.color)
            symbols.append(style.symbol)
            
        node_trace.marker.size = sizes
        node_trace.marker.color = colors
        node_trace.marker.symbol = symbols
        node_trace.marker.line.width = style.border_width
        node_trace.marker.line.color = style.border_color
        
        return node_trace
        
    def apply_edge_style(
        self,
        edge_trace: go.Scatter,
        edge_types: Dict[tuple, str]
    ) -> go.Scatter:
        """Apply styles to edge trace.
        
        Args:
            edge_trace: Plotly scatter trace for edges
            edge_types: Dictionary mapping edge tuples to their types
            
        Returns:
            Styled edge trace
        """
        widths = []
        colors = []
        
        for edge in edge_types:
            edge_type = edge_types.get(edge, "default")
            style = self.edge_styles.get(edge_type, self.edge_styles["default"])
            
            widths.append(style.width)
            colors.append(style.color)
            
        edge_trace.line.width = widths
        edge_trace.line.color = colors
        
        return edge_trace
        
    def apply_graph_style(self, fig: go.Figure) -> go.Figure:
        """Apply overall graph styling.
        
        Args:
            fig: Plotly figure object
            
        Returns:
            Styled figure
        """
        fig.update_layout(
            plot_bgcolor=self.graph_style.background_color,
            paper_bgcolor=self.graph_style.background_color,
            font=dict(
                size=self.graph_style.font_size,
                color=self.graph_style.font_color
            )
        )
        
        return fig
