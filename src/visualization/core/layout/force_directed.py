"""Force-directed layout implementation."""

from typing import Dict, Any, List
import networkx as nx
from pydantic import BaseModel, Field

class LayoutConfig(BaseModel):
    """Configuration for force-directed layout."""
    k: float = Field(default=1.0, description="Optimal distance between nodes")
    iterations: int = Field(default=50, description="Number of iterations")
    seed: int = Field(default=42, description="Random seed for reproducibility")

class ForceDirectedLayout:
    """Force-directed layout implementation using networkx."""
    
    def __init__(self, config: LayoutConfig = None):
        """Initialize layout engine.
        
        Args:
            config: Layout configuration
        """
        self.config = config or LayoutConfig()
        
    def compute_layout(self, graph: nx.Graph) -> Dict[str, Dict[float, float]]:
        """Compute force-directed layout for graph.
        
        Args:
            graph: NetworkX graph object
            
        Returns:
            Dictionary mapping node ids to positions
        """
        pos = nx.spring_layout(
            graph,
            k=self.config.k,
            iterations=self.config.iterations,
            seed=self.config.seed
        )
        
        # Convert positions to serializable format
        return {
            node: {"x": float(coords[0]), "y": float(coords[1])}
            for node, coords in pos.items()
        }
        
    def apply_constraints(
        self,
        positions: Dict[str, Dict[float, float]],
        constraints: List[Dict[str, Any]]
    ) -> Dict[str, Dict[float, float]]:
        """Apply position constraints to layout.
        
        Args:
            positions: Current node positions
            constraints: List of position constraints
            
        Returns:
            Updated node positions
        """
        for constraint in constraints:
            node_id = constraint["node_id"]
            if node_id in positions:
                if "x" in constraint:
                    positions[node_id]["x"] = constraint["x"]
                if "y" in constraint:
                    positions[node_id]["y"] = constraint["y"]
                    
        return positions
