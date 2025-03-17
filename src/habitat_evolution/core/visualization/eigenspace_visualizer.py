"""Eigenspace Visualizer for Vector + Tonic-Harmonic patterns.

This module provides visualization tools for exploring patterns in eigenspace,
visualizing dimensional resonance, and navigating through transition zones.
It supports the core principles of pattern evolution and co-evolution by enabling
the observation of semantic change across the system.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging


class EigenspaceVisualizer:
    """Visualizer for patterns in eigenspace.
    
    This class provides methods for visualizing patterns in eigenspace, highlighting
    communities, boundary patterns, and resonance relationships. It supports the
    observation of pattern evolution through dimensional resonance and fuzzy boundaries.
    """
    
    def __init__(self, field_data: Optional[Dict[str, Any]] = None):
        """Initialize the eigenspace visualizer.
        
        Args:
            field_data: Field data with patterns, communities, and eigenspace coordinates
        """
        self.field_data = field_data
        self.logger = logging.getLogger(__name__)
        
        # Default color map for communities
        self.community_colors = plt.cm.tab10
        
        # Default marker styles
        self.marker_styles = {
            "normal": "o",      # Regular pattern
            "boundary": "s",    # Boundary pattern
            "resonant": "^",    # Resonant pattern
            "selected": "*"     # Selected pattern
        }
        
        # Default figure size
        self.figsize = (12, 10)
    
    def load_data(self, data_path: str):
        """Load field data from a JSON file.
        
        Args:
            data_path: Path to the JSON file with field data
        """
        try:
            with open(data_path, 'r') as f:
                self.field_data = json.load(f)
            self.logger.info(f"Loaded field data from {data_path}")
        except Exception as e:
            self.logger.error(f"Failed to load field data: {str(e)}")
    
    def visualize_eigenspace_2d(self, dim1: int = 0, dim2: int = 1, 
                               highlight_boundaries: bool = True,
                               highlight_communities: bool = True,
                               selected_patterns: List[int] = None,
                               title: str = "Patterns in 2D Eigenspace",
                               save_path: Optional[str] = None):
        """Visualize patterns in 2D eigenspace.
        
        This method creates a 2D visualization of patterns in eigenspace, highlighting
        community structure and boundary patterns. It supports the observation of
        pattern relationships and transitions between communities.
        
        Args:
            dim1: First dimension to use (default: 0)
            dim2: Second dimension to use (default: 1)
            highlight_boundaries: Whether to highlight boundary patterns (default: True)
            highlight_communities: Whether to highlight communities (default: True)
            selected_patterns: List of pattern indices to highlight (default: None)
            title: Plot title (default: "Patterns in 2D Eigenspace")
            save_path: Path to save the visualization (default: None)
        """
        if not self.field_data:
            self.logger.error("No field data available")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get pattern data
        patterns = self.field_data.get("patterns", [])
        
        # Get boundary patterns
        boundaries = self.field_data.get("boundaries", [])
        boundary_indices = [b.get("pattern_idx") for b in boundaries]
        
        # Get communities
        communities = {}
        if "communities" in self.field_data:
            communities = self.field_data["communities"]
        
        # Prepare data for plotting
        x_coords = []
        y_coords = []
        colors = []
        markers = []
        pattern_ids = []
        
        for pattern in patterns:
            # Get pattern index
            pattern_idx = pattern.get("index")
            pattern_ids.append(pattern.get("id", f"pattern_{pattern_idx}"))
            
            # Get coordinates
            coords = pattern.get("coordinates", [0, 0, 0])
            if len(coords) > dim1 and len(coords) > dim2:
                x_coords.append(coords[dim1])
                y_coords.append(coords[dim2])
            else:
                x_coords.append(0)
                y_coords.append(0)
            
            # Determine marker style
            if selected_patterns and pattern_idx in selected_patterns:
                markers.append(self.marker_styles["selected"])
            elif highlight_boundaries and pattern_idx in boundary_indices:
                markers.append(self.marker_styles["boundary"])
            else:
                markers.append(self.marker_styles["normal"])
            
            # Determine color based on community
            if highlight_communities:
                community = pattern.get("community")
                if isinstance(community, list):
                    community = community[0] if community else 0
                colors.append(int(community) % 10)  # Limit to 10 colors
            else:
                colors.append(0)
        
        # Create scatter plot
        for i, marker in enumerate(set(markers)):
            mask = [m == marker for m in markers]
            scatter = ax.scatter(
                [x for x, m in zip(x_coords, mask) if m],
                [y for y, m in zip(y_coords, mask) if m],
                c=[c for c, m in zip(colors, mask) if m],
                marker=marker,
                cmap=self.community_colors,
                alpha=0.7,
                s=100
            )
        
        # Add labels for selected patterns
        if selected_patterns:
            for i, pattern_idx in enumerate(selected_patterns):
                if pattern_idx < len(patterns):
                    pattern = patterns[pattern_idx]
                    coords = pattern.get("coordinates", [0, 0, 0])
                    if len(coords) > dim1 and len(coords) > dim2:
                        ax.annotate(
                            pattern.get("id", f"pattern_{pattern_idx}"),
                            (coords[dim1], coords[dim2]),
                            xytext=(5, 5),
                            textcoords="offset points"
                        )
        
        # Add legend for communities
        if highlight_communities and communities:
            legend_handles = []
            for i, community_id in enumerate(communities):
                color = self.community_colors(i % 10)
                patch = mpatches.Patch(color=color, label=f"Community {community_id}")
                legend_handles.append(patch)
            ax.legend(handles=legend_handles, loc="upper right")
        
        # Add legend for marker styles
        marker_handles = []
        if highlight_boundaries:
            marker_handles.append(plt.Line2D([0], [0], marker=self.marker_styles["normal"], 
                                           color='w', markerfacecolor='gray', 
                                           markersize=10, label='Normal Pattern'))
            marker_handles.append(plt.Line2D([0], [0], marker=self.marker_styles["boundary"], 
                                           color='w', markerfacecolor='gray', 
                                           markersize=10, label='Boundary Pattern'))
        if selected_patterns:
            marker_handles.append(plt.Line2D([0], [0], marker=self.marker_styles["selected"], 
                                           color='w', markerfacecolor='gray', 
                                           markersize=10, label='Selected Pattern'))
        
        if marker_handles:
            ax.legend(handles=marker_handles, loc="lower right")
        
        # Set axis labels and title
        ax.set_xlabel(f"Dimension {dim1}")
        ax.set_ylabel(f"Dimension {dim2}")
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved visualization to {save_path}")
        
        plt.tight_layout()
        return fig, ax
