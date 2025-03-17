"""Eigenspace Visualizer for Vector + Tonic-Harmonic patterns.

This module provides visualization tools for exploring patterns in eigenspace,
visualizing dimensional resonance, and navigating through transition zones.
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
    communities, boundary patterns, and resonance relationships.
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
    
    def visualize_eigenspace_3d(self, dim1: int = 0, dim2: int = 1, dim3: int = 2,
                               highlight_boundaries: bool = True,
                               highlight_communities: bool = True,
                               selected_patterns: List[int] = None,
                               title: str = "Patterns in 3D Eigenspace",
                               save_path: Optional[str] = None):
        """Visualize patterns in 3D eigenspace.
        
        Args:
            dim1: First dimension to use (default: 0)
            dim2: Second dimension to use (default: 1)
            dim3: Third dimension to use (default: 2)
            highlight_boundaries: Whether to highlight boundary patterns (default: True)
            highlight_communities: Whether to highlight communities (default: True)
            selected_patterns: List of pattern indices to highlight (default: None)
            title: Plot title (default: "Patterns in 3D Eigenspace")
            save_path: Path to save the visualization (default: None)
        """
        if not self.field_data:
            self.logger.error("No field data available")
            return
        
        # Create figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
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
        z_coords = []
        colors = []
        markers = []
        pattern_ids = []
        
        for pattern in patterns:
            # Get pattern index
            pattern_idx = pattern.get("index")
            pattern_ids.append(pattern.get("id", f"pattern_{pattern_idx}"))
            
            # Get coordinates
            coords = pattern.get("coordinates", [0, 0, 0])
            if len(coords) > dim1 and len(coords) > dim2 and len(coords) > dim3:
                x_coords.append(coords[dim1])
                y_coords.append(coords[dim2])
                z_coords.append(coords[dim3])
            else:
                x_coords.append(0)
                y_coords.append(0)
                z_coords.append(0)
            
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
        scatter = ax.scatter(
            x_coords, y_coords, z_coords,
            c=colors,
            marker='o',
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
                    if len(coords) > dim1 and len(coords) > dim2 and len(coords) > dim3:
                        ax.text(
                            coords[dim1], coords[dim2], coords[dim3],
                            pattern.get("id", f"pattern_{pattern_idx}"),
                            size=8
                        )
        
        # Add legend for communities
        if highlight_communities and communities:
            legend_handles = []
            for i, community_id in enumerate(communities):
                color = self.community_colors(i % 10)
                patch = mpatches.Patch(color=color, label=f"Community {community_id}")
                legend_handles.append(patch)
            ax.legend(handles=legend_handles, loc="upper right")
        
        # Set axis labels and title
        ax.set_xlabel(f"Dimension {dim1}")
        ax.set_ylabel(f"Dimension {dim2}")
        ax.set_zlabel(f"Dimension {dim3}")
        ax.set_title(title)
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved visualization to {save_path}")
        
        plt.tight_layout()
        return fig, ax
    
    def visualize_dimensional_resonance(self, dim: int = 0, 
                                      threshold: float = 0.3,
                                      title: str = "Dimensional Resonance",
                                      save_path: Optional[str] = None):
        """Visualize dimensional resonance for a specific dimension.
        
        Args:
            dim: Dimension to visualize (default: 0)
            threshold: Threshold for strong projections (default: 0.3)
            title: Plot title (default: "Dimensional Resonance")
            save_path: Path to save the visualization (default: None)
        """
        if not self.field_data:
            self.logger.error("No field data available")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get pattern data
        patterns = self.field_data.get("patterns", [])
        
        # Prepare data for plotting
        pattern_ids = []
        projections = []
        
        for pattern in patterns:
            # Get pattern ID
            pattern_idx = pattern.get("index")
            pattern_id = pattern.get("id", f"pattern_{pattern_idx}")
            
            # Get projection for this dimension
            projection = 0.0
            if "projections" in pattern:
                dim_key = f"dim_{dim}"
                if dim_key in pattern["projections"]:
                    projection = pattern["projections"][dim_key]
            
            pattern_ids.append(pattern_id)
            projections.append(projection)
        
        # Sort by projection value
        sorted_indices = np.argsort(projections)
        pattern_ids = [pattern_ids[i] for i in sorted_indices]
        projections = [projections[i] for i in sorted_indices]
        
        # Create bar chart
        bars = ax.barh(pattern_ids, projections, height=0.7)
        
        # Color bars based on projection value
        for i, bar in enumerate(bars):
            if projections[i] > threshold:
                bar.set_color('green')
            elif projections[i] < -threshold:
                bar.set_color('red')
            else:
                bar.set_color('gray')
        
        # Add threshold lines
        ax.axvline(x=threshold, color='green', linestyle='--', alpha=0.7)
        ax.axvline(x=-threshold, color='red', linestyle='--', alpha=0.7)
        
        # Set axis labels and title
        ax.set_xlabel(f"Projection on Dimension {dim}")
        ax.set_ylabel("Pattern ID")
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved visualization to {save_path}")
        
        plt.tight_layout()
        return fig, ax
    
    def visualize_navigation_path(self, path: List[Dict[str, Any]],
                                title: str = "Navigation Path in Eigenspace",
                                save_path: Optional[str] = None):
        """Visualize a navigation path in eigenspace.
        
        Args:
            path: List of patterns in the navigation path
            title: Plot title (default: "Navigation Path in Eigenspace")
            save_path: Path to save the visualization (default: None)
        """
        if not self.field_data or not path:
            self.logger.error("No field data or path available")
            return
        
        # Create figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get pattern indices in the path
        path_indices = [p.get("index") for p in path]
        
        # Visualize all patterns in eigenspace
        self.visualize_eigenspace_3d(
            highlight_boundaries=True,
            highlight_communities=True,
            selected_patterns=path_indices,
            title=title,
            save_path=None
        )
        
        # Add path lines
        for i in range(len(path) - 1):
            start_pattern = path[i]
            end_pattern = path[i+1]
            
            start_coords = start_pattern.get("coordinates", [0, 0, 0])
            end_coords = end_pattern.get("coordinates", [0, 0, 0])
            
            if len(start_coords) >= 3 and len(end_coords) >= 3:
                ax.plot(
                    [start_coords[0], end_coords[0]],
                    [start_coords[1], end_coords[1]],
                    [start_coords[2], end_coords[2]],
                    'r-', linewidth=2
                )
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved visualization to {save_path}")
        
        plt.tight_layout()
        return fig, ax
    
    def visualize_community_boundaries(self, community1: int, community2: int,
                                     title: str = "Community Boundaries",
                                     save_path: Optional[str] = None):
        """Visualize boundaries between two communities.
        
        Args:
            community1: ID of the first community
            community2: ID of the second community
            title: Plot title (default: "Community Boundaries")
            save_path: Path to save the visualization (default: None)
        """
        if not self.field_data:
            self.logger.error("No field data available")
            return
        
        # Get boundary patterns
        boundaries = self.field_data.get("boundaries", [])
        
        # Get patterns in the communities
        patterns = self.field_data.get("patterns", [])
        community1_patterns = []
        community2_patterns = []
        boundary_patterns = []
        
        for pattern in patterns:
            pattern_idx = pattern.get("index")
            community = pattern.get("community")
            
            # Check if pattern is in one of the communities
            if isinstance(community, list):
                if community1 in community and community2 in community:
                    boundary_patterns.append(pattern_idx)
                elif community1 in community:
                    community1_patterns.append(pattern_idx)
                elif community2 in community:
                    community2_patterns.append(pattern_idx)
            else:
                if community == community1:
                    community1_patterns.append(pattern_idx)
                elif community == community2:
                    community2_patterns.append(pattern_idx)
        
        # Add boundary patterns from boundary info
        for boundary in boundaries:
            pattern_idx = boundary.get("pattern_idx")
            if pattern_idx not in boundary_patterns:
                pattern = patterns[pattern_idx] if pattern_idx < len(patterns) else None
                if pattern:
                    community = pattern.get("community")
                    if isinstance(community, list):
                        if community1 in community and community2 in community:
                            boundary_patterns.append(pattern_idx)
                    elif community == community1 or community == community2:
                        # Check if this is a boundary between the two communities
                        fuzziness = boundary.get("fuzziness", 0.0)
                        if fuzziness > 0.5:  # High fuzziness threshold
                            boundary_patterns.append(pattern_idx)
        
        # Create figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Visualize patterns in eigenspace
        for pattern_idx in community1_patterns:
            pattern = patterns[pattern_idx]
            coords = pattern.get("coordinates", [0, 0, 0])
            if len(coords) >= 3:
                ax.scatter(
                    coords[0], coords[1], coords[2],
                    color='blue', marker='o', s=50, alpha=0.7,
                    label=f"Community {community1}" if pattern_idx == community1_patterns[0] else ""
                )
        
        for pattern_idx in community2_patterns:
            pattern = patterns[pattern_idx]
            coords = pattern.get("coordinates", [0, 0, 0])
            if len(coords) >= 3:
                ax.scatter(
                    coords[0], coords[1], coords[2],
                    color='green', marker='o', s=50, alpha=0.7,
                    label=f"Community {community2}" if pattern_idx == community2_patterns[0] else ""
                )
        
        for pattern_idx in boundary_patterns:
            pattern = patterns[pattern_idx]
            coords = pattern.get("coordinates", [0, 0, 0])
            if len(coords) >= 3:
                ax.scatter(
                    coords[0], coords[1], coords[2],
                    color='red', marker='s', s=100, alpha=0.7,
                    label="Boundary" if pattern_idx == boundary_patterns[0] else ""
                )
        
        # Set axis labels and title
        ax.set_xlabel("Dimension 0")
        ax.set_ylabel("Dimension 1")
        ax.set_zlabel("Dimension 2")
        ax.set_title(title)
        
        # Add legend
        ax.legend()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved visualization to {save_path}")
        
        plt.tight_layout()
        return fig, ax
