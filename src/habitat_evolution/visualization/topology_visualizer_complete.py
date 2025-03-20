"""Topology Visualizer for semantic landscape features.

This module provides visualization tools for exploring topology states, including
frequency domains, boundaries, and resonance points in the semantic landscape.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime

from ..pattern_aware_rag.topology.models import (
    FrequencyDomain,
    Boundary,
    ResonancePoint,
    TopologyState,
    FieldMetrics
)


class TopologyVisualizer:
    """Visualizer for topology states in the semantic landscape.
    
    This class provides methods for visualizing frequency domains, boundaries,
    resonance points, and complete topology states.
    """
    
    def __init__(self):
        """Initialize the topology visualizer."""
        self.logger = logging.getLogger(__name__)
        
        # Store visualization data for testing and analysis
        self.visualization_data = {
            "domains": {},
            "boundaries": {},
            "resonance_points": {}
        }
        
        # Default figure size
        self.figsize = (12, 10)
        
        # Default color maps
        self.domain_cmap = plt.cm.viridis
        self.boundary_cmap = plt.cm.coolwarm
        self.resonance_cmap = plt.cm.plasma
        
        # Default alpha values for visualization
        self.domain_alpha = 0.5
        self.boundary_alpha = 0.7
        self.resonance_alpha = 0.8
    
    def visualize_frequency_domains(self, topology_state: TopologyState, ax=None, 
                                   highlight_patterns: bool = True,
                                   show_labels: bool = True,
                                   title: str = "Frequency Domains in Semantic Landscape",
                                   save_path: Optional[str] = None) -> Figure:
        """Visualize frequency domains in the topology state.
        
        Args:
            topology_state: The topology state to visualize
            ax: Optional matplotlib axis to plot on
            highlight_patterns: Whether to highlight patterns within domains
            show_labels: Whether to show domain labels
            title: Plot title
            save_path: Path to save the visualization
            
        Returns:
            Figure: Matplotlib figure with the visualization
        """
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        # Clear visualization data for this component
        self.visualization_data["domains"] = {}
        
        # Get frequency domains
        domains = topology_state.frequency_domains
        
        # Create a colormap based on frequency
        freq_values = [domain.dominant_frequency for domain in domains.values()]
        norm = Normalize(vmin=min(freq_values) if freq_values else 0, 
                         vmax=max(freq_values) if freq_values else 1)
        
        # Plot each domain as a circle
        for domain_id, domain in domains.items():
            # Get domain properties
            center = domain.center_coordinates
            radius = domain.radius
            frequency = domain.dominant_frequency
            coherence = domain.phase_coherence
            
            # Skip if center is not 2D
            if len(center) < 2:
                self.logger.warning(f"Domain {domain_id} has less than 2D coordinates. Skipping.")
                continue
            
            # Create circle patch
            circle = plt.Circle(
                (center[0], center[1]),
                radius,
                color=self.domain_cmap(norm(frequency)),
                alpha=self.domain_alpha * coherence,  # More coherent domains are more opaque
                fill=True,
                edgecolor='black',
                linewidth=1.5
            )
            ax.add_patch(circle)
            
            # Add domain label if requested
            if show_labels:
                ax.annotate(
                    f"D{domain_id.split('-')[-1]}: {frequency:.2f}Hz",
                    (center[0], center[1]),
                    ha='center',
                    va='center',
                    fontsize=10,
                    fontweight='bold',
                    color='black'
                )
            
            # Highlight patterns if requested
            if highlight_patterns and domain.pattern_ids:
                # Plot a small dot for each pattern
                pattern_count = len(domain.pattern_ids)
                for i, pattern_id in enumerate(domain.pattern_ids):
                    # Calculate position within domain (spread patterns out)
                    angle = 2 * np.pi * i / pattern_count
                    pattern_x = center[0] + 0.7 * radius * np.cos(angle)
                    pattern_y = center[1] + 0.7 * radius * np.sin(angle)
                    
                    # Plot pattern
                    ax.plot(
                        pattern_x,
                        pattern_y,
                        'o',
                        color='white',
                        markersize=4,
                        alpha=0.8
                    )
            
            # Store visualization data for testing
            self.visualization_data["domains"][domain_id] = {
                "center": center,
                "radius": radius,
                "frequency": frequency,
                "coherence": coherence,
                "pattern_count": len(domain.pattern_ids) if domain.pattern_ids else 0
            }
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=self.domain_cmap), cax=cax)
        cbar.set_label("Dominant Frequency")
        
        # Set axis properties
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved frequency domain visualization to {save_path}")
        
        return fig
    
    def visualize_boundaries(self, topology_state: TopologyState, ax=None,
                            show_labels: bool = True,
                            title: str = "Boundaries Between Frequency Domains",
                            save_path: Optional[str] = None) -> Figure:
        """Visualize boundaries between frequency domains.
        
        Args:
            topology_state: The topology state to visualize
            ax: Optional matplotlib axis to plot on
            show_labels: Whether to show boundary labels
            title: Plot title
            save_path: Path to save the visualization
            
        Returns:
            Figure: Matplotlib figure with the visualization
        """
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        # Clear visualization data for this component
        self.visualization_data["boundaries"] = {}
        
        # Get boundaries and domains
        boundaries = topology_state.boundaries
        domains = topology_state.frequency_domains
        
        # Create a colormap based on permeability
        perm_values = [boundary.permeability for boundary in boundaries.values()]
        norm = Normalize(vmin=min(perm_values) if perm_values else 0, 
                         vmax=max(perm_values) if perm_values else 1)
        
        # Plot each boundary as a line or gradient
        for boundary_id, boundary in boundaries.items():
            # Get boundary properties
            domain_ids = boundary.domain_ids
            sharpness = boundary.sharpness
            permeability = boundary.permeability
            stability = boundary.stability
            coordinates = boundary.coordinates
            
            # Skip if we don't have coordinates
            if not coordinates or len(coordinates) < 2:
                self.logger.warning(f"Boundary {boundary_id} has insufficient coordinates. Skipping.")
                continue
            
            # Get domain centers if available
            domain_centers = []
            for domain_id in domain_ids:
                if domain_id in domains:
                    center = domains[domain_id].center_coordinates
                    if len(center) >= 2:
                        domain_centers.append((center[0], center[1]))
            
            # Use domain centers if no coordinates are provided
            if not coordinates and len(domain_centers) == 2:
                coordinates = domain_centers
            
            # Extract x and y coordinates
            x_coords = [coord[0] for coord in coordinates]
            y_coords = [coord[1] for coord in coordinates]
            
            # Line width based on sharpness (sharper boundaries are thicker)
            linewidth = 1.0 + 3.0 * sharpness
            
            # Line style based on stability (more stable boundaries are solid)
            linestyle = '-' if stability > 0.5 else '--'
            
            # Plot boundary
            ax.plot(
                x_coords,
                y_coords,
                linestyle=linestyle,
                color=self.boundary_cmap(norm(permeability)),
                linewidth=linewidth,
                alpha=self.boundary_alpha,
                zorder=10  # Ensure boundaries are drawn on top of domains
            )
            
            # Add boundary label if requested
            if show_labels:
                # Calculate midpoint of boundary
                mid_x = sum(x_coords) / len(x_coords)
                mid_y = sum(y_coords) / len(y_coords)
                
                ax.annotate(
                    f"B{boundary_id.split('-')[-1]}: {permeability:.2f}p",
                    (mid_x, mid_y),
                    ha='center',
                    va='center',
                    fontsize=8,
                    fontweight='bold',
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
                )
            
            # Store visualization data for testing
            self.visualization_data["boundaries"][boundary_id] = {
                "domain_ids": domain_ids,
                "sharpness": sharpness,
                "permeability": permeability,
                "stability": stability,
                "coordinates": coordinates
            }
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=self.boundary_cmap), cax=cax)
        cbar.set_label("Permeability")
        
        # Set axis properties
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved boundary visualization to {save_path}")
        
        return fig
    
    def visualize_resonance_points(self, topology_state: TopologyState, ax=None,
                                show_labels: bool = True,
                                title: str = "Resonance Points in Semantic Landscape",
                                save_path: Optional[str] = None) -> Figure:
        """Visualize resonance points in the topology state.
        
        Args:
            topology_state: The topology state to visualize
            ax: Optional matplotlib axis to plot on
            show_labels: Whether to show resonance point labels
            title: Plot title
            save_path: Path to save the visualization
            
        Returns:
            Figure: Matplotlib figure with the visualization
        """
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        # Clear visualization data for this component
        self.visualization_data["resonance_points"] = {}
        
        # Get resonance points
        resonance_points = topology_state.resonance_points
        
        # Create a colormap based on strength
        strength_values = [point.strength for point in resonance_points.values()]
        norm = Normalize(vmin=min(strength_values) if strength_values else 0, 
                         vmax=max(strength_values) if strength_values else 1)
        
        # Plot each resonance point
        for point_id, point in resonance_points.items():
            # Get point properties
            coordinates = point.coordinates
            strength = point.strength
            stability = point.stability
            radius = point.attractor_radius
            contributing_patterns = point.contributing_pattern_ids
            
            # Skip if coordinates are not 2D
            if len(coordinates) < 2:
                self.logger.warning(f"Resonance point {point_id} has less than 2D coordinates. Skipping.")
                continue
            
            # Marker size based on strength and radius
            markersize = 100 * strength * radius
            
            # Plot resonance point
            ax.scatter(
                coordinates[0],
                coordinates[1],
                s=markersize,
                color=self.resonance_cmap(norm(strength)),
                alpha=self.resonance_alpha * stability,  # More stable points are more opaque
                edgecolor='black',
                linewidth=1.5,
                zorder=20  # Ensure resonance points are drawn on top
            )
            
            # Add label if requested
            if show_labels:
                ax.annotate(
                    f"R{point_id.split('-')[-1]}: {strength:.2f}s",
                    (coordinates[0], coordinates[1]),
                    ha='center',
                    va='center',
                    fontsize=8,
                    fontweight='bold',
                    color='white',
                    zorder=30  # Ensure labels are on top
                )
            
            # Store visualization data for testing
            self.visualization_data["resonance_points"][point_id] = {
                "coordinates": coordinates,
                "strength": strength,
                "stability": stability,
                "radius": radius,
                "contributing_patterns": contributing_patterns
            }
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=self.resonance_cmap), cax=cax)
        cbar.set_label("Resonance Strength")
        
        # Set axis properties
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved resonance point visualization to {save_path}")
        
        return fig
    
    def visualize_topology(self, topology_state: TopologyState,
                          show_domains: bool = True,
                          show_boundaries: bool = True,
                          show_resonance_points: bool = True,
                          show_labels: bool = True,
                          title: str = "Complete Topology State",
                          save_path: Optional[str] = None) -> Figure:
        """Visualize the complete topology state with all components.
        
        Args:
            topology_state: The topology state to visualize
            show_domains: Whether to show frequency domains
            show_boundaries: Whether to show boundaries
            show_resonance_points: Whether to show resonance points
            show_labels: Whether to show labels
            title: Plot title
            save_path: Path to save the visualization
            
        Returns:
            Figure: Matplotlib figure with the visualization
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Visualize components in order (domains at bottom, resonance points on top)
        if show_domains:
            self.visualize_frequency_domains(topology_state, ax=ax, 
                                           highlight_patterns=True, 
                                           show_labels=show_labels,
                                           title="")
        
        if show_boundaries:
            self.visualize_boundaries(topology_state, ax=ax, 
                                    show_labels=show_labels,
                                    title="")
        
        if show_resonance_points:
            self.visualize_resonance_points(topology_state, ax=ax, 
                                         show_labels=show_labels,
                                         title="")
        
        # Set title for the combined visualization
        ax.set_title(title)
        
        # Add timestamp
        timestamp_str = topology_state.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        ax.annotate(
            f"Timestamp: {timestamp_str}",
            (0.98, 0.02),
            ha='right',
            va='bottom',
            fontsize=8,
            color='gray',
            xycoords='axes fraction'
        )
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=self.domain_cmap(0.3), alpha=self.domain_alpha, 
                         label='Frequency Domain'),
            plt.Line2D([0], [0], color=self.boundary_cmap(0.5), lw=2, 
                      label='Boundary'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=self.resonance_cmap(0.7), markersize=10, 
                      label='Resonance Point')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved complete topology visualization to {save_path}")
        
        return fig
    
    def visualize_topology_evolution(self, topology_states: List[TopologyState],
                                   show_labels: bool = True,
                                   title: str = "Topology Evolution Over Time",
                                   save_path: Optional[str] = None) -> Figure:
        """Visualize the evolution of topology states over time.
        
        Args:
            topology_states: List of topology states to visualize
            show_labels: Whether to show labels
            title: Plot title
            save_path: Path to save the visualization
            
        Returns:
            Figure: Matplotlib figure with the visualization
        """
        # Check if we have states to visualize
        if not topology_states:
            self.logger.warning("No topology states provided for evolution visualization.")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No topology states available", 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Sort states by timestamp
        sorted_states = sorted(topology_states, key=lambda state: state.timestamp)
        
        # Create figure with subplots for each state
        n_states = len(sorted_states)
        n_cols = min(3, n_states)  # Maximum 3 columns
        n_rows = (n_states + n_cols - 1) // n_cols  # Ceiling division
        
        fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
        
        # Visualize each state
        for i, state in enumerate(sorted_states):
            # Create subplot
            row = i // n_cols
            col = i % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            # Visualize topology state
            self.visualize_topology(state, ax=ax, show_labels=show_labels, 
                                  title=f"State {i+1}: {state.timestamp.strftime('%Y-%m-%d %H:%M')}")
        
        # Add overall title
        fig.suptitle(title, fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved topology evolution visualization to {save_path}")
        
        return fig
    
    def create_interactive_visualization(self, topology_state: TopologyState,
                                       title: str = "Interactive Topology Visualization",
                                       save_path: Optional[str] = None) -> Figure:
        """Create an interactive visualization of the topology state.
        
        Note: This is a placeholder for interactive visualization capabilities.
        In a real implementation, this would use interactive plotting libraries
        like Plotly or Bokeh.
        
        Args:
            topology_state: The topology state to visualize
            title: Plot title
            save_path: Path to save the visualization
            
        Returns:
            Figure: Matplotlib figure with the visualization
        """
        # For now, just create a static visualization with a note
        fig = self.visualize_topology(topology_state, title=title)
        
        # Add note about interactivity
        fig.text(
            0.5, 0.01,
            "Note: This is a placeholder for interactive visualization. " +
            "In a full implementation, this would provide interactive elements.",
            ha='center', va='bottom', fontsize=10, style='italic'
        )
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved interactive visualization to {save_path}")
        
        return fig
