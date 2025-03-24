"""
Propensity Visualizer Module

This module implements visualization tools for emergent propensities,
allowing for the visual representation of propensity gradients, state-change
vectors, and community conditions across domains.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import io
import base64

from habitat_evolution.adaptive_core.transformation.semantic_proposition_patterns import SemanticProposition
from habitat_evolution.adaptive_core.emergence.emergent_propensity import EmergentPropensity
from habitat_evolution.adaptive_core.emergence.multi_proposition_dynamics import PropositionEcosystem
from habitat_evolution.adaptive_core.emergence.feedback_loops import PropensityGradient


class PropensityVisualizer:
    """
    Visualizes emergent propensities and their dynamics.
    
    Provides methods to create visual representations of propensity
    gradients, state-change vectors, and community conditions.
    """
    
    def __init__(self, figure_size: Tuple[int, int] = (10, 8)):
        """
        Initialize a propensity visualizer.
        
        Args:
            figure_size: Default figure size for plots
        """
        self.figure_size = figure_size
        self.color_map = LinearSegmentedColormap.from_list(
            'propensity_cmap', ['#1a237e', '#4fc3f7', '#4caf50', '#ffc107', '#d84315']
        )
    
    def visualize_propensity_heatmap(self, gradient: PropensityGradient, 
                                    domain_labels: List[str] = None) -> plt.Figure:
        """
        Create a heatmap of manifestation potentials across domains.
        
        Args:
            gradient: The propensity gradient to visualize
            domain_labels: Optional labels for domains
            
        Returns:
            Matplotlib figure
        """
        # Get manifestation potentials
        potentials = gradient.get_manifestation_gradient()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create heatmap
        im = ax.imshow([potentials], cmap=self.color_map, aspect='auto')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Manifestation Potential', rotation=-90, va="bottom")
        
        # Set labels
        ax.set_title(f'Propensity Gradient: {gradient.proposition.name}')
        ax.set_yticks([])
        
        # Set domain labels if provided
        if domain_labels:
            ax.set_xticks(np.arange(len(domain_labels)))
            ax.set_xticklabels(domain_labels, rotation=45, ha="right")
        else:
            ax.set_xticks(np.arange(len(potentials)))
            ax.set_xticklabels([f'Domain {i+1}' for i in range(len(potentials))], 
                             rotation=45, ha="right")
        
        # Add values as text
        for i, potential in enumerate(potentials):
            ax.text(i, 0, f'{potential:.2f}', ha="center", va="center", 
                   color="white" if potential < 0.7 else "black")
        
        plt.tight_layout()
        return fig
    
    def visualize_direction_radar(self, propensity: EmergentPropensity) -> plt.Figure:
        """
        Create a radar chart of state-change vectors.
        
        Args:
            propensity: The emergent propensity to visualize
            
        Returns:
            Matplotlib figure
        """
        # Get directions and weights
        directions = list(propensity.state_change_vectors.keys())
        weights = [propensity.state_change_vectors[d] for d in directions]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size, subplot_kw=dict(polar=True))
        
        # Number of variables
        N = len(directions)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add the first point at the end to close the loop
        weights += weights[:1]
        
        # Draw the chart
        ax.plot(angles, weights, linewidth=2, linestyle='solid')
        ax.fill(angles, weights, alpha=0.4)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(directions)
        
        # Draw y-axis labels
        ax.set_rlabel_position(0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])
        ax.set_ylim(0, 1)
        
        # Add title
        plt.title(f'State-Change Vectors: {propensity.source_proposition.name}')
        
        return fig
    
    def visualize_condition_bar(self, propensity: EmergentPropensity) -> plt.Figure:
        """
        Create a bar chart of community condition indices.
        
        Args:
            propensity: The emergent propensity to visualize
            
        Returns:
            Matplotlib figure
        """
        # Get conditions and indices
        conditions = list(propensity.community_condition_indices.keys())
        indices = [propensity.community_condition_indices[c] for c in conditions]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create bar chart
        bars = ax.bar(conditions, indices, color=cm.viridis(np.array(indices)))
        
        # Add labels and title
        ax.set_xlabel('Community Conditions')
        ax.set_ylabel('Condition Index')
        ax.set_title(f'Community Condition Indices: {propensity.source_proposition.name}')
        
        # Add values above bars
        for bar, index in zip(bars, indices):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{index:.2f}', ha='center', va='bottom')
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        return fig
    
    def visualize_ecosystem_network(self, ecosystem: PropositionEcosystem) -> plt.Figure:
        """
        Create a network visualization of proposition interactions.
        
        Args:
            ecosystem: The proposition ecosystem to visualize
            
        Returns:
            Matplotlib figure
        """
        # Get interaction network
        G = ecosystem.interaction_network
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create layout
        pos = nx.spring_layout(G)
        
        # Get node colors based on manifestation potential
        node_colors = []
        for node in G.nodes():
            propensity = G.nodes[node]['propensity']
            node_colors.append(propensity.manifestation_potential)
        
        # Get edge colors based on interaction type
        edge_colors = []
        edge_widths = []
        for u, v, data in G.edges(data=True):
            interaction = data['interaction']
            
            # Map interaction type to color
            type_colors = {
                'reinforcing': 'green',
                'conflicting': 'red',
                'catalyzing': 'blue',
                'inhibiting': 'orange',
                'transforming': 'purple',
                'neutral': 'gray'
            }
            
            edge_colors.append(type_colors.get(interaction.interaction_type.value, 'black'))
            edge_widths.append(1 + 3 * interaction.interaction_strength)
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=self.color_map,
                              node_size=500, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        # Add a colorbar with explicit axis reference
        sm = plt.cm.ScalarMappable(cmap=self.color_map)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)  # Pass the axis explicitly
        cbar.set_label('Manifestation Potential')
        
        # Add a legend for edge colors
        legend_elements = [plt.Line2D([0], [0], color=c, lw=2, label=t) 
                          for t, c in type_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set title and remove axis
        plt.title('Proposition Ecosystem Interactions')
        plt.axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_feedback_loop(self, history: List[Dict[str, Any]], 
                              proposition_names: List[str]) -> plt.Figure:
        """
        Create a line chart showing proposition potentials over time.
        
        Args:
            history: Feedback loop history
            proposition_names: Names of propositions to track
            
        Returns:
            Matplotlib figure
        """
        # Extract steps and potentials
        steps = [state['step'] for state in history]
        potentials = {}
        
        for name in proposition_names:
            potentials[name] = [
                state['propensities'][name]['manifestation_potential']
                for state in history
            ]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot potentials
        for name, values in potentials.items():
            ax.plot(steps, values, marker='o', linewidth=2, label=name)
        
        # Add labels and title
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Manifestation Potential')
        ax.set_title('Proposition Potentials Over Time')
        
        # Add legend
        ax.legend()
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        # Set x-axis as integer steps
        ax.set_xticks(steps)
        
        plt.tight_layout()
        return fig
    
    def visualize_3d_propensity_landscape(self, x_coords: List[float], y_coords: List[float],
                                        potentials: List[float], title: str) -> plt.Figure:
        """
        Create a 3D surface plot of propensity landscape.
        
        Args:
            x_coords: X coordinates (e.g., geographic)
            y_coords: Y coordinates (e.g., geographic)
            potentials: Manifestation potentials at each point
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig = plt.figure(figsize=self.figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create surface plot
        surf = ax.plot_trisurf(x_coords, y_coords, potentials, cmap=self.color_map,
                             linewidth=0.2, antialiased=True)
        
        # Add labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Manifestation Potential')
        ax.set_title(title)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set z-axis limits
        ax.set_zlim(0, 1)
        
        plt.tight_layout()
        return fig
    
    def visualize_direction_flow(self, domains: List[Dict[str, Any]], 
                               flows: Dict[str, List[float]],
                               domain_labels: List[str] = None) -> plt.Figure:
        """
        Create a stacked area chart of semantic flows across domains.
        
        Args:
            domains: List of domains
            flows: Dictionary mapping flow names to values across domains
            domain_labels: Optional labels for domains
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Set domain indices
        x = np.arange(len(domains))
        
        # Create stacked area chart
        y_stack = np.zeros(len(domains))
        for flow, values in flows.items():
            ax.fill_between(x, y_stack, y_stack + values, label=flow, alpha=0.7)
            y_stack += values
        
        # Add labels and title
        if domain_labels:
            ax.set_xticks(x)
            ax.set_xticklabels(domain_labels, rotation=45, ha='right')
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([f'Domain {i+1}' for i in range(len(domains))], 
                             rotation=45, ha='right')
            
        ax.set_xlabel('Domain')
        ax.set_ylabel('Semantic Flow Strength')
        ax.set_title('Semantic Flows Across Domains')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        return fig
    
    def save_figure_to_file(self, fig: plt.Figure, filename: str) -> str:
        """
        Save a figure to a file.
        
        Args:
            fig: The figure to save
            filename: The filename to save to
            
        Returns:
            The full path to the saved file
        """
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return filename
    
    def figure_to_base64(self, fig: plt.Figure) -> str:
        """
        Convert a figure to a base64-encoded string.
        
        Args:
            fig: The figure to convert
            
        Returns:
            Base64-encoded string representation of the figure
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
