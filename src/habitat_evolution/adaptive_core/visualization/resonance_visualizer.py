"""
Resonance Visualizer for tonic-harmonic resonance patterns.

This module provides visualization capabilities for resonance patterns,
cascades, and harmonic relationships in the semantic network.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import logging
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from datetime import datetime
import math
import io
import base64

logger = logging.getLogger(__name__)

class ResonanceVisualizer:
    """
    Visualizes resonance patterns and cascades in the semantic network.
    
    This class provides methods to create visual representations of tonic-harmonic
    resonance patterns, including wave visualizations, cascade paths, and
    convergence points.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ResonanceVisualizer with configuration parameters.
        
        Args:
            config: Configuration dictionary with the following optional parameters:
                - node_size_factor: Factor for node size scaling (default: 300)
                - edge_width_factor: Factor for edge width scaling (default: 2)
                - color_map: Color map name for visualization (default: 'viridis')
                - figsize: Figure size tuple (default: (10, 8))
                - dpi: Figure DPI (default: 100)
                - animation_frames: Number of frames for animations (default: 30)
                - animation_interval: Interval between frames in ms (default: 100)
        """
        default_config = {
            "node_size_factor": 300,      # Factor for node size scaling
            "edge_width_factor": 2,       # Factor for edge width scaling
            "color_map": "viridis",       # Color map for visualization
            "figsize": (10, 8),           # Figure size
            "dpi": 100,                   # Figure DPI
            "animation_frames": 30,       # Number of frames for animations
            "animation_interval": 100     # Interval between frames in ms
        }
        
        self.config = default_config
        if config:
            self.config.update(config)
    
    def visualize_resonance_network(self, resonance_data: Dict[str, Any], 
                                   title: str = "Resonance Network") -> Dict[str, Any]:
        """
        Create a network visualization of resonance patterns.
        
        Args:
            resonance_data: Dictionary with nodes and links from ResonanceCascadeTracker
            title: Title for the visualization
            
        Returns:
            Dictionary with visualization data including base64-encoded image
        """
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in resonance_data["nodes"]:
            G.add_node(node["id"], **node)
        
        # Add edges
        for link in resonance_data["links"]:
            G.add_edge(link["source"], link["target"], **link)
        
        # Create figure with axes for better control
        fig, ax = plt.subplots(figsize=self.config["figsize"], dpi=self.config["dpi"])
        
        # Create layout
        pos = nx.spring_layout(G, seed=42)
        
        # Get node attributes for visualization
        node_sizes = [G.nodes[n].get("frequency", 0.5) * self.config["node_size_factor"] + 100 for n in G.nodes()]
        
        # Color nodes by amplitude
        node_colors = [G.nodes[n].get("amplitude", 0.5) for n in G.nodes()]
        
        # Highlight convergence points
        node_borders = ['red' if G.nodes[n].get("is_convergence_point", False) else 'black' for n in G.nodes()]
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(G, pos, 
                              ax=ax,
                              node_size=node_sizes,
                              node_color=node_colors, 
                              edgecolors=node_borders,
                              linewidths=2,
                              alpha=0.8,
                              cmap=plt.cm.get_cmap(self.config["color_map"]))
        
        # Get edge attributes for visualization
        edge_widths = [G[u][v].get("strength", 0.5) * self.config["edge_width_factor"] for u, v in G.edges()]
        edge_colors = [G[u][v].get("strength", 0.5) for u, v in G.edges()]
        
        # Draw edges
        edges = nx.draw_networkx_edges(G, pos, 
                                     ax=ax,
                                     width=edge_widths,
                                     edge_color=edge_colors,
                                     edge_cmap=plt.cm.get_cmap(self.config["color_map"]),
                                     alpha=0.6,
                                     arrows=True,
                                     arrowsize=10,
                                     connectionstyle='arc3,rad=0.1')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_family='sans-serif')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(self.config["color_map"]))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Resonance Strength")
        
        # Set title and layout
        ax.set_title(title)
        ax.axis('off')
        fig.tight_layout()
        
        # Save figure to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode as base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Close figure to free memory
        plt.close()
        
        return {
            "image_base64": img_str,
            "title": title,
            "node_count": len(G.nodes()),
            "edge_count": len(G.edges()),
            "timestamp": datetime.now().isoformat()
        }
    
    def visualize_resonance_cascade(self, resonance_data: Dict[str, Any], 
                                   cascade_id: str,
                                   title: str = "Resonance Cascade") -> Dict[str, Any]:
        """
        Create a visualization of a specific resonance cascade.
        
        Args:
            resonance_data: Dictionary with nodes, links and cascades
            cascade_id: ID of the cascade to visualize
            title: Title for the visualization
            
        Returns:
            Dictionary with visualization data including base64-encoded image
        """
        # Find the specified cascade
        cascade = None
        for c in resonance_data["cascades"]:
            if c["id"] == cascade_id:
                cascade = c
                break
        
        if not cascade:
            logger.error(f"Cascade with ID {cascade_id} not found")
            return {"error": f"Cascade with ID {cascade_id} not found"}
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Create a subgraph with only the nodes and edges in the cascade
        cascade_path = cascade["path"]
        
        # Add nodes
        for node_id in cascade_path:
            # Find the node data
            node_data = None
            for n in resonance_data["nodes"]:
                if n["id"] == node_id:
                    node_data = n
                    break
            
            if node_data:
                G.add_node(node_id, **node_data)
            else:
                G.add_node(node_id)
        
        # Add edges
        for i in range(len(cascade_path) - 1):
            source = cascade_path[i]
            target = cascade_path[i + 1]
            
            # Find the edge data
            edge_data = None
            for e in resonance_data["links"]:
                if e["source"] == source and e["target"] == target:
                    edge_data = e
                    break
            
            if edge_data:
                G.add_edge(source, target, **edge_data)
            else:
                G.add_edge(source, target)
        
        # Create figure with axes for better control
        fig, ax = plt.subplots(figsize=self.config["figsize"], dpi=self.config["dpi"])
        
        # Create layout - use a path layout for cascades
        pos = {}
        for i, node_id in enumerate(cascade_path):
            pos[node_id] = (i, 0)
        
        # Get node attributes for visualization
        node_sizes = [G.nodes[n].get("frequency", 0.5) * self.config["node_size_factor"] + 100 for n in G.nodes()]
        
        # Color nodes by position in cascade
        node_colors = [i / (len(cascade_path) - 1) if len(cascade_path) > 1 else 0.5 
                      for i, _ in enumerate(cascade_path)]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              ax=ax,
                              node_size=node_sizes,
                              node_color=node_colors, 
                              cmap=plt.cm.get_cmap(self.config["color_map"]),
                              alpha=0.8)
        
        # Get edge attributes for visualization
        edge_widths = [G[u][v].get("strength", 0.5) * self.config["edge_width_factor"] for u, v in G.edges()]
        
        # Draw edges
        edges = nx.draw_networkx_edges(G, pos, 
                                     ax=ax,
                                     width=edge_widths,
                                     alpha=0.6,
                                     arrows=True,
                                     arrowsize=15)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_family='sans-serif')
        
        # Add cascade information
        fig.text(0.02, 0.02, f"Cascade Strength: {cascade['strength']:.3f}", fontsize=10)
        fig.text(0.02, 0.05, f"Cascade Length: {cascade['length']}", fontsize=10)
        
        # Set title and layout
        ax.set_title(f"{title} - {cascade_id}")
        ax.axis('off')
        fig.tight_layout()
        
        # Save figure to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode as base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Close figure to free memory
        plt.close()
        
        return {
            "image_base64": img_str,
            "title": title,
            "cascade_id": cascade_id,
            "cascade_path": cascade_path,
            "cascade_strength": cascade["strength"],
            "cascade_length": cascade["length"],
            "timestamp": datetime.now().isoformat()
        }
    
    def visualize_harmonic_waves(self, domains: List[Dict[str, Any]], 
                               title: str = "Harmonic Waves") -> Dict[str, Any]:
        """
        Create a visualization of harmonic waves for domains.
        
        Args:
            domains: List of domain dictionaries with wave properties
            title: Title for the visualization
            
        Returns:
            Dictionary with visualization data including base64-encoded image
        """
        # Create figure with axes for better control
        fig, ax = plt.subplots(figsize=self.config["figsize"], dpi=self.config["dpi"])
        
        # Time points for wave visualization
        t = np.linspace(0, 2*np.pi, 1000)
        
        # Plot each domain as a wave
        for i, domain in enumerate(domains):
            frequency = domain.get("frequency", 1.0)
            amplitude = domain.get("amplitude", 1.0)
            phase = domain.get("phase", 0.0)
            
            # Generate wave
            wave = amplitude * np.sin(frequency * t + phase)
            
            # Plot with domain-specific color from colormap
            color = plt.cm.get_cmap(self.config["color_map"])(i / len(domains))
            ax.plot(t, wave, label=domain["id"], color=color, linewidth=2)
        
        # Add legend and labels
        ax.legend(loc='upper right')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Save figure to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode as base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Close figure to free memory
        plt.close()
        
        return {
            "image_base64": img_str,
            "title": title,
            "domain_count": len(domains),
            "timestamp": datetime.now().isoformat()
        }
    
    def create_resonance_animation(self, domains: List[Dict[str, Any]], 
                                 title: str = "Resonance Animation") -> Dict[str, Any]:
        """
        Create an animation of resonating waves.
        
        Args:
            domains: List of domain dictionaries with wave properties
            title: Title for the animation
            
        Returns:
            Dictionary with animation data including base64-encoded gif
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.config["figsize"], dpi=self.config["dpi"])
        
        # Time points for wave visualization
        t = np.linspace(0, 2*np.pi, 1000)
        
        # Initialize lines for each domain
        lines = []
        for i, domain in enumerate(domains):
            # Plot with domain-specific color from colormap
            color = plt.cm.get_cmap(self.config["color_map"])(i / len(domains))
            line, = ax.plot([], [], label=domain["id"], color=color, linewidth=2)
            lines.append(line)
        
        # Add legend and labels
        ax.legend(loc='upper right')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Set axis limits
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(-1.5, 1.5)
        
        # Animation initialization function
        def init():
            for line in lines:
                line.set_data([], [])
            return lines
        
        # Animation update function
        def update(frame):
            phase_offset = frame / self.config["animation_frames"] * 2 * np.pi
            
            for i, (line, domain) in enumerate(zip(lines, domains)):
                frequency = domain.get("frequency", 1.0)
                amplitude = domain.get("amplitude", 1.0)
                phase = domain.get("phase", 0.0)
                
                # Generate wave with phase offset
                wave = amplitude * np.sin(frequency * t + phase + phase_offset)
                
                # Update line data
                line.set_data(t, wave)
            
            return lines
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=self.config["animation_frames"],
                                     init_func=init, blit=True, 
                                     interval=self.config["animation_interval"])
        
        # Create a temporary file path for the animation
        import tempfile
        import time
        import os
        
        # Create a temporary directory if it doesn't exist
        temp_dir = os.path.join(tempfile.gettempdir(), 'habitat_visualizations')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create a unique filename
        temp_file = os.path.join(temp_dir, f"resonance_animation_{int(time.time())}.gif")
        
        # Save animation to file
        anim.save(temp_file, writer='pillow', fps=10)
        
        # Read the file back for base64 encoding
        with open(temp_file, 'rb') as f:
            animation_data = f.read()
            gif_str = base64.b64encode(animation_data).decode('utf-8')
        
        # Close figure to free memory
        plt.close(fig)
        
        return {
            "animation_base64": gif_str,
            "title": title,
            "domain_count": len(domains),
            "frame_count": self.config["animation_frames"],
            "timestamp": datetime.now().isoformat(),
            "file_path": temp_file
        }
    
    def visualize_convergence_points(self, resonance_data: Dict[str, Any],
                                   title: str = "Convergence Points") -> Dict[str, Any]:
        """
        Create a visualization highlighting convergence points in the resonance network.
        
        Args:
            resonance_data: Dictionary with nodes, links and convergence points
            title: Title for the visualization
            
        Returns:
            Dictionary with visualization data including base64-encoded image
        """
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in resonance_data["nodes"]:
            G.add_node(node["id"], **node)
        
        # Add edges
        for link in resonance_data["links"]:
            G.add_edge(link["source"], link["target"], **link)
        
        # Create figure with axes for better control
        fig, ax = plt.subplots(figsize=self.config["figsize"], dpi=self.config["dpi"])
        
        # Create layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw all nodes and edges with low alpha
        nx.draw_networkx_nodes(G, pos, 
                              ax=ax,
                              node_size=100,
                              node_color='lightgray', 
                              alpha=0.3)
        
        nx.draw_networkx_edges(G, pos, 
                              ax=ax,
                              width=0.5,
                              alpha=0.2,
                              arrows=False)
        
        # Highlight convergence points
        convergence_nodes = []
        for cp in resonance_data["convergence_points"]:
            convergence_nodes.append(cp["node_id"])
        
        if convergence_nodes:
            # Draw convergence points
            nx.draw_networkx_nodes(G, pos, 
                                  ax=ax,
                                  nodelist=convergence_nodes,
                                  node_size=[300 * cp["independent_sources"] for cp in resonance_data["convergence_points"]],
                                  node_color='red', 
                                  alpha=0.8)
            
            # Draw incoming edges to convergence points
            for cp in resonance_data["convergence_points"]:
                for source in cp["source_nodes"]:
                    if G.has_edge(source, cp["node_id"]):
                        nx.draw_networkx_edges(G, pos, 
                                             ax=ax,
                                             edgelist=[(source, cp["node_id"])],
                                             width=2,
                                             alpha=0.8,
                                             arrows=True,
                                             arrowsize=15,
                                             edge_color='red')
            
            # Draw labels for convergence points
            nx.draw_networkx_labels(G, pos, 
                                   ax=ax,
                                   font_size=10, 
                                   font_family='sans-serif',
                                   labels={n: n for n in convergence_nodes})
        
        # Set title and layout
        ax.set_title(title)
        ax.axis('off')
        fig.tight_layout()
        
        # Add legend
        fig.text(0.02, 0.02, f"Convergence Points: {len(convergence_nodes)}", fontsize=10)
        
        # Save figure to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode as base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Close figure to free memory
        plt.close()
        
        return {
            "image_base64": img_str,
            "title": title,
            "convergence_point_count": len(convergence_nodes),
            "timestamp": datetime.now().isoformat()
        }
