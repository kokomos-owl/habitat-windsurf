"""
Pattern Correlation Visualizer for Habitat Evolution.

This module provides visualization tools for displaying correlations between
statistical and semantic patterns, including time-based visualizations and
network diagrams showing pattern relationships.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

class PatternCorrelationVisualizer:
    """
    Visualizer for pattern correlations across domains.
    
    This class provides methods for visualizing the correlations between
    statistical patterns from time-series data and semantic patterns
    extracted from text documents.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize a new pattern correlation visualizer.
        
        Args:
            figsize: Size of the figure for visualizations
        """
        self.figsize = figsize
        
        # Set up color schemes
        self.statistical_color = "#1f77b4"  # Blue
        self.semantic_color = "#ff7f0e"     # Orange
        self.correlation_color = "#2ca02c"  # Green
        
        # Quality state colors
        self.quality_colors = {
            "hypothetical": "#d3d3d3",  # Light gray
            "emergent": "#ffcc99",      # Light orange
            "stable": "#66cc66"         # Light green
        }
        
        # Set up Seaborn style
        sns.set_style("whitegrid")
        
        # Custom colormap for correlation strength
        self.correlation_cmap = LinearSegmentedColormap.from_list(
            "correlation_strength", 
            ["#f7fbff", "#08306b"]
        )
    
    def plot_time_series_with_patterns(self, 
                                     time_series_data: Dict[str, Any],
                                     statistical_patterns: List[Dict[str, Any]],
                                     title: str = "Time Series with Detected Patterns",
                                     show_anomalies: bool = True,
                                     figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot time series data with highlighted statistical patterns.
        
        Args:
            time_series_data: Dictionary containing time series data
            statistical_patterns: List of detected statistical patterns
            title: Title for the plot
            show_anomalies: Whether to show anomalies in addition to values
            figsize: Optional custom figure size
            
        Returns:
            Matplotlib figure object
        """
        fig_size = figsize or self.figsize
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Extract time series data
        timestamps = []
        values = []
        anomalies = []
        
        for timestamp, data in time_series_data.get("data", {}).items():
            try:
                # Convert timestamp string to datetime
                dt = datetime.strptime(timestamp, "%Y%m")
                timestamps.append(dt)
                values.append(data.get("value", 0))
                anomalies.append(data.get("anomaly", 0))
            except (ValueError, TypeError):
                continue
        
        # Plot the main time series
        ax.plot(timestamps, values, color=self.statistical_color, 
                linewidth=2, label="Temperature")
        
        # Plot anomalies if requested
        if show_anomalies:
            ax2 = ax.twinx()
            ax2.plot(timestamps, anomalies, color="#d62728", 
                    linewidth=1.5, linestyle="--", label="Anomaly")
            ax2.set_ylabel("Anomaly (°F)", color="#d62728")
            ax2.tick_params(axis="y", labelcolor="#d62728")
            ax2.spines["right"].set_color("#d62728")
        
        # Highlight pattern regions
        for pattern in statistical_patterns:
            # Get pattern time range
            start_time = pattern.get("start_time", "")
            end_time = pattern.get("end_time", "")
            quality = pattern.get("quality_state", "hypothetical")
            
            if start_time and end_time:
                try:
                    start_dt = datetime.strptime(start_time, "%Y%m")
                    end_dt = datetime.strptime(end_time, "%Y%m")
                    
                    # Highlight the pattern region
                    ax.axvspan(start_dt, end_dt, 
                              alpha=0.3, 
                              color=self.quality_colors.get(quality, "#d3d3d3"),
                              label=f"{quality.capitalize()} Pattern")
                    
                    # Add pattern label
                    pattern_type = pattern.get("type", "Unknown")
                    ax.text(start_dt, max(values) * 0.95, 
                           f"{pattern_type}", 
                           fontsize=9, 
                           bbox=dict(facecolor="white", alpha=0.7))
                except (ValueError, TypeError):
                    continue
        
        # Format the plot
        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature (°F)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis to show years
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        plt.xticks(rotation=45)
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        if show_anomalies:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        else:
            ax.legend(loc="upper left")
        
        plt.tight_layout()
        return fig
    
    def plot_pattern_correlation_heatmap(self,
                                       statistical_patterns: List[Dict[str, Any]],
                                       semantic_patterns: List[Dict[str, Any]],
                                       correlations: List[Dict[str, Any]],
                                       title: str = "Pattern Correlation Heatmap",
                                       figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot a heatmap showing correlations between statistical and semantic patterns.
        
        Args:
            statistical_patterns: List of statistical patterns
            semantic_patterns: List of semantic patterns
            correlations: List of correlations between patterns
            title: Title for the plot
            figsize: Optional custom figure size
            
        Returns:
            Matplotlib figure object
        """
        fig_size = figsize or (max(10, len(semantic_patterns) * 0.8), 
                              max(8, len(statistical_patterns) * 0.6))
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Create correlation matrix
        correlation_matrix = np.zeros((len(statistical_patterns), len(semantic_patterns)))
        
        # Map pattern IDs to indices
        stat_id_to_idx = {p["id"]: i for i, p in enumerate(statistical_patterns)}
        sem_id_to_idx = {p["id"]: j for j, p in enumerate(semantic_patterns)}
        
        # Fill correlation matrix
        for corr in correlations:
            stat_id = corr.get("statistical_pattern_id")
            sem_id = corr.get("semantic_pattern_id")
            strength = corr.get("correlation_strength", 0)
            
            if stat_id in stat_id_to_idx and sem_id in sem_id_to_idx:
                i = stat_id_to_idx[stat_id]
                j = sem_id_to_idx[sem_id]
                correlation_matrix[i, j] = strength
        
        # Create heatmap
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap=self.correlation_cmap,
                   vmin=0, 
                   vmax=1,
                   linewidths=0.5,
                   ax=ax)
        
        # Set labels
        stat_labels = [self._get_pattern_label(p) for p in statistical_patterns]
        sem_labels = [self._get_pattern_label(p) for p in semantic_patterns]
        
        ax.set_yticks(np.arange(len(statistical_patterns)) + 0.5)
        ax.set_yticklabels(stat_labels, rotation=0)
        
        ax.set_xticks(np.arange(len(semantic_patterns)) + 0.5)
        ax.set_xticklabels(sem_labels, rotation=45, ha="right")
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel("Semantic Patterns")
        ax.set_ylabel("Statistical Patterns")
        
        plt.tight_layout()
        return fig
    
    def plot_pattern_network(self,
                           statistical_patterns: List[Dict[str, Any]],
                           semantic_patterns: List[Dict[str, Any]],
                           correlations: List[Dict[str, Any]],
                           title: str = "Pattern Correlation Network",
                           min_correlation: float = 0.3,
                           figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot a network diagram showing pattern relationships.
        
        Args:
            statistical_patterns: List of statistical patterns
            semantic_patterns: List of semantic patterns
            correlations: List of correlations between patterns
            title: Title for the plot
            min_correlation: Minimum correlation strength to show
            figsize: Optional custom figure size
            
        Returns:
            Matplotlib figure object
        """
        fig_size = figsize or self.figsize
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Create graph
        G = nx.Graph()
        
        # Add statistical pattern nodes
        for pattern in statistical_patterns:
            pattern_id = pattern.get("id")
            quality = pattern.get("quality_state", "hypothetical")
            label = self._get_pattern_label(pattern)
            
            G.add_node(pattern_id, 
                      domain="statistical", 
                      quality=quality,
                      label=label)
        
        # Add semantic pattern nodes
        for pattern in semantic_patterns:
            pattern_id = pattern.get("id")
            quality = pattern.get("quality_state", "hypothetical")
            label = self._get_pattern_label(pattern)
            
            G.add_node(pattern_id, 
                      domain="semantic", 
                      quality=quality,
                      label=label)
        
        # Add correlation edges
        for corr in correlations:
            stat_id = corr.get("statistical_pattern_id")
            sem_id = corr.get("semantic_pattern_id")
            strength = corr.get("correlation_strength", 0)
            corr_type = corr.get("correlation_type", "unknown")
            
            if strength >= min_correlation:
                G.add_edge(stat_id, sem_id, 
                          weight=strength, 
                          type=corr_type)
        
        # Remove nodes without edges
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        
        if len(G.nodes) == 0:
            ax.text(0.5, 0.5, "No correlations above threshold", 
                   ha="center", va="center", fontsize=14)
            ax.set_title(title)
            ax.axis("off")
            return fig
        
        # Set node positions using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        
        for node in G.nodes:
            domain = G.nodes[node]["domain"]
            quality = G.nodes[node]["quality"]
            
            # Set color based on domain
            if domain == "statistical":
                color = self.statistical_color
            else:
                color = self.semantic_color
            
            # Set size based on quality
            if quality == "stable":
                size = 800
            elif quality == "emergent":
                size = 600
            else:
                size = 400
            
            node_colors.append(color)
            node_sizes.append(size)
        
        # Draw edges with width based on correlation strength
        edge_widths = [G[u][v]["weight"] * 5 for u, v in G.edges]
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors, 
                              node_size=node_sizes,
                              alpha=0.8,
                              ax=ax)
        
        nx.draw_networkx_edges(G, pos, 
                              width=edge_widths, 
                              alpha=0.6,
                              edge_color="gray",
                              ax=ax)
        
        # Add node labels
        labels = {node: G.nodes[node]["label"] for node in G.nodes}
        nx.draw_networkx_labels(G, pos, 
                               labels=labels, 
                               font_size=9,
                               font_weight="bold",
                               ax=ax)
        
        # Add legend
        stat_patch = plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=self.statistical_color, 
                               markersize=15, label='Statistical Pattern')
        
        sem_patch = plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=self.semantic_color, 
                              markersize=15, label='Semantic Pattern')
        
        stable_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor='gray', 
                                 markersize=15, label='Stable Pattern')
        
        emergent_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor='gray', 
                                   markersize=12, label='Emergent Pattern')
        
        hypothetical_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                       markerfacecolor='gray', 
                                       markersize=10, label='Hypothetical Pattern')
        
        ax.legend(handles=[stat_patch, sem_patch, 
                          stable_patch, emergent_patch, hypothetical_patch],
                 loc="upper right")
        
        ax.set_title(title)
        ax.axis("off")
        
        plt.tight_layout()
        return fig
    
    def plot_sliding_window_view(self,
                               time_series_data: Dict[str, Any],
                               statistical_patterns: List[Dict[str, Any]],
                               semantic_patterns: List[Dict[str, Any]],
                               correlations: List[Dict[str, Any]],
                               window_start: str,
                               window_end: str,
                               title: str = "Sliding Window Pattern View",
                               figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot a sliding window view of patterns and correlations.
        
        Args:
            time_series_data: Dictionary containing time series data
            statistical_patterns: List of detected statistical patterns
            semantic_patterns: List of semantic patterns
            correlations: List of correlations between patterns
            window_start: Start time for the window (YYYYMM format)
            window_end: End time for the window (YYYYMM format)
            title: Title for the plot
            figsize: Optional custom figure size
            
        Returns:
            Matplotlib figure object
        """
        fig_size = figsize or (self.figsize[0], self.figsize[1] * 1.5)
        fig = plt.figure(figsize=fig_size)
        
        # Create a 2x1 grid
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
        
        # Convert window times to datetime
        try:
            window_start_dt = datetime.strptime(window_start, "%Y%m")
            window_end_dt = datetime.strptime(window_end, "%Y%m")
        except ValueError:
            # Default to full range if invalid format
            timestamps = []
            for timestamp in time_series_data.get("data", {}).keys():
                try:
                    timestamps.append(datetime.strptime(timestamp, "%Y%m"))
                except ValueError:
                    continue
            
            if timestamps:
                window_start_dt = min(timestamps)
                window_end_dt = max(timestamps)
            else:
                # Fallback if no valid timestamps
                window_start_dt = datetime(1990, 1, 1)
                window_end_dt = datetime(2025, 1, 1)
        
        # Filter patterns by time window
        filtered_stat_patterns = []
        for pattern in statistical_patterns:
            start_time = pattern.get("start_time", "")
            end_time = pattern.get("end_time", "")
            
            try:
                pattern_start_dt = datetime.strptime(start_time, "%Y%m")
                pattern_end_dt = datetime.strptime(end_time, "%Y%m")
                
                # Include if pattern overlaps with window
                if (pattern_start_dt <= window_end_dt and 
                    pattern_end_dt >= window_start_dt):
                    filtered_stat_patterns.append(pattern)
            except ValueError:
                continue
        
        # Filter semantic patterns (assuming they have temporal markers)
        filtered_sem_patterns = []
        for pattern in semantic_patterns:
            temporal_markers = pattern.get("temporal_markers", [])
            
            if temporal_markers:
                # Check if any marker falls within window
                for marker in temporal_markers:
                    marker_time = marker.get("time", "")
                    
                    try:
                        marker_dt = datetime.strptime(marker_time, "%Y%m")
                        
                        if window_start_dt <= marker_dt <= window_end_dt:
                            filtered_sem_patterns.append(pattern)
                            break
                    except ValueError:
                        continue
            else:
                # If no temporal markers, include based on text analysis
                # This is a simplified approach
                text = pattern.get("text", "")
                
                # Extract years from text
                import re
                years = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
                
                for year_str in years:
                    year = int(year_str)
                    if (window_start_dt.year <= year <= window_end_dt.year):
                        filtered_sem_patterns.append(pattern)
                        break
        
        # Filter correlations to only include filtered patterns
        filtered_correlations = []
        stat_ids = {p["id"] for p in filtered_stat_patterns}
        sem_ids = {p["id"] for p in filtered_sem_patterns}
        
        for corr in correlations:
            stat_id = corr.get("statistical_pattern_id")
            sem_id = corr.get("semantic_pattern_id")
            
            if stat_id in stat_ids and sem_id in sem_ids:
                filtered_correlations.append(corr)
        
        # Plot time series in top subplot
        ax1 = fig.add_subplot(gs[0])
        
        # Extract time series data within window
        timestamps = []
        values = []
        anomalies = []
        
        for timestamp, data in time_series_data.get("data", {}).items():
            try:
                dt = datetime.strptime(timestamp, "%Y%m")
                
                if window_start_dt <= dt <= window_end_dt:
                    timestamps.append(dt)
                    values.append(data.get("value", 0))
                    anomalies.append(data.get("anomaly", 0))
            except ValueError:
                continue
        
        # Plot the main time series
        ax1.plot(timestamps, values, color=self.statistical_color, 
                linewidth=2, label="Temperature")
        
        # Plot anomalies
        ax1_twin = ax1.twinx()
        ax1_twin.plot(timestamps, anomalies, color="#d62728", 
                     linewidth=1.5, linestyle="--", label="Anomaly")
        ax1_twin.set_ylabel("Anomaly (°F)", color="#d62728")
        ax1_twin.tick_params(axis="y", labelcolor="#d62728")
        
        # Highlight pattern regions
        for pattern in filtered_stat_patterns:
            start_time = pattern.get("start_time", "")
            end_time = pattern.get("end_time", "")
            quality = pattern.get("quality_state", "hypothetical")
            
            try:
                start_dt = datetime.strptime(start_time, "%Y%m")
                end_dt = datetime.strptime(end_time, "%Y%m")
                
                # Ensure within window
                start_dt = max(start_dt, window_start_dt)
                end_dt = min(end_dt, window_end_dt)
                
                # Highlight the pattern region
                ax1.axvspan(start_dt, end_dt, 
                           alpha=0.3, 
                           color=self.quality_colors.get(quality, "#d3d3d3"))
                
                # Add pattern label
                pattern_type = pattern.get("type", "Unknown")
                ax1.text(start_dt, max(values) * 0.95, 
                        f"{pattern_type}", 
                        fontsize=9, 
                        bbox=dict(facecolor="white", alpha=0.7))
            except ValueError:
                continue
        
        # Format the top subplot
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Temperature (°F)")
        ax1.set_title(f"Time Series Patterns ({window_start} to {window_end})")
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis to show years
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Add legend for top subplot
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        
        # Plot correlation network in bottom subplot
        ax2 = fig.add_subplot(gs[1])
        
        # Create graph
        G = nx.Graph()
        
        # Add statistical pattern nodes
        for pattern in filtered_stat_patterns:
            pattern_id = pattern.get("id")
            quality = pattern.get("quality_state", "hypothetical")
            label = self._get_pattern_label(pattern)
            
            G.add_node(pattern_id, 
                      domain="statistical", 
                      quality=quality,
                      label=label)
        
        # Add semantic pattern nodes
        for pattern in filtered_sem_patterns:
            pattern_id = pattern.get("id")
            quality = pattern.get("quality_state", "hypothetical")
            label = self._get_pattern_label(pattern)
            
            G.add_node(pattern_id, 
                      domain="semantic", 
                      quality=quality,
                      label=label)
        
        # Add correlation edges
        for corr in filtered_correlations:
            stat_id = corr.get("statistical_pattern_id")
            sem_id = corr.get("semantic_pattern_id")
            strength = corr.get("correlation_strength", 0)
            
            G.add_edge(stat_id, sem_id, weight=strength)
        
        # Remove nodes without edges
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        
        if len(G.nodes) == 0:
            ax2.text(0.5, 0.5, "No correlations in this time window", 
                    ha="center", va="center", fontsize=14)
            ax2.axis("off")
        else:
            # Set node positions using spring layout
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            node_colors = []
            node_sizes = []
            
            for node in G.nodes:
                domain = G.nodes[node]["domain"]
                quality = G.nodes[node]["quality"]
                
                # Set color based on domain
                if domain == "statistical":
                    color = self.statistical_color
                else:
                    color = self.semantic_color
                
                # Set size based on quality
                if quality == "stable":
                    size = 600
                elif quality == "emergent":
                    size = 400
                else:
                    size = 300
                
                node_colors.append(color)
                node_sizes.append(size)
            
            # Draw edges with width based on correlation strength
            edge_widths = [G[u][v]["weight"] * 4 for u, v in G.edges]
            
            # Draw the network
            nx.draw_networkx_nodes(G, pos, 
                                  node_color=node_colors, 
                                  node_size=node_sizes,
                                  alpha=0.8,
                                  ax=ax2)
            
            nx.draw_networkx_edges(G, pos, 
                                  width=edge_widths, 
                                  alpha=0.6,
                                  edge_color="gray",
                                  ax=ax2)
            
            # Add node labels
            labels = {node: G.nodes[node]["label"] for node in G.nodes}
            nx.draw_networkx_labels(G, pos, 
                                   labels=labels, 
                                   font_size=8,
                                   font_weight="bold",
                                   ax=ax2)
            
            # Add legend for bottom subplot
            stat_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=self.statistical_color, 
                                   markersize=10, label='Statistical Pattern')
            
            sem_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=self.semantic_color, 
                                  markersize=10, label='Semantic Pattern')
            
            ax2.legend(handles=[stat_patch, sem_patch], loc="upper right")
            
            ax2.set_title("Pattern Correlations")
            ax2.axis("off")
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        return fig
    
    def _get_pattern_label(self, pattern: Dict[str, Any]) -> str:
        """
        Get a short label for a pattern.
        
        Args:
            pattern: Pattern data
            
        Returns:
            Short label for the pattern
        """
        # Try different fields that might contain a good label
        if "name" in pattern:
            return pattern["name"]
        elif "type" in pattern:
            return pattern["type"]
        elif "text" in pattern:
            # Truncate text to a reasonable length
            text = pattern["text"]
            if len(text) > 30:
                return text[:27] + "..."
            return text
        elif "id" in pattern:
            # Use ID as last resort
            return pattern["id"]
        else:
            return "Unknown Pattern"
