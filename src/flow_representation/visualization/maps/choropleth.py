"""Create choropleth maps for Martha's Vineyard climate risk metrics."""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import geopandas as gpd
from typing import Dict, List, Tuple, Optional
import numpy as np

from src.flow_representation.visualization.style.theme import (
    COLORS, BACKGROUND_COLOR, MAP_EDGE_COLOR, MAP_EDGE_WIDTH,
    FIGURE_SIZE, MAP_BOUNDS, FONT_FAMILY, TEXT_COLOR
)

class ClimateRiskMapVisualizer:
    """Create multi-panel choropleth maps for climate risk metrics."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.figure: Optional[Figure] = None
        self.axes: Optional[List[Axes]] = None
        
    def create_gradient_colormap(self, metric: str) -> LinearSegmentedColormap:
        """Create a gradient colormap for a metric.
        
        Args:
            metric: Name of the metric to create colormap for
            
        Returns:
            LinearSegmentedColormap for the metric
        """
        colors = COLORS[metric]
        return LinearSegmentedColormap.from_list(
            f"{metric}_gradient",
            colors,
            N=256
        )
    
    def plot_map_on_ax(
        self,
        data: gpd.GeoDataFrame,
        metric: str,
        ax: Axes,
        cmap: LinearSegmentedColormap
    ) -> None:
        """Plot a choropleth map on a given axis.
        
        Args:
            data: GeoDataFrame containing geometries and metrics
            metric: Name of the metric to plot
            ax: Matplotlib axis to plot on
            cmap: Colormap to use
        """
        data.plot(
            column=metric,
            cmap=cmap,
            edgecolor=MAP_EDGE_COLOR,
            linewidth=MAP_EDGE_WIDTH,
            ax=ax
        )
        ax.set_xlim(MAP_BOUNDS['x_min'], MAP_BOUNDS['x_max'])
        ax.set_ylim(MAP_BOUNDS['y_min'], MAP_BOUNDS['y_max'])
        ax.axis('off')
    
    def plot_lollipop(
        self,
        min_max_df: Dict[str, Tuple[float, float]],
        ax: Axes
    ) -> None:
        """Plot lollipop chart showing metric ranges.
        
        Args:
            min_max_df: Dictionary of (min, max) tuples for each metric
            ax: Matplotlib axis to plot on
        """
        metrics = list(min_max_df.keys())
        
        for i, metric in enumerate(metrics):
            min_val, max_val = min_max_df[metric]
            min_color = COLORS[metric][0]
            max_color = COLORS[metric][1]
            
            # Plot points
            ax.scatter(
                min_val, i,
                zorder=2,
                s=160,
                edgecolor='black',
                linewidth=0.5,
                color=min_color
            )
            ax.scatter(
                max_val, i,
                zorder=2,
                s=160,
                edgecolor='black',
                linewidth=0.5,
                color=max_color
            )
            
            # Plot connecting line
            ax.hlines(
                y=i,
                xmin=min_val,
                xmax=max_val,
                color='white',
                linewidth=0.8,
                zorder=1
            )
        
        # Customize axis
        ax.spines[['right', 'top', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.set_yticks([])
        ax.set_ylim(-1, len(metrics))
        
        # Add metric labels
        for i, metric in enumerate(metrics):
            ax.text(
                -0.1, i,
                metric.replace('_', ' ').title(),
                color=TEXT_COLOR,
                ha='right',
                va='center',
                fontfamily=FONT_FAMILY,
                transform=ax.get_yaxis_transform()
            )
    
    def create_visualization(
        self,
        data: gpd.GeoDataFrame,
        min_max_values: Dict[str, Tuple[float, float]],
        output_path: Optional[str] = None
    ) -> Figure:
        """Create the complete visualization.
        
        Args:
            data: GeoDataFrame with geometries and metrics
            min_max_values: Dictionary of (min, max) tuples for each metric
            output_path: Optional path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        self.figure, self.axes = plt.subplots(
            nrows=3,
            ncols=2,
            figsize=FIGURE_SIZE
        )
        self.axes = self.axes.flatten()
        
        # Set background color
        self.figure.set_facecolor(BACKGROUND_COLOR)
        self.axes[1].set_facecolor(BACKGROUND_COLOR)
        
        # List of metrics to display
        metrics = list(min_max_values.keys())
        
        # Plot maps
        for i, (ax, metric) in enumerate(zip(self.axes[2:], metrics)):
            cmap = self.create_gradient_colormap(metric)
            self.plot_map_on_ax(data, metric, ax, cmap)
        
        # Plot lollipop chart
        self.plot_lollipop(min_max_values, self.axes[1])
        
        # Add title
        self.figure.text(
            0.5, 0.95,
            "Climate Risk Metrics by Town",
            ha='center',
            color=TEXT_COLOR,
            fontsize=14,
            fontfamily=FONT_FAMILY
        )
        
        # Add description
        self.figure.text(
            0.5, 0.92,
            "Share of risk explained by different factors across Martha's Vineyard",
            ha='center',
            color=TEXT_COLOR,
            fontsize=10,
            fontfamily=FONT_FAMILY
        )
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(
                output_path,
                dpi=100,
                bbox_inches='tight',
                facecolor=BACKGROUND_COLOR
            )
        
        return self.figure
