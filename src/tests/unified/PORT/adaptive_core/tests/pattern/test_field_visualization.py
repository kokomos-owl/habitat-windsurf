"""Visualization helpers for pattern field testing."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

@dataclass
class VisualizationConfig:
    """Configuration for field visualization."""
    # Color schemes
    coherence_cmap: str = 'viridis'  # For coherence fields
    flow_cmap: str = 'coolwarm'      # For flow fields
    density_cmap: str = 'YlOrRd'     # For density fields
    
    # Plot settings
    plot_size: Tuple[int, int] = (10, 8)
    dpi: int = 100
    contour_levels: int = 20
    quiver_scale: float = 50
    
    # Animation
    animation_frames: int = 30
    frame_interval: int = 100  # milliseconds

class FieldVisualizer:
    """Visualizes pattern fields and their properties."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        sns.set_style('whitegrid')
    
    def plot_field_state(self, 
                        field: np.ndarray,
                        patterns: List[Dict[str, Any]],
                        title: str = "Field State") -> Tuple[Figure, Axes]:
        """Plot current field state with patterns."""
        fig, ax = plt.subplots(figsize=self.config.plot_size, dpi=self.config.dpi)
        
        # Plot field intensity
        im = ax.imshow(field, 
                      cmap=self.config.coherence_cmap,
                      interpolation='gaussian')
        plt.colorbar(im, ax=ax, label='Field Strength')
        
        # Plot patterns
        for pattern in patterns:
            pos = pattern["context"]["position"]
            strength = pattern["metrics"]["energy_state"]
            coherence = pattern["metrics"]["coherence"]
            
            # Pattern marker size based on strength
            size = 100 * strength
            # Color based on coherence
            color = plt.cm.viridis(coherence)
            
            ax.scatter(pos[0], pos[1], s=size, c=[color], alpha=0.6,
                      edgecolor='white', linewidth=2)
        
        ax.set_title(title)
        return fig, ax
    
    def plot_flow_field(self,
                       field: np.ndarray,
                       flow_vectors: np.ndarray,
                       patterns: List[Dict[str, Any]]) -> Tuple[Figure, Axes]:
        """Plot flow field with vectors."""
        fig, ax = plt.subplots(figsize=self.config.plot_size, dpi=self.config.dpi)
        
        # Create grid for flow vectors
        y, x = np.mgrid[0:field.shape[0], 0:field.shape[1]]
        
        # Plot background field
        im = ax.imshow(field, cmap=self.config.flow_cmap, alpha=0.3)
        plt.colorbar(im, ax=ax, label='Field Potential')
        
        # Plot flow vectors
        ax.quiver(x, y, flow_vectors[..., 0], flow_vectors[..., 1],
                 scale=self.config.quiver_scale)
        
        # Add patterns
        for pattern in patterns:
            pos = pattern["context"]["position"]
            ax.scatter(pos[0], pos[1], c='red', s=100, zorder=5)
        
        ax.set_title("Pattern Flow Field")
        return fig, ax
    
    def plot_coherence_landscape(self,
                               field: np.ndarray,
                               patterns: List[Dict[str, Any]]) -> Tuple[Figure, Axes]:
        """Plot coherence landscape with contours."""
        fig, ax = plt.subplots(figsize=self.config.plot_size, dpi=self.config.dpi)
        
        # Create coherence field
        coherence_field = np.zeros_like(field)
        for pattern in patterns:
            pos = pattern["context"]["position"]
            coherence = pattern["metrics"]["coherence"]
            # Add Gaussian contribution
            y, x = np.ogrid[-pos[0]:field.shape[0]-pos[0],
                          -pos[1]:field.shape[1]-pos[1]]
            r2 = x*x + y*y
            coherence_field += coherence * np.exp(-r2 / (2.0 * 2.0**2))
        
        # Plot contours
        contours = ax.contour(coherence_field,
                            levels=self.config.contour_levels,
                            cmap=self.config.coherence_cmap)
        plt.colorbar(contours, ax=ax, label='Coherence')
        
        # Add patterns
        for pattern in patterns:
            pos = pattern["context"]["position"]
            coherence = pattern["metrics"]["coherence"]
            ax.scatter(pos[1], pos[0], c='red', s=100*coherence,
                      label=f'Pattern (c={coherence:.2f})')
        
        ax.set_title("Coherence Landscape")
        return fig, ax
    
    def plot_pattern_evolution(self,
                             field_states: List[np.ndarray],
                             pattern_states: List[List[Dict[str, Any]]]) -> Figure:
        """Create animation of pattern evolution."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=self.config.dpi)
        fig.suptitle("Pattern Evolution Analysis")
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        # Plot different aspects
        self.plot_field_state(field_states[-1], pattern_states[-1], 
                            title="Final Field State")
        self.plot_flow_field(field_states[-1], self._calculate_flow(field_states),
                           pattern_states[-1])
        self.plot_coherence_landscape(field_states[-1], pattern_states[-1])
        
        # Plot evolution metrics
        self._plot_evolution_metrics(axes_flat[3], pattern_states)
        
        plt.tight_layout()
        return fig
    
    def _calculate_flow(self, field_states: List[np.ndarray]) -> np.ndarray:
        """Calculate flow vectors from field evolution."""
        if len(field_states) < 2:
            return np.zeros((*field_states[0].shape, 2))
        
        # Calculate gradient between last two states
        grad_y, grad_x = np.gradient(field_states[-1] - field_states[-2])
        return np.stack([grad_x, grad_y], axis=-1)
    
    def _plot_evolution_metrics(self,
                              ax: Axes,
                              pattern_states: List[List[Dict[str, Any]]]) -> None:
        """Plot evolution of key metrics over time."""
        times = range(len(pattern_states))
        
        # Extract metrics
        coherence = [np.mean([p["metrics"]["coherence"] for p in state])
                    for state in pattern_states]
        energy = [np.mean([p["metrics"]["energy_state"] for p in state])
                 for state in pattern_states]
        
        # Plot metrics
        ax.plot(times, coherence, 'b-', label='Coherence')
        ax.plot(times, energy, 'r-', label='Energy')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Metric Value')
        ax.set_title('Evolution Metrics')
        ax.legend()

def visualize_test_results(field: np.ndarray,
                          patterns: List[Dict[str, Any]],
                          save_path: Optional[str] = None) -> None:
    """Convenience function to visualize test results."""
    visualizer = FieldVisualizer()
    fig = visualizer.plot_pattern_evolution([field], [patterns])
    
    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)
