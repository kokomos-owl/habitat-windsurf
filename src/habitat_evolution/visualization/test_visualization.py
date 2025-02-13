"""Test-focused visualization toolset with Neo4j export capabilities."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Protocol
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from datetime import datetime
from neo4j import GraphDatabase

@dataclass
class TestVisualizationConfig:
    """Enhanced configuration for test visualization with Neo4j bridge."""
    # Core visualization settings
    coherence_cmap: str = 'viridis'
    flow_cmap: str = 'coolwarm'
    density_cmap: str = 'YlOrRd'
    plot_size: Tuple[int, int] = (10, 8)
    dpi: int = 100
    
    # Neo4j settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_container: str = "cc7b03d4b96692134a32a67b1324fc9ec3d2319630de47e3e3af6d7e2da11e3f"
    
    # Climate-specific settings
    hazard_types: List[str] = field(default_factory=lambda: ['precipitation', 'drought', 'wildfire'])
    
    # Time periods and risk factors
    time_periods: Dict[str, int] = field(default_factory=lambda: {
        'current': 2025,
        'mid_century': 2050,
        'late_century': 2100
    })
    
    # Risk evolution factors by time period
    risk_factors: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'precipitation': {
            'current': 1.0,     # baseline
            'mid_century': 1.2,  # slight increase
            'late_century': 5.0  # 5x more likely
        },
        'drought': {
            'current': 0.085,   # 8.5% baseline
            'mid_century': 0.13, # 13% probability
            'late_century': 0.26 # 26% probability
        },
        'wildfire': {
            'current': 1.0,     # baseline
            'mid_century': 1.44, # 44% increase
            'late_century': 1.94 # 94% increase
        }
    })
    
    # Late-century thresholds for visualization
    hazard_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'precipitation': 7.34,  # inches (historical 100-year event * 5.0)
        'drought': 0.26,       # probability (late-century)
        'wildfire': 0.94       # increase factor (94% increase)
    })
    
    # Visualization parameters
    contour_levels: int = 20
    quiver_scale: float = 50
    pattern_marker_scale: float = 100

class PatternVisualizer(Protocol):
    """Protocol for pattern visualization implementations."""
    def visualize_field_state(self, field: np.ndarray, patterns: List[Dict[str, Any]]) -> Any: ...
    def visualize_flow(self, vectors: np.ndarray) -> Any: ...
    def visualize_coherence(self, field: np.ndarray) -> Any: ...

class TestPatternVisualizer:
    """Test-focused pattern visualization with Neo4j export capability."""
    
    def __init__(self, config: Optional[TestVisualizationConfig] = None):
        self.config = config or TestVisualizationConfig()
        self.test_results: List[Dict[str, Any]] = []
        sns.set_style('whitegrid')
        
        # Initialize Neo4j connection
        self._neo4j_driver = None
    
    def capture_test_state(self, 
                          test_name: str,
                          field: np.ndarray,
                          patterns: List[Dict[str, Any]],
                          metrics: Dict[str, float]) -> None:
        """Capture test state for visualization and Neo4j export."""
        state = {
            'test_name': test_name,
            'timestamp': datetime.now(),
            'field_state': field.copy(),
            'patterns': patterns.copy(),
            'metrics': metrics.copy()
        }
        self.test_results.append(state)
    
    def visualize_field_state(self, field: np.ndarray, patterns: List[Dict[str, Any]]) -> Figure:
        """Create visualization for field state with patterns.
        
        Args:
            field (np.ndarray): Field state
            patterns (list): List of pattern dictionaries
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        fig, ax = plt.subplots(figsize=self.config.plot_size)
        
        # Plot field as heatmap
        im = ax.imshow(field, cmap=self.config.density_cmap, origin='lower')
        plt.colorbar(im, ax=ax, label='Field intensity')
        
        # Plot pattern positions
        for pattern in patterns:
            pos = pattern['position']
            energy = pattern['metrics']['energy_state']
            hazard_type = pattern.get('hazard_type', 'unknown')
            ax.scatter(pos[0], pos[1], 
                      s=self.config.pattern_marker_scale * energy,
                      alpha=0.6,
                      label=f'{hazard_type} (E={energy:.2f})')
            
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.tight_layout()
        
        return fig
        
    def visualize_climate_patterns(self,
                                 field: np.ndarray,
                                 patterns: List[Dict[str, Any]],
                                 hazard_type: str) -> Tuple[Figure, Dict[str, float]]:
        """Visualize climate patterns with hazard-specific settings."""
        if hazard_type not in self.config.hazard_types:
            raise ValueError(f"Unsupported hazard type: {hazard_type}")
            
        threshold = self.config.hazard_thresholds[hazard_type]
        
        # Create figure with multiple views
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f"{hazard_type.title()} Pattern Analysis")
        
        # Generate visualizations
        gs = plt.GridSpec(2, 2, figure=fig)
        
        # Field state
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_field_state(ax1, field, patterns, hazard_type)
        
        # Flow field
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_flow_field(ax2, field, patterns)
        
        # Coherence landscape
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_coherence_landscape(ax3, field, patterns)
        
        # Metrics
        ax4 = fig.add_subplot(gs[1, 1])
        metrics = self._plot_hazard_metrics(ax4, field, patterns, threshold)
        
        plt.tight_layout()
        return fig, metrics
    
    def visualize_pattern_graph(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], field: np.ndarray) -> Figure:
        """Create graph visualization of pattern relationships.
        
        Args:
            nodes (List[Dict]): List of pattern nodes with embedded data
            edges (List[Dict]): List of edges representing pattern relationships
            field (np.ndarray): Underlying field state for reference
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot 1: Graph representation
        pos = {node['id']: node['position'] for node in nodes}
        
        # Draw nodes
        node_colors = [node['metrics']['coherence'] for node in nodes]
        node_sizes = [node['metrics']['energy_state'] * 1000 for node in nodes]
        nx.draw_networkx_nodes(G=nx.Graph(),
                             pos=pos,
                             node_color=node_colors,
                             node_size=node_sizes,
                             cmap=self.config.coherence_cmap,
                             ax=ax1)
        
        # Draw edges
        edge_weights = [edge['metrics']['combined_strength'] * 2 for edge in edges]
        edge_list = [(edge['source'], edge['target']) for edge in edges]
        nx.draw_networkx_edges(G=nx.Graph(edge_list),
                             pos=pos,
                             width=edge_weights,
                             alpha=0.6,
                             ax=ax1)
        
        # Add node labels
        labels = {node['id']: f"{node['embedded_data']['hazard_type']}\n{node['metrics']['energy_state']:.2f}"
                 for node in nodes}
        nx.draw_networkx_labels(G=nx.Graph(),
                              pos=pos,
                              labels=labels,
                              font_size=8,
                              ax=ax1)
        
        ax1.set_title('Pattern Interaction Graph')
        
        # Plot 2: Field state with pattern overlay
        im = ax2.imshow(field, cmap=self.config.density_cmap, origin='lower')
        plt.colorbar(im, ax=ax2, label='Field intensity')
        
        # Overlay patterns
        for node in nodes:
            pos = node['position']
            energy = node['metrics']['energy_state']
            coherence = node['metrics']['coherence']
            ax2.scatter(pos[0], pos[1],
                       s=energy * 500,
                       c=[coherence],
                       cmap=self.config.coherence_cmap,
                       alpha=0.6)
            
        # Draw edges as lines
        for edge in edges:
            node1 = next(n for n in nodes if n['id'] == edge['source'])
            node2 = next(n for n in nodes if n['id'] == edge['target'])
            ax2.plot([node1['position'][0], node2['position'][0]],
                     [node1['position'][1], node2['position'][1]],
                     'w--', alpha=0.3,
                     linewidth=edge['metrics']['combined_strength'])
        
        ax2.set_title('Field State with Pattern Overlay')
        
        plt.tight_layout()
        return fig
    
    def export_to_neo4j(self, test_name: str) -> None:
        """Export test results to Neo4j for further analysis."""
        if not self._neo4j_driver:
            self._connect_to_neo4j()
            
        relevant_results = [
            result for result in self.test_results 
            if result['test_name'] == test_name
        ]
        
        with self._neo4j_driver.session() as session:
            # Create test state nodes
            for result in relevant_results:
                session.run("""
                    CREATE (t:TestState {
                        test_name: $test_name,
                        timestamp: $timestamp,
                        metrics: $metrics
                    })
                """, {
                    'test_name': result['test_name'],
                    'timestamp': result['timestamp'].isoformat(),
                    'metrics': result['metrics']
                })
            
            # Create relationships between consecutive states
            for i in range(len(relevant_results) - 1):
                session.run("""
                    MATCH (t1:TestState), (t2:TestState)
                    WHERE t1.timestamp = $timestamp1 AND t2.timestamp = $timestamp2
                    CREATE (t1)-[:EVOLVES_TO {delta_time: $delta_time}]->(t2)
                """, {
                    'timestamp1': relevant_results[i]['timestamp'].isoformat(),
                    'timestamp2': relevant_results[i + 1]['timestamp'].isoformat(),
                    'delta_time': str(relevant_results[i + 1]['timestamp'] - 
                                    relevant_results[i]['timestamp'])
                })
    
    def _plot_field_state(self,
                         ax: Axes,
                         field: np.ndarray,
                         patterns: List[Dict[str, Any]],
                         hazard_type: str) -> None:
        """Plot current field state with patterns."""
        im = ax.imshow(field, 
                      cmap=self.config.coherence_cmap,
                      interpolation='gaussian')
        plt.colorbar(im, ax=ax, label=f'{hazard_type.title()} Intensity')
        
        # Plot patterns
        for pattern in patterns:
            pos = pattern["position"]
            strength = pattern["metrics"]["energy_state"]
            coherence = pattern["metrics"]["coherence"]
            
            size = self.config.pattern_marker_scale * strength
            color = plt.cm.viridis(coherence)
            
            ax.scatter(pos[0], pos[1], s=size, c=[color], alpha=0.6,
                      edgecolor='white', linewidth=2)
        
        ax.set_title(f"{hazard_type.title()} Field State")
    
    def _plot_flow_field(self,
                        ax: Axes,
                        field: np.ndarray,
                        patterns: List[Dict[str, Any]]) -> None:
        """Plot flow field with vectors."""
        # Calculate gradients for flow
        grad_y, grad_x = np.gradient(field)
        flow_vectors = np.stack([grad_x, grad_y], axis=-1)
        
        # Create grid for flow vectors
        y, x = np.mgrid[0:field.shape[0], 0:field.shape[1]]
        
        # Plot background field
        im = ax.imshow(field, cmap=self.config.flow_cmap, alpha=0.3)
        plt.colorbar(im, ax=ax, label='Flow Potential')
        
        # Plot flow vectors
        ax.quiver(x, y, flow_vectors[..., 0], flow_vectors[..., 1],
                 scale=self.config.quiver_scale)
        
        # Add patterns
        for pattern in patterns:
            pos = pattern["position"]
            ax.scatter(pos[0], pos[1], c='red', s=100, zorder=5)
        
        ax.set_title("Pattern Flow Field")
    
    def _plot_coherence_landscape(self,
                                ax: Axes,
                                field: np.ndarray,
                                patterns: List[Dict[str, Any]]) -> None:
        """Plot coherence landscape with contours."""
        coherence_field = np.zeros_like(field)
        for pattern in patterns:
            pos = pattern["position"]
            coherence = pattern["metrics"]["coherence"]
            # Add Gaussian contribution
            y, x = np.ogrid[-pos[0]:field.shape[0]-pos[0],
                          -pos[1]:field.shape[1]-pos[1]]
            r2 = x*x + y*y
            coherence_field += coherence * np.exp(-r2 / (2.0 * 2.0**2))
        
        contours = ax.contour(coherence_field,
                            levels=self.config.contour_levels,
                            cmap=self.config.coherence_cmap)
        plt.colorbar(contours, ax=ax, label='Coherence')
        
        # Add patterns
        for pattern in patterns:
            pos = pattern["position"]
            coherence = pattern["metrics"]["coherence"]
            ax.scatter(pos[1], pos[0], c='red', s=100*coherence,
                      label=f'Pattern (c={coherence:.2f})')
        
        ax.set_title("Coherence Landscape")
    
    def _plot_hazard_metrics(self,
                           ax: Axes,
                           field: np.ndarray,
                           patterns: List[Dict[str, Any]],
                           threshold: float) -> Dict[str, float]:
        """Plot hazard-specific metrics and return calculated values."""
        metrics = self._calculate_hazard_metrics(field, patterns, threshold)
        
        # Plot metrics as bar chart
        x = range(len(metrics))
        ax.bar(x, list(metrics.values()))
        ax.set_xticks(x)
        ax.set_xticklabels(list(metrics.keys()), rotation=45)
        ax.set_title("Hazard Metrics")
        
        return metrics
    
    def _calculate_hazard_metrics(self,
                                field: np.ndarray,
                                patterns: List[Dict[str, Any]],
                                threshold: float) -> Dict[str, float]:
        """Calculate hazard-specific metrics."""
        return {
            'coherence': np.mean([p['metrics']['coherence'] for p in patterns]),
            'energy': np.mean([p['metrics']['energy_state'] for p in patterns]),
            'above_threshold': np.sum(field > threshold) / field.size,
            'max_intensity': np.max(field)
        }
    
    def _connect_to_neo4j(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self._neo4j_driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=("neo4j", "password")  # Should be configured securely
            )
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self._neo4j_driver = None
    
    def __del__(self):
        """Clean up Neo4j connection."""
        if self._neo4j_driver:
            self._neo4j_driver.close()
