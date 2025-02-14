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

from .pattern_id import PatternAdaptiveID

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
    neo4j_uri: str = "bolt://localhost:7476"
    neo4j_container: str = "33e58c02bb2d"
    
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
    
    def visualize_pattern_graph(self, patterns: List[PatternAdaptiveID], field: np.ndarray) -> Figure:
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
        pos = {}
        node_colors = []
        node_sizes = []
        edge_list = []
        edge_weights = []
        
        # Process patterns for visualization
        for pattern in patterns:
            data = pattern.to_dict()
            pos[pattern.id] = data['position']
            node_colors.append(data['coherence'])
            node_sizes.append(data['energy_state'] * 1000)
            
            # Process relationships
            for target_id, relationships in pattern.relationships.items():
                latest = sorted(relationships, key=lambda x: x['timestamp'])[-1]
                edge_list.append((pattern.id, target_id))
                edge_weights.append(latest['metrics']['combined_strength'] * 2)
        
        # Draw nodes
        nx.draw_networkx_nodes(G=nx.Graph(),
                             pos=pos,
                             node_color=node_colors,
                             node_size=node_sizes,
                             cmap=self.config.coherence_cmap,
                             ax=ax1)
        
        # Draw edges
        nx.draw_networkx_edges(G=nx.Graph(edge_list),
                             pos=pos,
                             width=edge_weights,
                             alpha=0.6,
                             ax=ax1)
        
        # Add node labels
        labels = {}
        for pattern in patterns:
            data = pattern.to_dict()
            labels[pattern.id] = f"{data['hazard_type']}\n{data['energy_state']:.2f}"
        nx.draw_networkx_labels(G=nx.Graph(),
                              pos=pos,
                              labels=labels,
                              font_size=8,
                              ax=ax1)
        
        ax1.set_title('Pattern Interaction Graph')
        
        # Plot 2: Field state with pattern overlay
        im = ax2.imshow(field, cmap=self.config.density_cmap, origin='lower')
        plt.colorbar(im, ax=ax2, label='Field intensity')
        
        # Overlay patterns with AdaptiveID data
        for pattern in patterns:
            node_data = pattern.to_dict()
            pos = node_data['position']
            energy = node_data['energy_state']
            coherence = node_data['coherence']
            ax2.scatter(pos[0], pos[1],
                       s=energy * 500,
                       c=[coherence],
                       cmap=self.config.coherence_cmap,
                       alpha=0.6)
            
        # Draw edges from AdaptiveID relationships
        for pattern in patterns:
            for target_id, relationships in pattern.relationships.items():
                # Use the latest relationship
                latest = sorted(relationships, key=lambda x: x['timestamp'])[-1]
                source_pos = pattern.spatial_context['position']
                target_pattern = next(p for p in patterns if p.id == target_id)
                target_pos = target_pattern.spatial_context['position']
                ax2.plot([source_pos[0], target_pos[0]],
                         [source_pos[1], target_pos[1]],
                         'w--', alpha=0.3,
                         linewidth=latest['metrics']['combined_strength'])
        
        ax2.set_title('Field State with Pattern Overlay')
        
        plt.tight_layout()
        return fig
    
    def create_pattern_id(self, pattern_type: str, hazard_type: str, position: tuple,
                         field_state: float, coherence: float, energy_state: float) -> PatternAdaptiveID:
        """Create a new PatternAdaptiveID instance for a pattern.
        
        Args:
            pattern_type: Type of pattern (core, satellite)
            hazard_type: Type of hazard (precipitation, drought, etc)
            position: (x, y) position in field
            field_state: Current field state value
            coherence: Pattern coherence value
            energy_state: Pattern energy state
            
        Returns:
            PatternAdaptiveID instance
        """
        pattern_id = PatternAdaptiveID(pattern_type, hazard_type)
        pattern_id.update_metrics(position, field_state, coherence, energy_state)
        return pattern_id
    
    def export_pattern_graph_to_neo4j(self, patterns: List[PatternAdaptiveID], field: np.ndarray) -> None:
        """Export pattern graph to Neo4j with rich data embedding using AdaptiveIDs.
        
        Args:
            patterns: List of PatternAdaptiveID instances
            field: The underlying field state
        """
        if not self._neo4j_driver:
            self._connect_to_neo4j()
            
        with self._neo4j_driver.session() as session:
            # Clear existing pattern graph
            session.run("""
                MATCH (n:Pattern)
                DETACH DELETE n
            """)
            
            # Create pattern nodes with embedded data
            for pattern in patterns:
                node_data = pattern.to_dict()
                session.run("""
                    CREATE (p:Pattern {
                        id: $id,
                        pattern_type: $pattern_type,
                        hazard_type: $hazard_type,
                        position_x: $pos_x,
                        position_y: $pos_y,
                        field_state: $field_state,
                        coherence: $coherence,
                        energy_state: $energy_state,
                        weight: $weight,
                        confidence: $confidence,
                        version_id: $version_id,
                        created_at: $created_at,
                        last_modified: $last_modified
                    })
                """, {
                    'id': node_data['id'],
                    'pattern_type': node_data['pattern_type'],
                    'hazard_type': node_data['hazard_type'],
                    'pos_x': float(node_data['position'][0]),
                    'pos_y': float(node_data['position'][1]),
                    'field_state': float(node_data['field_state']),
                    'coherence': float(node_data['coherence']),
                    'energy_state': float(node_data['energy_state']),
                    'weight': float(node_data['weight']),
                    'confidence': float(node_data['confidence']),
                    'version_id': node_data['version_id'],
                    'created_at': node_data['created_at'],
                    'last_modified': node_data['last_modified']
                })
            
            # Create pattern relationships from AdaptiveID relationships
            processed_pairs = set()
            for pattern in patterns:
                for target_id, relationships in pattern.relationships.items():
                    # Skip if we've already processed this pair
                    pair = tuple(sorted([pattern.id, target_id]))
                    if pair in processed_pairs:
                        continue
                    processed_pairs.add(pair)
                    
                    # Use the latest relationship
                    latest = sorted(relationships, key=lambda x: x['timestamp'])[-1]
                    session.run("""
                        MATCH (p1:Pattern {id: $source}), (p2:Pattern {id: $target})
                        CREATE (p1)-[:INTERACTS_WITH {
                            type: $type,
                            spatial_distance: $distance,
                            coherence_similarity: $similarity,
                            combined_strength: $strength,
                            timestamp: $timestamp
                        }]->(p2)
                    """, {
                        'source': pattern.id,
                        'target': target_id,
                        'type': latest['type'],
                        'distance': float(latest['metrics']['spatial_distance']),
                        'similarity': float(latest['metrics']['coherence_similarity']),
                        'strength': float(latest['metrics']['combined_strength']),
                        'timestamp': latest['timestamp']
                    })
            
            # Add field context with fixed timestamp
            timestamp = datetime.now().isoformat()
            session.run("""
                CREATE (f:FieldState {
                    timestamp: $timestamp,
                    dimensions: $dimensions,
                    mean_intensity: $mean_intensity,
                    max_intensity: $max_intensity,
                    min_intensity: $min_intensity
                })
            """, {
                'timestamp': timestamp,
                'dimensions': field.shape,
                'mean_intensity': float(np.mean(field)),
                'max_intensity': float(np.max(field)),
                'min_intensity': float(np.min(field))
            })
            
            # Link patterns to field state
            session.run("""
                MATCH (p:Pattern), (f:FieldState)
                CREATE (p)-[:EXISTS_IN]->(f)
            """)
    
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
