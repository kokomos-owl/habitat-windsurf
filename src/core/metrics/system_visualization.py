"""System-centric visualization of pattern evolution and coherence metrics."""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import scipy.sparse as sparse
from scipy.spatial.distance import cdist

class VectorSpace(Enum):
    """Different vector spaces for pattern analysis."""
    COHERENCE = "coherence_space"
    EMERGENCE = "emergence_space"
    TEMPORAL = "temporal_space"

@dataclass
class PatternVector:
    """Represents a pattern in vector space."""
    pattern_id: str
    coordinates: np.ndarray
    velocity: np.ndarray
    coherence: float
    emergence_potential: float
    last_update: datetime

class SystemVisualizer:
    """System-centric visualization of pattern evolution."""
    
    def __init__(self, dimensions: int = 4):
        self.dimensions = dimensions
        self.pattern_vectors: Dict[str, PatternVector] = {}
        self.coherence_matrix = sparse.lil_matrix((0, 0))
        self.emergence_field = np.array([])
        self.temporal_grid = np.array([])
        
    def update_pattern_space(self, 
                           pattern_metrics: Dict[str, Dict[str, float]],
                           temporal_context: Dict[str, Any]) -> None:
        """Update the pattern vector space based on new metrics."""
        # Convert pattern metrics to vectors
        new_vectors = self._metrics_to_vectors(pattern_metrics)
        
        # Update existing vectors or add new ones
        for pattern_id, vector_data in new_vectors.items():
            if pattern_id in self.pattern_vectors:
                self._update_vector(pattern_id, vector_data)
            else:
                self.pattern_vectors[pattern_id] = vector_data
                
        # Update coherence matrix
        self._update_coherence_matrix()
        
        # Update emergence field
        self._update_emergence_field()
        
        # Update temporal grid
        self._update_temporal_grid(temporal_context)
    
    def _metrics_to_vectors(self, 
                          pattern_metrics: Dict[str, Dict[str, float]]) -> Dict[str, PatternVector]:
        """Convert pattern metrics to vectors in our space."""
        vectors = {}
        for pattern_id, metrics in pattern_metrics.items():
            # Create vector components from metrics
            components = [
                metrics.get('coherence', 0.0),
                metrics.get('success_rate', 0.0),
                metrics.get('stability', 0.0),
                metrics.get('emergence_potential', 0.0)
            ]
            
            # Pad or truncate to match dimensions
            while len(components) < self.dimensions:
                components.append(0.0)
            components = components[:self.dimensions]
            
            # Create velocity vector (rate of change)
            if pattern_id in self.pattern_vectors:
                old_coords = self.pattern_vectors[pattern_id].coordinates
                new_coords = np.array(components)
                velocity = new_coords - old_coords
            else:
                velocity = np.zeros(self.dimensions)
            
            vectors[pattern_id] = PatternVector(
                pattern_id=pattern_id,
                coordinates=np.array(components),
                velocity=velocity,
                coherence=metrics.get('coherence', 0.0),
                emergence_potential=metrics.get('emergence_potential', 0.0),
                last_update=datetime.now()
            )
            
        return vectors
    
    def _update_vector(self, pattern_id: str, new_vector: PatternVector) -> None:
        """Update an existing pattern vector."""
        old_vector = self.pattern_vectors[pattern_id]
        
        # Calculate new velocity
        time_delta = (new_vector.last_update - old_vector.last_update).total_seconds()
        if time_delta > 0:
            velocity = (new_vector.coordinates - old_vector.coordinates) / time_delta
        else:
            velocity = old_vector.velocity
            
        # Update vector with momentum
        momentum = 0.8
        new_vector.velocity = velocity * momentum + old_vector.velocity * (1 - momentum)
        self.pattern_vectors[pattern_id] = new_vector
    
    def _update_coherence_matrix(self) -> None:
        """Update the coherence matrix between patterns."""
        n_patterns = len(self.pattern_vectors)
        if n_patterns == 0:
            self.coherence_matrix = sparse.lil_matrix((0, 0))
            return
            
        # Create coordinate matrix
        coordinates = np.vstack([
            v.coordinates for v in self.pattern_vectors.values()
        ])
        
        # Calculate pairwise distances
        distances = cdist(coordinates, coordinates)
        
        # Convert distances to coherence values (inverse relationship)
        coherence = 1 / (1 + distances)
        
        # Sparsify the matrix (only keep significant relationships)
        threshold = 0.3
        coherence[coherence < threshold] = 0
        self.coherence_matrix = sparse.lil_matrix(coherence)
    
    def _update_emergence_field(self) -> None:
        """Update the emergence field based on pattern movements."""
        if not self.pattern_vectors:
            self.emergence_field = np.array([])
            return
            
        # Create grid for emergence field
        grid_size = 10
        grid = np.zeros((grid_size,) * self.dimensions)
        
        # Calculate emergence potential at each grid point
        for vector in self.pattern_vectors.values():
            # Normalize coordinates to grid space
            grid_coords = (vector.coordinates * (grid_size - 1)).astype(int)
            grid_coords = np.clip(grid_coords, 0, grid_size - 1)
            
            # Add emergence potential
            grid[tuple(grid_coords)] += vector.emergence_potential
            
            # Add velocity influence
            velocity_magnitude = np.linalg.norm(vector.velocity)
            if velocity_magnitude > 0:
                grid[tuple(grid_coords)] += velocity_magnitude * 0.5
                
        self.emergence_field = grid
    
    def _update_temporal_grid(self, temporal_context: Dict[str, Any]) -> None:
        """Update the temporal grid based on context."""
        if not temporal_context:
            self.temporal_grid = np.array([])
            return
            
        # Create temporal grid
        time_points = temporal_context.get('time_points', [])
        if not time_points:
            return
            
        # Convert time points to relative positions
        base_time = min(time_points)
        max_time = max(time_points)
        time_span = (max_time - base_time).total_seconds()
        
        if time_span == 0:
            return
            
        # Create normalized temporal positions
        positions = [
            ((t - base_time).total_seconds() / time_span)
            for t in time_points
        ]
        
        self.temporal_grid = np.array(positions)
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current state of the system visualization."""
        return {
            'pattern_space': {
                pattern_id: {
                    'coordinates': vector.coordinates.tolist(),
                    'velocity': vector.velocity.tolist(),
                    'coherence': vector.coherence,
                    'emergence_potential': vector.emergence_potential
                }
                for pattern_id, vector in self.pattern_vectors.items()
            },
            'coherence_matrix': self.coherence_matrix.toarray().tolist(),
            'emergence_field': self.emergence_field.tolist(),
            'temporal_grid': self.temporal_grid.tolist(),
            'system_metrics': self._calculate_system_metrics()
        }
    
    def _calculate_system_metrics(self) -> Dict[str, float]:
        """Calculate overall system metrics."""
        if not self.pattern_vectors:
            return {
                'global_coherence': 1.0,
                'emergence_intensity': 0.0,
                'pattern_stability': 1.0
            }
            
        # Calculate global coherence
        coherence_values = [v.coherence for v in self.pattern_vectors.values()]
        global_coherence = np.mean(coherence_values)
        
        # Calculate emergence intensity
        emergence_values = [v.emergence_potential for v in self.pattern_vectors.values()]
        emergence_intensity = np.mean(emergence_values)
        
        # Calculate pattern stability (inverse of average velocity)
        velocities = [np.linalg.norm(v.velocity) for v in self.pattern_vectors.values()]
        pattern_stability = 1 / (1 + np.mean(velocities))
        
        return {
            'global_coherence': float(global_coherence),
            'emergence_intensity': float(emergence_intensity),
            'pattern_stability': float(pattern_stability)
        }
    
    def detect_emergent_structures(self) -> List[Dict[str, Any]]:
        """Detect emergent structures in the pattern space."""
        if not self.pattern_vectors:
            return []
            
        structures = []
        
        # Find clusters of patterns with high coherence
        coherence_matrix = self.coherence_matrix.toarray()
        threshold = 0.7
        
        for i in range(len(coherence_matrix)):
            related_patterns = np.where(coherence_matrix[i] > threshold)[0]
            if len(related_patterns) > 1:  # At least one other pattern
                pattern_ids = list(self.pattern_vectors.keys())
                structure = {
                    'core_pattern': pattern_ids[i],
                    'related_patterns': [pattern_ids[j] for j in related_patterns if j != i],
                    'coherence': float(np.mean(coherence_matrix[i, related_patterns])),
                    'emergence_potential': float(np.mean([
                        self.pattern_vectors[pattern_ids[j]].emergence_potential
                        for j in related_patterns
                    ]))
                }
                structures.append(structure)
                
        return structures
