"""Anomaly detection for pattern evolution and coherence metrics."""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import numpy as np
from enum import Enum

class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    COHERENCE_BREAK = "coherence_break"         # Sudden loss of pattern coherence
    RAPID_EMERGENCE = "rapid_emergence"         # Unusually fast pattern emergence
    PATTERN_COLLAPSE = "pattern_collapse"       # Pattern degradation
    TEMPORAL_DISCORD = "temporal_discord"       # Temporal inconsistency
    STRUCTURAL_SHIFT = "structural_shift"       # Major change in pattern relationships

@dataclass
class AnomalySignal:
    """Represents a detected anomaly in the system."""
    anomaly_type: AnomalyType
    severity: float  # 0.0 to 1.0
    timestamp: datetime
    affected_patterns: List[str]
    vector_space_coordinates: Optional[np.ndarray] = None
    context: Dict[str, Any] = None
    
    @property
    def is_critical(self) -> bool:
        """Determine if anomaly requires immediate attention."""
        return self.severity > 0.8

class AnomalyDetector:
    """Detects anomalies in pattern evolution and system coherence."""
    
    def __init__(self):
        self.history_window = 100
        self.pattern_history: Dict[str, List[Dict[str, Any]]] = {}
        self.anomaly_history: List[AnomalySignal] = []
        
        # Vector space configuration
        self.dimensions = {
            'coherence': {'weight': 2.0, 'threshold': 0.7, 'break_threshold': 0.4},
            'emergence': {'weight': 1.5, 'threshold': 0.6, 'rapid_threshold': 0.8},
            'stability': {'weight': 1.0, 'threshold': 0.8, 'collapse_threshold': 0.3},
            'temporal': {'weight': 1.0, 'threshold': 0.5, 'discord_threshold': 0.6}
        }
        
        # Flow field parameters
        self.field_resolution = 0.1
        self.attractor_radius = 0.2
        
        # Anomaly detection thresholds
        self.coherence_window = 5  # Window for coherence break detection
        self.collapse_window = 10   # Window for pattern collapse detection
        
        # Anomaly detection thresholds
        self.thresholds = {
            'vector_magnitude': 0.3,    # Significant movement in vector space
            'attractor_strength': 0.6,  # Strong attractor formation
            'field_divergence': 0.4,    # Flow field instability
            'topology_change': 0.25     # Changes in flow field topology
        }
    
    def detect_anomalies(self, 
                        current_state: Dict[str, Any],
                        pattern_vectors: Dict[str, Any],
                        coherence_matrix: np.ndarray) -> List[AnomalySignal]:
        """Detect anomalies in the current system state."""
        anomalies = []
        timestamp = datetime.now()
        
        # Update pattern history
        self._update_history(current_state, pattern_vectors)
        
        # Check for coherence breaks
        coherence_anomalies = self._detect_coherence_breaks(pattern_vectors)
        anomalies.extend(coherence_anomalies)
        
        # Check for rapid emergence
        emergence_anomalies = self._detect_rapid_emergence(pattern_vectors)
        anomalies.extend(emergence_anomalies)
        
        # Check for pattern collapse
        collapse_anomalies = self._detect_pattern_collapse(pattern_vectors)
        anomalies.extend(collapse_anomalies)
        
        # Check for temporal discord
        temporal_anomalies = self._detect_temporal_discord(current_state)
        anomalies.extend(temporal_anomalies)
        
        # Check for structural shifts
        structural_anomalies = self._detect_structural_shifts(coherence_matrix)
        anomalies.extend(structural_anomalies)
        
        # Update anomaly history
        self.anomaly_history.extend(anomalies)
        self._prune_history()
        
        return anomalies
    
    def _update_history(self, 
                       current_state: Dict[str, Any],
                       pattern_vectors: Dict[str, Any]) -> None:
        """Update pattern history with current state."""
        timestamp = datetime.now()
        
        for pattern_id, vector in pattern_vectors.items():
            if pattern_id not in self.pattern_history:
                self.pattern_history[pattern_id] = []
                
            self.pattern_history[pattern_id].append({
                'timestamp': timestamp,
                'vector': vector,
                'state': current_state.get('system_metrics', {})
            })
    
    def _calculate_vector_position(self, metrics: Dict[str, float]) -> np.ndarray:
        """Calculate position in vector space."""
        position = np.zeros(len(self.dimensions))
        
        for i, (dim, params) in enumerate(self.dimensions.items()):
            if dim == 'coherence':
                position[i] = metrics.get('coherence', 0) * params['weight']
            elif dim == 'emergence':
                position[i] = metrics.get('emergence_potential', 0) * params['weight']
            elif dim == 'stability':
                position[i] = metrics.get('temporal_stability', 0) * params['weight']
            elif dim == 'temporal':
                position[i] = (1 - metrics.get('temporal_variance', 0)) * params['weight']
        
        return position
    
    def _detect_coherence_breaks(self, 
                               pattern_vectors: Dict[str, Any]) -> List[AnomalySignal]:
        """Detect coherence breaks using vector space analysis."""
        anomalies = []
        
        for pattern_id, history in self.pattern_history.items():
            if len(history) < self.coherence_window:
                continue
                
            # Get coherence value directly from vector
            coherence = pattern_vectors[pattern_id].get('coherence', 0.0)
            
            # Check for coherence break threshold
            if coherence < self.dimensions['coherence']['break_threshold']:
                severity = (self.dimensions['coherence']['threshold'] - coherence) / \
                          self.dimensions['coherence']['threshold']
                
                anomalies.append(AnomalySignal(
                    anomaly_type=AnomalyType.COHERENCE_BREAK,
                    severity=severity,
                    timestamp=datetime.now(),
                    affected_patterns=[pattern_id],
                    vector_space_coordinates=self._calculate_vector_position(pattern_vectors[pattern_id]),
                    context={
                        'coherence': float(coherence),
                        'threshold': float(self.dimensions['coherence']['threshold'])
                    }
                ))

        return anomalies
    
    def _detect_rapid_emergence(self, 
                              pattern_vectors: Dict[str, Any]) -> List[AnomalySignal]:
        """Detect unusually rapid pattern emergence."""
        anomalies = []
        
        for pattern_id, vector in pattern_vectors.items():
            # Calculate position in vector space
            position = self._calculate_vector_position(vector)
            
            # Check emergence characteristics
            emergence_dim = list(self.dimensions.keys()).index('emergence')
            emergence_strength = position[emergence_dim] / self.dimensions['emergence']['weight']
            
            # Calculate emergence velocity
            if 'velocity' in vector:
                velocity = np.array(vector['velocity'])
                speed = np.linalg.norm(velocity)
            else:
                speed = 0.0
            
            # Detect rapid emergence
            if emergence_strength > self.dimensions['emergence']['threshold'] or speed > self.field_resolution:
                # Calculate severity based on emergence and movement
                severity_factors = [
                    emergence_strength * 2.0,
                    speed / self.field_resolution if speed < self.field_resolution else 1.0
                ]
                severity = sum(severity_factors) / 3.0
                
                if severity > self.thresholds['vector_magnitude']:
                    anomalies.append(AnomalySignal(
                        anomaly_type=AnomalyType.RAPID_EMERGENCE,
                        severity=severity,
                        timestamp=datetime.now(),
                        affected_patterns=[pattern_id],
                        vector_space_coordinates=position,
                        context={
                            'emergence_strength': float(emergence_strength),
                            'velocity': float(speed),
                            'position': position.tolist()
                        }
                    ))
                
        return anomalies
    
    def _detect_pattern_collapse(self, 
                               pattern_vectors: Dict[str, Any]) -> List[AnomalySignal]:
        """Detect patterns that are collapsing using flow dynamics."""
        anomalies = []
        
        for pattern_id, vector in pattern_vectors.items():
            # Calculate collapse indicators
            velocity = np.linalg.norm(vector['velocity'])
            coherence = vector.get('coherence', 0.0)
            emergence = vector.get('emergence_potential', 0.0)
            
            # Consider multiple collapse indicators
            collapse_factors = [
                (1.0 - (1.0 - velocity)) * 2.0,  # Weight instability heavily
                (1.0 - coherence) * 1.5,  # Weight low coherence
                emergence * velocity * 0.5  # Rapid emergence can indicate instability
            ]
            
            # Calculate overall severity
            severity = min(sum(collapse_factors) / 4.0, 1.0)  # Normalize and cap at 1.0
            
            if severity > 0.7:  # Higher threshold for collapse detection
                anomalies.append(AnomalySignal(
                    anomaly_type=AnomalyType.PATTERN_COLLAPSE,
                    severity=severity,
                    timestamp=datetime.now(),
                    affected_patterns=[pattern_id],
                    vector_space_coordinates=self._calculate_vector_position(vector),
                    context={
                        'velocity': float(velocity),
                        'coherence': float(coherence),
                        'emergence': float(emergence),
                        'collapse_factors': [float(f) for f in collapse_factors]
                    }
                ))
                
        return anomalies
    
    def _detect_temporal_discord(self, 
                               current_state: Dict[str, Any]) -> List[AnomalySignal]:
        """Detect temporal inconsistencies in pattern evolution."""
        anomalies = []
        temporal_grid = current_state.get('temporal_grid', [])
        
        if not temporal_grid:
            return anomalies
            
        # Calculate temporal variance
        temporal_diffs = np.diff(temporal_grid)
        variance = np.var(temporal_diffs) if len(temporal_diffs) > 0 else 0
        
        if variance > self.thresholds['temporal_variance']:
            severity = min(variance / self.thresholds['temporal_variance'], 1.0)
            anomalies.append(AnomalySignal(
                anomaly_type=AnomalyType.TEMPORAL_DISCORD,
                severity=severity,
                timestamp=datetime.now(),
                affected_patterns=[],  # Affects whole system
                context={
                    'temporal_variance': variance,
                    'grid_points': len(temporal_grid)
                }
            ))
            
        return anomalies
    
    def _detect_structural_shifts(self, 
                                coherence_matrix: np.ndarray) -> List[AnomalySignal]:
        """Detect structural shifts using vector field topology analysis."""
        anomalies = []
        
        if len(self.anomaly_history) > 0:
            # Compare with previous coherence matrix
            previous_signals = [a for a in self.anomaly_history 
                              if a.anomaly_type == AnomalyType.STRUCTURAL_SHIFT]
            
            if previous_signals:
                previous_matrix = previous_signals[-1].context.get('coherence_matrix')
                if previous_matrix is not None:
                    # Calculate structural characteristics
                    current_positions = np.array([self._calculate_vector_position(v) for v in pattern_vectors.values()])
                    previous_positions = np.array([self._calculate_vector_position(h[-2]['vector']) 
                                                  for h in self.pattern_history.values() if len(h) > 1])
                    
                    if len(current_positions) == 0 or len(previous_positions) == 0:
                        return anomalies
                    
                    # Calculate centroid movement
                    current_centroid = np.mean(current_positions, axis=0)
                    previous_centroid = np.mean(previous_positions, axis=0)
                    centroid_shift = np.linalg.norm(current_centroid - previous_centroid)
                    
                    # Calculate dispersion change
                    current_dispersion = np.mean([np.linalg.norm(p - current_centroid) for p in current_positions])
                    previous_dispersion = np.mean([np.linalg.norm(p - previous_centroid) for p in previous_positions])
                    dispersion_change = abs(current_dispersion - previous_dispersion)
                    
                    # Calculate structural shift severity
                    severity = min((
                        centroid_shift / self.field_resolution + 
                        dispersion_change / self.attractor_radius
                    ) / 2.0, 1.0)
                    
                    if severity > self.thresholds['topology_change']:
                        anomalies.append(AnomalySignal(
                            anomaly_type=AnomalyType.STRUCTURAL_SHIFT,
                            severity=severity,
                            timestamp=datetime.now(),
                            affected_patterns=list(pattern_vectors.keys()),
                            context={
                                'centroid_shift': float(centroid_shift),
                                'dispersion_change': float(dispersion_change),
                                'current_centroid': current_centroid.tolist(),
                                'previous_centroid': previous_centroid.tolist()
                            }
                        ))
                        
        return anomalies
        
    def _calculate_field_topology(self, matrix: np.ndarray) -> np.ndarray:
        """Calculate the topological structure of the vector field."""
        # Create a grid for the vector field
        x, y = np.meshgrid(np.linspace(0, 1, matrix.shape[0]), 
                          np.linspace(0, 1, matrix.shape[1]))
        
        # Calculate field gradients
        dx, dy = np.gradient(matrix)
        
        # Calculate field topology characteristics
        magnitude = np.sqrt(dx**2 + dy**2)
        direction = np.arctan2(dy, dx)
        
        # Combine into topology tensor
        topology = np.stack([magnitude, direction], axis=-1)
        
        return topology
        
    def _detect_field_singularities(self, topology: np.ndarray) -> List[tuple]:
        """Detect singularities (critical points) in the vector field."""
        singularities = []
        
        # Look for points where magnitude is near zero but surrounding points aren't
        magnitude = topology[..., 0]
        threshold = np.mean(magnitude) * 0.1
        
        for i in range(1, magnitude.shape[0]-1):
            for j in range(1, magnitude.shape[1]-1):
                if magnitude[i,j] < threshold:
                    # Check if surrounding points have higher magnitude
                    neighborhood = magnitude[i-1:i+2, j-1:j+2]
                    if np.any(neighborhood > threshold):
                        singularities.append((i,j))
                        
        return singularities
                        
        return anomalies
    
    def _prune_history(self) -> None:
        """Prune old entries from history."""
        # Prune pattern history
        for pattern_id in self.pattern_history:
            if len(self.pattern_history[pattern_id]) > self.history_window:
                self.pattern_history[pattern_id] = self.pattern_history[pattern_id][-self.history_window:]
                
        # Prune anomaly history
        if len(self.anomaly_history) > self.history_window:
            self.anomaly_history = self.anomaly_history[-self.history_window:]
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of recent anomalies for both system and human consumption."""
        if not self.anomaly_history:
            return {
                'system_view': {'anomaly_count': 0, 'critical_count': 0},
                'human_view': 'System operating normally with no detected anomalies.'
            }
            
        recent_anomalies = self.anomaly_history[-10:]  # Last 10 anomalies
        critical_anomalies = [a for a in recent_anomalies if a.is_critical]
        
        return {
            'system_view': {
                'anomaly_count': len(recent_anomalies),
                'critical_count': len(critical_anomalies),
                'anomaly_types': {
                    atype.value: len([a for a in recent_anomalies if a.anomaly_type == atype])
                    for atype in AnomalyType
                },
                'severity_distribution': {
                    'high': len([a for a in recent_anomalies if a.severity > 0.8]),
                    'medium': len([a for a in recent_anomalies if 0.5 <= a.severity <= 0.8]),
                    'low': len([a for a in recent_anomalies if a.severity < 0.5])
                }
            },
            'human_view': self._generate_human_readable_summary(recent_anomalies)
        }
    
    def _generate_human_readable_summary(self, anomalies: List[AnomalySignal]) -> str:
        """Generate a human-readable summary of anomalies."""
        if not anomalies:
            return "No anomalies detected."
            
        critical_count = len([a for a in anomalies if a.is_critical])
        summary_parts = []
        
        if critical_count > 0:
            summary_parts.append(f"⚠️ {critical_count} critical anomalies detected!")
            
        type_counts = {}
        for anomaly in anomalies:
            type_counts[anomaly.anomaly_type] = type_counts.get(anomaly.anomaly_type, 0) + 1
            
        summary_parts.append("Recent anomalies:")
        for atype, count in type_counts.items():
            summary_parts.append(f"- {atype.value}: {count} occurrences")
            
        return "\n".join(summary_parts)
