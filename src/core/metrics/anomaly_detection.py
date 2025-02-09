"""Anomaly detection for pattern evolution and coherence metrics."""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
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
        
        # Thresholds for different types of anomalies
        self.thresholds = {
            'coherence_delta': 0.2,      # Sudden coherence change
            'emergence_rate': 0.4,       # Rate of emergence
            'pattern_stability': 0.25,   # Minimum stability
            'temporal_variance': 0.5,    # Maximum temporal variance
            'structural_delta': 0.25     # Structural change threshold
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
    
    def _detect_coherence_breaks(self, 
                               pattern_vectors: Dict[str, Any]) -> List[AnomalySignal]:
        """Detect sudden breaks in pattern coherence."""
        anomalies = []
        
        for pattern_id, history in self.pattern_history.items():
            if len(history) < 2:
                continue
                
            # Calculate coherence delta
            current_coherence = pattern_vectors[pattern_id]['coherence']
            previous_coherence = history[-2]['vector']['coherence']
            coherence_delta = previous_coherence - current_coherence  # Check for drops specifically
            
            if coherence_delta > self.thresholds['coherence_delta']:
                severity = min(coherence_delta / 0.5, 1.0)  # Normalize to 0-1
                anomalies.append(AnomalySignal(
                    anomaly_type=AnomalyType.COHERENCE_BREAK,
                    severity=severity,
                    timestamp=datetime.now(),
                    affected_patterns=[pattern_id],
                    vector_space_coordinates=np.array(pattern_vectors[pattern_id]['coordinates']),
                    context={
                        'previous_coherence': previous_coherence,
                        'current_coherence': current_coherence,
                        'delta': coherence_delta
                    }
                ))
                
        return anomalies
    
    def _detect_rapid_emergence(self, 
                              pattern_vectors: Dict[str, Any]) -> List[AnomalySignal]:
        """Detect unusually rapid pattern emergence."""
        anomalies = []
        
        for pattern_id, vector in pattern_vectors.items():
            emergence_rate = vector['emergence_potential']
            velocity_magnitude = np.linalg.norm(vector['velocity'])
            
            if emergence_rate > self.thresholds['emergence_rate'] or velocity_magnitude > 0.5:
                severity = min((emergence_rate * velocity_magnitude) / 0.5, 1.0)
                anomalies.append(AnomalySignal(
                    anomaly_type=AnomalyType.RAPID_EMERGENCE,
                    severity=severity,
                    timestamp=datetime.now(),
                    affected_patterns=[pattern_id],
                    vector_space_coordinates=np.array(vector['coordinates']),
                    context={
                        'emergence_rate': emergence_rate,
                        'velocity': velocity_magnitude
                    }
                ))
                
        return anomalies
    
    def _detect_pattern_collapse(self, 
                               pattern_vectors: Dict[str, Any]) -> List[AnomalySignal]:
        """Detect patterns that are collapsing or degrading."""
        anomalies = []
        
        for pattern_id, vector in pattern_vectors.items():
            stability = 1.0 - np.linalg.norm(vector['velocity'])  # High velocity = low stability
            coherence = vector['coherence']
            emergence_potential = vector['emergence_potential']
            
            if stability < self.thresholds['pattern_stability'] or (coherence < 0.5 and emergence_potential > 0.7):
                severity = min((1.0 - stability) * (1.0 - coherence), 1.0)
                anomalies.append(AnomalySignal(
                    anomaly_type=AnomalyType.PATTERN_COLLAPSE,
                    severity=severity,
                    timestamp=datetime.now(),
                    affected_patterns=[pattern_id],
                    vector_space_coordinates=np.array(vector['coordinates']),
                    context={
                        'stability': stability,
                        'coherence': coherence
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
        """Detect significant shifts in pattern relationships."""
        anomalies = []
        
        if len(self.anomaly_history) > 0:
            # Compare with previous coherence matrix
            previous_signals = [a for a in self.anomaly_history 
                              if a.anomaly_type == AnomalyType.STRUCTURAL_SHIFT]
            
            if previous_signals:
                previous_matrix = previous_signals[-1].context.get('coherence_matrix')
                if previous_matrix is not None:
                    # Calculate structural change
                    matrix_diff = np.abs(coherence_matrix - previous_matrix)
                    max_diff = np.max(matrix_diff)
                    
                    if max_diff > self.thresholds['structural_delta']:
                        severity = min(max_diff / 0.5, 1.0)
                        anomalies.append(AnomalySignal(
                            anomaly_type=AnomalyType.STRUCTURAL_SHIFT,
                            severity=severity,
                            timestamp=datetime.now(),
                            affected_patterns=[],  # Affects whole system
                            context={
                                'matrix_difference': float(max_diff),
                                'coherence_matrix': coherence_matrix.copy()
                            }
                        ))
                        
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
