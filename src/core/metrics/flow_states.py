"""Flow state management for metric evolution."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import numpy as np

class FlowState(Enum):
    """States of a metric flow with transition awareness."""
    ACTIVE = "active"          # Normal operation, pattern is functioning as expected
    EMERGING = "emerging"      # Pattern showing signs of differentiation
    LEARNING = "learning"      # Pattern actively being improved
    STABLE = "stable"         # Pattern performing consistently well
    DEPRECATED = "deprecated"  # Pattern no longer in use

@dataclass
class FlowTransition:
    """Captures the transition between flow states."""
    from_state: FlowState
    to_state: FlowState
    timestamp: datetime
    trigger_metrics: Dict[str, float]
    confidence_delta: float

class FlowStateManager:
    """Manages flow states and transitions with emergence detection."""
    
    def __init__(self):
        self.emergence_threshold = 0.6  # Minimum emergence potential to trigger emergence
        self.emergence_window = timedelta(days=7)  # Time window for emergence detection
        self.stability_threshold = 0.85
        self.learning_threshold = 0.65
        
        # Vector space dimensions and their characteristics
        self.dimensions = {
            'coherence': {'weight': 2.0, 'attractor_threshold': 0.7},
            'emergence': {'weight': 1.5, 'attractor_threshold': 0.6},
            'stability': {'weight': 1.0, 'attractor_threshold': 0.8},
            'temporal': {'weight': 1.0, 'attractor_threshold': 0.5}
        }
        
        # Flow field parameters
        self.field_resolution = 0.1  # Grid resolution for flow field
        self.attractor_radius = 0.2  # Radius for attractor detection
        
        # Track metric history in vector space
        self.metric_history: List[Dict[str, float]] = []
        self.history_window = 100
        
        # Track state transitions
        self.transitions: list[FlowTransition] = []
    
    def assess_state(self, 
                    current_state: FlowState,
                    metrics: Dict[str, float],
                    pattern_stats: Dict[str, Any]) -> FlowState:
        """Assess if flow state should change based on metrics and pattern stats."""
        
        # Update metric history
        self._update_metric_history(metrics)
        
        # Calculate vector position
        position = self._calculate_vector_position(metrics)
        
        # Get emergence dimension index
        emergence_dim = list(self.dimensions.keys()).index('emergence')
        emergence_strength = position[emergence_dim] / self.dimensions['emergence']['weight']
        
        # Calculate state transition metrics
        stability_score = self._calculate_stability_score(metrics, pattern_stats)
        learning_need = self._calculate_learning_need(metrics, pattern_stats)
        
        # State transition logic using vector space analysis
        if current_state == FlowState.ACTIVE:
            if emergence_strength > self.dimensions['emergence']['attractor_threshold']:
                self._record_transition(current_state, FlowState.EMERGING, metrics, 0.0)
                return FlowState.EMERGING
                
        elif current_state == FlowState.EMERGING:
            # Prioritize learning transition when confidence is low
            confidence = metrics.get('confidence', 1.0)
            if confidence < self.learning_threshold:
                self._record_transition(current_state, FlowState.LEARNING, metrics, 0.0)
                return FlowState.LEARNING
            elif stability_score > self.stability_threshold:
                self._record_transition(current_state, FlowState.STABLE, metrics, 0.0)
                return FlowState.STABLE
                
        elif current_state == FlowState.LEARNING:
            if stability_score > self.stability_threshold:
                self._record_transition(current_state, FlowState.STABLE, metrics, 0.0)
                return FlowState.STABLE
            
        # From STABLE, check for deprecation
        elif current_state == FlowState.STABLE:
            if self._should_deprecate(metrics, pattern_stats):
                self._record_transition(current_state, FlowState.DEPRECATED, metrics, 0.0)
                return FlowState.DEPRECATED
            elif learning_need > self.learning_threshold:
                self._record_transition(current_state, FlowState.LEARNING, metrics, 0.0)
                return FlowState.LEARNING
                
        # Check for deprecation in any state
        if self._should_deprecate(metrics, pattern_stats):
            self._record_transition(current_state, FlowState.DEPRECATED, metrics, 0.0)
            return FlowState.DEPRECATED
            
        # Default transitions based on stability and learning needs
        if stability_score > self.stability_threshold:
            self._record_transition(current_state, FlowState.STABLE, metrics, 0.0)
            return FlowState.STABLE
            
        if learning_need > self.learning_threshold:
            self._record_transition(current_state, FlowState.LEARNING, metrics, 0.0)
            return FlowState.LEARNING
            
        return current_state
    
    def _update_metric_history(self, metrics: Dict[str, float]) -> None:
        """Update historical metrics."""
        # Add new metrics to history
        self.metric_history.append(metrics)
        
        # Keep only the last history_window entries
        if len(self.metric_history) > self.history_window:
            self.metric_history = self.metric_history[-self.history_window:]
    
    def _calculate_differentials(self) -> Dict[str, float]:
        """Calculate rate of change for each metric."""
        differentials = {}
        
        if len(self.metric_history) < 2:
            return differentials
            
        # Get current and previous metrics
        current = self.metric_history[-1]
        previous = self.metric_history[-2]
        
        # Calculate differentials for key metrics
        for metric in ['coherence', 'emergence_potential', 'temporal_stability', 'confidence']:
            current_val = current.get(metric, 0.0)
            prev_val = previous.get(metric, 0.0)
            differentials[metric] = current_val - prev_val
                
        return differentials

    
    def _calculate_stability_score(self, metrics: Dict[str, float], pattern_stats: Dict[str, Any]) -> float:
        """Calculate overall stability score using multiple metrics."""
        # Get relevant metrics
        temporal_stability = metrics.get('temporal_stability', 0.0)
        coherence = metrics.get('coherence', 0.0)
        confidence = metrics.get('confidence', 0.0)
        pattern_count = len(pattern_stats.get('active_patterns', []))
        
        # Calculate stability components
        temporal_weight = 2.0
        coherence_weight = 1.5
        confidence_weight = 1.0
        pattern_weight = 0.5
        
        stability_factors = [
            temporal_stability * temporal_weight,
            coherence * coherence_weight,
            confidence * confidence_weight,
            min(pattern_count / 10.0, 1.0) * pattern_weight  # Normalize pattern count
        ]
        
        return sum(stability_factors) / (temporal_weight + coherence_weight + confidence_weight + pattern_weight)
        
    def _calculate_learning_need(self, metrics: Dict[str, float], pattern_stats: Dict[str, Any]) -> float:
        """Calculate the need for learning based on pattern behavior."""
        # Get relevant metrics
        emergence_potential = metrics.get('emergence_potential', 0.0)
        pattern_matches = metrics.get('pattern_matches', 0)
        temporal_variance = metrics.get('temporal_variance', 0.0)
        
        learning_factors = [
            emergence_potential * 2.0,  # Weight emergence heavily
            (1.0 - pattern_matches/100.0) if pattern_matches > 0 else 1.0,  # Inverse of pattern matches
            temporal_variance
        ]
        
        return sum(learning_factors) / 4.0  # Normalize
        
    def _calculate_vector_position(self, metrics: Dict[str, float]) -> np.ndarray:
        """Calculate position in vector space using metric values."""
        position = np.zeros(len(self.dimensions))
        
        # Map metrics to dimensions with fallbacks
        metric_mapping = {
            'coherence': metrics.get('coherence', 0.0),
            'emergence': metrics.get('emergence_potential', metrics.get('confidence', 0.0)),
            'stability': metrics.get('stability', metrics.get('success_rate', 0.0)),
            'temporal': metrics.get('temporal_stability', 1.0 - metrics.get('temporal_variance', 0.0))
        }
        
        # Calculate weighted position
        for i, (dim, params) in enumerate(self.dimensions.items()):
            position[i] = metric_mapping.get(dim, 0.0) * params['weight']
                
        return position
    
    def _detect_emergence(self, differentials: Dict[str, float]) -> float:
        """Detect emergence using vector space analysis."""
        if len(self.metric_history) < 2:
            return 0.0
            
        # Calculate current and previous positions
        current_metrics = self.metric_history[-1]
        prev_metrics = self.metric_history[-2]
        
        current_pos = self._calculate_vector_position(current_metrics)
        prev_pos = self._calculate_vector_position(prev_metrics)
        
        # Calculate emergence characteristics
        emergence_dim = list(self.dimensions.keys()).index('emergence')
        emergence_strength = current_pos[emergence_dim] / self.dimensions['emergence']['weight']
        
        # Calculate flow characteristics
        velocity = current_pos - prev_pos
        speed = np.linalg.norm(velocity)
        
        # Weight emergence by both position and movement
        emergence_score = emergence_strength * (1.0 + min(speed / self.field_resolution, 1.0))
        
        return emergence_score



    
    def _is_pattern_stable(self, metrics: Dict[str, float], pattern_stats: Dict[str, Any]) -> bool:
        """Check if pattern has stabilized."""
        return (metrics['confidence'] > self.stability_threshold and
                metrics['temporal_stability'] > self.stability_threshold and
                pattern_stats.get('success_rate', 0) > self.stability_threshold)
    
    def _should_deprecate(self, metrics: Dict[str, float], pattern_stats: Dict[str, Any]) -> bool:
        """Determine if pattern should be deprecated."""
        # Consider deprecation if:
        # 1. Pattern has very low success rate
        # 2. Better patterns exist
        # 3. Pattern hasn't been used recently
        return (pattern_stats.get('success_rate', 0) < 0.2 or
                metrics['confidence'] < 0.2 or
                (pattern_stats.get('days_since_last_use', 0) > 30 and
                 pattern_stats.get('total_uses', 0) < 5))
    
    def _record_transition(self, 
                         from_state: FlowState,
                         to_state: FlowState,
                         metrics: Dict[str, float],
                         confidence_delta: float) -> None:
        """Record a state transition."""
        self.transitions.append(FlowTransition(
            from_state=from_state,
            to_state=to_state,
            timestamp=datetime.now(),
            trigger_metrics=metrics.copy(),
            confidence_delta=confidence_delta
        ))
    
    def get_transition_history(self) -> list[FlowTransition]:
        """Get history of state transitions."""
        return self.transitions.copy()
    
    def get_emergence_metrics(self) -> Dict[str, Any]:
        """Get metrics related to pattern emergence."""
        return {
            'current_differentials': self._calculate_differentials(),
            'emergence_detected': any(
                self._detect_emergence(self._calculate_differentials())
                for _ in range(1)  # Currently checking only once
            ),
            'metric_stability': {
                metric_type: self._calculate_stability(history)
                for metric_type, history in self.metric_history.items()
            }
        }
    
    def _calculate_stability(self, history: list) -> float:
        """Calculate stability score for a metric history."""
        if len(history) < 2:
            return 1.0
            
        # Calculate variance in changes
        changes = [abs(history[i][1] - history[i-1][1]) for i in range(1, len(history))]
        avg_change = sum(changes) / len(changes)
        return max(0, 1 - (avg_change / 0.5))  # 0.5 is max expected change
