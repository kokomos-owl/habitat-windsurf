"""Flow state management for metric evolution."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

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
        
        # Track metric differentials over time
        self.metric_history: Dict[str, list] = {
            'confidence': [],
            'temporal_stability': [],
            'pattern_matches': []
        }
        
        # Track state transitions
        self.transitions: list[FlowTransition] = []
    
    def assess_state(self, 
                    current_state: FlowState,
                    metrics: Dict[str, float],
                    pattern_stats: Dict[str, Any]) -> FlowState:
        """Assess if flow state should change based on metrics and pattern stats."""
        
        # Update metric history
        self._update_metric_history(metrics)
        
        # Calculate differentials
        differentials = self._calculate_differentials()
        
        # Check for emergence first
        if self._detect_emergence(differentials) and current_state == FlowState.ACTIVE:
            self._record_transition(current_state, FlowState.EMERGING, metrics, differentials['confidence'])
            return FlowState.EMERGING
            
        # From EMERGING, we can go to LEARNING or back to ACTIVE
        if current_state == FlowState.EMERGING:
            if metrics['confidence'] < self.learning_threshold:
                self._record_transition(current_state, FlowState.LEARNING, metrics, differentials['confidence'])
                return FlowState.LEARNING
            elif self._is_pattern_stable(metrics, pattern_stats):
                self._record_transition(current_state, FlowState.ACTIVE, metrics, differentials['confidence'])
                return FlowState.ACTIVE
                
        # Check other state transitions
        if metrics['confidence'] > self.stability_threshold and current_state in [FlowState.ACTIVE, FlowState.LEARNING]:
            self._record_transition(current_state, FlowState.STABLE, metrics, differentials['confidence'])
            return FlowState.STABLE
            
        if metrics['confidence'] < self.learning_threshold and current_state != FlowState.LEARNING:
            self._record_transition(current_state, FlowState.LEARNING, metrics, differentials['confidence'])
            return FlowState.LEARNING
            
        # Check for deprecation
        if self._should_deprecate(metrics, pattern_stats):
            self._record_transition(current_state, FlowState.DEPRECATED, metrics, differentials['confidence'])
            return FlowState.DEPRECATED
            
        return current_state
    
    def _update_metric_history(self, metrics: Dict[str, float]) -> None:
        """Update historical metrics."""
        timestamp = datetime.now()
        for metric_type, value in metrics.items():
            if metric_type in self.metric_history:
                self.metric_history[metric_type].append((timestamp, value))
                
        # Prune old data
        cutoff = timestamp - self.emergence_window
        for metric_type in self.metric_history:
            self.metric_history[metric_type] = [
                (ts, val) for ts, val in self.metric_history[metric_type]
                if ts > cutoff
            ]
    
    def _calculate_differentials(self) -> Dict[str, float]:
        """Calculate rate of change for each metric."""
        differentials = {}
        for metric_type, history in self.metric_history.items():
            if len(history) < 2:
                differentials[metric_type] = 0.0
                continue
                
            # Calculate weighted moving average of changes
            changes = []
            weights = []
            for i in range(1, len(history)):
                time_diff = (history[i][0] - history[i-1][0]).total_seconds()
                value_diff = history[i][1] - history[i-1][1]
                if time_diff > 0:
                    rate = value_diff / time_diff
                    changes.append(rate)
                    weights.append(1 / (i + 1))  # More recent changes weighted higher
                    
            if changes:
                differentials[metric_type] = sum(c * w for c, w in zip(changes, weights)) / sum(weights)
            else:
                differentials[metric_type] = 0.0
                
        return differentials
    
    def _detect_emergence(self, differentials: Dict[str, float]) -> bool:
        """Detect if pattern is showing signs of emergence."""
        # Check emergence potential in metrics
        return differentials.get('emergence_potential', 0) > self.emergence_threshold
    
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
