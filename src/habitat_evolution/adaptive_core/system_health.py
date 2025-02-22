"""System health monitoring service with rhythmic pattern detection."""

from enum import Enum
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
from .dimensional_context import DimensionalContext, DimensionType, WindowState

class HealthMetricType(Enum):
    TONIC = "tonic"          # Baseline system state
    PULSE = "pulse"          # Regular rhythmic patterns
    RESONANCE = "resonance"  # Cross-dimensional harmony
    TENSION = "tension"      # System stress indicators

class SystemPulse:
    """Tracks rhythmic patterns in system behavior."""
    def __init__(self, window_size: timedelta = timedelta(minutes=5)):
        self.window_size = window_size
        self.pulse_history: List[Dict[str, Any]] = []
        self.baseline_tonic: Dict[str, float] = {}
        self.resonance_patterns: Dict[str, List[float]] = defaultdict(list)
        
    def record_pulse(self, metrics: Dict[str, float]) -> None:
        """Record a system pulse with timestamp."""
        self.pulse_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        self._update_tonic(metrics)
        
    def _update_tonic(self, metrics: Dict[str, float]) -> None:
        """Update baseline tonic levels using exponential moving average."""
        alpha = 0.1  # Smoothing factor
        for metric, value in metrics.items():
            if metric not in self.baseline_tonic:
                self.baseline_tonic[metric] = value
            else:
                self.baseline_tonic[metric] = (
                    alpha * value + (1 - alpha) * self.baseline_tonic[metric]
                )

    def get_rhythm_patterns(self) -> Dict[str, Any]:
        """Analyze rhythmic patterns in recent pulses."""
        recent_pulses = [p for p in self.pulse_history 
                        if datetime.now() - p['timestamp'] <= self.window_size]
        
        patterns = {}
        for metric in self.baseline_tonic.keys():
            values = [p['metrics'].get(metric, 0) for p in recent_pulses]
            if values:
                patterns[metric] = {
                    'tonic': self.baseline_tonic[metric],
                    'variance': sum((v - self.baseline_tonic[metric])**2 
                                  for v in values) / len(values),
                    'rhythm': self._detect_rhythm(values)
                }
        return patterns

    def _detect_rhythm(self, values: List[float]) -> Dict[str, Any]:
        """Detect rhythmic patterns in value sequence."""
        if len(values) < 3:
            return {'type': 'insufficient_data'}
            
        # Detect basic rhythm type
        differences = [b - a for a, b in zip(values[:-1], values[1:])]
        if all(d > 0 for d in differences):
            return {'type': 'ascending'}
        elif all(d < 0 for d in differences):
            return {'type': 'descending'}
        elif all(d * differences[0] > 0 for d in differences[1::2]):
            return {'type': 'oscillating'}
        return {'type': 'irregular'}

class SystemHealthService:
    """Monitors system health through dimensional context and rhythmic patterns."""
    
    def __init__(self, history_dir: str = None):
        self.context = DimensionalContext()
        self.pulse = SystemPulse()
        self.active_thresholds: Dict[str, float] = {}
        self.health_status = defaultdict(lambda: defaultdict(float))
        
        # Initialize history store
        if history_dir:
            from .system_health_history import SystemHealthHistory
            self.history = SystemHealthHistory(history_dir)
        else:
            self.history = None
        
    def observe(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Record observation and update system health metrics."""
        # Record in dimensional context
        dim_results = self.context.observe_pattern(observation)
        
        # Extract health metrics
        metrics = self._extract_health_metrics(observation, dim_results)
        self.pulse.record_pulse(metrics)
        
        # Update health status
        self._update_health_status(metrics, dim_results)
        
        health_report = self.get_health_report()
        
        # Record in history if available
        if self.history:
            snapshot_id = self.history.record_health_snapshot(health_report)
            if hasattr(self, '_last_snapshot_id'):
                self.history.record_health_transition(
                    self._last_snapshot_id,
                    snapshot_id,
                    {'pressure': self.pulse.get_rhythm_patterns()}
                )
            self._last_snapshot_id = snapshot_id
        
        return health_report
    
    def _extract_health_metrics(self, observation: Dict[str, Any], 
                              dim_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract health metrics from observation and dimensional results."""
        metrics = {}
        
        # System-level metrics
        if 'severity' in observation:
            metrics['system_stress'] = float(observation['severity'])
            
        # Dimensional metrics
        for dim_type, result in dim_results.items():
            metrics[f'{dim_type}_tension'] = result.get('boundary_tension', 0.0)
            if 'suggestions' in result:
                metrics[f'{dim_type}_activity'] = len(result['suggestions'])
        
        return metrics
    
    def _update_health_status(self, metrics: Dict[str, float], 
                            dim_results: Dict[str, Any]) -> None:
        """Update system health status based on new metrics."""
        # Update basic health metrics
        for metric, value in metrics.items():
            self.health_status['current'][metric] = value
            
        # Check for cross-dimensional resonance
        active_dims = [dim for dim, res in dim_results.items() 
                      if res.get('window_state') == WindowState.OPEN.value]
        if len(active_dims) > 1:
            self.health_status['resonance']['cross_dimensional'] += 0.1
            
        # Check thresholds
        for metric, value in metrics.items():
            if metric in self.active_thresholds:
                if value > self.active_thresholds[metric]:
                    self.health_status['alerts'][f'{metric}_threshold'] = True
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        rhythm_patterns = self.pulse.get_rhythm_patterns()
        
        return {
            'timestamp': datetime.now(),
            'current_status': dict(self.health_status['current']),
            'rhythm_patterns': rhythm_patterns,
            'resonance_levels': dict(self.health_status['resonance']),
            'active_dimensions': self.context.get_active_dimensions(),
            'alerts': dict(self.health_status['alerts']),
            'evolution_summary': self.context.get_evolution_summary()
        }

    def set_threshold(self, metric: str, value: float) -> None:
        """Set threshold for health metric alerting."""
        self.active_thresholds[metric] = value
