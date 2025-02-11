"""
Pattern quality assessment and signal processing.

This module provides tools for evaluating pattern quality, processing signals,
and managing pattern lifecycle states. It helps distinguish real patterns
from noise and tracks pattern health over time.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
import math

class PatternState(Enum):
    """Lifecycle states of a pattern."""
    EMERGING = "emerging"     # Pattern is starting to form
    STABLE = "stable"        # Pattern is well-established
    DECLINING = "declining"  # Pattern is losing strength
    TRANSFORMING = "transforming"  # Pattern is changing form
    NOISE = "noise"         # Pattern may be just noise
    MERGED = "merged"       # Pattern has merged with another

@dataclass
class SignalMetrics:
    """Metrics for pattern signal quality."""
    strength: float  # 0-1: How clear/strong the pattern signal is
    noise_ratio: float  # 0-1: Amount of noise in the signal
    persistence: float  # 0-1: How long pattern maintains integrity
    reproducibility: float  # 0-1: How consistently pattern appears

@dataclass
class FlowMetrics:
    """Metrics for pattern flow dynamics."""
    viscosity: float  # 0-1: Resistance to pattern propagation
    back_pressure: float  # 0-1: Counter-forces to emergence
    volume: float  # 0-1: Quantity of pattern instances
    current: float  # -1 to 1: Rate and direction of flow

class PatternQualityAnalyzer:
    """Analyzes pattern quality and dynamics."""
    
    def __init__(self,
                 signal_threshold: float = 0.3,
                 noise_threshold: float = 0.7,
                 persistence_window: int = 10):
        """Initialize analyzer.
        
        Args:
            signal_threshold: Minimum signal strength for valid pattern
            noise_threshold: Maximum noise ratio for valid pattern
            persistence_window: Number of updates to track for persistence
        """
        self.signal_threshold = signal_threshold
        self.noise_threshold = noise_threshold
        self.persistence_window = persistence_window
        
        # Dynamic thresholds
        self._dynamic_signal_threshold = signal_threshold
        self._dynamic_noise_threshold = noise_threshold
        
        # Historical metrics
        self._signal_history: List[float] = []
        self._noise_history: List[float] = []
    
    def analyze_signal(self,
                      pattern: Dict[str, Any],
                      history: List[Dict[str, Any]]) -> SignalMetrics:
        """Analyze pattern signal quality.
        
        Args:
            pattern: Current pattern state
            history: Historical pattern states
            
        Returns:
            SignalMetrics for the pattern
        """
        # Calculate base signal strength from pattern metrics
        metrics = pattern.get("metrics", {})
        base_strength = (
            metrics.get("coherence", 0) * 0.3 +
            metrics.get("stability", 0) * 0.3 +
            metrics.get("emergence_rate", 0) * 0.2 +
            metrics.get("energy_state", 0) * 0.2
        )
        
        # Calculate noise ratio from metric volatility
        if history:
            volatility = self._calculate_volatility(history)
            noise_ratio = min(1.0, volatility * 2)  # Scale up for sensitivity
        else:
            noise_ratio = 0.5  # Default for new patterns
        
        # Calculate persistence
        persistence = self._calculate_persistence(pattern, history)
        
        # Calculate reproducibility from similar patterns
        reproducibility = self._calculate_reproducibility(pattern, history)
        
        # Update history
        self._signal_history.append(base_strength)
        self._noise_history.append(noise_ratio)
        
        # Maintain window size
        if len(self._signal_history) > self.persistence_window:
            self._signal_history.pop(0)
            self._noise_history.pop(0)
        
        # Update dynamic thresholds
        self._update_dynamic_thresholds()
        
        return SignalMetrics(
            strength=base_strength,
            noise_ratio=noise_ratio,
            persistence=persistence,
            reproducibility=reproducibility
        )
    
    def analyze_flow(self,
                    pattern: Dict[str, Any],
                    related_patterns: List[Dict[str, Any]]) -> FlowMetrics:
        """Analyze pattern flow dynamics.
        
        Args:
            pattern: Current pattern state
            related_patterns: Connected patterns
            
        Returns:
            FlowMetrics for the pattern
        """
        metrics = pattern.get("metrics", {})
        
        # Calculate viscosity from cross_pattern_flow
        viscosity = 1.0 - metrics.get("cross_pattern_flow", 0)
        
        # Calculate back pressure from opposing patterns
        back_pressure = self._calculate_back_pressure(pattern, related_patterns)
        
        # Calculate volume from pattern instances
        volume = len(related_patterns) / 10.0  # Normalize to 0-1
        volume = min(1.0, volume)
        
        # Calculate current from rate of change
        current = metrics.get("adaptation_rate", 0) * 2 - 1  # Scale to -1 to 1
        
        return FlowMetrics(
            viscosity=viscosity,
            back_pressure=back_pressure,
            volume=volume,
            current=current
        )
    
    def determine_state(self,
                       signal_metrics: SignalMetrics,
                       flow_metrics: FlowMetrics,
                       current_state: Optional[PatternState] = None) -> PatternState:
        """Determine pattern lifecycle state.
        
        Args:
            signal_metrics: Current signal metrics
            flow_metrics: Current flow metrics
            current_state: Optional current state for context
            
        Returns:
            PatternState indicating lifecycle state
        """
        # Check for noise
        if (signal_metrics.strength < self._dynamic_signal_threshold or
            signal_metrics.noise_ratio > self._dynamic_noise_threshold):
            return PatternState.NOISE
        
        # Check for emergence
        if (signal_metrics.strength > self._dynamic_signal_threshold and
            signal_metrics.persistence < 0.3):
            return PatternState.EMERGING
        
        # Check for stability
        if (signal_metrics.strength > 0.6 and
            signal_metrics.persistence > 0.7 and
            signal_metrics.reproducibility > 0.6):
            return PatternState.STABLE
        
        # Check for decline
        if (signal_metrics.strength < 0.4 and
            flow_metrics.back_pressure > 0.7):
            return PatternState.DECLINING
        
        # Check for transformation
        if (flow_metrics.current > 0.7 and
            signal_metrics.persistence < 0.5):
            return PatternState.TRANSFORMING
        
        # Default to current state or emerging
        return current_state or PatternState.EMERGING
    
    def _calculate_volatility(self, history: List[Dict[str, Any]]) -> float:
        """Calculate metric volatility from history."""
        if len(history) < 2:
            return 0.0
            
        # Calculate average change in metrics
        changes = []
        for i in range(1, len(history)):
            prev = history[i-1].get("metrics", {})
            curr = history[i].get("metrics", {})
            
            # Calculate change for each metric
            metric_changes = []
            for key in ["coherence", "stability", "emergence_rate", "energy_state"]:
                prev_val = prev.get(key, 0)
                curr_val = curr.get(key, 0)
                if prev_val != 0:
                    change = abs((curr_val - prev_val) / prev_val)
                    metric_changes.append(change)
            
            if metric_changes:
                changes.append(sum(metric_changes) / len(metric_changes))
        
        if not changes:
            return 0.0
            
        return sum(changes) / len(changes)
    
    def _calculate_persistence(self,
                             pattern: Dict[str, Any],
                             history: List[Dict[str, Any]]) -> float:
        """Calculate pattern persistence."""
        if not history:
            return 0.0
            
        # Calculate how consistently the pattern maintains its characteristics
        stability_scores = []
        for hist in history[-self.persistence_window:]:
            score = self._compare_pattern_state(pattern, hist)
            stability_scores.append(score)
        
        if not stability_scores:
            return 0.0
            
        return sum(stability_scores) / len(stability_scores)
    
    def _calculate_reproducibility(self,
                                 pattern: Dict[str, Any],
                                 history: List[Dict[str, Any]]) -> float:
        """Calculate pattern reproducibility."""
        if not history:
            return 0.0
            
        # Find similar patterns in history
        similarity_scores = []
        for hist in history:
            if hist.get("type") == pattern.get("type"):
                score = self._compare_pattern_state(pattern, hist)
                similarity_scores.append(score)
        
        if not similarity_scores:
            return 0.0
            
        return sum(similarity_scores) / len(similarity_scores)
    
    def _calculate_back_pressure(self,
                               pattern: Dict[str, Any],
                               related_patterns: List[Dict[str, Any]]) -> float:
        """Calculate back pressure from related patterns."""
        if not related_patterns:
            return 0.0
            
        # Calculate opposing forces from related patterns
        opposing_forces = []
        pattern_energy = pattern.get("metrics", {}).get("energy_state", 0)
        
        for related in related_patterns:
            related_energy = related.get("metrics", {}).get("energy_state", 0)
            if related_energy > pattern_energy:
                force = (related_energy - pattern_energy) / related_energy
                opposing_forces.append(force)
        
        if not opposing_forces:
            return 0.0
            
        return sum(opposing_forces) / len(opposing_forces)
    
    def _compare_pattern_state(self,
                             pattern1: Dict[str, Any],
                             pattern2: Dict[str, Any]) -> float:
        """Compare similarity of two pattern states."""
        metrics1 = pattern1.get("metrics", {})
        metrics2 = pattern2.get("metrics", {})
        
        if not metrics1 or not metrics2:
            return 0.0
        
        # Calculate Euclidean distance between metric vectors
        distance = 0.0
        count = 0
        
        for key in ["coherence", "stability", "emergence_rate", "energy_state"]:
            val1 = metrics1.get(key, 0)
            val2 = metrics2.get(key, 0)
            distance += (val1 - val2) ** 2
            count += 1
        
        if count == 0:
            return 0.0
            
        distance = math.sqrt(distance / count)
        
        # Convert distance to similarity score (0-1)
        return max(0.0, 1.0 - distance)
    
    def _update_dynamic_thresholds(self) -> None:
        """Update dynamic thresholds based on history."""
        if not self._signal_history:
            return
            
        # Calculate mean and standard deviation
        mean_signal = sum(self._signal_history) / len(self._signal_history)
        mean_noise = sum(self._noise_history) / len(self._noise_history)
        
        # Adjust thresholds gradually
        self._dynamic_signal_threshold = (
            self._dynamic_signal_threshold * 0.8 +
            mean_signal * 0.2
        )
        
        self._dynamic_noise_threshold = (
            self._dynamic_noise_threshold * 0.8 +
            mean_noise * 0.2
        )
