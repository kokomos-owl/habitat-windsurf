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

    def _asdict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'strength': self.strength,
            'noise_ratio': self.noise_ratio,
            'persistence': self.persistence,
            'reproducibility': self.reproducibility
        }

@dataclass
class FlowMetrics:
    """Metrics for pattern flow dynamics."""
    viscosity: float  # 0-1: Resistance to pattern propagation
    back_pressure: float  # 0-1: Counter-forces to emergence
    volume: float  # 0-1: Quantity of pattern instances
    current: float  # -1 to 1: Rate and direction of flow

    def _asdict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'viscosity': self.viscosity,
            'back_pressure': self.back_pressure,
            'volume': self.volume,
            'current': self.current
        }

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
        # Calculate base signal strength from pattern metrics and context
        metrics = pattern.get("metrics", {})
        context = pattern.get("context", {})
        
        # Initial strength is a major factor
        initial_strength = context.get("initial_strength", 0.0)
        
        # Calculate signal strength based on pattern state and metrics
        metrics = pattern.get("metrics", {})
        context = pattern.get("context", {})
        
        # Get pattern metrics with appropriate defaults
        coherence = metrics.get("coherence", 0.0)
        stability = metrics.get("stability", 0.0)
        energy = metrics.get("energy_state", 0.0)
        emergence = metrics.get("emergence_rate", 0.0)
        
        # Get context values
        initial_strength = context.get("initial_strength", 0.0)
        phase = abs(context.get("phase", 0.0))
        wavelength = context.get("wavelength", 2*math.pi)
        
        # For initial patterns (no history)
        if not history:
            if initial_strength >= 0.9 and phase == 0.0:
                # Core patterns maintain maximum strength
                return SignalMetrics(
                    strength=1.0,  # Always maximum for core patterns
                    noise_ratio=0.1,
                    persistence=1.0,
                    reproducibility=1.0
                )
            
            # For satellite patterns, calculate initial signal metrics
            phase_factor = min(1.0, phase / wavelength)
            strength = initial_strength * (1.0 - 0.3 * phase_factor)
            noise_ratio = 0.3 + (0.5 * (1.0 - initial_strength))
            persistence = 0.7 * initial_strength
            reproducibility = persistence
            
            return SignalMetrics(
                strength=strength,
                noise_ratio=min(0.8, noise_ratio),
                persistence=persistence,
                reproducibility=reproducibility
            )
        
        # For established patterns, use full metric model
        # For wave-like patterns, coherence and energy are more important
        metric_strength = (
            coherence * 0.4 +  # Increased weight for coherence
            stability * 0.2 +
            emergence * 0.2 +
            energy * 0.2     # Energy helps maintain signal
        )
        
        # For wave patterns, use sqrt of initial strength to reduce decay rate
        base_strength = math.sqrt(initial_strength) * 0.7 + metric_strength * 0.3
        
        # Boost strength if coherence is high relative to distance
        if coherence > self.signal_threshold:
            base_strength *= (1.0 + coherence * 0.2)  # Up to 20% boost
        
        # Calculate temporal metrics
        history_length = len(history) if history else 0
        temporal_window = min(self.persistence_window, history_length)
        
        # Calculate persistence
        persistence = self._calculate_persistence(pattern, history)
        
        # Calculate reproducibility from similar patterns
        reproducibility = self._calculate_reproducibility(pattern, history)
        
        # Calculate base metrics
        if coherence >= 0.8 or (initial_strength >= 0.9 and phase == 0.0):
            # Core patterns maintain high quality
            noise_ratio = 0.1  # Minimal noise
            signal_strength = 1.0  # Maximum signal for core patterns
            persistence = 1.0  # Perfect persistence
            reproducibility = 1.0  # Perfect reproducibility
            
            # Boost metrics if energy and stability are also high
            if energy > 0.3 and stability > 0.4:
                signal_strength = 1.0  # Maximum signal
                persistence = 1.0  # Maximum persistence
                reproducibility = 1.0  # Maximum reproducibility
        
        elif coherence <= 0.3:
            # Incoherent patterns have poor quality
            incoherence = 1.0 - coherence
            noise_ratio = min(0.8, self.noise_threshold + (0.5 * incoherence))
            signal_strength = min(0.4, initial_strength)  # Weak signal
            persistence = min(0.4, persistence)  # Low persistence
            reproducibility = min(0.4, reproducibility)  # Low reproducibility
        
        else:
            # Coherent patterns (0.3 < coherence < 0.8)
            # Calculate base noise from coherence
            coherence_factor = (coherence - 0.3) / 0.5  # Normalized coherence (0-1)
            base_noise = 0.5 * (1.0 - coherence_factor)  # Start with lower base noise
            
            # Apply cubic reduction for stronger coherence effects
            noise_reduction = coherence_factor * coherence_factor * coherence_factor
            
            # Apply stability bonus (up to 40% reduction)
            if stability > 0.5:
                noise_reduction += (stability - 0.5) * 0.4
            
            # Apply energy bonus (up to 30% reduction)
            if energy > 0.3:
                noise_reduction += (energy - 0.3) * 0.3
            
            # Strengthen reduction for coherent satellite patterns
            if coherence > 0.35:
                noise_reduction *= 1.5  # 50% stronger reduction
            
            # Calculate final noise with phase impact
            phase_impact = min(0.1, phase / (2 * math.pi * wavelength))  # Small increase based on phase
            noise_ratio = min(0.5, max(0.1, base_noise * (1.0 - noise_reduction) + phase_impact))
            
            # Calculate signal strength based on coherence and initial strength
            signal_strength = max(coherence, initial_strength * 0.8)
            
            # Scale persistence and reproducibility with coherence
            persistence = min(0.8, max(0.3, persistence * coherence))
            reproducibility = min(0.8, max(0.3, reproducibility * coherence))
        
        # Update and maintain history
        if hasattr(self, '_signal_history'):
            self._signal_history.append(signal_strength)
            if len(self._signal_history) > self.persistence_window:
                self._signal_history = self._signal_history[-self.persistence_window:]
                
        if hasattr(self, '_noise_history'):
            self._noise_history.append(noise_ratio)
            if len(self._noise_history) > self.persistence_window:
                self._noise_history = self._noise_history[-self.persistence_window:]
        
        # Update dynamic thresholds
        if hasattr(self, '_update_dynamic_thresholds'):
            self._update_dynamic_thresholds()
        
        # Return final metrics
        return SignalMetrics(
            strength=signal_strength,
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
        
        # Get key metrics
        coherence = metrics.get("coherence", 0.0)
        stability = metrics.get("stability", 0.0)
        energy = metrics.get("energy_state", 0.0)
        cross_flow = metrics.get("cross_pattern_flow", 0.0)
        
        # Calculate viscosity based on coherence and stability
        base_viscosity = 1.0 - coherence  # Higher coherence = lower viscosity
        stability_factor = 1.0 - (stability * 0.5)  # Stability reduces viscosity by up to 50%
        energy_factor = 1.0 - (energy * 0.3)  # Energy reduces viscosity by up to 30%
        
        # For incoherent patterns, increase viscosity significantly
        if coherence <= 0.3:
            # Double base viscosity and add 0.2 to ensure it exceeds noise threshold
            viscosity = min(1.0, base_viscosity * 2.0 + 0.2)
        else:
            viscosity = base_viscosity * stability_factor * energy_factor
        
        # Calculate back pressure from opposing patterns
        back_pressure = self._calculate_back_pressure(pattern, related_patterns)
        
        # Calculate volume (affected by energy and coherence)
        volume = (energy * 0.7 + coherence * 0.3) * len(related_patterns) / 10.0
        volume = min(1.0, volume)
        
        # Calculate current (flow direction and rate)
        current = -1.0 if coherence < 0.3 else cross_flow * 2 - 1  # Incoherent patterns flow outward
        
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
