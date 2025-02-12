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
            # Calculate base noise from coherence and adaptation
            coherence_factor = (coherence - 0.3) / 0.5  # Normalized coherence (0-1)
            adaptation_factor = metrics.get("adaptation_rate", 0.0)
            base_noise = 0.6 * (1.0 - coherence_factor) + 0.4 * adaptation_factor
            
            # Calculate volatility from history
            volatility = self._calculate_volatility(history)
            
            # Amplify noise for volatile patterns
            volatility_boost = volatility * volatility  # Quadratic scaling
            noise_boost = volatility_boost * 0.8  # Up to 80% boost from volatility
            
            # Apply stability dampening (up to 40% reduction)
            if stability > 0.5:
                noise_boost *= (1.0 - (stability - 0.5) * 0.8)
            
            # Apply energy influence
            energy_factor = 1.0 - (energy * 0.3)  # Up to 30% reduction
            
            # Calculate final noise with all factors
            phase_impact = min(0.2, phase / (2 * math.pi * wavelength))  # Increased phase impact
            noise_ratio = min(0.9, base_noise * (1.0 + noise_boost) * energy_factor + phase_impact)
            
            # Calculate signal strength based on coherence and initial strength
            signal_strength = max(coherence, initial_strength * 0.8)
            
            # Scale persistence and reproducibility with coherence and volatility
            volatility_impact = volatility * volatility  # Quadratic scaling
            persistence_factor = coherence * (1.0 - volatility_impact * 0.8)  # Up to 80% reduction from volatility
            persistence = min(0.8, max(0.2, persistence * persistence_factor))
            reproducibility = min(0.8, max(0.2, reproducibility * persistence_factor))
        
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
        """Analyze pattern flow dynamics with granular noise and feedback.
        
        Args:
            pattern: Current pattern state
            related_patterns: Connected patterns
            
        Returns:
            FlowMetrics for the pattern
        """
        metrics = pattern.get("metrics", {})
        history = pattern.get("history", [])
        context = pattern.get("context", {})
        
        # Get key metrics with noise granularity
        coherence = metrics.get("coherence", 0.0)
        stability = metrics.get("stability", 0.0)
        energy = metrics.get("energy_state", 0.0)
        cross_flow = metrics.get("cross_pattern_flow", 0.0)
        
        # Get field gradients and turbulence
        field_gradients = context.get('field_gradients', {})
        turbulence = field_gradients.get('turbulence', 0.0)
        coherence_gradient = field_gradients.get('coherence', coherence)
        energy_gradient = field_gradients.get('energy', energy)
        density = field_gradients.get('density', 1.0)
        
        # Calculate time-based noise threshold
        time_steps = len(history)
        if time_steps > 0:
            # Add slight oscillation to noise threshold
            noise_freq = 0.5  # Frequency of noise variation
            noise_amp = 0.05   # Amplitude of variation
            noise_offset = math.sin(time_steps * noise_freq) * noise_amp
            
            # Dynamic noise threshold based on pattern age
            age_factor = min(1.0, time_steps / 8.0)  # Saturates after 8 steps
            base_noise = 0.3 + noise_offset
            dynamic_noise = base_noise * (1.0 + age_factor * 0.2)  # Up to 20% increase
            
            # Store in context for evolution manager
            context["dynamic_noise_threshold"] = dynamic_noise
        
        # Calculate base viscosity from coherence gradient
        base_viscosity = 1.0 - coherence_gradient
        
        # Add granular variations based on local field state
        field_variations = []
        for related in related_patterns:
            rel_coherence = related.get("metrics", {}).get("coherence", 0.0)
            rel_energy = related.get("metrics", {}).get("energy_state", 0.0)
            
            # Calculate local field gradient
            gradient = abs(coherence - rel_coherence)
            field_variations.append(gradient * rel_energy)
        
        # Compute field feedback factor
        field_feedback = sum(field_variations) / (len(field_variations) + 1) if field_variations else 0.0
        
        # Dynamic viscosity calculation
        if coherence <= 0.3:
            # Non-linear viscosity increase for incoherent patterns
            time_factor = min(1.0, time_steps / 5.0)
            viscosity_growth = math.exp(time_factor) - 1  # Exponential growth
            
            # Add turbulence effects
            turbulence_factor = 1.0 + (turbulence * 2.0)  # Turbulence doubles viscosity at max
            
            # Calculate effective viscosity
            local_factor = 1.0 + field_feedback
            viscosity = min(1.0, base_viscosity * turbulence_factor * (1.0 + viscosity_growth) * local_factor)
        else:
            # Normal viscosity with field influence and turbulence
            stability_factor = 1.0 - (stability * 0.5)
            energy_factor = 1.0 - (energy_gradient * 0.3)
            turbulence_factor = 1.0 + (turbulence * 0.5)  # Less turbulence impact for coherent patterns
            viscosity = base_viscosity * stability_factor * energy_factor * turbulence_factor * (1.0 + field_feedback * 0.3)
        
        # Enhanced back pressure calculation with density and gradients
        base_pressure = self._calculate_back_pressure(pattern, related_patterns)
        gradient_pressure = abs(coherence_gradient - coherence) + abs(energy_gradient - energy)
        pressure_factor = 1.0 + (density * 0.5) + (gradient_pressure * 0.3)
        back_pressure = base_pressure * pressure_factor + (density * 0.2)  # Add density baseline
        
        # Volume calculation with density and turbulence
        volume_base = energy * 0.6 + coherence * 0.4  # Increase coherence influence
        volume_factor = density * (1.0 - turbulence * 0.7)  # Reduce turbulence impact
        volume = volume_base * volume_factor
        volume = min(1.0, max(0.2, volume))  # Ensure minimum volume
        
        # Calculate current with gradient effects
        coherence_diff = abs(coherence_gradient - coherence)
        energy_diff = abs(energy_gradient - energy)
        gradient_strength = coherence_diff + energy_diff
        
        if coherence <= 0.3:
            # Strong dissipation for incoherent patterns
            current = -1.0 * (1.0 + turbulence + gradient_strength)
        else:
            # Flow driven by gradients and cross-pattern interactions
            gradient_direction = 1.0 if coherence_gradient > coherence else -1.0
            base_flow = gradient_direction * gradient_strength * 2.0
            
            # Add cross-pattern flow contribution
            cross_flow_contribution = cross_flow * (1.0 - gradient_strength)
            
            # Combine and apply turbulence damping
            current = (base_flow + cross_flow_contribution) * (1.0 - turbulence * 0.3)
        
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
