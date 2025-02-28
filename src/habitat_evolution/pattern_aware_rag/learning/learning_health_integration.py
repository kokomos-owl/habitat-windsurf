"""Integration between learning system and system health monitoring.

This module provides the integration layer between the learning window control system 
and the system health monitoring services, enabling:

1. Health-aware learning windows
2. Pattern-sensitive health metrics  
3. Cross-layer coherence through natural observation
4. Dual-mode support (Neo4j and Direct LLM)
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import deque, defaultdict
import asyncio

from ...adaptive_core.system_health import SystemHealthService, HealthMetricType
from ...adaptive_core.dimensional_context import DimensionalContext, DimensionType, WindowState
from .learning_control import LearningWindow, BackPressureController, EventCoordinator

class LearningMetricProfile:
    """Tracks learning metrics for health integration."""
    
    def __init__(self, window_size: int = 20):
        self.metrics = {
            "stability": deque(maxlen=window_size),
            "coherence": deque(maxlen=window_size),
            "density": deque(maxlen=window_size),
            "rhythm": deque(maxlen=window_size)
        }
        self.last_update = datetime.now()
        
    def record_metric(self, metric_type: str, value: float) -> None:
        """Record a new metric value."""
        if metric_type in self.metrics:
            self.metrics[metric_type].append(value)
            self.last_update = datetime.now()
    
    def get_average(self, metric_type: str) -> float:
        """Get average value for a metric type."""
        if metric_type in self.metrics and self.metrics[metric_type]:
            return float(np.mean(self.metrics[metric_type]))
        return 0.0
    
    def get_rhythm_pattern(self, metric_type: str) -> Dict[str, Any]:
        """Calculate rhythm pattern for a metric."""
        if metric_type not in self.metrics or len(self.metrics[metric_type]) < 3:
            return {"type": "insufficient_data"}
            
        values = list(self.metrics[metric_type])
        differences = [b - a for a, b in zip(values[:-1], values[1:])]
        
        if all(d > 0 for d in differences):
            return {"type": "ascending"}
        elif all(d < 0 for d in differences):
            return {"type": "descending"}
        elif all(d * differences[0] > 0 for d in differences[1::2]):
            return {"type": "oscillating"}
        return {"type": "irregular"}


class HealthAwareLearningWindow(LearningWindow):
    """Learning window with integrated health awareness.
    
    Extends the basic learning window to incorporate system health metrics
    and adapts its behavior based on system health patterns.
    """
    
    def __init__(
        self, 
        start_time: datetime,
        end_time: datetime,
        stability_threshold: float,
        coherence_threshold: float,
        max_changes_per_window: int,
        health_service: Optional[SystemHealthService] = None
    ):
        super().__init__(
            start_time=start_time,
            end_time=end_time,
            stability_threshold=stability_threshold,
            coherence_threshold=coherence_threshold,
            max_changes_per_window=max_changes_per_window
        )
        self.health_service = health_service
        self.metric_profile = LearningMetricProfile()
        self.dimension_resonance: Dict[DimensionType, float] = defaultdict(float)
        self.last_health_report: Optional[Dict[str, Any]] = None
        
    def record_health_metrics(self, metrics: Dict[str, float]) -> None:
        """Record learning-related health metrics."""
        for metric_type, value in metrics.items():
            self.metric_profile.record_metric(metric_type, value)
            
    def get_health_observation(self) -> Dict[str, Any]:
        """Generate health observation from current state."""
        observation = {
            "window_state": self.state.value,
            "stability": self.metric_profile.get_average("stability"),
            "coherence": self.metric_profile.get_average("coherence"),
            "change_saturation": self.change_count / max(1, self.max_changes_per_window),
            "time_remaining": (self.end_time - datetime.now()).total_seconds()
        }
        
        # Add rhythm patterns when available
        for metric in ["stability", "coherence"]:
            pattern = self.metric_profile.get_rhythm_pattern(metric)
            if pattern["type"] != "insufficient_data":
                observation[f"{metric}_rhythm"] = pattern["type"]
        
        return observation
    
    def update_from_health_report(self, health_report: Dict[str, Any]) -> None:
        """Update window behavior based on system health report."""
        self.last_health_report = health_report
        
        # Extract resonance information
        if 'resonance_levels' in health_report:
            for dim_type, level in health_report['resonance_levels'].items():
                try:
                    dimension = DimensionType(dim_type)
                    self.dimension_resonance[dimension] = level
                except ValueError:
                    pass  # Not a valid dimension type
        
        # Adapt window behavior based on health metrics
        self._adapt_window_from_health()
    
    def _adapt_window_from_health(self) -> None:
        """Adapt window behavior based on health metrics."""
        if not self.last_health_report:
            return
            
        # Extract rhythm patterns for adaptation
        rhythm_patterns = self.last_health_report.get('rhythm_patterns', {})
        
        # Check for significant system stress
        current_status = self.last_health_report.get('current_status', {})
        stress_level = current_status.get('system_stress', 0.0)
        
        # Adjust window parameters based on health conditions
        if stress_level > 0.8:
            # High stress - reduce max changes
            self.max_changes_per_window = max(1, int(self.max_changes_per_window * 0.8))
        elif stress_level < 0.3 and self.dimension_resonance:
            # Low stress with good resonance - allow more changes
            # But only if there's dimensional resonance (indicating coordinated state)
            avg_resonance = sum(self.dimension_resonance.values()) / len(self.dimension_resonance)
            if avg_resonance > 0.6:
                self.max_changes_per_window = int(self.max_changes_per_window * 1.2)


class HealthAwareEventCoordinator(EventCoordinator):
    """Event coordinator with integrated health awareness.
    
    Extends the basic event coordinator to incorporate system health metrics
    and adapts its event processing based on health patterns.
    """
    
    def __init__(
        self, 
        max_queue_size: int = 1000, 
        persistence_mode: bool = True,
        health_service: Optional[SystemHealthService] = None
    ):
        super().__init__(max_queue_size, persistence_mode)
        self.health_service = health_service
        self.health_metrics = LearningMetricProfile()
        self.last_health_report: Optional[Dict[str, Any]] = None
        
    def create_health_aware_window(
        self,
        duration: timedelta = timedelta(minutes=10),
        stability_threshold: float = 0.7,
        coherence_threshold: float = 0.6,
        max_changes: int = 20
    ) -> HealthAwareLearningWindow:
        """Create a health-aware learning window."""
        now = datetime.now()
        health_window = HealthAwareLearningWindow(
            start_time=now,
            end_time=now + duration,
            stability_threshold=stability_threshold,
            coherence_threshold=coherence_threshold,
            max_changes_per_window=max_changes,
            health_service=self.health_service
        )
        
        self.current_window = health_window
        return health_window
        
    def update_health_metrics(self, stability: float, coherence: float) -> Dict[str, Any]:
        """Update health metrics and report to health service."""
        # Record metrics
        self.health_metrics.record_metric("stability", stability)
        self.health_metrics.record_metric("coherence", coherence)
        
        # Calculate derived metrics
        density = 0.6 * stability + 0.4 * coherence  # Simple derived metric
        self.health_metrics.record_metric("density", density)
        
        # Generate observation
        observation = {
            "coordinator_state": "active" if self.current_window and self.current_window.is_active else "inactive",
            "stability": stability,
            "coherence": coherence,
            "persistence_mode": self.persistence_mode,
            "queue_saturation": len(self.event_queue) / self.event_queue.maxlen,
        }
        
        # Report to health service if available
        if self.health_service:
            self.last_health_report = self.health_service.observe(observation)
            
            # Update window if available
            if self.current_window and isinstance(self.current_window, HealthAwareLearningWindow):
                self.current_window.update_from_health_report(self.last_health_report)
                
            return self.last_health_report
            
        return {"status": "health_service_unavailable"}
        
    def process_event_with_health(self, event_id: str, event_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Process event with health awareness."""
        # Extract health-related metrics from event
        stability = event_data.get("stability", 0.5)
        coherence = event_data.get("coherence", 0.5)
        
        # Update health metrics
        health_report = self.update_health_metrics(stability, coherence)
        
        # Use regular event processing but with health-informed back pressure
        result = self.process_event(event_id, event_data)
        
        return result, health_report


class HealthIntegratedBackPressure(BackPressureController):
    """Back pressure controller with system health integration.
    
    Extends the basic back pressure controller to incorporate system health
    metrics and adapts its delay calculations based on health patterns.
    """
    
    def __init__(
        self,
        base_delay: float = 0.1,
        max_delay: float = 5.0,
        stability_threshold: float = 0.7,
        window_size: int = 10,
        health_service: Optional[SystemHealthService] = None
    ):
        super().__init__(
            base_delay=base_delay,
            max_delay=max_delay,
            stability_threshold=stability_threshold,
            window_size=window_size
        )
        self.health_service = health_service
        self.health_influence = 0.3  # How much health metrics influence delays
        self.last_health_report: Optional[Dict[str, Any]] = None
        self.system_resonance = 0.5  # Default middle resonance
        
    def calculate_delay_with_health(self, stability_score: float) -> float:
        """Calculate delay incorporating system health metrics."""
        # Calculate base delay from regular algorithm
        base_delay = self.calculate_delay(stability_score)
        
        # If no health service, return base delay
        if not self.health_service:
            return base_delay
            
        # Get observation about current back pressure state
        observation = {
            "stability_score": stability_score,
            "base_delay": base_delay,
            "current_pressure": self.current_pressure,
            "component": "back_pressure_controller"
        }
        
        # Get health report
        self.last_health_report = self.health_service.observe(observation)
        
        # Extract health metrics that influence pressure
        current_status = self.last_health_report.get('current_status', {})
        system_stress = current_status.get('system_stress', 0.5)
        
        # Get resonance for natural rhythm alignment
        resonance = self.last_health_report.get('resonance_levels', {})
        system_resonance = resonance.get('cross_dimensional', 0.5)
        self.system_resonance = system_resonance
        
        # Modify delay based on health metrics
        # - High stress increases delay (protective)
        # - High resonance decreases delay (things are flowing naturally)
        stress_factor = 1.0 + (system_stress - 0.5) * self.health_influence
        resonance_factor = 1.0 - (system_resonance - 0.5) * self.health_influence
        
        # Combine factors with natural alignment
        health_adjusted_delay = base_delay * stress_factor * resonance_factor
        
        # Ensure within bounds
        return np.clip(health_adjusted_delay, self.base_delay, self.max_delay)


class FieldObserver:
    """Base class for field observers.
    
    Field observers record contextual conditions and detect pattern shifts
    through harmonic analysis.
    """
    
    def __init__(self, field_id: str):
        """Initialize the field observer.
        
        Args:
            field_id: Unique identifier for the field
        """
        self.field_id = field_id
        self.observations = []
        self.field_metrics = {}
        self.wave_history = []  # Store stability wave
        self.tonic_history = [] # Store tonic values
    
    async def observe(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Observe and record field conditions without enforcing.
        
        Args:
            context: The context of the observation
            
        Returns:
            Updated field metrics
        """
        self.observations.append({"context": context, "time": datetime.now()})
        
        # Extract stability from context if available
        if "stability" in context:
            self.wave_history.append(context["stability"])
            
        # Store default tonic if not present
        if len(self.tonic_history) < len(self.wave_history):
            self.tonic_history.append(0.5)  # Default tonic value
        
        # Update standard field metrics
        self.field_metrics.update({
            "observation_count": len(self.observations),
            "latest_observation_time": datetime.now().isoformat(),
            "latest_context": context
        })
        
        return self.field_metrics
    
    def get_field_metrics(self) -> Dict[str, Any]:
        """Get current field metrics.
        
        Returns:
            Dictionary of field metrics
        """
        # Add computed metrics
        if self.observations:
            states = [obs["context"].get("state", "") for obs in self.observations]
            self.field_metrics["state_transitions"] = len(set(states))
            
            # Add tonic value if available
            if self.tonic_history:
                self.field_metrics["tonic"] = self.tonic_history[-1]
            
            # Add stability trend if available
            if len(self.wave_history) >= 2:
                self.field_metrics["stability_trend"] = sum(
                    self.wave_history[-1] - self.wave_history[-2]
                    for _ in range(min(3, len(self.wave_history) - 1))
                ) / min(3, len(self.wave_history) - 1)
        
        return self.field_metrics
    
    def perform_harmonic_analysis(self, stability_series: List[float], tonic_series: List[float]) -> Dict[str, Any]:
        """Perform tonic-harmonic analysis on stability and tonic series.
        
        Args:
            stability_series: List of stability values
            tonic_series: List of tonic values
            
        Returns:
            Dictionary with harmonic analysis results
        """
        if len(stability_series) < 3 or len(tonic_series) < 3:
            return {"harmonic": [], "boundaries": []}
            
        # Ensure same length
        min_len = min(len(stability_series), len(tonic_series))
        stability = stability_series[-min_len:]
        tonic = tonic_series[-min_len:]
        
        # Calculate harmonic series as the product of stability and tonic
        harmonic = [s * t for s, t in zip(stability, tonic)]
        
        # Identify boundaries (significant changes in harmonic value)
        boundaries = []
        for i in range(1, len(harmonic)):
            # Detect significant change (threshold of 10%)
            if abs(harmonic[i] - harmonic[i-1]) > 0.1:
                boundaries.append(i)
                
        # Detect resonance patterns (alternating high-low values)
        resonance = []
        if len(harmonic) >= 4:
            for i in range(len(harmonic) - 3):
                segment = harmonic[i:i+4]
                # Check for alternating pattern
                if (segment[0] > segment[1] < segment[2] > segment[3]) or \
                   (segment[0] < segment[1] > segment[2] < segment[3]):
                    resonance.append(i+1)  # Middle point of pattern
        
        return {
            "harmonic": harmonic,
            "boundaries": boundaries,
            "resonance": resonance,
            "avg_harmonic": sum(harmonic) / len(harmonic) if harmonic else 0,
            "max_harmonic": max(harmonic) if harmonic else 0,
            "stability_trend": sum(s2 - s1 for s1, s2 in zip(stability[:-1], stability[1:])) / (len(stability) - 1) if len(stability) > 1 else 0,
            "tonic_trend": sum(t2 - t1 for t1, t2 in zip(tonic[:-1], tonic[1:])) / (len(tonic) - 1) if len(tonic) > 1 else 0
        }
    
    @property
    def tonic_values(self) -> List[float]:
        """Get the current tonic values history.
        
        Returns:
            List of tonic values
        """
        return self.tonic_history
    
    def get_optimal_transition_time(self, 
                                  window_start: datetime,
                                  window_duration: timedelta) -> Optional[datetime]:
        """Find optimal time for window transition based on field harmonics.
        
        Analyzes tonic-harmonic patterns to detect natural boundaries for transitions.
        """
        if len(self.tonic_history) < 3 or "observation_rhythm" not in self.field_metrics:
            return None
            
        # Estimate tonic cycle period from observations
        if len(self.field_metrics["observation_rhythm"]) > 2:
            rhythm_mean = np.mean(self.field_metrics["observation_rhythm"])
            
            # Find pattern in tonic values
            tonic_values = np.array(self.tonic_history)
            peak_indices = []
            
            # Find peaks in tonic pattern
            for i in range(1, len(tonic_values)-1):
                if tonic_values[i] > tonic_values[i-1] and tonic_values[i] > tonic_values[i+1]:
                    peak_indices.append(i)
            
            # Calculate tonic period if peaks found
            if len(peak_indices) >= 2:
                avg_peak_distance = np.mean([peak_indices[i+1] - peak_indices[i] 
                                           for i in range(len(peak_indices)-1)])
                tonic_period = avg_peak_distance * rhythm_mean
                
                # Calculate next peak time
                now = datetime.now()
                last_peak_time = self.observations[peak_indices[-1]]["timestamp"]
                time_since_peak = (now - last_peak_time).total_seconds()
                time_to_next_peak = tonic_period - (time_since_peak % tonic_period)
                
                # Calculate optimal transition time (at tonic peak)
                return now + timedelta(seconds=time_to_next_peak)
        
        # Default to half the window duration if pattern analysis failed
        return window_start + (window_duration / 2)

class HealthFieldObserver(FieldObserver):
    """Field observer that integrates with system health.
    
    Connects field observation with system health metrics to create a
    tonic-harmonic approach to detecting contextual boundaries.
    """
    
    def __init__(self, field_id: str, health_service: SystemHealthService):
        super().__init__(field_id)
        self.health_service = health_service
    
    async def observe(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Add observation and notify health service if configured."""
        # Record observation
        now = datetime.now()
        self.observations.append({"context": context, "time": now})
        
        # Get stability from context (handle both field name formats)
        stability = context.get("stability", context.get("stability_score", 0.8))
        
        # Add to wave history for analysis
        self.wave_history.append(stability)
        
        # Call parent observe method
        parent_response = await super().observe(context)
        
        # Update field metrics
        self.update_field_metrics(context)
        
        # Request health observation if service available
        if self.health_service:
            health_data = self.health_service.observe(context)
            
            # Extract tonic value if available
            if "rhythm_patterns" in health_data and "stability" in health_data["rhythm_patterns"]:
                tonic = health_data["rhythm_patterns"]["stability"].get("tonic", 0.5)
                self.tonic_history.append(tonic)
                self.field_metrics["tonic"] = tonic
                
                # Add resonance metrics if available
                if "resonance" in health_data["rhythm_patterns"]["stability"]:
                    self.field_metrics["resonance"] = health_data["rhythm_patterns"]["stability"]["resonance"]
            
            return {**parent_response, **health_data}
        
        return parent_response
