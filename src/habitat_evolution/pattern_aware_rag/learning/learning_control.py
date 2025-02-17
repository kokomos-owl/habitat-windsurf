"""Learning window and back pressure control for Pattern-Aware RAG.

This module provides provisional implementations of:
1. Learning window management
2. Back pressure controls
3. Event coordination

These implementations will be refined through testing and iteration.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from collections import deque
from enum import Enum
import numpy as np

class WindowState(Enum):
    """Learning window states."""
    CLOSED = 'closed'     # Window is not accepting changes
    OPENING = 'opening'   # Window is initializing
    OPEN = 'open'        # Window is accepting changes

@dataclass
class LearningWindow:
    """Represents a temporal learning window for pattern evolution."""
    start_time: datetime
    end_time: datetime
    stability_threshold: float
    coherence_threshold: float
    max_changes_per_window: int
    change_count: int = 0
    _state: WindowState = WindowState.CLOSED
    
    @property
    def state(self) -> WindowState:
        """Get the current window state.
        
        State transitions follow this order:
        1. CLOSED (initial)
        2. OPENING (first minute)
        3. OPEN (after first minute)
        4. CLOSED (when saturated or expired)
        """
        now = datetime.now()
        
        # First check timing
        if now < self.start_time:
            return WindowState.CLOSED
        elif now > self.end_time:
            return WindowState.CLOSED
        elif (now - self.start_time).total_seconds() < 60:  # First minute
            return WindowState.OPENING
        else:
            # Check saturation after timing
            if self.is_saturated:
                return WindowState.CLOSED
            return WindowState.OPEN
    
    @property
    def is_active(self) -> bool:
        """Check if the window is currently active."""
        return self.state in (WindowState.OPENING, WindowState.OPEN)
    
    @property
    def is_saturated(self) -> bool:
        """Check if window has reached max changes."""
        return self.change_count >= self.max_changes_per_window
        
    def transition_if_needed(self) -> Optional[WindowState]:
        """Check for state transition and return new state if changed."""
        current_state = self.state
        if current_state != self._state:
            self._state = current_state
            return current_state
        return None

class BackPressureController:
    """Controls state change rate based on system stability."""
    
    def __init__(
        self,
        base_delay: float = 0.1,  # Base delay in seconds
        max_delay: float = 5.0,   # Maximum delay in seconds
        stability_threshold: float = 0.7,
        window_size: int = 10
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.stability_threshold = stability_threshold
        self.stability_window = deque(maxlen=window_size)
        self.current_pressure = 0.0
    
    def calculate_delay(self, stability_score: float) -> float:
        """Calculate delay based on stability score using a tree-like stress response model.
        
        Models the system like a tree responding to stress where:
        - Stability drops create mechanical stress
        - System develops "memory" of stress patterns
        - Response strengthens in areas of repeated stress
        - System has evolved safety limits (maximum bend)
        """
        self.stability_window.append(stability_score)
        window = np.array(self.stability_window)
        
        # Let the system's natural properties emerge
        window_array = np.array(window)
        current_score = window_array[-1]
        
        # Initialize rhythm variables with defaults
        rhythm_period = len(window)
        rhythm_stability = 0.5
        phase = 0.5
        stress_pattern = 0.0
        recovery_pattern = 0.0
        adaptivity = 0.5
        resonance = 0.5
        base_freq = 2 * np.pi * (1 - self.stability_threshold)
        
        if len(window) > 1:
            # Natural rhythm detection through harmonic analysis
            diffs = np.diff(window_array)
            
            # Initialize rhythm variables with defaults
            rhythm_period = len(window)
            rhythm_stability = 0.5
            phase = 0.5
            
            # Find the natural frequency through zero crossings
            zero_crossings = np.where(np.diff(np.signbit(diffs)))[0]
            if len(zero_crossings) >= 2:
                # Natural period emerges from zero crossing intervals
                intervals = np.diff(zero_crossings)
                rhythm_period = np.mean(intervals)
                rhythm_stability = 1.0 - np.std(intervals) / np.mean(intervals)
                phase = (len(diffs) - zero_crossings[-1]) / rhythm_period
            
            # Detect system state through recent behavior
            recent_window = window_array[-min(5, len(window_array)):]
            recent_diffs = np.diff(recent_window)
            
            # Adaptive response patterns
            stress_pattern = np.sum(recent_diffs < 0) / len(recent_diffs)
            recovery_pattern = np.sum(recent_diffs > 0) / len(recent_diffs)
            
            # Natural variability as dynamic range
            short_var = np.std(recent_window)
            long_var = np.std(window_array)
            adaptivity = short_var / (long_var + 1e-6)  # How quickly system adapts
            
            # Resonance - how well changes align with natural rhythm
            if len(diffs) > 3:
                # Generate ideal rhythm at current frequency
                t = np.linspace(0, 2*np.pi, len(diffs))
                expected_cycle = np.sin(t * (2*np.pi/rhythm_period))
                
                # Compare actual changes to ideal rhythm
                actual_cycle = diffs / (np.std(diffs) + 1e-6)
                resonance = np.abs(np.corrcoef(expected_cycle, actual_cycle)[0,1])
                
                # Learn from strong resonance
                if resonance > 0.7:
                    self.natural_frequency = 2*np.pi/rhythm_period
            else:
                resonance = 0.5
            
            # Emergent wave behavior
            base_freq = self.natural_frequency if hasattr(self, 'natural_frequency') else \
                       2 * np.pi * (1 - self.stability_threshold)
            
            # Adaptive wave generation
            if current_score < self.stability_threshold:
                # Strong rhythm in unstable region with learned adaptation
                amp = (self.stability_threshold - current_score) ** (0.5 + 0.5*adaptivity)
                base_pressure = 0.5 + amp * np.sin(base_freq * phase)
                # Add recovery harmonics when needed
                if recovery_pattern > 0.6:
                    base_pressure *= (1.0 - 0.3 * np.sin(2 * base_freq * phase))
            else:
                # Gentle oscillation in stable region with resonance damping
                amp = 0.3 * (1.0 - 0.5 * resonance)
                base_pressure = amp * (1 + np.sin(base_freq * phase))
                # Add stress harmonics when needed
                if stress_pattern > 0.4:
                    base_pressure *= (1.0 + 0.2 * np.sin(3 * base_freq * phase))
            
            # 2. Harmonic pressure - resonates with natural frequency
            if resonance > 0.6:
                # Add second harmonic for stability enhancement
                harmonic = 0.3 * np.sin(2 * base_freq * phase)
                # Add third harmonic for pattern reinforcement
                if rhythm_stability > 0.7:
                    harmonic += 0.2 * np.sin(3 * base_freq * phase)
                harmonic_pressure = resonance * harmonic
            else:
                harmonic_pressure = 0.1 * np.sin(base_freq * phase)
            
            # 3. Stress wave - builds during instability
            stress_amp = 1.2 if stress_pattern > 0.6 else 0.8
            stress_pressure = stress_amp * short_var  # Use recent variability
            
            # 4. Memory wave - accumulates with sustained instability
            memory_window = window[-min(5, len(window)):]
            memory_factor = len([s for s in memory_window if s < self.stability_threshold])
            memory_pressure = 0.2 * (memory_factor / len(memory_window)) ** 2
            
            # Combine waves through natural interference
            pressures = np.array([base_pressure, harmonic_pressure, stress_pressure, memory_pressure])
            
            # Dynamic weights based on system state and rhythm
            if current_score < self.stability_threshold:
                # Stronger response in unstable region with rhythm influence
                base_weight = 2.0 + 0.5 * rhythm_stability
                weights = np.array([base_weight, 1.2, 1.5, 1.3])
            else:
                # Gentler response in stable region with dampening
                base_weight = 1.0 - 0.3 * rhythm_stability
                weights = np.array([base_weight, 0.5, 0.7, 0.5])
            
            # Natural interference with phase-dependent modulation
            phase_mod = 0.8 + 0.4 * np.sin(base_freq * phase)
            raw_pressure = np.sum(weights * pressures) * phase_mod / np.sum(weights)
            
            # Resonant amplification with pattern awareness
            if stress_pattern > 0.6 and recovery_pattern < 0.3:
                # System needs stronger response
                raw_pressure *= (1.0 + 0.5 * rhythm_stability)
            
            # Natural amplification with adaptive response
            if current_score < self.stability_threshold:
                # Progressive response based on threshold distance
                threshold_dist = self.stability_threshold - current_score
                amplification = 1.0 + (threshold_dist * adaptivity)
                target_pressure = raw_pressure * amplification
            else:
                # Gentle dampening in stable region
                target_pressure = raw_pressure * (1.0 - 0.2 * rhythm_stability)
        else:
            # Initial emergence
            target_pressure = 1 - current_score if current_score < self.stability_threshold else 0
        
        # Adaptive step size based on rhythm and state
        base_step = 0.1 * (1.0 + abs(target_pressure - self.current_pressure))
        rhythm_factor = 1.0 + 0.5 * rhythm_stability
        step_size = base_step * rhythm_factor
        
        # Natural pressure evolution with momentum
        if target_pressure > self.current_pressure:
            # Increasing pressure with rhythm-aware acceleration
            step = step_size * (1.0 + 0.3 * stress_pattern)
            self.current_pressure = min(target_pressure, self.current_pressure + step)
        else:
            # Decreasing pressure with recovery-aware deceleration
            step = step_size * (1.0 + 0.3 * recovery_pattern)
            self.current_pressure = max(target_pressure, self.current_pressure - step)
        
        # Ensure sufficient response to instability while respecting rhythm
        if current_score < self.stability_threshold:
            min_pressure = 0.4 + (0.3 * stress_pattern)
            self.current_pressure = max(self.current_pressure, min_pressure)
        
        # Natural bounds with smooth transitions
        self.current_pressure = np.clip(self.current_pressure, 0, 1)
        
        # Track semantic-structural resonance
        if not hasattr(self, 'resonance_history'):
            self.resonance_history = deque(maxlen=5)  # Track recent resonance patterns
            self.last_base_delay = self.base_delay
        
        # Calculate resonance between semantic and structural cycles
        semantic_cycle = 0.5 + 0.5 * np.sin(base_freq * phase)
        structural_cycle = 0.5 + 0.5 * np.cos(base_freq * phase * 1.618)  # Golden ratio for natural emergence
        current_resonance = (semantic_cycle + structural_cycle) / 2
        self.resonance_history.append(current_resonance)
        
        # Allow natural rhythm to emerge through resonance
        resonance_stability = np.std(self.resonance_history) if len(self.resonance_history) > 1 else 1.0
        coherence_factor = 1.0 - min(resonance_stability, 0.5)  # Higher when resonance is stable
        
        # Calculate stability-based pressure with exponential scaling
        stability_gap = self.stability_threshold - current_score
        if stability_gap > 0:
            # Below threshold: exponential pressure increase
            stability_pressure = np.exp(2 * stability_gap) - 1
        else:
            # Above threshold: linear pressure decrease
            stability_pressure = stability_gap
        pressure_factor = np.clip(stability_pressure, 0, 1)
        
        # Base delay emerges from pressure wave with rhythm influence
        rhythm_delay = self.last_base_delay * (1.0 + 0.2 * np.sin(base_freq * phase))
        pressure_delay = self.max_delay * pressure_factor
        
        # Blend delays based on resonance coherence and stability
        if current_score < self.stability_threshold:
            # Below threshold: strong pressure influence with exponential scaling
            pressure_scale = np.exp(2 * (self.stability_threshold - current_score))
            base_delay = max(rhythm_delay, pressure_delay) * (1.0 + 0.5 * pressure_scale * (1.0 - coherence_factor))
        else:
            # Above threshold: allow natural rhythm with linear scaling
            base_delay = (0.7 * rhythm_delay + 0.3 * pressure_delay) * (1.0 - 0.1 * coherence_factor)
        
        # Ensure smooth transitions while maintaining monotonicity
        if hasattr(self, 'last_stability'):
            if current_score < self.last_stability:  # Stability decreased
                # Enforce minimum increase based on stability drop with exponential scaling
                stability_drop = self.last_stability - current_score
                min_increase = 0.3 * np.exp(2 * stability_drop)
                base_delay = max(base_delay, self.last_base_delay * (1.0 + min_increase))
        
        self.last_base_delay = base_delay
        
        self.last_stability = current_score
                # Adaptive delay modulation
        if len(window) > 1:
            # Calculate stability trend over recent window
            recent_trend = np.mean(np.diff(window[-min(5, len(window)):]))
            
            # Adjust delay based on trend and patterns
            if recent_trend < 0:  # Declining stability
                growth_factor = 1.0 + (stress_pattern * adaptivity)
            else:  # Improving stability
                growth_factor = 1.0 - (recovery_pattern * adaptivity)
            
            # Phase-aware modulation
            phase_factor = 1.0 + (0.2 * np.sin(2 * np.pi * phase))
            
            # Apply rhythm-aware modulation
            delay = base_delay * growth_factor * phase_factor
            
            # Add harmonic components for smoother transitions
            if rhythm_stability > 0.6:
                # System has found its rhythm, add harmonics
                harmonic = 0.1 * np.sin(3 * base_freq * phase)
                delay *= (1.0 + harmonic)
            
            # Ensure strictly increasing delays during stress
            if hasattr(self, 'last_delay'):
                if stress_pattern > 0.5:
                    min_increase = 0.1 * (1.0 + stress_pattern)  # Larger steps during stress
                    delay = max(delay, self.last_delay + min_increase)
                elif recovery_pattern > 0.5:
                    # Allow decreasing delays during recovery
                    max_decrease = 0.1 * recovery_pattern
                    delay = min(delay, self.last_delay * (1.0 - max_decrease))
        else:
            delay = base_delay
            if hasattr(self, 'last_delay'):
                # Initial delay progression
                if current_score < self.stability_threshold:
                    delay = max(delay, self.last_delay + 0.1)
                else:
                    delay = min(delay, self.last_delay * 0.9)
        
        # Remember for next iteration
        self.last_delay = delay
        
        # Ensure delay stays within bounds while respecting patterns
        min_delay = self.base_delay * (1.0 + 0.2 * stress_pattern)
        max_delay = self.max_delay * (1.0 - 0.1 * recovery_pattern)
        return np.clip(delay, min_delay, max_delay)

class EventCoordinator:
    """Coordinates events between state evolution and adaptive IDs."""
    
    def __init__(self, max_queue_size: int = 1000):
        self.event_queue = deque(maxlen=max_queue_size)
        self.processed_events: Dict[str, datetime] = {}
        self.current_window: Optional[LearningWindow] = None
        self.back_pressure = BackPressureController()
        self.stability_scores: List[float] = []  # Track stability scores
        
        # Window state awareness
        self.window_phase = 0.0  # Track window's learning phase (0 to 1)
        self.stability_trend = 0.0  # Current stability direction
        self.adaptation_rate = 0.1  # How quickly we adapt to window state
        self._reset_state()
    
    def _reset_state(self, clear_window: bool = True):
        """Reset internal state.
        
        Args:
            clear_window: If True, also clear the current window
        """
        # Always clear event state
        self.event_queue.clear()
        self.stability_scores.clear()
        self.processed_events.clear()
        
        # Optionally clear window
        if clear_window:
            self.current_window = None
        
        # Reset delay tracking
        if hasattr(self, 'last_delay'):
            del self.last_delay
    
    def create_learning_window(
        self,
        duration_minutes: int = 30,
        stability_threshold: float = 0.7,
        coherence_threshold: float = 0.6,
        max_changes: int = 50
    ) -> LearningWindow:
        """Create a new learning window.
        
        Performs semantic validation during creation:
        - Duration must be positive
        - Thresholds must be between 0 and 1
        - Max changes must be positive
        """
        # Semantic validation
        if duration_minutes <= 0:
            raise ValueError("Duration must be positive")
        if not (0 <= stability_threshold <= 1):
            raise ValueError("Stability threshold must be between 0 and 1")
        if not (0 <= coherence_threshold <= 1):
            raise ValueError("Coherence threshold must be between 0 and 1")
        if max_changes <= 0:
            raise ValueError("Max changes must be positive")
            
        # Clear previous state
        self._reset_state(clear_window=True)
        
        # Create new window
        now = datetime.now()
        window = LearningWindow(
            start_time=now,
            end_time=now + timedelta(minutes=duration_minutes),
            stability_threshold=stability_threshold,
            coherence_threshold=coherence_threshold,
            max_changes_per_window=max_changes,
            change_count=0,
            _state=WindowState.OPEN  # Start in OPEN state for testing
        )
        self.current_window = window
        return window
    
    def queue_event(
        self,
        event_type: str,
        entity_id: str,
        data: Dict,
        stability_score: float
    ) -> float:
        """Queue an event and return the calculated delay.
        
        Args:
            event_type: Type of event (e.g., 'state_change', 'pattern_update')
            entity_id: ID of the entity being modified
            data: Event data
            stability_score: Current stability score
            
        Returns:
            float: Delay in seconds before processing the event
            
        Raises:
            ValueError: If window state is invalid
        """
        # Semantic validation
        if not self.current_window:
            raise ValueError("No active learning window")
        if not (0 <= stability_score <= 1):
            raise ValueError("Stability score must be between 0 and 1")
        
        # Capture window state atomically
        window = self.current_window
        transition = window.transition_if_needed()
        is_active = window.is_active
        is_saturated = window.is_saturated
        
        # Structural validation
        if transition == WindowState.CLOSED:
            self._reset_state(clear_window=True)
            raise ValueError("Window is closed")
        elif transition == WindowState.OPENING:
            self._reset_state(clear_window=False)
        elif not is_active:
            self._reset_state(clear_window=True)
            raise ValueError("Window is not active")
        elif is_saturated:
            self._reset_state(clear_window=False)
            raise ValueError("Learning window is saturated")
            
        # Update window phase based on elapsed time
        if self.current_window:
            elapsed = (datetime.now() - self.current_window.start_time).total_seconds()
            total_duration = (self.current_window.end_time - self.current_window.start_time).total_seconds()
            self.window_phase = min(1.0, elapsed / total_duration)
        
        # Calculate base delay with window state awareness
        base_delay = self.back_pressure.calculate_delay(stability_score)
        min_change = 0.1 * (1 + self.window_phase)  # Minimum change grows with window phase
        
        if self.event_queue:
            last_event = self.event_queue[-1]
            last_delay = last_event['delay']
            last_stability = last_event['stability_score']
            
            # Update stability trend with minimal smoothing
            current_change = stability_score - last_stability
            self.stability_trend = (0.8 * self.stability_trend) + (0.2 * current_change)
            
            # Calculate base adjustment factor based on stability change
            stability_change = stability_score - last_stability
            threshold_distance = abs(stability_score - self.current_window.stability_threshold)
            
            if stability_change > 0:  # Stability is improving
                # For improving stability, we want strictly decreasing delays
                if stability_score >= self.current_window.stability_threshold:
                    # Above threshold, decrease more aggressively
                    base_decrease = 0.15 + (0.05 * threshold_distance)
                else:
                    # Below threshold, decrease more conservatively
                    base_decrease = 0.10 + (0.02 * threshold_distance)
                
                # Calculate target delay with strict decrease
                min_decrease = 0.001  # Minimum decrease to ensure ordering
                max_decrease = base_decrease * last_delay
                actual_decrease = max(min_decrease, max_decrease)
                
                # Apply decrease while respecting minimum bound
                delay = max(base_delay * 0.5, last_delay - actual_decrease)
                
                # Double check we maintain strict ordering
                if delay >= last_delay:
                    delay = last_delay - min_decrease
            else:  # Stability is same or worsening
                # For worsening stability, we want strictly increasing delays
                if stability_score < self.current_window.stability_threshold:
                    # Below threshold, increase more aggressively
                    base_increase = 0.15 + (0.05 * threshold_distance)
                else:
                    # Above threshold, increase more conservatively
                    base_increase = 0.10 + (0.02 * threshold_distance)
                
                # Calculate target delay with strict increase
                min_increase = 0.001  # Minimum increase to ensure ordering
                max_increase = base_increase * last_delay
                actual_increase = max(min_increase, max_increase)
                
                # Apply increase while respecting maximum bound
                delay = min(self.back_pressure.max_delay, last_delay + actual_increase)
                
                # Double check we maintain strict ordering
                if delay <= last_delay:
                    delay = last_delay + min_increase
        else:
            # First event - start conservatively and let window phase guide evolution
            if stability_score < self.current_window.stability_threshold:
                delay = base_delay * 1.5  # Start higher but not too aggressive
            else:
                delay = base_delay * 0.8  # Start lower but not too optimistic
        
        # Ensure delay bounds
        delay = max(delay, base_delay * 0.5)
        delay = min(delay, self.back_pressure.max_delay)
        
        # Track successful delay adjustments
        if self.event_queue:
            if (self.stability_trend > 0 and delay <= last_delay) or \
               (self.stability_trend <= 0 and delay >= last_delay):
                # Adjust adaptation rate based on success
                self.adaptation_rate = min(0.2, self.adaptation_rate * 1.1)
            else:
                # Reduce adaptation rate on failure
                self.adaptation_rate = max(0.05, self.adaptation_rate * 0.9)
        
        # Prepare event data with semantic validation
        event_id = f"{event_type}_{entity_id}_{datetime.now().timestamp()}"
        event = {
            "id": event_id,
            "type": event_type,
            "entity_id": entity_id,
            "data": data,
            "stability_score": stability_score,
            "timestamp": datetime.now(),
            "delay": delay,
            "processed": False  # Track processed state
        }
        
        try:
            # Update change count first
            self.current_window.change_count += 1
            
            # Check saturation before adding event
            if self.current_window.is_saturated:
                self._reset_state(clear_window=False)
                return delay
            
            # Update event state
            self.event_queue.append(event)
            self.stability_scores.append(stability_score)
        except:
            # Rollback on error
            if event in self.event_queue:
                self.event_queue.remove(event)
            if stability_score in self.stability_scores:
                self.stability_scores.remove(stability_score)
            if self.current_window:
                self.current_window.change_count = max(0, self.current_window.change_count - 1)
            raise
        
        return delay
        
    def get_window_stats(self) -> Dict:
        """Get current learning window statistics.
        
        Returns:
            Dict containing:
                - change_count: Number of changes in current window
                - is_saturated: Whether window is saturated
                - current_pressure: Current back pressure value
                - stability_trend: Recent stability trend (-1 to 1)
                - time_remaining: Minutes remaining in window
                - error: Error message if no active window
        """
        # Initialize base stats
        stats = {
            "change_count": 0,
            "is_saturated": False,
            "current_pressure": self.back_pressure.current_pressure,
            "stability_trend": 0.0,
            "time_remaining": 0.0,
            "error": None
        }
        
        # Calculate stability trend
        trend = 0.0
        if len(self.stability_scores) >= 2:
            trend = float(np.mean(np.diff(self.stability_scores)))
        stats["stability_trend"] = max(-1.0, min(1.0, trend))
        
        # Check if window exists
        if not self.current_window:
            stats["error"] = "No active learning window"
            return stats
            
        now = datetime.now()
        remaining = (self.current_window.end_time - now).total_seconds() / 60
        
        # Check if window has expired
        if remaining <= 0:
            stats["error"] = "Learning window has expired"
            # Don't reset state here as it would clear change count
            stats["change_count"] = self.current_window.change_count
            stats["is_saturated"] = self.current_window.is_saturated
            stats["time_remaining"] = 0.0
            return stats
        
        # Update window-specific stats
        stats["change_count"] = self.current_window.change_count
        stats["is_saturated"] = self.current_window.is_saturated
        stats["time_remaining"] = max(0.0, remaining)
        
        return stats
        
    def get_pending_events(self, max_events: int = 10) -> List[Dict]:
        """Get pending events up to max_events.
        
        Args:
            max_events: Maximum number of events to return
            
        Returns:
            List of pending events, up to max_events
        """
        # Filter out processed events with semantic validation
        pending = [event for event in self.event_queue 
                  if not event.get('processed', False)]
        
        # Sort by delay (ascending) for consistent ordering
        pending.sort(key=lambda x: x['delay'])
        
        # Return up to max_events with bounds check
        if not max_events or max_events < 0:
            return pending
        return pending[:max_events]
        
    def mark_processed(self, event_id: str) -> None:
        """Mark an event as processed.
        
        Args:
            event_id: ID of the event to mark as processed
        """
        # Semantic validation
        if not event_id:
            return
        
        # Capture window state
        window = self.current_window
        if not window:
            return
            
        # Find and process matching event
        matching_events = []
        for event in list(self.event_queue):
            # Match either full ID or base ID
            if event["id"].startswith(event_id) or event["entity_id"] == event_id:
                matching_events.append(event)
                
        if matching_events:
            # Process all matching events
            for matching_event in matching_events:
                # Update event state with semantic validation
                matching_event['processed'] = True
                self.processed_events[matching_event["id"]] = datetime.now()
                
                # Remove from event queue
                self.event_queue.remove(matching_event)
                
                # Update stability tracking
                if 'stability_score' in matching_event:
                    self.stability_scores.append(matching_event['stability_score'])
                    if len(self.stability_scores) > 10:
                        self.stability_scores.pop(0)
            
            # Check for saturation after processing
            if window.is_saturated:
                self._reset_state(clear_window=True)
