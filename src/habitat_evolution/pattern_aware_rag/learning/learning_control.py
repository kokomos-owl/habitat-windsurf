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
        
        if len(window) > 1:
            # Natural rhythm detection through harmonic analysis
            diffs = np.diff(window_array)
            
            # Find the natural frequency of stability changes
            zero_crossings = np.where(np.diff(np.signbit(diffs)))[0]
            if len(zero_crossings) >= 2:
                rhythm_period = np.mean(np.diff(zero_crossings))
                # How far are we into current cycle?
                phase = (len(diffs) - zero_crossings[-1]) / rhythm_period
            else:
                rhythm_period = len(window)
                phase = 0.5
            
            # Detect if we're in stress-building or recovery phase
            recent_diffs = diffs[-min(3, len(diffs)):]
            stress_building = np.sum(recent_diffs < 0) > len(recent_diffs)/2
            
            # Natural variability as amplitude of oscillation
            variability = np.std(window_array)
            
            # Resonance - how well stability changes match natural rhythm
            if len(diffs) > 2:
                expected_cycle = np.sin(np.linspace(0, 2*np.pi, len(diffs)))
                actual_cycle = diffs / np.std(diffs)
                resonance = np.abs(np.corrcoef(expected_cycle, actual_cycle)[0,1])
            else:
                resonance = 0.5
            
            # Natural wave emergence
            # Base frequency from stability threshold
            base_freq = 2 * np.pi * (1 - self.stability_threshold)
            
            # 1. Fundamental wave - stronger below threshold
            if current_score < self.stability_threshold:
                # Non-linear amplification in unstable region
                amp = 2 * (self.stability_threshold - current_score) ** 0.5
                base_pressure = 0.5 + amp * np.sin(base_freq * phase)
            else:
                # Gentle oscillation in stable region
                base_pressure = 0.3 * (1 + np.sin(base_freq * phase))
            
            # 2. Harmonic pressure - resonates with natural frequency
            if resonance > 0.6:
                harmonic = 0.4 * np.sin(2 * base_freq * phase)
                harmonic_pressure = resonance * harmonic
            else:
                harmonic_pressure = 0
            
            # 3. Stress wave - builds during instability
            stress_amp = 1.5 if stress_building else 0.5
            stress_pressure = stress_amp * variability
            
            # 4. Memory wave - accumulates with sustained instability
            memory_factor = len([s for s in window if s < self.stability_threshold])
            memory_pressure = 0.2 * (memory_factor / len(window)) ** 2
            
            # Combine waves through natural interference
            pressures = np.array([base_pressure, harmonic_pressure, stress_pressure, memory_pressure])
            
            # Dynamic weights based on system state
            if current_score < self.stability_threshold:
                # Stronger response in unstable region
                weights = np.array([2.5, 1.5, 2.0, 2.0])
            else:
                # Gentler response in stable region
                weights = np.array([1.0, 0.5, 0.8, 0.7])
            
            # Constructive interference
            raw_pressure = np.sum(weights * pressures) / np.sum(weights)
            
            # Resonant amplification
            if stress_building and resonance > 0.7:
                # System is in deep harmony with stress pattern
                raw_pressure *= 2.0
            
            # Natural amplification when system needs change
            if current_score < self.stability_threshold:
                # Stronger response when further from stability
                amplification = 1 + (self.stability_threshold - current_score)
                target_pressure = raw_pressure * amplification
            else:
                target_pressure = raw_pressure
        else:
            # Initial emergence
            target_pressure = 1 - current_score if current_score < self.stability_threshold else 0
        
        # Ensure natural growth (monotonic but organic)
        target_pressure = max(target_pressure, self.current_pressure)
        
        # Natural step size based on distance
        step_size = 0.1 * (1 + abs(target_pressure - self.current_pressure))
        
        # Move pressure toward target naturally
        if target_pressure > self.current_pressure:
            self.current_pressure = min(target_pressure,
                                      self.current_pressure + step_size)
        
        # Ensure sufficient response to instability
        if current_score < self.stability_threshold:
            self.current_pressure = max(self.current_pressure, 0.6)
        
        # Natural bounds
        self.current_pressure = np.clip(self.current_pressure, 0, 1)
        
        # Base delay emerges from pressure wave
        base_delay = self.base_delay + (self.max_delay * self.current_pressure)
        
        # Natural delay modulation
        if len(window) > 1:
            # Delay grows with sustained instability
            instability_duration = np.sum(window < self.stability_threshold) / len(window)
            growth_factor = 1 + (instability_duration ** 2)
            
            # Phase affects growth rate
            if stress_building:
                # Faster growth during stress phase
                phase_factor = 1 + (phase ** 0.5)
            else:
                # Slower growth during recovery
                phase_factor = 1 + phase
            
            # Calculate natural delay progression
            delay = base_delay * growth_factor * phase_factor
            
            # Ensure strictly increasing delays
            if hasattr(self, 'last_delay'):
                min_increase = 0.1 * (1 + instability_duration)  # Larger steps during instability
                delay = max(delay, self.last_delay + min_increase)
        else:
            delay = base_delay
            if hasattr(self, 'last_delay'):
                delay = max(delay, self.last_delay + 0.1)
        
        # Remember for monotonicity
        self.last_delay = delay
        
        return np.clip(delay, self.base_delay, self.max_delay)

class EventCoordinator:
    """Coordinates events between state evolution and adaptive IDs."""
    
    def __init__(self, max_queue_size: int = 1000):
        self.event_queue = deque(maxlen=max_queue_size)
        self.processed_events: Dict[str, datetime] = {}
        self.current_window: Optional[LearningWindow] = None
        self.back_pressure = BackPressureController()
        self.stability_scores: List[float] = []  # Track stability scores
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
            change_count=0
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
            
        # Calculate base delay following natural rhythm
        base_delay = self.back_pressure.calculate_delay(stability_score)
        
        # Calculate base delay based on stability
        base_delay = self.back_pressure.calculate_delay(stability_score)
        min_increase = 0.1
        
        if self.event_queue:
            # Get last event info
            last_event = self.event_queue[-1]
            last_delay = last_event['delay']
            last_stability = last_event['stability_score']
            
            # Calculate stability trend
            stability_change = stability_score - last_stability
            
            if stability_change > 0:
                # Improving stability - decrease delay
                trend_impact = -stability_change * 0.5
                delay = max(base_delay, last_delay + trend_impact)
            else:
                # Worsening stability - increase delay
                trend_impact = abs(stability_change)
                if stability_score < self.current_window.stability_threshold:
                    # Below threshold - aggressive increase
                    trend_impact *= 2.0
                    delay = last_delay + min_increase * 2 + trend_impact
                else:
                    # Above threshold - gentle increase
                    delay = last_delay + min_increase + trend_impact
        else:
            # First event - base calculation
            if stability_score < self.current_window.stability_threshold:
                # Start higher when below threshold
                instability = self.current_window.stability_threshold - stability_score
                delay = base_delay * (1 + instability)
            else:
                delay = base_delay
        
        # Ensure delay bounds
        delay = max(delay, base_delay)
        delay = min(delay, self.back_pressure.max_delay)
        
        # Update last delay
        if not hasattr(self, 'last_delay') or stability_change > 0:
            self.last_delay = delay
        else:
            # Ensure strictly increasing delays for decreasing stability
            delay = max(delay, self.last_delay + min_increase)
            self.last_delay = delay
        
        # Prepare event data
        event_id = f"{event_type}_{entity_id}_{datetime.now().timestamp()}"
        event = {
            "id": event_id,
            "type": event_type,
            "entity_id": entity_id,
            "data": data,
            "stability_score": stability_score,
            "timestamp": datetime.now(),
            "delay": delay
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
            self._reset_state()
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
        # Filter out processed events
        pending = [event for event in self.event_queue 
                  if event['id'] not in self.processed_events]
        
        # Sort by delay (ascending)
        pending.sort(key=lambda x: x['delay'])
        
        # Return up to max_events
        return pending[:max_events] if max_events else pending
        
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
            
        # Find matching event
        matching_event = None
        for event in list(self.event_queue):
            if event["id"].startswith(event_id):
                matching_event = event
                break
                
        if matching_event:
            try:
                # Update processed state
                self.processed_events[matching_event["id"]] = datetime.now()
                
                # Update event state
                self.event_queue.remove(matching_event)
                if self.stability_scores:
                    self.stability_scores.pop(0)
                
                # Check saturation
                if window.is_saturated:
                    self._reset_state(clear_window=False)
            except:
                # Rollback on error
                if matching_event["id"] in self.processed_events:
                    del self.processed_events[matching_event["id"]]
                if matching_event not in self.event_queue:
                    self.event_queue.append(matching_event)
                raise
