"""
Learning window control module for Pattern-Aware RAG.

This module provides the learning window abstraction that controls when
semantic changes are allowed to occur in the system, along with back
pressure mechanisms to maintain stability during periods of change.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
from collections import deque
import asyncio
import uuid

from dataclasses import dataclass

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
    field_observers: List = None  # Track field observers
    field_aware_transitions: bool = False  # Enable/disable field awareness
    next_transition_time: datetime = None  # Next optimal transition time
    stability_metrics: List = None  # Track stability metrics
    
    def __post_init__(self):
        self.field_observers = []
        self.stability_scores = []
        self.stability_metrics = []
        
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
    def stability_score(self) -> float:
        """Calculate current stability score.
        
        Returns:
            Current stability score between 0 and 1
        """
        if not self.stability_scores:
            return 1.0  # Fully stable if no scores recorded
            
        # Use recent history for stability trend
        recent_scores = self.stability_scores[-10:]
        return sum(recent_scores) / len(recent_scores)
        
    @property
    def is_saturated(self) -> bool:
        """Check if window is saturated with changes.
        
        Returns:
            True if window has reached max changes
        """
        return self.change_count >= self.max_changes_per_window
        
    def transition_if_needed(self) -> Optional[WindowState]:
        """Check for state transition and return new state if changed."""
        current_state = self.state
        if current_state != self._state:
            self._state = current_state
            return current_state
        return None

    def register_field_observer(self, observer):
        """Register a field observer to receive state notifications.
        
        Args:
            observer: The field observer to register
        """
        self.field_observers.append(observer)
        
        # Immediately notify observer of current state
        try:
            import asyncio
            if asyncio.get_event_loop().is_running():
                asyncio.create_task(self.notify_field_observers())
            else:
                # Sync fallback for when there's no running event loop (test environment)
                current_state = self.state
                context = {
                    "state": current_state.value,
                    "stability": self.stability_score,
                    "coherence": self.coherence_threshold,
                    "saturation": self.change_count / max(1, self.max_changes_per_window)
                }
                # Direct append to observations
                observer.observations.append({"context": context, "time": datetime.now()})
        except (RuntimeError, ImportError):
            # Fallback if no event loop can be accessed
            current_state = self.state
            context = {
                "state": current_state.value,
                "stability": self.stability_score,
                "coherence": self.coherence_threshold,
                "saturation": self.change_count / max(1, self.max_changes_per_window)
            }
            observer.observations.append({"context": context, "time": datetime.now()})

    async def notify_field_observers(self):
        """Notify all field observers of current state."""
        current_state = self.state
        context = {
            "state": current_state.value,
            "stability": self.stability_score,
            "coherence": self.coherence_threshold,
            "saturation": self.change_count / max(1, self.max_changes_per_window)
        }
        
        # Notify each observer asynchronously
        for observer in self.field_observers:
            try:
                await observer.observe(context)
            except Exception as e:
                print(f"Error notifying field observer: {e}")

    async def notify_state_change(self) -> None:
        """Notify observers of state changes without coupling."""
        context = {
            "state": self.state.value,
            "stability": self.stability_score,
            "coherence": self.coherence_threshold,
            "saturation": self.change_count / max(1, self.max_changes_per_window)
        }
        
        # Notify field observers
        await self.notify_field_observers()

    def get_harmonic_analysis(self) -> List[float]:
        """Get harmonic analysis from field observers if available."""
        if not self.field_observers:
            return []
            
        # Find first observer with harmonic analysis
        for observer in self.field_observers:
            if hasattr(observer, 'harmonic_analysis') and observer.harmonic_analysis:
                return observer.harmonic_analysis
                
        return []
    
    def find_optimal_transition_time(self) -> Optional[datetime]:
        """Find optimal time for next state transition based on field observers."""
        if not self.field_observers or not self.field_aware_transitions:
            return None
            
        transition_times = []
        for observer in self.field_observers:
            if hasattr(observer, 'get_optimal_transition_time'):
                time = observer.get_optimal_transition_time(self.start_time, 
                                                         self.end_time - self.start_time)
                if time:
                    transition_times.append(time)
                    
        if transition_times:
            # Choose earliest reasonable transition time
            now = datetime.now()
            future_times = [t for t in transition_times if t > now]
            if future_times:
                self.next_transition_time = min(future_times)
                return self.next_transition_time
                
        return None

    def record_change(self, stability_score: float) -> None:
        """Record a change event with its stability score.
        
        Args:
            stability_score: The stability score of the change
        """
        self.change_count += 1
        self.stability_metrics.append(stability_score)
        
        # Trigger field observers notification on significant state changes
        try:
            import asyncio
            if asyncio.get_event_loop().is_running():
                asyncio.create_task(self.notify_state_change())
        except (RuntimeError, ImportError):
            pass

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
    """Coordinates events and changes during learning windows."""
    
    def __init__(self, max_queue_size: int = 1000, persistence_mode: bool = True):
        """Initialize event coordinator.
        
        Args:
            max_queue_size: Maximum size of event queue
            persistence_mode: Whether to use persistence (Neo4j) or direct mode
        """
        self.event_queue = deque(maxlen=max_queue_size)
        self.current_window = None
        self.processed_events = {}  # Changed back to Dict for backwards compatibility
        self.persistence_mode = persistence_mode
        self.back_pressure = BackPressureController()
        self.health_service = None
        self.field_observers = []
        self.stability_scores = []  # Track stability scores
        self.stability_trend = 0.0  # Track stability direction
        self.adaptation_rate = 0.1  # How quickly we adapt to window state
        self.window_phase = 0.0  # Track window's learning phase (0 to 1)

    def create_learning_window(
            self,
            duration_minutes: int = 10,
            stability_threshold: float = 0.7,
            coherence_threshold: float = 0.6,
            max_changes: int = 20,
            health_service = None
        ) -> LearningWindow:
        """Create a new learning window.
        
        Args:
            duration_minutes: Length of learning window in minutes
            stability_threshold: Minimum stability score for changes
            coherence_threshold: Minimum coherence for window to remain open
            max_changes: Maximum changes allowed in window
            health_service: Optional health service for field awareness
            
        Returns:
            Newly created learning window
        """
        now = datetime.now()
        window = LearningWindow(
            start_time=now,
            end_time=now + timedelta(minutes=duration_minutes),
            stability_threshold=stability_threshold,
            coherence_threshold=coherence_threshold,
            max_changes_per_window=max_changes
        )
        
        # Store health service if provided
        if health_service:
            self.health_service = health_service
            
        # Register field observers if available
        for observer in self.field_observers:
            window.register_field_observer(observer)
            
        # If we have a health service but no observers, create one
        if self.health_service and not self.field_observers:
            observer = self._create_field_observer()
            window.register_field_observer(observer)
            
        self.current_window = window
        return window
    
    def _create_field_observer(self):
        """Create a field observer based on available services."""
        # Use imported class if available, otherwise return generic observer
        if hasattr(self, 'health_service') and self.health_service:
            # Try importing HealthFieldObserver dynamically to avoid circular imports
            try:
                from ..learning.learning_health_integration import HealthFieldObserver
                return HealthFieldObserver(f"window_{id(self.current_window)}", self.health_service)
            except ImportError:
                pass
        
        # Fallback to generic observer using dynamic import
        try:
            from ..learning.learning_health_integration import FieldObserver
            return FieldObserver(f"window_{id(self.current_window)}")
        except ImportError:
            # Last resort - create minimal compatible observer
            class MinimalObserver:
                def __init__(self, field_id):
                    self.field_id = field_id
                    self.observations = []
                
                async def observe(self, context):
                    self.observations.append({"context": context})
                    return {"observed": True}
                
                def get_field_metrics(self):
                    return {}
            
            return MinimalObserver(f"window_{id(self.current_window)}")

    def queue_event(
        self,
        event_type: str,
        entity_id: str,
        data: Dict[str, Any],
        stability_score: float = 1.0
    ) -> float:
        """Queue an event and verify against back-pressure.
        
        Args:
            event_type: Type of event
            entity_id: Entity identifier
            data: Event data
            stability_score: Score indicating event stability (0.0-1.0)
            
        Returns:
            Delay in seconds
        """
        if not self.current_window or self.current_window.state not in [WindowState.OPEN, WindowState.OPENING]:
            raise ValueError("No active learning window")
            
        # Check if window is saturated
        if self.current_window.is_saturated:
            raise ValueError("Learning window is saturated")
            
        # Create event
        event = {
            "id": str(uuid.uuid4()),
            "type": event_type,
            "entity_id": entity_id,
            "data": data,
            "stability": stability_score,
            "timestamp": datetime.now().isoformat(),
            "processed": False
        }
        
        # Calculate delay based on stability
        delay = self.back_pressure.calculate_delay(stability_score)
        
        # Add delay to event for sorting in get_pending_events
        event["delay"] = delay
        
        # Add to window and check back pressure
        self.event_queue.append(event)
        
        # Record the change in the current window
        if self.current_window:
            self.current_window.record_change(stability_score)
        
        # Store result for retrieval
        self.processed_events[event["id"]] = {
            "status": "accepted", 
            "event_id": event["id"],
            "delay": delay
        }
        
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
                self.processed_events[matching_event["id"]] = {"context": matching_event}
                
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

    def _reset_state(self, clear_window: bool = True):
        """Reset internal state.
        
        Args:
            clear_window: If True, also clear the current window
        """
        # Always clear event state
        self.event_queue.clear()
        self.stability_scores.clear()
        self.processed_events = {}
        
        # Optionally clear window
        if clear_window:
            self.current_window = None
        
        # Reset delay tracking
        if hasattr(self, 'last_delay'):
            del self.last_delay
