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
import numpy as np

@dataclass
class LearningWindow:
    """Represents a temporal learning window for pattern evolution."""
    start_time: datetime
    end_time: datetime
    stability_threshold: float
    coherence_threshold: float
    max_changes_per_window: int
    change_count: int = 0
    
    @property
    def is_active(self) -> bool:
        """Check if the window is currently active."""
        now = datetime.now()
        return self.start_time <= now <= self.end_time
    
    @property
    def is_saturated(self) -> bool:
        """Check if window has reached max changes."""
        return self.change_count >= self.max_changes_per_window

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
        """Calculate delay based on stability score."""
        self.stability_window.append(stability_score)
        
        # Calculate trend
        if len(self.stability_window) >= 2:
            trend = np.mean(np.diff(list(self.stability_window)))
        else:
            trend = 0
            
        # Update pressure based on stability and trend
        if stability_score < self.stability_threshold:
            # Increase pressure when stability is low
            self.current_pressure = min(
                1.0,
                self.current_pressure + (0.1 * abs(trend))
            )
        else:
            # Gradually release pressure when stable
            self.current_pressure = max(
                0.0,
                self.current_pressure - 0.05
            )
            
        return self.base_delay + (self.max_delay * self.current_pressure)

class EventCoordinator:
    """Coordinates events between state evolution and adaptive IDs."""
    
    def __init__(self, max_queue_size: int = 1000):
        self.event_queue = deque(maxlen=max_queue_size)
        self.processed_events: Dict[str, datetime] = {}
        self.current_window: Optional[LearningWindow] = None
        self.back_pressure = BackPressureController()
    
    def create_learning_window(
        self,
        duration_minutes: int = 30,
        stability_threshold: float = 0.7,
        coherence_threshold: float = 0.6,
        max_changes: int = 50
    ) -> LearningWindow:
        """Create a new learning window."""
        now = datetime.now()
        window = LearningWindow(
            start_time=now,
            end_time=now + timedelta(minutes=duration_minutes),
            stability_threshold=stability_threshold,
            coherence_threshold=coherence_threshold,
            max_changes_per_window=max_changes
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
        """Queue an event and return recommended delay."""
        # Check window status
        if not self.current_window or not self.current_window.is_active:
            self.create_learning_window()
        
        # Apply back pressure
        delay = self.back_pressure.calculate_delay(stability_score)
        
        # Queue event if window not saturated
        if not self.current_window.is_saturated:
            event = {
                'id': f"{event_type}_{entity_id}_{datetime.now().isoformat()}",
                'type': event_type,
                'entity_id': entity_id,
                'timestamp': datetime.now(),
                'data': data,
                'stability': stability_score,
                'window_id': id(self.current_window)
            }
            self.event_queue.append(event)
            self.current_window.change_count += 1
            
        return delay
    
    def get_pending_events(
        self,
        max_events: int = 10
    ) -> List[Dict]:
        """Get pending events up to max_events."""
        events = []
        while len(events) < max_events and self.event_queue:
            event = self.event_queue.popleft()
            events.append(event)
        return events
    
    def mark_processed(self, event_id: str):
        """Mark an event as processed."""
        self.processed_events[event_id] = datetime.now()
        
    def get_window_stats(self) -> Dict:
        """Get statistics for the current learning window."""
        if not self.current_window:
            return {}
            
        return {
            'start_time': self.current_window.start_time,
            'end_time': self.current_window.end_time,
            'change_count': self.current_window.change_count,
            'is_saturated': self.current_window.is_saturated,
            'current_pressure': self.back_pressure.current_pressure,
            'pending_events': len(self.event_queue)
        }
