"""Temporal window tracking with error discovery focus."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from enum import Enum

class WindowState(Enum):
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"

@dataclass
class WindowError:
    timestamp: datetime
    error_type: str
    details: str
    
@dataclass
class TemporalWindow:
    start_time: datetime
    duration: timedelta
    state: WindowState = WindowState.ACTIVE
    errors: List[WindowError] = None
    
    def __post_init__(self):
        self.errors = self.errors or []
    
    @property
    def end_time(self) -> datetime:
        return self.start_time + self.duration
        
    @property
    def is_active(self) -> bool:
        return datetime.now() < self.end_time and self.state == WindowState.ACTIVE

class WindowTracker:
    def __init__(self):
        self.windows: Dict[str, TemporalWindow] = {}
        self.error_log: List[WindowError] = []
        
    def create_window(
        self,
        window_id: str,
        duration: timedelta
    ) -> TemporalWindow:
        """Create a new temporal window."""
        if window_id in self.windows:
            self._log_error(
                window_id,
                "window_exists",
                f"Window {window_id} already exists"
            )
            raise ValueError(f"Window {window_id} already exists")
            
        window = TemporalWindow(
            start_time=datetime.now(),
            duration=duration
        )
        self.windows[window_id] = window
        return window
        
    def close_window(self, window_id: str):
        """Close a temporal window."""
        if window_id not in self.windows:
            self._log_error(
                window_id,
                "window_not_found",
                f"Window {window_id} not found"
            )
            raise ValueError(f"Window {window_id} not found")
            
        window = self.windows[window_id]
        if window.state != WindowState.ACTIVE:
            self._log_error(
                window_id,
                "invalid_state",
                f"Window {window_id} is not active"
            )
            raise ValueError(f"Window {window_id} is not active")
            
        window.state = WindowState.CLOSING
        
    def get_active_windows(self) -> List[TemporalWindow]:
        """Get all currently active windows."""
        return [
            window for window in self.windows.values()
            if window.is_active
        ]
        
    def _log_error(
        self,
        window_id: str,
        error_type: str,
        details: str
    ):
        """Log a window-related error."""
        error = WindowError(
            timestamp=datetime.now(),
            error_type=error_type,
            details=details
        )
        self.error_log.append(error)
        
        if window_id in self.windows:
            self.windows[window_id].errors.append(error)
            self.windows[window_id].state = WindowState.ERROR
