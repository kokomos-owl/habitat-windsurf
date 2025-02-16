"""Learning window manager for Pattern-Aware RAG.

This module provides window management and back pressure control
for pattern evolution and state changes.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .learning_control import LearningWindow, BackPressureController

class LearningWindowManager:
    """Manages learning windows and applies back pressure controls."""
    
    def __init__(self):
        """Initialize window manager with default settings."""
        self.back_pressure = BackPressureController()
        self.current_window = self._create_default_window()
    
    def apply_constraints(self, pressure: float) -> float:
        """Apply window constraints to pressure value.
        
        Args:
            pressure: Current pressure value
            
        Returns:
            Adjusted pressure value between 0 and 1
        """
        # Ensure window is active
        if not self.current_window.is_active:
            self.current_window = self._create_default_window()
            
        # Apply back pressure if window is saturated
        if self.current_window.is_saturated:
            pressure = min(1.0, pressure * 1.5)
            
        # Keep pressure within bounds
        return max(0.0, min(1.0, pressure))
    
    def _create_default_window(self) -> LearningWindow:
        """Create a default learning window."""
        now = datetime.now()
        return LearningWindow(
            start_time=now,
            end_time=now + timedelta(minutes=30),
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes_per_window=50
        )
