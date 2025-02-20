"""Learning window manager for Pattern-Aware RAG.

This module provides window management and back pressure control
for pattern evolution and state changes.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np

from .learning_control import LearningWindow, BackPressureController

class LearningWindowManager:
    """Manages learning windows and applies back pressure controls."""
    
    def __init__(self):
        """Initialize window manager with default settings."""
        self.back_pressure = BackPressureController()
        self.current_window = self._create_default_window()
        self._windows = [self.current_window]
        self._stability_scores = []
        self._flow_rates = []
        self._coordination_metrics = {}
    
    def apply_constraints(self, pressure: float) -> float:
        """Apply pressure constraints to the current window.
        
        Args:
            pressure: Current pressure value to apply
            
        Returns:
            float: Calculated delay based on pressure
        """
        self.current_window.change_count = min(
            self.current_window.change_count + 1,
            self.current_window.max_changes_per_window
        )
        return self.back_pressure.calculate_delay(pressure)
    
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
    
    async def get_stability_metrics(self) -> Dict[str, float]:
        """Get current stability metrics."""
        stability = sum(self._stability_scores[-10:]) / 10 if self._stability_scores else 0.5
        return {
            'stability': stability,
            'trend': stability - (sum(self._stability_scores[-20:-10]) / 10 if len(self._stability_scores) >= 20 else stability),
            'variance': np.var(self._stability_scores[-30:]) if len(self._stability_scores) >= 30 else 0.0
        }
    
    async def get_recovery_metrics(self) -> Dict[str, float]:
        """Get current recovery metrics."""
        return {
            'recovery_rate': 1.0 - (self.current_window.change_count / self.current_window.max_changes_per_window),
            'stability_recovery': sum(s > self.current_window.stability_threshold 
                                   for s in self._stability_scores[-5:]) / 5 if self._stability_scores else 0.0
        }
    
    def get_flow_rate(self) -> float:
        """Get current pattern flow rate."""
        return sum(self._flow_rates[-5:]) / 5 if self._flow_rates else 0.0
    
    async def get_coordination_metrics(self) -> Dict[str, float]:
        """Get window coordination metrics."""
        return {
            w.state.value: sum(1 for other in self._windows 
                              if other.state == w.state) / len(self._windows)
            for w in self._windows
        }
    
    def get_system_stability(self) -> float:
        """Get overall system stability."""
        return sum(self._stability_scores[-30:]) / 30 if len(self._stability_scores) >= 30 else 0.5
    
    async def get_boundary_metrics(self) -> Dict[str, float]:
        """Get boundary formation metrics."""
        boundaries = {
            'coherence': sum(w.coherence_threshold for w in self._windows) / len(self._windows),
            'stability': sum(w.stability_threshold for w in self._windows) / len(self._windows),
            'pressure': self.back_pressure.current_pressure
        }
        return boundaries
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
