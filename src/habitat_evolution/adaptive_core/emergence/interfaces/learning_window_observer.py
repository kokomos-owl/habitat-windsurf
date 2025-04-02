"""
Learning Window Observer Interface for Vector-Tonic Persistence Integration.

This module defines the interface for observing learning window events
in the Vector-Tonic Window system. It supports the pattern evolution and 
co-evolution principles of Habitat Evolution by enabling the observation
of semantic change across the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum


class LearningWindowState(Enum):
    """Enumeration of possible learning window states."""
    CLOSED = "CLOSED"
    OPENING = "OPENING"
    OPEN = "OPEN"
    CLOSING = "CLOSING"


class LearningWindowObserverInterface(ABC):
    """
    Interface for observers of learning window events.
    
    This interface defines the methods that must be implemented by any class
    that wants to observe learning window events in the Vector-Tonic Window system.
    It enables components to react to window state changes and process patterns
    that emerge during learning windows.
    """
    
    @abstractmethod
    def on_window_state_change(self, window_id: str, previous_state: LearningWindowState, 
                              new_state: LearningWindowState, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when a learning window changes state.
        
        Args:
            window_id: The ID of the learning window.
            previous_state: The previous state of the window.
            new_state: The new state of the window.
            metadata: Optional metadata about the state change.
        """
        pass
    
    @abstractmethod
    def on_window_open(self, window_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when a learning window opens.
        
        Args:
            window_id: The ID of the learning window.
            metadata: Optional metadata about the window.
        """
        pass
    
    @abstractmethod
    def on_window_close(self, window_id: str, patterns_detected: Dict[str, Any], 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when a learning window closes.
        
        Args:
            window_id: The ID of the learning window.
            patterns_detected: Patterns detected during the window.
            metadata: Optional metadata about the window.
        """
        pass
    
    @abstractmethod
    def on_back_pressure(self, window_id: str, pressure_level: float, 
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when back pressure is detected in a learning window.
        
        Back pressure indicates that the system is experiencing stress and may need
        to adjust its processing rate or parameters.
        
        Args:
            window_id: The ID of the learning window.
            pressure_level: The level of back pressure (0.0 to 1.0).
            metadata: Optional metadata about the back pressure.
        """
        pass
