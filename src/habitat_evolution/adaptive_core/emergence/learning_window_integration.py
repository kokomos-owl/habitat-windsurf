"""
Learning Window Integration for Dynamic Pattern Detection.

This module provides components to integrate dynamic pattern detection with
the learning window system, enabling field-aware state transitions and
back pressure control for pattern evolution.
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import logging
import math
from collections import deque
import numpy as np

# Use absolute imports to avoid module path issues
from src.habitat_evolution.pattern_aware_rag.learning.learning_control import LearningWindow, WindowState, BackPressureController
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from src.habitat_evolution.adaptive_core.emergence.event_bus_integration import PatternEventPublisher
from src.habitat_evolution.adaptive_core.emergence.event_aware_detector import EventAwarePatternDetector

logger = logging.getLogger(__name__)

class LearningWindowAwareDetector:
    """
    Pattern detector that respects learning window states.
    
    This class wraps an EventAwarePatternDetector and controls its behavior
    based on learning window states, ensuring that pattern detection only
    occurs when appropriate according to the learning window system.
    """
    
    def __init__(
        self, 
        detector: EventAwarePatternDetector,
        pattern_publisher: PatternEventPublisher,
        back_pressure_controller: Optional[BackPressureController] = None
    ):
        """
        Initialize a learning window aware detector.
        
        Args:
            detector: The pattern detector to wrap
            pattern_publisher: Publisher for pattern events
            back_pressure_controller: Optional controller for rate regulation
        """
        self.detector = detector
        self.pattern_publisher = pattern_publisher
        self.back_pressure_controller = back_pressure_controller or BackPressureController()
        
        # Learning window state
        self.current_window: Optional[LearningWindow] = None
        self.window_state = WindowState.CLOSED
        self.last_detection_time = datetime.now()
        self.detection_delay = 0.0
        
        # Field metrics
        self.field_metrics = {
            "coherence": 0.7,
            "turbulence": 0.3,
            "stability": 0.8
        }
        
        # Detection history for stability calculation
        self.detection_history = deque(maxlen=10)
        
    def set_learning_window(self, window: LearningWindow) -> None:
        """
        Set the current learning window.
        
        Args:
            window: The learning window to use
        """
        self.current_window = window
        self.window_state = window.state
        
        # Publish window state event
        if self.pattern_publisher:
            self.pattern_publisher.publish_learning_window_state(
                window_id=str(id(window)),
                state=self.window_state.value,
                metrics=self.field_metrics,
                source="learning_window_aware_detector"
            )
        
        logger.info(f"Set learning window with state: {self.window_state}")
    
    def update_window_state(self, state: WindowState) -> None:
        """
        Update the current window state.
        
        Args:
            state: The new window state
        """
        if self.window_state != state:
            self.window_state = state
            
            # Adjust detector threshold based on window state
            if state == WindowState.OPEN:
                # Lower threshold when window is open
                self.detector.dynamic_threshold = max(1, self.detector.threshold - 1)
            else:
                # Reset threshold when window is closed
                self.detector.dynamic_threshold = self.detector.threshold
            
            # Publish window state event
            if self.pattern_publisher:
                self.pattern_publisher.publish_learning_window_state(
                    window_id=str(id(self.current_window)) if self.current_window else "default",
                    state=state.value,
                    metrics=self.field_metrics,
                    source="learning_window_aware_detector"
                )
            
            logger.info(f"Updated window state to: {state}")
    
    def update_field_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Update field metrics.
        
        Args:
            metrics: New field metrics
        """
        self.field_metrics.update(metrics)
        
        # If we have a current window and it's field-aware, check for state transitions
        if self.current_window and self.current_window.field_aware_transitions:
            self._check_field_aware_transitions()
    
    def _check_field_aware_transitions(self) -> None:
        """Check for field-aware state transitions."""
        if not self.current_window:
            return
            
        # Get current coherence and stability
        coherence = self.field_metrics.get("coherence", 0.5)
        stability = self.field_metrics.get("stability", 0.5)
        
        # Determine if we should transition based on field metrics
        if self.window_state == WindowState.CLOSED:
            # Open window if coherence and stability are high
            if coherence > self.current_window.coherence_threshold and stability > self.current_window.stability_threshold:
                self.update_window_state(WindowState.OPENING)
                
                # Schedule transition to OPEN
                self.current_window.next_transition_time = datetime.now() + timedelta(seconds=30)
        
        elif self.window_state == WindowState.OPENING:
            # Check if it's time to transition to OPEN
            if self.current_window.next_transition_time and datetime.now() >= self.current_window.next_transition_time:
                self.update_window_state(WindowState.OPEN)
        
        elif self.window_state == WindowState.OPEN:
            # Close window if coherence or stability drop too low
            if coherence < self.current_window.coherence_threshold or stability < self.current_window.stability_threshold:
                self.update_window_state(WindowState.CLOSED)
    
    def detect_patterns(self) -> List[Dict[str, Any]]:
        """
        Detect patterns respecting learning window state.
        
        Returns:
            List of detected patterns, or empty list if window is closed
        """
        # Check if we can detect patterns based on window state
        if self.window_state != WindowState.OPEN:
            logger.debug(f"Pattern detection skipped: window state is {self.window_state}")
            return []
        
        # Check if we need to wait based on back pressure
        now = datetime.now()
        time_since_last = (now - self.last_detection_time).total_seconds()
        
        if time_since_last < self.detection_delay:
            logger.debug(f"Pattern detection delayed by back pressure: {self.detection_delay - time_since_last:.2f}s remaining")
            return []
        
        # Calculate stability score based on recent detections
        stability_score = self._calculate_stability_score()
        
        # Update detection delay based on stability
        if self.back_pressure_controller:
            self.detection_delay = self.back_pressure_controller.calculate_delay(stability_score)
        
        # Detect patterns
        patterns = self.detector.detect_patterns()
        
        # Update detection history
        self.last_detection_time = now
        self.detection_history.append({
            "time": now,
            "count": len(patterns),
            "stability": stability_score
        })
        
        # If we have a current window, increment its change count
        if self.current_window:
            self.current_window.change_count += len(patterns)
            
            # Check if window is saturated
            if self.current_window.is_saturated:
                self.update_window_state(WindowState.CLOSED)
        
        return patterns
    
    def _calculate_stability_score(self) -> float:
        """
        Calculate stability score based on recent pattern detections.
        
        Returns:
            Stability score between 0.0 and 1.0
        """
        if not self.detection_history:
            return 1.0
        
        # Calculate variance in pattern counts
        counts = [entry["count"] for entry in self.detection_history]
        if len(counts) < 2:
            return 1.0
            
        mean_count = sum(counts) / len(counts)
        if mean_count == 0:
            return 1.0
            
        variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
        
        # Normalize variance to a stability score (higher variance = lower stability)
        # Using a sigmoid function to map variance to a 0-1 range
        normalized_variance = min(10, variance) / 10.0  # Cap at 10 for stability
        stability = 1.0 - (1.0 / (1.0 + math.exp(-5 * (normalized_variance - 0.5))))
        
        return stability


class FieldAwarePatternController:
    """
    Controller for field-aware pattern detection.
    
    This class integrates pattern detection with field gradients and
    learning windows, enabling sophisticated control over when and how
    patterns are detected and evolved.
    """
    
    def __init__(
        self,
        detector: LearningWindowAwareDetector,
        event_bus,
        field_gradient_service=None
    ):
        """
        Initialize a field-aware pattern controller.
        
        Args:
            detector: The learning window aware detector
            event_bus: Event bus for publishing and subscribing to events
            field_gradient_service: Optional service for field gradients
        """
        self.detector = detector
        self.event_bus = event_bus
        self.field_gradient_service = field_gradient_service
        
        # Subscribe to field gradient events
        if self.event_bus:
            self.event_bus.subscribe("field.gradient.update", self._handle_field_gradient_update)
            self.event_bus.subscribe("learning.window.state", self._handle_learning_window_state)
        
        # Create a learning window
        self.learning_window = self._create_default_learning_window()
        self.detector.set_learning_window(self.learning_window)
        
        logger.info("Initialized field-aware pattern controller")
    
    def _create_default_learning_window(self) -> LearningWindow:
        """
        Create a default learning window.
        
        Returns:
            Default learning window
        """
        now = datetime.now()
        return LearningWindow(
            start_time=now,
            end_time=now + timedelta(minutes=30),
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes_per_window=50,
            field_aware_transitions=True
        )
    
    def _handle_field_gradient_update(self, event) -> None:
        """
        Handle field gradient update events.
        
        Args:
            event: Field gradient update event
        """
        try:
            gradients = event.data.get("gradients", {})
            
            # Update field metrics
            metrics = {}
            
            if "coherence" in gradients:
                metrics["coherence"] = gradients["coherence"]
            
            if "turbulence" in gradients:
                metrics["turbulence"] = gradients["turbulence"]
            
            if "stability" in gradients:
                metrics["stability"] = gradients["stability"]
            
            if metrics:
                self.detector.update_field_metrics(metrics)
                logger.debug(f"Updated field metrics: {metrics}")
        
        except Exception as e:
            logger.error(f"Error handling field gradient update: {e}")
    
    def _handle_learning_window_state(self, event) -> None:
        """
        Handle learning window state events.
        
        Args:
            event: Learning window state event
        """
        try:
            state_str = event.data.get("state")
            if state_str:
                # Convert string to WindowState enum
                if state_str == "open":
                    self.detector.update_window_state(WindowState.OPEN)
                elif state_str == "closed":
                    self.detector.update_window_state(WindowState.CLOSED)
                elif state_str == "opening":
                    self.detector.update_window_state(WindowState.OPENING)
        
        except Exception as e:
            logger.error(f"Error handling learning window state event: {e}")
    
    def detect_patterns(self) -> List[Dict[str, Any]]:
        """
        Detect patterns in a field-aware manner.
        
        Returns:
            List of detected patterns
        """
        return self.detector.detect_patterns()
    
    def create_new_learning_window(
        self,
        duration_minutes: int = 30,
        stability_threshold: float = 0.7,
        coherence_threshold: float = 0.6,
        max_changes: int = 50,
        field_aware: bool = True
    ) -> LearningWindow:
        """
        Create a new learning window.
        
        Args:
            duration_minutes: Duration of the window in minutes
            stability_threshold: Minimum stability for changes
            coherence_threshold: Minimum coherence for window to remain open
            max_changes: Maximum changes allowed in window
            field_aware: Whether to enable field-aware transitions
            
        Returns:
            Newly created learning window
        """
        now = datetime.now()
        window = LearningWindow(
            start_time=now,
            end_time=now + timedelta(minutes=duration_minutes),
            stability_threshold=stability_threshold,
            coherence_threshold=coherence_threshold,
            max_changes_per_window=max_changes,
            field_aware_transitions=field_aware
        )
        
        self.detector.set_learning_window(window)
        return window
