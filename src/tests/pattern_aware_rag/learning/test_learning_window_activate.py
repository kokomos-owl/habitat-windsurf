"""
Unit tests for the LearningWindow activate method.

These tests verify the behavior of the activate method in the LearningWindow class,
ensuring proper state transitions and observer notifications.
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Add src to path if not already there
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import modules using relative imports
sys.path.append(os.path.join(src_path, 'src'))
from habitat_evolution.pattern_aware_rag.learning.learning_control import LearningWindow, WindowState

# Mock observer for testing
class MockObserver:
    """Mock observer for testing observer notifications."""
    
    def __init__(self):
        """Initialize with empty observations list."""
        self.observations = []
    
    async def observe(self, context):
        """Observe context and store in observations."""
        self.observations.append({"context": context, "time": datetime.now()})
        return {"observed": True}
    
    def observe_pattern_evolution(self, context):
        """Observe pattern evolution and store in observations."""
        self.observations.append({"context": context, "time": datetime.now()})


class TestLearningWindowActivate:
    """Test suite for LearningWindow activate method."""
    
    def test_activate_basic(self):
        """Test basic activation of a learning window."""
        # Create window with fixed times
        now = datetime.now()
        window = LearningWindow(
            start_time=now,
            end_time=now + timedelta(minutes=10),
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes_per_window=20
        )
        
        # Verify initial state
        assert window._state == WindowState.CLOSED
        
        # Activate the window
        result = window.activate(origin="test")
        
        # Verify activation was successful
        assert result is True
        assert window._state == WindowState.OPENING
    
    def test_activate_already_open(self):
        """Test activation of an already open window."""
        # Create window with fixed times
        now = datetime.now()
        window = LearningWindow(
            start_time=now,
            end_time=now + timedelta(minutes=10),
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes_per_window=20
        )
        
        # First activation should succeed
        assert window.activate(origin="test") is True
        
        # Second activation should fail
        assert window.activate(origin="test") is False
        
        # State should remain OPENING
        assert window._state == WindowState.OPENING
    
    def test_activate_with_stability_score(self):
        """Test activation with a custom stability score."""
        # Create window with fixed times
        now = datetime.now()
        window = LearningWindow(
            start_time=now,
            end_time=now + timedelta(minutes=10),
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes_per_window=20
        )
        
        # Activate with custom stability score
        window.activate(origin="test", stability_score=0.85)
        
        # Verify stability score was set
        assert 0.85 in window.stability_scores
        
    def test_activate_notifies_observers(self):
        """Test that activate notifies field observers."""
        # Create window with fixed times
        now = datetime.now()
        window = LearningWindow(
            start_time=now,
            end_time=now + timedelta(minutes=10),
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes_per_window=20
        )
        
        # Create and register mock observer
        observer = MockObserver()
        window.register_field_observer(observer)
        
        # Activate window
        window.activate(origin="test")
        
        # Verify observer was notified
        assert len(observer.observations) > 0
        
        # Check notification content - the context will be from notify_field_observers
        # which doesn't include the window_activated event
        notification = observer.observations[0]["context"]
        assert "state" in notification
        assert notification["state"] == "opening"
    
    def test_activate_with_pattern_observers(self):
        """Test activation with pattern observers."""
        # Create window with fixed times
        now = datetime.now()
        window = LearningWindow(
            start_time=now,
            end_time=now + timedelta(minutes=10),
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes_per_window=20
        )
        
        # Create mock pattern observer
        pattern_observer = MockObserver()
        
        # Add pattern observer attributes to window
        window.pattern_observers = [pattern_observer]
        window.notify_pattern_observers = lambda context: [
            observer.observe_pattern_evolution(context) 
            for observer in window.pattern_observers
        ]
        
        # Activate window
        window.activate(origin="test")
        
        # Verify pattern observer was notified
        assert len(pattern_observer.observations) > 0
        
        # Check notification content
        notification = pattern_observer.observations[0]["context"]
        assert "window_activated" in str(notification)
