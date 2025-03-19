"""
Tests for AdaptiveID learning window integration.

These tests validate the integration between AdaptiveID and learning windows,
ensuring proper registration, notification, and version history retrieval.
"""

import unittest
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import the AdaptiveID class
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

class MockLearningWindow:
    """Mock learning window for testing AdaptiveID integration."""
    
    def __init__(self):
        """Initialize the mock learning window."""
        self.state_changes = []
        self.registered_ids = []
    
    def record_state_change(self, entity_id, change_type, old_value, new_value, origin):
        """Record a state change from an AdaptiveID."""
        self.state_changes.append({
            "entity_id": entity_id,
            "change_type": change_type,
            "old_value": old_value,
            "new_value": new_value,
            "origin": origin,
            "timestamp": datetime.now().isoformat()
        })
        return True


class TestAdaptiveIDLearningIntegration(unittest.TestCase):
    """Test suite for AdaptiveID learning window integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.adaptive_id = AdaptiveID("test_concept", "test_creator")
        self.learning_window = MockLearningWindow()
    
    def test_register_with_learning_window(self):
        """Test registering an AdaptiveID with a learning window."""
        # Register the AdaptiveID with the learning window
        self.adaptive_id.register_with_learning_window(self.learning_window)
        
        # Verify the learning window is registered
        self.assertTrue(hasattr(self.adaptive_id, 'learning_windows'))
        self.assertIn(self.learning_window, self.adaptive_id.learning_windows)
    
    def test_notify_state_change(self):
        """Test notification of state changes to learning windows."""
        # Register the AdaptiveID with the learning window
        self.adaptive_id.register_with_learning_window(self.learning_window)
        
        # Notify state change
        self.adaptive_id.notify_state_change(
            "test_change", 
            "old_value", 
            "new_value", 
            "test_origin"
        )
        
        # Verify the learning window received the notification
        self.assertEqual(len(self.learning_window.state_changes), 1)
        change = self.learning_window.state_changes[0]
        self.assertEqual(change["entity_id"], self.adaptive_id.id)
        self.assertEqual(change["change_type"], "test_change")
        self.assertEqual(change["old_value"], "old_value")
        self.assertEqual(change["new_value"], "new_value")
        self.assertEqual(change["origin"], "test_origin")
    
    def test_get_version_history(self):
        """Test retrieving version history within a time window."""
        # Create some versions
        self.adaptive_id.update_temporal_context("test_key1", "test_value1", "test_origin")
        
        # Wait a bit to ensure timestamps are different
        import time
        time.sleep(0.1)
        
        # Create more versions
        self.adaptive_id.update_temporal_context("test_key2", "test_value2", "test_origin")
        
        # Get version history
        history = self.adaptive_id.get_version_history()
        
        # Verify we got all versions
        self.assertGreaterEqual(len(history), 2)
        
        # Test with time window
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        future = now + timedelta(hours=1)
        
        # All versions should be within this window
        history_time_window = self.adaptive_id.get_version_history(
            start_time=one_hour_ago.isoformat(),
            end_time=future.isoformat()
        )
        self.assertEqual(len(history), len(history_time_window))
    
    def test_update_with_notification(self):
        """Test that updates trigger notifications to learning windows."""
        # Register the AdaptiveID with the learning window
        self.adaptive_id.register_with_learning_window(self.learning_window)
        
        # Update temporal context (should trigger notification)
        self.adaptive_id.update_temporal_context("test_key", "test_value", "test_origin")
        
        # Verify the learning window received the notification
        self.assertEqual(len(self.learning_window.state_changes), 1)
        change = self.learning_window.state_changes[0]
        self.assertEqual(change["entity_id"], self.adaptive_id.id)
        self.assertEqual(change["change_type"], "temporal_context")
        self.assertEqual(change["new_value"]["key"], "test_key")
        self.assertEqual(change["new_value"]["value"], "test_value")
        self.assertEqual(change["origin"], "test_origin")


if __name__ == '__main__':
    unittest.main()
