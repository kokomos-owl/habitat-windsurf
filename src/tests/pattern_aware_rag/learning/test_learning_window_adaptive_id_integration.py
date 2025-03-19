"""
Tests for LearningWindow integration with AdaptiveID.

These tests validate the integration between LearningWindow and AdaptiveID,
ensuring proper state change tracking and field observer notification.
"""

import unittest
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Any, Optional

from habitat_evolution.pattern_aware_rag.learning.learning_control import LearningWindow, WindowState
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

class TestLearningWindowAdaptiveIDIntegration(unittest.TestCase):
    """Test suite for LearningWindow integration with AdaptiveID."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a learning window
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(minutes=10)
        self.learning_window = LearningWindow(
            start_time=self.start_time,
            end_time=self.end_time,
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes_per_window=20
        )
        
        # Create an AdaptiveID instance
        self.adaptive_id = AdaptiveID("test_concept", "test_creator")
        
        # Create a mock field observer
        self.field_observer = MagicMock()
        self.field_observer.observations = []
        self.field_observer.observe = MagicMock(return_value={"status": "observed"})
        
        # Register the field observer with the learning window
        self.learning_window.register_field_observer(self.field_observer)
    
    def test_record_state_change_basic(self):
        """Test basic state change recording."""
        # Record a state change
        self.learning_window.record_state_change(
            change_type="test_change",
            old_value="old_value",
            new_value="new_value",
            origin="test_origin",
            entity_id="test_entity",
            tonic_value=0.6,
            stability=0.8
        )
        
        # Verify change was recorded
        self.assertEqual(self.learning_window.change_count, 1)
        self.assertEqual(len(self.learning_window.stability_metrics), 1)
        self.assertEqual(self.learning_window.stability_metrics[0], 0.8)
        
        # Verify field observer was notified
        self.assertEqual(len(self.field_observer.observations), 1)
        context = self.field_observer.observations[0]["context"]
        self.assertEqual(context["entity_id"], "test_entity")
        self.assertEqual(context["change_type"], "test_change")
        self.assertEqual(context["old_value"], "old_value")
        self.assertEqual(context["new_value"], "new_value")
        self.assertEqual(context["origin"], "test_origin")
        self.assertEqual(context["stability"], 0.8)
        self.assertEqual(context["tonic_value"], 0.6)
        
        # Verify harmonic value was calculated
        self.assertEqual(context["harmonic_value"], 0.8 * 0.6)
    
    def test_record_state_change_with_default_stability(self):
        """Test state change recording with default stability."""
        # Set up a stability score for the learning window
        self.learning_window.stability_scores = [0.75, 0.8, 0.85]
        
        # Record a state change without providing stability
        self.learning_window.record_state_change(
            change_type="test_change",
            old_value="old_value",
            new_value="new_value",
            origin="test_origin",
            entity_id="test_entity"
        )
        
        # Verify change was recorded with default stability
        self.assertEqual(self.learning_window.change_count, 1)
        self.assertEqual(len(self.learning_window.stability_metrics), 1)
        
        # Verify field observer was notified with default stability
        context = self.field_observer.observations[0]["context"]
        self.assertEqual(context["stability"], self.learning_window.stability_score)
    
    def test_adaptive_id_integration(self):
        """Test integration with AdaptiveID."""
        # Register the AdaptiveID with the field observer
        self.adaptive_id.register_with_field_observer(self.field_observer)
        
        # Clear initial observation
        self.field_observer.observations = []
        
        # Register the learning window with the AdaptiveID
        self.adaptive_id.learning_windows = [self.learning_window]
        
        # Notify state change from AdaptiveID
        self.adaptive_id.notify_state_change(
            "test_change", 
            "old_value", 
            "new_value", 
            "test_origin"
        )
        
        # Verify field observer was notified
        self.assertEqual(len(self.field_observer.observations), 1)
        
        # Verify learning window recorded the change
        self.assertEqual(self.learning_window.change_count, 1)
    
    def test_tonic_harmonic_properties(self):
        """Test tonic-harmonic properties in state change notifications."""
        # Record a state change with tonic value
        self.learning_window.record_state_change(
            change_type="temporal_context",
            old_value={"key": "old_key", "value": "old_value"},
            new_value={"key": "new_key", "value": "new_value"},
            origin="test_origin",
            entity_id="test_entity",
            tonic_value=0.7,  # Higher tonic for temporal context
            stability=0.9
        )
        
        # Verify tonic-harmonic properties in notification
        context = self.field_observer.observations[0]["context"]
        self.assertEqual(context["tonic_value"], 0.7)
        self.assertEqual(context["harmonic_value"], 0.9 * 0.7)
        
        # Verify field properties
        self.assertIn("field_properties", context)
        self.assertIn("coherence", context["field_properties"])
        self.assertIn("stability", context["field_properties"])
        self.assertIn("navigability", context["field_properties"])
        self.assertIn("saturation", context["field_properties"])
    
    def test_window_state_transition(self):
        """Test window state transition after recording changes."""
        # Set up a learning window close to saturation
        self.learning_window.max_changes_per_window = 3
        self.learning_window.change_count = 2
        
        # Record a state change that will saturate the window
        self.learning_window.record_state_change(
            change_type="final_change",
            old_value="old_value",
            new_value="new_value",
            origin="test_origin",
            entity_id="test_entity"
        )
        
        # Verify window is now saturated
        self.assertTrue(self.learning_window.is_saturated)
        
        # Verify state is CLOSED due to saturation
        # Note: This assumes we're within the time window but saturated
        if datetime.now() < self.end_time:
            self.assertEqual(self.learning_window.state, WindowState.CLOSED)

if __name__ == '__main__':
    unittest.main()
