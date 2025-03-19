"""
Tests for pattern co-evolution in the tonic-harmonic system.

This module tests the ability of the system to detect and track patterns
that evolve together, verifying that the tonic-harmonic approach can
identify relationships between patterns that would be invisible when
testing components in isolation.
"""

import pytest
import sys
import os
import unittest
import numpy as np
from datetime import datetime, timedelta
import asyncio
from unittest.mock import MagicMock, patch

# Add src to path if not already there
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import modules using relative imports
sys.path.append(os.path.join(src_path, 'src'))
from habitat_evolution.pattern_aware_rag.learning.learning_control import (
    LearningWindow, WindowState, EventCoordinator
)

# Mock classes for testing
class MockAdaptiveID:
    """Mock AdaptiveID class for testing."""
    
    def __init__(self):
        """Initialize a mock AdaptiveID."""
        self.id = "mock_adaptive_id"
        self.updates = []
        
    def update_from_pattern_id(self, pattern_id, context):
        """Record updates from pattern ID."""
        self.updates.append({
            "pattern_id": pattern_id.pattern_id,
            "context": context
        })

class PatternID:
    """Mock PatternID class for testing pattern evolution tracking."""
    
    def __init__(self, pattern_id: str, pattern_type: str):
        """Initialize PatternID with id and type."""
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type
        self.evolution_count = 0
        self.evolution_history = []
        self.adaptive_ids = []
        
    def observe_pattern_evolution(self, context):
        """Record pattern evolution event."""
        self.evolution_count += 1
        self.evolution_history.append(context)
        
        # Notify associated AdaptiveIDs
        for adaptive_id in self.adaptive_ids:
            adaptive_id.update_from_pattern_id(self, context)
        
    def reset(self):
        """Reset pattern evolution tracking."""
        self.evolution_count = 0
        self.evolution_history = []
        
    def register_adaptive_id(self, adaptive_id):
        """Register an AdaptiveID with this pattern."""
        if adaptive_id not in self.adaptive_ids:
            self.adaptive_ids.append(adaptive_id)


class MockFieldObserver:
    """Mock field observer for testing."""
    
    def __init__(self, field_id):
        self.field_id = field_id
        self.observations = []
        
    def observe(self, context):
        self.observations.append({"context": context})


class TestPatternCoevolution(unittest.TestCase):
    """Test pattern co-evolution in the tonic-harmonic system."""
    
    def setUp(self):
        """Set up test environment."""
        # Create learning window
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=1)
        self.learning_window = LearningWindow(
            start_time=self.start_time,
            end_time=self.end_time,
            stability_threshold=0.6,
            coherence_threshold=0.7,
            max_changes_per_window=10
        )
        
        # Create field observer
        self.field_observer = MockFieldObserver(field_id="test_field")
        self.learning_window.register_field_observer(self.field_observer)
        
        # Create adaptive ID
        self.adaptive_id = MockAdaptiveID()
        
        # Create pattern IDs for different pattern types
        self.pattern_a = PatternID(pattern_id="pattern_a", pattern_type="harmonic")
        self.pattern_b = PatternID(pattern_id="pattern_b", pattern_type="sequential")
        self.pattern_c = PatternID(pattern_id="pattern_c", pattern_type="complementary")
        
        # Register adaptive ID with patterns
        self.pattern_a.register_adaptive_id(self.adaptive_id)
        self.pattern_b.register_adaptive_id(self.adaptive_id)
        self.pattern_c.register_adaptive_id(self.adaptive_id)
        
        # Register patterns with learning window
        self.learning_window.register_pattern_observer(self.pattern_a)
        self.learning_window.register_pattern_observer(self.pattern_b)
        self.learning_window.register_pattern_observer(self.pattern_c)
        
        # Ensure patterns start with clean state
        self.reset_all_patterns()
        
    def tearDown(self):
        """Clean up after each test."""
        # Reset all observers and patterns
        self.field_observer.observations = []
        self.reset_all_patterns()
        
    def reset_all_patterns(self):
        """Reset all pattern observers to a clean state."""
        self.pattern_a.reset()
        self.pattern_b.reset()
        self.pattern_c.reset()
        
    def test_basic_pattern_coevolution(self):
        """Test that patterns evolve together when state changes occur."""
        # Reset all patterns to ensure a clean state
        self.reset_all_patterns()
        
        # Activate the learning window
        self.learning_window.activate(stability_score=0.8)
        
        # Record a state change that should affect all patterns
        self.learning_window.record_state_change(
            change_type="semantic_shift",
            old_value={"concept": "initial"},
            new_value={"concept": "evolved"},
            origin="test_origin",
            entity_id="shared_entity",
            tonic_value=0.7,
            stability=0.8
        )
        
        # Verify all patterns received at least one notification
        # Note: The LearningWindow may send multiple notifications during state changes
        self.assertGreaterEqual(self.pattern_a.evolution_count, 1)
        self.assertGreaterEqual(self.pattern_b.evolution_count, 1)
        self.assertGreaterEqual(self.pattern_c.evolution_count, 1)
        
        # Verify all patterns received the same number of notifications
        self.assertEqual(self.pattern_a.evolution_count, self.pattern_b.evolution_count)
        self.assertEqual(self.pattern_b.evolution_count, self.pattern_c.evolution_count)
        
        # Verify the context in each pattern's evolution history
        for pattern in [self.pattern_a, self.pattern_b, self.pattern_c]:
            # Filter evolution history to find semantic_shift events
            semantic_shifts = [ctx for ctx in pattern.evolution_history 
                              if ctx.get("change_type") == "semantic_shift"]
            
            # Verify we have at least one semantic shift
            self.assertGreaterEqual(len(semantic_shifts), 1, 
                                  f"Pattern {pattern.pattern_id} should have at least one semantic_shift")
            
            # Use the first semantic shift for verification
            context = semantic_shifts[0]
            self.assertEqual(context["change_type"], "semantic_shift")
            self.assertEqual(context["entity_id"], "shared_entity")
            self.assertEqual(context["tonic_value"], 0.7)
            self.assertEqual(context["stability"], 0.8)
            self.assertEqual(context["harmonic_value"], 0.7 * 0.8)
    
    def test_differential_tonic_values(self):
        """Test that different patterns respond differently to tonic values."""
        # Activate the learning window
        self.learning_window.activate(stability_score=0.9)
        
        # Record state changes with different tonic values for different pattern types
        
        # Harmonic pattern should respond strongly to harmonic changes
        self.learning_window.record_state_change(
            change_type="harmonic_shift",
            old_value={"harmonic": 0.3},
            new_value={"harmonic": 0.8},
            origin="test_origin",
            entity_id="harmonic_entity",
            tonic_value=0.9,  # High tonic value for harmonic pattern
            stability=0.8
        )
        
        # Sequential pattern should respond strongly to sequential changes
        self.learning_window.record_state_change(
            change_type="sequential_shift",
            old_value={"sequence": 1},
            new_value={"sequence": 2},
            origin="test_origin",
            entity_id="sequential_entity",
            tonic_value=0.8,  # Medium-high tonic value for sequential pattern
            stability=0.8
        )
        
        # Complementary pattern should respond strongly to complementary changes
        self.learning_window.record_state_change(
            change_type="complementary_shift",
            old_value={"complement": "A"},
            new_value={"complement": "B"},
            origin="test_origin",
            entity_id="complementary_entity",
            tonic_value=0.7,  # Medium tonic value for complementary pattern
            stability=0.8
        )
        
        # Verify differential response based on pattern type
        
        # Find the harmonic shift in pattern_a's history
        harmonic_context = None
        for ctx in self.pattern_a.evolution_history:
            if ctx.get("change_type") == "harmonic_shift":
                harmonic_context = ctx
                break
        
        self.assertIsNotNone(harmonic_context, "Harmonic pattern did not record harmonic shift")
        self.assertEqual(harmonic_context["harmonic_value"], 0.9 * 0.8)
        
        # Find the sequential shift in pattern_b's history
        sequential_context = None
        for ctx in self.pattern_b.evolution_history:
            if ctx.get("change_type") == "sequential_shift":
                sequential_context = ctx
                break
        
        self.assertIsNotNone(sequential_context, "Sequential pattern did not record sequential shift")
        self.assertEqual(sequential_context["harmonic_value"], 0.8 * 0.8)
        
        # Find the complementary shift in pattern_c's history
        complementary_context = None
        for ctx in self.pattern_c.evolution_history:
            if ctx.get("change_type") == "complementary_shift":
                complementary_context = ctx
                break
        
        self.assertIsNotNone(complementary_context, "Complementary pattern did not record complementary shift")
        self.assertEqual(complementary_context["harmonic_value"], 0.7 * 0.8)
    
    def test_window_size_effects(self):
        """Test how different window sizes affect pattern co-evolution."""
        # Create windows of different sizes
        small_window = LearningWindow(
            start_time=self.start_time,
            end_time=self.start_time + timedelta(minutes=15),
            stability_threshold=0.6,
            coherence_threshold=0.7,
            max_changes_per_window=5
        )
        
        medium_window = LearningWindow(
            start_time=self.start_time,
            end_time=self.start_time + timedelta(hours=1),
            stability_threshold=0.6,
            coherence_threshold=0.7,
            max_changes_per_window=10
        )
        
        large_window = LearningWindow(
            start_time=self.start_time,
            end_time=self.start_time + timedelta(hours=4),
            stability_threshold=0.6,
            coherence_threshold=0.7,
            max_changes_per_window=20
        )
        
        # Register patterns with all windows
        for window in [small_window, medium_window, large_window]:
            window.register_pattern_observer(self.pattern_a)
            window.register_pattern_observer(self.pattern_b)
            window.register_pattern_observer(self.pattern_c)
            window.register_field_observer(self.field_observer)
        
        # Activate all windows
        for window in [small_window, medium_window, large_window]:
            window.activate(stability_score=0.8)
        
        # Record identical state changes in all windows
        for window in [small_window, medium_window, large_window]:
            window.record_state_change(
                change_type="test_change",
                old_value="old_value",
                new_value="new_value",
                origin="test_origin",
                entity_id="test_entity",
                tonic_value=0.7,
                stability=0.8
            )
        
        # Reset patterns to clear evolution history
        self.pattern_a.reset()
        self.pattern_b.reset()
        self.pattern_c.reset()
        
        # Record multiple changes in each window to simulate different rhythms
        # Small window - rapid changes
        for i in range(5):
            small_window.record_state_change(
                change_type=f"small_window_change_{i}",
                old_value=i,
                new_value=i+1,
                origin="test_origin",
                entity_id="test_entity",
                tonic_value=0.7,
                stability=0.8 - (i * 0.05)  # Decreasing stability with each change
            )
        
        # Medium window - moderate changes
        for i in range(3):
            medium_window.record_state_change(
                change_type=f"medium_window_change_{i}",
                old_value=i,
                new_value=i+1,
                origin="test_origin",
                entity_id="test_entity",
                tonic_value=0.7,
                stability=0.8
            )
        
        # Large window - few changes
        for i in range(2):
            large_window.record_state_change(
                change_type=f"large_window_change_{i}",
                old_value=i,
                new_value=i+1,
                origin="test_origin",
                entity_id="test_entity",
                tonic_value=0.7,
                stability=0.8 + (i * 0.05)  # Increasing stability with each change
            )
        
        # Verify that patterns received different numbers of notifications
        # from different window sizes
        small_window_changes = 0
        medium_window_changes = 0
        large_window_changes = 0
        
        for ctx in self.pattern_a.evolution_history:
            if "small_window_change" in ctx.get("change_type", ""):
                small_window_changes += 1
            elif "medium_window_change" in ctx.get("change_type", ""):
                medium_window_changes += 1
            elif "large_window_change" in ctx.get("change_type", ""):
                large_window_changes += 1
        
        self.assertEqual(small_window_changes, 5, "Small window should generate 5 changes")
        self.assertEqual(medium_window_changes, 3, "Medium window should generate 3 changes")
        self.assertEqual(large_window_changes, 2, "Large window should generate 2 changes")
        
        # Verify stability trends in different window sizes
        small_window_stabilities = []
        medium_window_stabilities = []
        large_window_stabilities = []
        
        for ctx in self.pattern_a.evolution_history:
            if "small_window_change" in ctx.get("change_type", ""):
                small_window_stabilities.append(ctx.get("stability", 0))
            elif "medium_window_change" in ctx.get("change_type", ""):
                medium_window_stabilities.append(ctx.get("stability", 0))
            elif "large_window_change" in ctx.get("change_type", ""):
                large_window_stabilities.append(ctx.get("stability", 0))
        
        # Verify stability trends match our expectations
        self.assertTrue(all(small_window_stabilities[i] >= small_window_stabilities[i+1] 
                           for i in range(len(small_window_stabilities)-1)), 
                       "Small window should show decreasing stability")
        
        self.assertTrue(all(medium_window_stabilities[i] == medium_window_stabilities[i+1] 
                           for i in range(len(medium_window_stabilities)-1)), 
                       "Medium window should show stable stability")
        
        self.assertTrue(all(large_window_stabilities[i] <= large_window_stabilities[i+1] 
                           for i in range(len(large_window_stabilities)-1)), 
                       "Large window should show increasing stability")
    
    def test_constructive_dissonance(self):
        """Test that apparent dissonance in individual patterns contributes to system harmony."""
        # Reset all patterns to ensure a clean state
        self.reset_all_patterns()
        
        # Create an event coordinator to manage multiple windows
        coordinator = EventCoordinator(max_queue_size=100, persistence_mode=False)
        
        # Create windows with overlapping time periods
        window1 = coordinator.create_learning_window(
            duration_minutes=30,
            stability_threshold=0.6,
            coherence_threshold=0.7,
            max_changes=10
        )
        
        window2 = coordinator.create_learning_window(
            duration_minutes=45,
            stability_threshold=0.7,
            coherence_threshold=0.8,
            max_changes=15
        )
        
        # Register patterns with both windows
        for window in [window1, window2]:
            window.register_pattern_observer(self.pattern_a)
            window.register_pattern_observer(self.pattern_b)
            window.register_pattern_observer(self.pattern_c)
        
        # Activate both windows
        window1.activate(stability_score=0.8)
        window2.activate(stability_score=0.9)
        
        # Record changes that appear to conflict when viewed in isolation
        window1.record_state_change(
            change_type="concept_shift",
            old_value={"concept": "initial"},
            new_value={"concept": "intermediate"},
            origin="window1",
            entity_id="shared_entity",
            tonic_value=0.6,
            stability=0.7
        )
        
        window2.record_state_change(
            change_type="concept_shift",
            old_value={"concept": "intermediate"},
            new_value={"concept": "final"},
            origin="window2",
            entity_id="shared_entity",
            tonic_value=0.8,
            stability=0.9
        )
        
        # Verify that patterns recorded changes from both windows
        for pattern in [self.pattern_a, self.pattern_b, self.pattern_c]:
            # The pattern may receive multiple notifications from each window
            # We just need to verify it received notifications from both windows
            self.assertGreaterEqual(pattern.evolution_count, 2, 
                           f"Pattern {pattern.pattern_id} should have recorded at least 2 changes")
            
            # Find the concept_shift changes from each window
            window1_change = None
            window2_change = None
            
            for ctx in pattern.evolution_history:
                if ctx.get("origin") == "window1" and ctx.get("change_type") == "concept_shift":
                    window1_change = ctx
                elif ctx.get("origin") == "window2" and ctx.get("change_type") == "concept_shift":
                    window2_change = ctx
            
            self.assertIsNotNone(window1_change, f"Pattern {pattern.pattern_id} missing window1 change")
            self.assertIsNotNone(window2_change, f"Pattern {pattern.pattern_id} missing window2 change")
            
            # Verify the progression of values
            self.assertEqual(window1_change["old_value"], {"concept": "initial"})
            self.assertEqual(window1_change["new_value"], {"concept": "intermediate"})
            self.assertEqual(window2_change["old_value"], {"concept": "intermediate"})
            self.assertEqual(window2_change["new_value"], {"concept": "final"})
            
            # Verify harmonic values reflect window settings
            self.assertEqual(window1_change["harmonic_value"], 0.6 * 0.7)
            self.assertEqual(window2_change["harmonic_value"], 0.8 * 0.9)
        
        # Verify that the combined changes form a coherent progression
        # that wouldn't be visible when testing windows in isolation
        progression = []
        
        # Filter for concept_shift events and sort by timestamp
        concept_shifts = [ctx for ctx in self.pattern_a.evolution_history 
                         if ctx.get("change_type") == "concept_shift" and 
                         "concept" in ctx.get("new_value", {})]
        
        for ctx in sorted(concept_shifts, key=lambda x: x.get("timestamp", "")):
            progression.append(ctx["new_value"]["concept"])
        
        self.assertEqual(progression, ["intermediate", "final"], 
                       "Changes should form a coherent progression")


if __name__ == "__main__":
    unittest.main()
