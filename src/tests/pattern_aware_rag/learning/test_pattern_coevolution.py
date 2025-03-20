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
    
    def test_dynamic_window_interactions(self):
        """Test how windows interact within a field, creating oscillation patterns and resonance.
        
        This test focuses on the dynamic interactions between windows in a semantic field,
        demonstrating how windows with different frequencies can create oscillation patterns
        that lead to emergent field behaviors.
        """
        # Create windows with different frequencies (not sizes)
        high_frequency_window = LearningWindow(
            start_time=self.start_time,
            end_time=self.start_time + timedelta(hours=1),
            stability_threshold=0.6,
            coherence_threshold=0.7,
            max_changes_per_window=15
        )
        
        medium_frequency_window = LearningWindow(
            start_time=self.start_time,
            end_time=self.start_time + timedelta(hours=2),
            stability_threshold=0.65,
            coherence_threshold=0.75,
            max_changes_per_window=10
        )
        
        low_frequency_window = LearningWindow(
            start_time=self.start_time,
            end_time=self.start_time + timedelta(hours=3),
            stability_threshold=0.7,
            coherence_threshold=0.8,
            max_changes_per_window=5
        )
        
        # Create a field by registering patterns with all windows
        windows = [high_frequency_window, medium_frequency_window, low_frequency_window]
        for window in windows:
            window.register_pattern_observer(self.pattern_a)
            window.register_pattern_observer(self.pattern_b)
            window.register_pattern_observer(self.pattern_c)
            window.register_field_observer(self.field_observer)
        
        # Activate all windows with slightly different stability scores
        high_frequency_window.activate(stability_score=0.75)
        medium_frequency_window.activate(stability_score=0.8)
        low_frequency_window.activate(stability_score=0.85)
        
        # Reset patterns to ensure clean evolution history
        self.reset_all_patterns()
        
        # Generate oscillating patterns in each window
        # High frequency window - rapid oscillations
        for i in range(12):
            # Create a sine wave pattern for stability
            stability = 0.7 + 0.2 * np.sin(i * np.pi / 3)
            # Create a cosine wave pattern for tonic value
            tonic = 0.6 + 0.3 * np.cos(i * np.pi / 3)
            
            high_frequency_window.record_state_change(
                change_type=f"high_freq_change_{i}",
                old_value={"frequency": "high", "phase": i-1},
                new_value={"frequency": "high", "phase": i},
                origin="high_frequency_field",
                entity_id="oscillating_entity",
                tonic_value=tonic,
                stability=stability
            )
        
        # Medium frequency window - moderate oscillations
        for i in range(8):
            # Create a sine wave pattern with different frequency
            stability = 0.75 + 0.15 * np.sin(i * np.pi / 4)
            # Create a cosine wave pattern with different frequency
            tonic = 0.65 + 0.25 * np.cos(i * np.pi / 4)
            
            medium_frequency_window.record_state_change(
                change_type=f"medium_freq_change_{i}",
                old_value={"frequency": "medium", "phase": i-1},
                new_value={"frequency": "medium", "phase": i},
                origin="medium_frequency_field",
                entity_id="oscillating_entity",
                tonic_value=tonic,
                stability=stability
            )
        
        # Low frequency window - slow oscillations
        for i in range(6):
            # Create a sine wave pattern with different frequency
            stability = 0.8 + 0.1 * np.sin(i * np.pi / 6)
            # Create a cosine wave pattern with different frequency
            tonic = 0.7 + 0.2 * np.cos(i * np.pi / 6)
            
            low_frequency_window.record_state_change(
                change_type=f"low_freq_change_{i}",
                old_value={"frequency": "low", "phase": i-1},
                new_value={"frequency": "low", "phase": i},
                origin="low_frequency_field",
                entity_id="oscillating_entity",
                tonic_value=tonic,
                stability=stability
            )
        
        # Verify patterns received notifications from all windows
        high_freq_changes = sum(1 for ctx in self.pattern_a.evolution_history 
                              if "high_freq_change" in str(ctx.get("change_type", "")))
        medium_freq_changes = sum(1 for ctx in self.pattern_a.evolution_history 
                                if "medium_freq_change" in str(ctx.get("change_type", "")))
        low_freq_changes = sum(1 for ctx in self.pattern_a.evolution_history 
                             if "low_freq_change" in str(ctx.get("change_type", "")))
        
        self.assertEqual(high_freq_changes, 12, "High frequency window should generate 12 changes")
        self.assertEqual(medium_freq_changes, 8, "Medium frequency window should generate 8 changes")
        self.assertEqual(low_freq_changes, 6, "Low frequency window should generate 6 changes")
        
        # Extract stability and tonic values to verify oscillation patterns
        high_freq_data = [(ctx.get("stability", 0), ctx.get("tonic_value", 0)) 
                         for ctx in self.pattern_a.evolution_history 
                         if "high_freq_change" in str(ctx.get("change_type", ""))]
        
        medium_freq_data = [(ctx.get("stability", 0), ctx.get("tonic_value", 0)) 
                           for ctx in self.pattern_a.evolution_history 
                           if "medium_freq_change" in str(ctx.get("change_type", ""))]
        
        low_freq_data = [(ctx.get("stability", 0), ctx.get("tonic_value", 0)) 
                        for ctx in self.pattern_a.evolution_history 
                        if "low_freq_change" in str(ctx.get("change_type", ""))]
        
        # Verify oscillation patterns in stability values
        # For oscillating patterns, we expect some values to increase and some to decrease
        # rather than a monotonic trend
        
        # Check high frequency oscillations (should have at least one peak and valley)
        high_freq_stability = [s for s, _ in high_freq_data]
        self.assertTrue(any(high_freq_stability[i] < high_freq_stability[i+1] 
                           for i in range(len(high_freq_stability)-1)), 
                       "High frequency window should show increasing stability at some point")
        self.assertTrue(any(high_freq_stability[i] > high_freq_stability[i+1] 
                           for i in range(len(high_freq_stability)-1)), 
                       "High frequency window should show decreasing stability at some point")
        
        # Check medium frequency oscillations
        medium_freq_stability = [s for s, _ in medium_freq_data]
        self.assertTrue(any(medium_freq_stability[i] < medium_freq_stability[i+1] 
                           for i in range(len(medium_freq_stability)-1)), 
                       "Medium frequency window should show increasing stability at some point")
        self.assertTrue(any(medium_freq_stability[i] > medium_freq_stability[i+1] 
                           for i in range(len(medium_freq_stability)-1)), 
                       "Medium frequency window should show decreasing stability at some point")
        
        # Check low frequency oscillations
        low_freq_stability = [s for s, _ in low_freq_data]
        self.assertTrue(any(low_freq_stability[i] < low_freq_stability[i+1] 
                           for i in range(len(low_freq_stability)-1)), 
                       "Low frequency window should show increasing stability at some point")
        self.assertTrue(any(low_freq_stability[i] > low_freq_stability[i+1] 
                           for i in range(len(low_freq_stability)-1)), 
                       "Low frequency window should show decreasing stability at some point")
        
        # Verify harmonic values reflect the product of tonic and stability
        for ctx in self.pattern_a.evolution_history:
            tonic = ctx.get("tonic_value", 0)
            stability = ctx.get("stability", 0)
            harmonic = ctx.get("harmonic_value", 0)
            self.assertAlmostEqual(harmonic, tonic * stability, places=5, 
                                 msg="Harmonic value should be the product of tonic and stability")
    
    def test_field_resonance(self):
        """Test the emergence of resonance patterns and natural boundary detection.
        
        This test verifies that when multiple windows interact within a field,
        resonance patterns emerge at specific frequencies, creating natural
        boundaries in the semantic field. These boundaries represent points where
        the system naturally organizes information.
        """
        # Create an event coordinator to manage multiple windows
        coordinator = EventCoordinator(max_queue_size=100, persistence_mode=False)
        
        # Create windows with specific frequencies that will create resonance
        # These frequencies are chosen to create constructive interference
        window1 = coordinator.create_learning_window(
            duration_minutes=30,
            stability_threshold=0.6,
            coherence_threshold=0.7,
            max_changes=15
        )
        
        window2 = coordinator.create_learning_window(
            duration_minutes=60,
            stability_threshold=0.65,
            coherence_threshold=0.75,
            max_changes=10
        )
        
        window3 = coordinator.create_learning_window(
            duration_minutes=90,
            stability_threshold=0.7,
            coherence_threshold=0.8,
            max_changes=8
        )
        
        # Register patterns with all windows
        for window in [window1, window2, window3]:
            window.register_pattern_observer(self.pattern_a)
            window.register_pattern_observer(self.pattern_b)
            window.register_pattern_observer(self.pattern_c)
            window.register_field_observer(self.field_observer)
        
        # Activate all windows
        window1.activate(stability_score=0.8)
        window2.activate(stability_score=0.85)
        window3.activate(stability_score=0.9)
        
        # Reset patterns to ensure clean evolution history
        self.reset_all_patterns()
        
        # Generate a series of changes with specific frequencies to create resonance
        # The key is to create changes that align at specific points (resonance points)
        # and cancel each other out at other points (boundary points)
        
        # Phase 1: Initial oscillations in each window
        for i in range(10):
            # Window 1: High frequency oscillations
            phase_shift1 = i * (2 * np.pi / 10)  # Complete 1 full cycle over 10 steps
            stability1 = 0.7 + 0.2 * np.sin(phase_shift1)
            tonic1 = 0.6 + 0.3 * np.cos(phase_shift1)
            
            # Window 2: Medium frequency oscillations
            phase_shift2 = i * (2 * np.pi / 5)  # Complete 2 full cycles over 10 steps
            stability2 = 0.75 + 0.15 * np.sin(phase_shift2)
            tonic2 = 0.65 + 0.25 * np.cos(phase_shift2)
            
            # Window 3: Low frequency oscillations
            phase_shift3 = i * (2 * np.pi / 20)  # Complete 0.5 cycles over 10 steps
            stability3 = 0.8 + 0.1 * np.sin(phase_shift3)
            tonic3 = 0.7 + 0.2 * np.cos(phase_shift3)
            
            # Record state changes in each window
            window1.record_state_change(
                change_type=f"resonance_test_w1_{i}",
                old_value={"phase": i-1, "frequency": "high"},
                new_value={"phase": i, "frequency": "high"},
                origin="resonance_field",
                entity_id="resonating_entity",
                tonic_value=tonic1,
                stability=stability1
            )
            
            window2.record_state_change(
                change_type=f"resonance_test_w2_{i}",
                old_value={"phase": i-1, "frequency": "medium"},
                new_value={"phase": i, "frequency": "medium"},
                origin="resonance_field",
                entity_id="resonating_entity",
                tonic_value=tonic2,
                stability=stability2
            )
            
            window3.record_state_change(
                change_type=f"resonance_test_w3_{i}",
                old_value={"phase": i-1, "frequency": "low"},
                new_value={"phase": i, "frequency": "low"},
                origin="resonance_field",
                entity_id="resonating_entity",
                tonic_value=tonic3,
                stability=stability3
            )
        
        # Extract all resonance test events
        resonance_events = [ctx for ctx in self.pattern_a.evolution_history 
                          if "resonance_test" in str(ctx.get("change_type", ""))]
        
        # Group events by phase to identify resonance points
        events_by_phase = {}
        for ctx in resonance_events:
            phase = ctx.get("new_value", {}).get("phase", -1)
            if phase not in events_by_phase:
                events_by_phase[phase] = []
            events_by_phase[phase].append(ctx)
        
        # Calculate combined harmonic values at each phase
        combined_harmonics = {}
        for phase, events in events_by_phase.items():
            # Sum the harmonic values for this phase
            combined_harmonics[phase] = sum(ctx.get("harmonic_value", 0) for ctx in events)
        
        # Identify resonance points (peaks) and boundary points (valleys)
        phases = sorted(combined_harmonics.keys())
        resonance_points = []
        boundary_points = []
        
        for i in range(1, len(phases) - 1):
            current_phase = phases[i]
            prev_phase = phases[i-1]
            next_phase = phases[i+1]
            
            current_value = combined_harmonics[current_phase]
            prev_value = combined_harmonics[prev_phase]
            next_value = combined_harmonics[next_phase]
            
            # Resonance point: higher than both neighbors (local maximum)
            if current_value > prev_value and current_value > next_value:
                resonance_points.append(current_phase)
            
            # Boundary point: lower than both neighbors (local minimum)
            if current_value < prev_value and current_value < next_value:
                boundary_points.append(current_phase)
        
        # Verify that we detected at least one resonance point and one boundary
        self.assertGreaterEqual(len(resonance_points), 1, 
                              "Should detect at least one resonance point")
        self.assertGreaterEqual(len(boundary_points), 1, 
                              "Should detect at least one boundary point")
        
        # Verify that the field observer received notifications about all events
        # Filter field observations to only include resonance test events
        field_resonance_observations = [obs for obs in self.field_observer.observations 
                                     if "resonance_test" in str(obs["context"].get("change_type", ""))]
        
        self.assertEqual(len(field_resonance_observations), len(resonance_events), 
                       "Field observer should receive all resonance events")
        
        # Verify that the harmonic values at resonance points are higher than at boundary points
        avg_resonance_value = sum(combined_harmonics[p] for p in resonance_points) / len(resonance_points)
        avg_boundary_value = sum(combined_harmonics[p] for p in boundary_points) / len(boundary_points)
        
        self.assertGreater(avg_resonance_value, avg_boundary_value, 
                         "Resonance points should have higher harmonic values than boundary points")
    
    def test_bidirectional_synchronization(self):
        """Test bidirectional synchronization between AdaptiveID, FieldObserver, and PatternID.
        
        This test verifies that changes propagate correctly between all components in the system,
        ensuring that tonic-harmonic properties are properly tracked and synchronized across
        different parts of the system.
        """
        # Reset all patterns to ensure a clean state
        self.reset_all_patterns()
        
        # Create a new adaptive ID for this test
        adaptive_id = MockAdaptiveID()
        
        # Create a new field observer for this test
        field_observer = MockFieldObserver(field_id="sync_test_field")
        
        # Create a new pattern for this test with clean evolution history
        sync_pattern = PatternID(pattern_id="sync_pattern", pattern_type="synchronization")
        sync_pattern.evolution_count = 0
        sync_pattern.evolution_history = []
        
        # Register the adaptive ID with the pattern
        sync_pattern.register_adaptive_id(adaptive_id)
        
        # Create a learning window
        window = LearningWindow(
            start_time=self.start_time,
            end_time=self.start_time + timedelta(hours=1),
            stability_threshold=0.65,
            coherence_threshold=0.75,
            max_changes_per_window=10
        )
        
        # Register the pattern and field observer with the window
        window.register_pattern_observer(sync_pattern)
        window.register_field_observer(field_observer)
        
        # Activate the window
        window.activate(stability_score=0.85)
        
        # Record a series of state changes with different tonic and stability values
        test_changes = [
            {"change_type": "sync_test_1", "tonic": 0.7, "stability": 0.8},
            {"change_type": "sync_test_2", "tonic": 0.75, "stability": 0.85},
            {"change_type": "sync_test_3", "tonic": 0.8, "stability": 0.9}
        ]
        
        for i, change in enumerate(test_changes):
            window.record_state_change(
                change_type=change["change_type"],
                old_value={"state": i},
                new_value={"state": i+1},
                origin="sync_test",
                entity_id="sync_entity",
                tonic_value=change["tonic"],
                stability=change["stability"]
            )
        
        # Filter for only our specific sync test events
        pattern_sync_events = [ctx for ctx in sync_pattern.evolution_history 
                             if any(change["change_type"] == ctx.get("change_type") 
                                  for change in test_changes)]
        
        # Verify that the pattern received all our specific changes
        self.assertEqual(len(pattern_sync_events), len(test_changes),
                       "Pattern should receive all sync test state changes")
        
        # Filter field observer observations for our specific sync test events
        field_sync_observations = [obs for obs in field_observer.observations 
                                 if any(change["change_type"] == obs["context"].get("change_type") 
                                      for change in test_changes)]
        
        # Verify that the field observer received all our specific changes
        self.assertEqual(len(field_sync_observations), len(test_changes),
                       "Field observer should receive all sync test state changes")
        
        # Filter adaptive ID updates for our specific sync test events
        adaptive_sync_updates = [update for update in adaptive_id.updates 
                               if any(change["change_type"] == update["context"].get("change_type") 
                                    for change in test_changes)]
        
        # Verify that the adaptive ID received updates for all our specific changes
        self.assertEqual(len(adaptive_sync_updates), len(test_changes),
                       "AdaptiveID should receive updates for all sync test changes")
        
        # Verify bidirectional synchronization by checking that the same context
        # was propagated to all components
        for i, change in enumerate(test_changes):
            # Get the context from the pattern
            pattern_ctx = next(ctx for ctx in sync_pattern.evolution_history 
                              if ctx.get("change_type") == change["change_type"])
            
            # Get the corresponding update in the adaptive ID
            adaptive_update = next(update for update in adaptive_id.updates 
                                 if update["context"].get("change_type") == change["change_type"])
            
            # Get the corresponding observation in the field observer
            field_observation = next(obs for obs in field_observer.observations 
                                   if obs["context"].get("change_type") == change["change_type"])
            
            # Verify that the context was properly propagated to the adaptive ID
            self.assertEqual(adaptive_update["pattern_id"], sync_pattern.pattern_id,
                           "AdaptiveID update should reference the correct pattern")
            
            # Verify that the harmonic value was correctly calculated and propagated
            expected_harmonic = change["tonic"] * change["stability"]
            
            self.assertAlmostEqual(pattern_ctx["harmonic_value"], expected_harmonic, 
                                 places=5, msg="Pattern should have correct harmonic value")
            
            self.assertAlmostEqual(adaptive_update["context"]["harmonic_value"], expected_harmonic, 
                                 places=5, msg="AdaptiveID should receive correct harmonic value")
            
            self.assertAlmostEqual(field_observation["context"]["harmonic_value"], expected_harmonic, 
                                 places=5, msg="FieldObserver should receive correct harmonic value")
            
            # Verify that all components received the same tonic and stability values
            self.assertEqual(pattern_ctx["tonic_value"], change["tonic"],
                           "Pattern should have correct tonic value")
            
            self.assertEqual(adaptive_update["context"]["tonic_value"], change["tonic"],
                           "AdaptiveID should receive correct tonic value")
            
            self.assertEqual(field_observation["context"]["tonic_value"], change["tonic"],
                           "FieldObserver should receive correct tonic value")
            
            self.assertEqual(pattern_ctx["stability"], change["stability"],
                           "Pattern should have correct stability value")
            
            self.assertEqual(adaptive_update["context"]["stability"], change["stability"],
                           "AdaptiveID should receive correct stability value")
            
            self.assertEqual(field_observation["context"]["stability"], change["stability"],
                           "FieldObserver should receive correct stability value")
    
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
        
        # Reset all patterns to ensure a clean state
        self.reset_all_patterns()
        
        # Record changes that appear to conflict when viewed in isolation
        # Make multiple changes to ensure at least one gets through
        for i in range(3):
            window1.record_state_change(
                change_type=f"concept_shift_{i}",
                old_value={"concept": "initial"},
                new_value={"concept": "intermediate"},
                origin="window1",
                entity_id="shared_entity",
                tonic_value=0.6,
                stability=0.7
            )
        
        for i in range(3):
            window2.record_state_change(
                change_type=f"concept_shift_{i}",
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
                # Use a more flexible approach to find changes from each window
                if ctx.get("origin") == "window1":
                    window1_change = ctx
                elif ctx.get("origin") == "window2":
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
                         if "concept_shift" in str(ctx.get("change_type", "")) and 
                         "concept" in ctx.get("new_value", {})]
        
        # Group by concept value to get unique progression
        unique_concepts = {}
        for ctx in sorted(concept_shifts, key=lambda x: x.get("timestamp", "")):
            concept = ctx["new_value"]["concept"]
            # Only keep the first occurrence of each concept
            if concept not in unique_concepts:
                unique_concepts[concept] = ctx
        
        # Extract the progression in timestamp order
        for ctx in sorted(unique_concepts.values(), key=lambda x: x.get("timestamp", "")):
            progression.append(ctx["new_value"]["concept"])
        
        self.assertEqual(progression, ["intermediate", "final"], 
                       "Changes should form a coherent progression")


if __name__ == "__main__":
    unittest.main()
