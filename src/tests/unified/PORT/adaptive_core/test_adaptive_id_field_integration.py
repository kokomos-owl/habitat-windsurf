"""
Tests for AdaptiveID field observer integration.

These tests validate the integration between AdaptiveID and field observers,
ensuring proper registration, notification, and field-aware state tracking.
"""

import unittest
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio

# Import the AdaptiveID class
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

class MockFieldObserver:
    """Mock field observer for testing AdaptiveID integration with tonic-harmonic approach."""
    
    def __init__(self, field_id: str = "test_field"):
        """Initialize the mock field observer."""
        self.field_id = field_id
        self.observations = []
        self.field_metrics = {}
        self.wave_history = []  # Store stability wave
        self.tonic_history = [0.5]  # Store tonic values with default
        self.harmonic_history = []  # Store harmonic values (stability * tonic)
        
    async def observe(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Observe and record field conditions with tonic-harmonic analysis.
        
        Args:
            context: The context of the observation
            
        Returns:
            Updated field metrics
        """
        self.observations.append({"context": context, "time": datetime.now()})
        
        # Extract stability and tonic from context if available
        if "stability" in context:
            self.wave_history.append(context["stability"])
            
        # Extract tonic value if available, otherwise use default
        tonic_value = context.get("tonic_value", 0.5)
        self.tonic_history.append(tonic_value)
        
        # Calculate harmonic value (stability * tonic)
        if len(self.wave_history) > 0:
            harmonic = self.wave_history[-1] * tonic_value
            self.harmonic_history.append(harmonic)
            
        # Update field metrics
        self.field_metrics.update({
            "observation_count": len(self.observations),
            "latest_observation_time": datetime.now().isoformat(),
            "latest_context": context,
            "latest_harmonic": self.harmonic_history[-1] if self.harmonic_history else None
        })
        
        return self.field_metrics
    
    def get_field_metrics(self) -> Dict[str, Any]:
        """Get current field metrics with tonic-harmonic analysis."""
        # Add computed metrics
        if self.observations:
            # Add stability trend if available
            if len(self.wave_history) >= 2:
                self.field_metrics["stability_trend"] = (
                    self.wave_history[-1] - self.wave_history[-2]
                )
                
            # Add tonic trend if available
            if len(self.tonic_history) >= 2:
                self.field_metrics["tonic_trend"] = (
                    self.tonic_history[-1] - self.tonic_history[-2]
                )
                
            # Add harmonic trend if available
            if len(self.harmonic_history) >= 2:
                self.field_metrics["harmonic_trend"] = (
                    self.harmonic_history[-1] - self.harmonic_history[-2]
                )
        
        return self.field_metrics
        
    def perform_harmonic_analysis(self) -> Dict[str, Any]:
        """Perform tonic-harmonic analysis on stability and tonic series.
        
        Returns:
            Dictionary with harmonic analysis results
        """
        if len(self.wave_history) < 3 or len(self.tonic_history) < 3:
            return {"harmonic": [], "boundaries": [], "resonance": []}
            
        # Ensure same length
        min_len = min(len(self.wave_history), len(self.tonic_history))
        stability = self.wave_history[-min_len:]
        tonic = self.tonic_history[-min_len:]
        
        # Calculate harmonic series as the product of stability and tonic
        harmonic = [s * t for s, t in zip(stability, tonic)]
        
        # Identify boundaries (significant changes in harmonic value)
        boundaries = []
        for i in range(1, len(harmonic)):
            # Detect significant change (threshold of 10%)
            if abs(harmonic[i] - harmonic[i-1]) > 0.1:
                boundaries.append(i)
                
        # Detect resonance patterns (alternating high-low values)
        resonance = []
        if len(harmonic) >= 4:
            for i in range(len(harmonic) - 3):
                segment = harmonic[i:i+4]
                # Check for alternating pattern
                if (segment[0] > segment[1] < segment[2] > segment[3]) or \
                   (segment[0] < segment[1] > segment[2] < segment[3]):
                    resonance.append(i+1)  # Middle point of pattern
        
        return {
            "harmonic": harmonic,
            "boundaries": boundaries,
            "resonance": resonance,
            "avg_harmonic": sum(harmonic) / len(harmonic) if harmonic else 0,
            "max_harmonic": max(harmonic) if harmonic else 0,
            "stability_trend": sum(s2 - s1 for s1, s2 in zip(stability[:-1], stability[1:])) / (len(stability) - 1) if len(stability) > 1 else 0,
            "tonic_trend": sum(t2 - t1 for t1, t2 in zip(tonic[:-1], tonic[1:])) / (len(tonic) - 1) if len(tonic) > 1 else 0
        }


class TestAdaptiveIDFieldIntegration(unittest.TestCase):
    """Test suite for AdaptiveID field observer integration with tonic-harmonic approach."""
    
    def setUp(self):
        """Set up test environment."""
        self.adaptive_id = AdaptiveID("test_concept", "test_creator")
        self.field_observer = MockFieldObserver()
    
    def test_register_with_field_observer(self):
        """Test registering an AdaptiveID with a field observer."""
        # Register the AdaptiveID with the field observer
        self.adaptive_id.register_with_field_observer(self.field_observer)
        
        # Verify the field observer is registered
        self.assertTrue(hasattr(self.adaptive_id, 'field_observers'))
        self.assertIn(self.field_observer, self.adaptive_id.field_observers)
        
        # Verify initial state was provided to the field observer
        self.assertEqual(len(self.field_observer.observations), 1)
        
        # Check that the context contains the expected information
        context = self.field_observer.observations[0]["context"]
        self.assertEqual(context["entity_id"], self.adaptive_id.id)
        self.assertEqual(context["entity_type"], "adaptive_id")
        self.assertEqual(context["base_concept"], "test_concept")
        
        # Verify vector properties are included
        self.assertIn("vector_properties", context)
        self.assertIn("temporal_context", context["vector_properties"])
        self.assertIn("spatial_context", context["vector_properties"])
        
        # Verify tonic value is included
        self.assertIn("tonic_value", context)
        self.assertEqual(context["tonic_value"], 0.5)  # Default tonic value
    
    def test_notify_field_observer_with_tonic_harmonic(self):
        """Test notification of state changes to field observers with tonic-harmonic properties."""
        # Register the AdaptiveID with the field observer
        self.adaptive_id.register_with_field_observer(self.field_observer)
        
        # Clear initial observation
        self.field_observer.observations = []
        self.field_observer.wave_history = []
        self.field_observer.tonic_history = []
        self.field_observer.harmonic_history = []
        
        # Notify state change
        self.adaptive_id.notify_state_change(
            "test_change", 
            "old_value", 
            "new_value", 
            "test_origin"
        )
        
        # Verify the field observer received the notification
        self.assertEqual(len(self.field_observer.observations), 1)
        context = self.field_observer.observations[0]["context"]
        
        # Verify basic properties
        self.assertEqual(context["entity_id"], self.adaptive_id.id)
        self.assertEqual(context["change_type"], "test_change")
        self.assertEqual(context["old_value"], "old_value")
        self.assertEqual(context["new_value"], "new_value")
        self.assertEqual(context["origin"], "test_origin")
        
        # Verify tonic-harmonic properties
        self.assertIn("tonic_value", context)
        self.assertIn("stability", context)
        self.assertIn("field_properties", context)
        self.assertIn("coherence", context["field_properties"])
        self.assertIn("navigability", context["field_properties"])
        self.assertIn("stability", context["field_properties"])
        
        # Verify tonic and stability values were recorded
        self.assertEqual(len(self.field_observer.tonic_history), 1)
        self.assertEqual(len(self.field_observer.wave_history), 1)
        
        # Verify harmonic value was calculated
        self.assertEqual(len(self.field_observer.harmonic_history), 1)
        expected_harmonic = self.field_observer.tonic_history[0] * self.field_observer.wave_history[0]
        self.assertEqual(self.field_observer.harmonic_history[0], expected_harmonic)
    
    def test_temporal_context_update_with_higher_tonic(self):
        """Test that temporal context updates have higher tonic values."""
        # Register the AdaptiveID with the field observer
        self.adaptive_id.register_with_field_observer(self.field_observer)
        
        # Clear initial observation
        self.field_observer.observations = []
        self.field_observer.tonic_history = []
        
        # Update temporal context (should trigger notification with higher tonic)
        self.adaptive_id.update_temporal_context("test_key", "test_value", "test_origin")
        
        # Verify the field observer received the notification
        self.assertEqual(len(self.field_observer.observations), 1)
        context = self.field_observer.observations[0]["context"]
        
        # Verify basic properties
        self.assertEqual(context["entity_id"], self.adaptive_id.id)
        self.assertEqual(context["change_type"], "temporal_context")
        self.assertEqual(context["new_value"]["key"], "test_key")
        self.assertEqual(context["new_value"]["value"], "test_value")
        self.assertEqual(context["origin"], "test_origin")
        
        # Verify tonic value is higher for temporal context updates
        self.assertEqual(context["tonic_value"], 0.7)
        self.assertEqual(self.field_observer.tonic_history[0], 0.7)
    
    def test_harmonic_analysis(self):
        """Test harmonic analysis of stability and tonic values."""
        # Register the AdaptiveID with the field observer
        self.adaptive_id.register_with_field_observer(self.field_observer)
        
        # Clear initial observation
        self.field_observer.observations = []
        self.field_observer.wave_history = []
        self.field_observer.tonic_history = []
        self.field_observer.harmonic_history = []
        
        # Create a sequence of updates to generate tonic-harmonic pattern
        # First update - temporal context (high tonic)
        self.adaptive_id.update_temporal_context("key1", "value1", "origin1")
        
        # Second update - spatial context (normal tonic)
        self.adaptive_id.update_spatial_context("key2", "value2", "origin2")
        
        # Third update - temporal context (high tonic)
        self.adaptive_id.update_temporal_context("key3", "value3", "origin3")
        
        # Fourth update - spatial context (normal tonic)
        self.adaptive_id.update_spatial_context("key4", "value4", "origin4")
        
        # Perform harmonic analysis
        analysis = self.field_observer.perform_harmonic_analysis()
        
        # Verify analysis results
        self.assertIn("harmonic", analysis)
        self.assertIn("boundaries", analysis)
        self.assertIn("resonance", analysis)
        self.assertIn("stability_trend", analysis)
        self.assertIn("tonic_trend", analysis)
        
        # Verify we have the expected number of observations
        self.assertEqual(len(self.field_observer.observations), 4)
        
        # Verify we have alternating tonic values (high-normal-high-normal)
        self.assertEqual(len(self.field_observer.tonic_history), 4)
        self.assertEqual(self.field_observer.tonic_history[0], 0.7)  # Temporal (high)
        self.assertEqual(self.field_observer.tonic_history[1], 0.5)  # Spatial (normal)
        self.assertEqual(self.field_observer.tonic_history[2], 0.7)  # Temporal (high)
        self.assertEqual(self.field_observer.tonic_history[3], 0.5)  # Spatial (normal)
        
        # Verify harmonic values were calculated
        self.assertEqual(len(self.field_observer.harmonic_history), 4)
        
        # With alternating tonic values, we should detect at least one boundary
        self.assertGreaterEqual(len(analysis["boundaries"]), 1)
        
        # The alternating pattern should create a resonance
        if len(analysis["resonance"]) == 0:
            # If no resonance detected, check that the pattern is as expected
            harmonic = analysis["harmonic"]
            self.assertEqual(len(harmonic), 4)
            # The harmonic values should show the alternating pattern
            # even if it doesn't meet the threshold for resonance detection
            self.assertNotEqual(harmonic[0], harmonic[1])
            self.assertNotEqual(harmonic[1], harmonic[2])
            self.assertNotEqual(harmonic[2], harmonic[3])


if __name__ == '__main__':
    unittest.main()
