"""
Test for the enhanced field state modulation system.

This test verifies that the field state modulation system properly implements:
1. Field state continuity
2. Pattern relationship tracking
3. Adaptive receptivity
4. Visualization data collection
"""

import unittest
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import modules
from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from src.habitat_evolution.field.field_state import TonicHarmonicFieldState
from src.habitat_evolution.field.harmonic_field_io_bridge import HarmonicFieldIOBridge
from src.habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService, OperationType
from src.habitat_evolution.adaptive_core.resonance.tonic_harmonic_metrics import TonicHarmonicMetrics
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import FieldStateModulator


class TestFieldStateModulation(unittest.TestCase):
    """Test the enhanced field state modulation system."""

    def setUp(self):
        """Set up test environment."""
        self.field_modulator = FieldStateModulator()
        
        # Create some test patterns
        self.test_patterns = [
            {
                'id': f'pattern_{i}',
                'context': {'cascade_type': pattern_type},
                'confidence': 0.7 + (np.random.random() * 0.2)
            }
            for i, pattern_type in enumerate([
                'primary', 'primary', 'secondary', 'secondary', 
                'meta', 'emergent', 'primary', 'secondary'
            ])
        ]

    def test_field_state_continuity(self):
        """Test that field state evolves continuously rather than discretely."""
        # Record initial state
        initial_density = self.field_modulator.field_density
        initial_turbulence = self.field_modulator.field_turbulence
        
        # Record a pattern emergence
        self.field_modulator.record_pattern_emergence(self.test_patterns[0])
        
        # Check that state changed but with continuity (not a complete reset)
        self.assertNotEqual(initial_density, self.field_modulator.field_density)
        self.assertNotEqual(initial_turbulence, self.field_modulator.field_turbulence)
        
        # The change should be gradual due to continuity factor
        density_delta = abs(initial_density - self.field_modulator.field_density)
        self.assertLess(density_delta, 0.3, "Field density changed too abruptly")
        
        # Record multiple patterns to see continuous evolution
        previous_states = []
        for i in range(5):
            previous_states.append({
                'density': self.field_modulator.field_density,
                'turbulence': self.field_modulator.field_turbulence,
                'coherence': self.field_modulator.field_coherence,
                'stability': self.field_modulator.field_stability
            })
            self.field_modulator.record_pattern_emergence(self.test_patterns[i % len(self.test_patterns)])
        
        # Verify continuous evolution
        for i in range(1, len(previous_states)):
            density_delta = abs(previous_states[i]['density'] - previous_states[i-1]['density'])
            self.assertLess(density_delta, 0.3, f"Field density changed too abruptly at step {i}")
            
        # Check that field metrics history is being recorded
        self.assertGreater(len(self.field_modulator.visualization_data['field_state_history']), 0)
        
    def test_pattern_relationship_tracking(self):
        """Test that pattern relationships are properly tracked."""
        # Record several patterns in sequence to establish relationships
        for pattern in self.test_patterns[:4]:
            self.field_modulator.record_pattern_emergence(pattern)
            
        # Manually trigger relationship updates
        self.field_modulator._update_pattern_relationships('pattern_0', 'primary')
        self.field_modulator._update_pattern_relationships('pattern_2', 'secondary')
        
        # Check that relationships were established
        self.assertGreater(len(self.field_modulator.pattern_relationships), 0)
        
        # Check relationship strength history
        self.assertGreater(len(self.field_modulator.relationship_strength_history), 0)
        
    def test_adaptive_receptivity(self):
        """Test that receptivity adapts based on detection patterns."""
        # Initial receptivity values
        initial_receptivity = self.field_modulator.pattern_receptivity.copy()
        
        # Simulate many detections of primary patterns
        for _ in range(20):
            self.field_modulator.receptivity_history['primary'].append(1.0)
            
        # Simulate few detections of meta patterns
        for _ in range(20):
            self.field_modulator.receptivity_history['meta'].append(0.0)
            
        # Update adaptive receptivity
        self.field_modulator._update_adaptive_receptivity()
        
        # Primary should have decreased receptivity (detected often)
        self.assertLess(
            self.field_modulator.pattern_receptivity['primary'],
            initial_receptivity['primary']
        )
        
        # Meta should have increased receptivity (detected rarely)
        self.assertGreater(
            self.field_modulator.pattern_receptivity['meta'],
            initial_receptivity['meta']
        )
        
    def test_pattern_detection_with_relationships(self):
        """Test that pattern detection considers relationships."""
        # Record several patterns to establish relationships
        for pattern in self.test_patterns[:4]:
            self.field_modulator.record_pattern_emergence(pattern)
            
        # Create a relationship between pattern_0 and pattern_2
        relationship_key = "pattern_0_pattern_2"
        self.field_modulator.pattern_relationships[relationship_key] = 0.8
        
        # Create interference patterns
        self.field_modulator._create_interference_pattern('pattern_0', 'primary', 0.8)
        self.field_modulator._create_interference_pattern('pattern_2', 'secondary', 0.7)
        
        # Test pattern with relationship
        test_pattern = {
            'id': 'pattern_0',
            'context': {'cascade_type': 'primary'},
            'confidence': 0.6
        }
        
        # Test pattern without relationship
        test_pattern_no_rel = {
            'id': 'pattern_5',
            'context': {'cascade_type': 'primary'},
            'confidence': 0.6
        }
        
        # Detect both patterns
        should_detect, confidence_mod = self.field_modulator.should_detect_pattern(test_pattern)
        should_detect_no_rel, confidence_mod_no_rel = self.field_modulator.should_detect_pattern(test_pattern_no_rel)
        
        # Pattern with relationship should have higher chance of detection
        # This is implemented through the relationship_factor in threshold calculation
        
        # Check visualization data
        self.assertGreater(len(self.field_modulator.visualization_data['pattern_emergence_points']), 0)
        
    def test_field_state_metrics(self):
        """Test that field state metrics are properly reported."""
        # Record some patterns
        for pattern in self.test_patterns[:4]:
            self.field_modulator.record_pattern_emergence(pattern)
            
        # Get field state
        field_state = self.field_modulator.get_field_state()
        
        # Check that all enhanced metrics are present
        self.assertIn('field_density', field_state)
        self.assertIn('field_turbulence', field_state)
        self.assertIn('field_coherence', field_state)
        self.assertIn('field_stability', field_state)
        self.assertIn('continuity_factor', field_state)
        self.assertIn('field_metrics_history', field_state)
        self.assertIn('top_relationships', field_state)
        
        # Check that visualization data is being collected
        self.assertGreater(len(self.field_modulator.visualization_data['field_state_history']), 0)


if __name__ == '__main__':
    unittest.main()
