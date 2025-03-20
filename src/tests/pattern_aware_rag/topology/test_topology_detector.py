"""
Tests for the topology detector.

These tests verify the functionality of the topology detector, including
frequency domain detection, boundary detection, resonance point detection,
and field dynamics analysis.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from habitat_evolution.pattern_aware_rag.topology.detector import TopologyDetector
from habitat_evolution.pattern_aware_rag.topology.models import (
    FrequencyDomain, Boundary, ResonancePoint, FieldMetrics, TopologyState
)


class MockPattern:
    """Mock pattern class for testing."""
    
    def __init__(self, pattern_id, evolution_history=None):
        """Initialize mock pattern."""
        self.pattern_id = pattern_id
        self.evolution_history = evolution_history or []
        self.evolution_count = len(self.evolution_history)


class MockLearningWindow:
    """Mock learning window class for testing."""
    
    def __init__(self, window_id, frequency=0.5, stability_threshold=0.7):
        """Initialize mock learning window."""
        self.window_id = window_id
        self.frequency = frequency
        self.stability_threshold = stability_threshold
        self.observers = []
    
    def register_pattern_observer(self, observer):
        """Register pattern observer."""
        self.observers.append(observer)


class TestTopologyDetector(unittest.TestCase):
    """Test case for topology detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = TopologyDetector()
        
        # Create test patterns with evolution histories
        self.pattern_a = self.create_test_pattern(
            "pattern-a", 
            frequency=0.2, 
            num_events=20,
            tonic_base=0.6
        )
        
        self.pattern_b = self.create_test_pattern(
            "pattern-b", 
            frequency=0.2, 
            num_events=20,
            tonic_base=0.7,
            phase_shift=0.1  # Similar frequency but phase shifted
        )
        
        self.pattern_c = self.create_test_pattern(
            "pattern-c", 
            frequency=0.5, 
            num_events=20,
            tonic_base=0.5
        )
        
        self.pattern_d = self.create_test_pattern(
            "pattern-d", 
            frequency=0.8, 
            num_events=20,
            tonic_base=0.4
        )
        
        # Create test learning windows
        self.window_a = MockLearningWindow("window-a", frequency=0.2)
        self.window_b = MockLearningWindow("window-b", frequency=0.5)
        self.window_c = MockLearningWindow("window-c", frequency=0.8)
        
        # Time period for analysis
        self.time_period = {
            "start": datetime.now() - timedelta(hours=1),
            "end": datetime.now()
        }
    
    def create_test_pattern(self, pattern_id, frequency=0.5, num_events=20, 
                           tonic_base=0.5, phase_shift=0.0):
        """Create a test pattern with synthetic evolution history."""
        evolution_history = []
        
        # Create synthetic harmonic values based on frequency
        for i in range(num_events):
            # Create sinusoidal pattern of harmonic values
            t = i / num_events
            tonic = tonic_base + 0.2 * np.sin(2 * np.pi * frequency * t + phase_shift)
            stability = 0.5 + 0.3 * np.sin(2 * np.pi * frequency * t + phase_shift + 0.5)
            harmonic = tonic * stability
            
            # Create event
            event = {
                "timestamp": datetime.now() - timedelta(minutes=(num_events - i) * 3),
                "change_type": f"test_change_{i}",
                "old_value": {"value": i - 1},
                "new_value": {"value": i},
                "origin": f"test_origin_{i % 3}",
                "tonic_value": tonic,
                "stability": stability,
                "harmonic_value": harmonic
            }
            
            evolution_history.append(event)
        
        return MockPattern(pattern_id, evolution_history)
    
    def test_detect_frequency_domains(self):
        """Test detection of frequency domains."""
        # Create pattern histories dictionary
        pattern_histories = {
            "pattern-a": self.pattern_a.evolution_history,
            "pattern-b": self.pattern_b.evolution_history,
            "pattern-c": self.pattern_c.evolution_history,
            "pattern-d": self.pattern_d.evolution_history
        }
        
        # Detect frequency domains
        domains = self.detector.detect_frequency_domains(pattern_histories)
        
        # Verify domains were detected
        self.assertGreaterEqual(len(domains), 2, "Should detect at least 2 frequency domains")
        
        # Check that similar patterns are grouped together
        domain_patterns = {}
        for domain_id, domain in domains.items():
            domain_patterns[domain_id] = domain.pattern_ids
        
        # Find domain containing pattern-a
        domain_with_a = None
        for domain_id, patterns in domain_patterns.items():
            if "pattern-a" in patterns:
                domain_with_a = domain_id
                break
        
        # Verify pattern-b is in the same domain as pattern-a (similar frequency)
        self.assertIsNotNone(domain_with_a, "Should find domain containing pattern-a")
        self.assertIn("pattern-b", domain_patterns[domain_with_a], 
                    "pattern-a and pattern-b should be in the same domain")
        
        # Verify pattern-c and pattern-d are in different domains
        domain_with_c = None
        for domain_id, patterns in domain_patterns.items():
            if "pattern-c" in patterns:
                domain_with_c = domain_id
                break
        
        self.assertIsNotNone(domain_with_c, "Should find domain containing pattern-c")
        self.assertNotEqual(domain_with_a, domain_with_c, 
                          "pattern-a and pattern-c should be in different domains")
    
    def test_calculate_harmonic_landscape(self):
        """Test calculation of harmonic landscape."""
        # Calculate harmonic landscape
        patterns = [self.pattern_a, self.pattern_b, self.pattern_c, self.pattern_d]
        windows = [self.window_a, self.window_b, self.window_c]
        
        landscape = self.detector.calculate_harmonic_landscape(patterns, windows)
        
        # Verify landscape properties
        self.assertIsInstance(landscape, np.ndarray)
        self.assertEqual(landscape.shape, (50, 50))
        self.assertGreaterEqual(np.max(landscape), 0.0)
        self.assertLessEqual(np.max(landscape), 1.0)
    
    def test_detect_boundaries(self):
        """Test detection of boundaries."""
        # Create a simple test landscape with a clear boundary
        landscape = np.zeros((50, 50))
        landscape[10:20, 10:20] = 0.8  # High value region
        landscape[30:40, 30:40] = 0.6  # Medium value region
        
        # Add gradient at boundary
        for i in range(5):
            landscape[20+i, 20+i] = 0.8 - (i * 0.1)
        
        # Detect boundaries
        boundaries = self.detector.detect_boundaries(landscape)
        
        # Verify boundaries were detected
        self.assertGreaterEqual(len(boundaries), 1, "Should detect at least 1 boundary")
        
        # Check boundary properties
        for boundary_id, boundary in boundaries.items():
            self.assertGreaterEqual(boundary.sharpness, 0.0)
            self.assertLessEqual(boundary.sharpness, 1.0)
            self.assertGreaterEqual(boundary.permeability, 0.0)
            self.assertLessEqual(boundary.permeability, 1.0)
            self.assertGreaterEqual(len(boundary.coordinates), 1)
    
    def test_detect_resonance_points(self):
        """Test detection of resonance points."""
        # Create a simple test landscape with clear resonance points
        landscape = np.zeros((50, 50))
        landscape[15, 15] = 0.9  # Strong resonance point
        landscape[35, 35] = 0.8  # Medium resonance point
        
        # Add some noise
        landscape += np.random.random((50, 50)) * 0.1
        
        # Ensure max is still 1.0
        landscape = landscape / np.max(landscape)
        
        # Detect resonance points
        resonance_points = self.detector.detect_resonance_points(landscape)
        
        # Verify resonance points were detected
        self.assertGreaterEqual(len(resonance_points), 1, "Should detect at least 1 resonance point")
        
        # Check resonance point properties
        for point_id, point in resonance_points.items():
            self.assertGreaterEqual(point.strength, 0.0)
            self.assertLessEqual(point.strength, 1.0)
            self.assertGreaterEqual(point.attractor_radius, 0.0)
    
    def test_analyze_field_dynamics(self):
        """Test analysis of field dynamics."""
        # Create pattern histories dictionary
        pattern_histories = {
            "pattern-a": self.pattern_a.evolution_history,
            "pattern-b": self.pattern_b.evolution_history,
            "pattern-c": self.pattern_c.evolution_history,
            "pattern-d": self.pattern_d.evolution_history
        }
        
        # Analyze field dynamics
        metrics = self.detector.analyze_field_dynamics(pattern_histories, self.time_period)
        
        # Verify metrics
        self.assertIsInstance(metrics, FieldMetrics)
        self.assertGreaterEqual(metrics.coherence, 0.0)
        self.assertLessEqual(metrics.coherence, 1.0)
        self.assertGreaterEqual(metrics.entropy, 0.0)
        self.assertLessEqual(metrics.entropy, 1.0)
    
    def test_analyze_topology(self):
        """Test complete topology analysis."""
        # Perform complete analysis
        patterns = [self.pattern_a, self.pattern_b, self.pattern_c, self.pattern_d]
        windows = [self.window_a, self.window_b, self.window_c]
        
        state = self.detector.analyze_topology(patterns, windows, self.time_period)
        
        # Verify state
        self.assertIsInstance(state, TopologyState)
        self.assertGreaterEqual(len(state.frequency_domains), 1)
        self.assertGreaterEqual(len(state.boundaries), 0)
        self.assertGreaterEqual(len(state.resonance_points), 0)
        self.assertIsNotNone(state.field_metrics)


if __name__ == "__main__":
    unittest.main()
