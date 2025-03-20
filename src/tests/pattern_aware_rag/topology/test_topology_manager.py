"""
Tests for the topology manager.

These tests verify the functionality of the topology manager, including
topology detection, persistence to Neo4j, serialization/deserialization,
and topology evolution tracking.
"""

import unittest
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from habitat_evolution.pattern_aware_rag.topology.manager import TopologyManager
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


class MockNeo4jSession:
    """Mock Neo4j session for testing."""
    
    def __init__(self):
        """Initialize mock session."""
        self.queries = []
        self.results = {}
    
    def run(self, query, **params):
        """Run a query."""
        self.queries.append((query, params))
        return self.results.get(query, MockNeo4jResult([]))
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        pass


class MockNeo4jResult:
    """Mock Neo4j result for testing."""
    
    def __init__(self, records):
        """Initialize mock result."""
        self.records = records
    
    def single(self):
        """Get single record."""
        return self.records[0] if self.records else None


class MockNeo4jDriver:
    """Mock Neo4j driver for testing."""
    
    def __init__(self):
        """Initialize mock driver."""
        self.session = MockNeo4jSession()
    
    def session(self):
        """Get session."""
        return self.session


class TestTopologyManager(unittest.TestCase):
    """Test case for topology manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock Neo4j driver
        self.neo4j_driver = MockNeo4jDriver()
        
        # Create topology manager
        self.manager = TopologyManager(neo4j_driver=self.neo4j_driver, persistence_mode=True)
        
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
        
        # Create test learning windows
        self.window_a = MockLearningWindow("window-a", frequency=0.2)
        self.window_b = MockLearningWindow("window-b", frequency=0.5)
        
        # Time period for analysis
        self.time_period = {
            "start": datetime.now() - timedelta(hours=1),
            "end": datetime.now()
        }
    
    def create_test_pattern(self, pattern_id, frequency=0.5, num_events=20, 
                           tonic_base=0.5, phase_shift=0.0):
        """Create a test pattern with synthetic evolution history."""
        import numpy as np
        
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
    
    def test_analyze_patterns(self):
        """Test analysis of patterns."""
        # Analyze patterns
        patterns = [self.pattern_a, self.pattern_b, self.pattern_c]
        windows = [self.window_a, self.window_b]
        
        state = self.manager.analyze_patterns(patterns, windows, self.time_period)
        
        # Verify state
        self.assertIsNotNone(state)
        self.assertIsInstance(state, TopologyState)
        self.assertEqual(state, self.manager.current_state)
        self.assertIn(state, self.manager.state_history)
        
        # Verify Neo4j persistence was called
        self.assertGreater(len(self.neo4j_driver.session.queries), 0)
    
    def test_serialize_deserialize(self):
        """Test serialization and deserialization of topology state."""
        # First analyze patterns to create a state
        patterns = [self.pattern_a, self.pattern_b, self.pattern_c]
        windows = [self.window_a, self.window_b]
        
        state = self.manager.analyze_patterns(patterns, windows, self.time_period)
        
        # Serialize state
        json_str = self.manager.serialize_current_state()
        self.assertIsNotNone(json_str)
        
        # Create a new manager
        new_manager = TopologyManager(persistence_mode=False)
        
        # Deserialize state
        loaded_state = new_manager.load_from_serialized(json_str)
        
        # Verify loaded state
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state.id, state.id)
        self.assertEqual(len(loaded_state.frequency_domains), len(state.frequency_domains))
        self.assertEqual(len(loaded_state.boundaries), len(state.boundaries))
        self.assertEqual(len(loaded_state.resonance_points), len(state.resonance_points))
    
    def test_file_persistence(self):
        """Test saving and loading topology state to/from file."""
        # First analyze patterns to create a state
        patterns = [self.pattern_a, self.pattern_b, self.pattern_c]
        windows = [self.window_a, self.window_b]
        
        state = self.manager.analyze_patterns(patterns, windows, self.time_period)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            
            # Serialize state to file
            json_str = self.manager.serialize_current_state()
            temp_file.write(json_str.encode('utf-8'))
        
        try:
            # Create a new manager
            new_manager = TopologyManager(persistence_mode=False)
            
            # Load state from file
            with open(temp_path, 'r') as f:
                json_str = f.read()
            
            loaded_state = new_manager.load_from_serialized(json_str)
            
            # Verify loaded state
            self.assertIsNotNone(loaded_state)
            self.assertEqual(loaded_state.id, state.id)
            self.assertEqual(len(loaded_state.frequency_domains), len(state.frequency_domains))
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('habitat_evolution.pattern_aware_rag.topology.manager.TopologyManager.load_from_neo4j')
    def test_get_topology_diff(self, mock_load):
        """Test getting difference between topology states."""
        # Create two states with different properties
        state1 = TopologyState(
            id="ts-test-1",
            frequency_domains={
                "fd-1": FrequencyDomain(id="fd-1", dominant_frequency=0.2),
                "fd-2": FrequencyDomain(id="fd-2", dominant_frequency=0.5)
            },
            boundaries={
                "b-1": Boundary(id="b-1", sharpness=0.8)
            },
            field_metrics=FieldMetrics(coherence=0.7)
        )
        
        state2 = TopologyState(
            id="ts-test-2",
            frequency_domains={
                "fd-1": FrequencyDomain(id="fd-1", dominant_frequency=0.2),
                "fd-3": FrequencyDomain(id="fd-3", dominant_frequency=0.8)
            },
            boundaries={
                "b-1": Boundary(id="b-1", sharpness=0.9)  # Changed
            },
            field_metrics=FieldMetrics(coherence=0.8)  # Changed
        )
        
        # Set up mock to return state1
        mock_load.return_value = state1
        
        # Set current state to state2
        self.manager.current_state = state2
        
        # Get diff
        diff = self.manager.get_topology_diff("ts-test-1")
        
        # Verify diff
        self.assertIn("added_domains", diff)
        self.assertIn("removed_domains", diff)
        self.assertIn("modified_boundaries", diff)
        self.assertIn("field_metrics_changes", diff)
        
        self.assertIn("fd-3", diff["added_domains"])
        self.assertIn("fd-2", diff["removed_domains"])
        self.assertIn("b-1", diff["modified_boundaries"])
        self.assertIn("coherence", diff["field_metrics_changes"])


class TestTopologyManagerIntegration(unittest.TestCase):
    """Integration tests for topology manager with pattern co-evolution."""
    
    @unittest.skip("Requires Neo4j database")
    def test_integration_with_pattern_coevolution(self):
        """Test integration with pattern co-evolution system."""
        from habitat_evolution.pattern_aware_rag.learning.learning_control import LearningWindow
        from habitat_evolution.pattern_aware_rag.learning.pattern_id import PatternID
        from habitat_evolution.pattern_aware_rag.learning.event_coordinator import EventCoordinator
        
        # Create event coordinator
        coordinator = EventCoordinator(max_queue_size=100, persistence_mode=False)
        
        # Create learning windows with different frequencies
        high_freq_window = coordinator.create_learning_window(
            duration_minutes=10,
            stability_threshold=0.6,
            coherence_threshold=0.7,
            max_changes=5
        )
        
        medium_freq_window = coordinator.create_learning_window(
            duration_minutes=30,
            stability_threshold=0.7,
            coherence_threshold=0.8,
            max_changes=10
        )
        
        low_freq_window = coordinator.create_learning_window(
            duration_minutes=60,
            stability_threshold=0.8,
            coherence_threshold=0.9,
            max_changes=15
        )
        
        # Create patterns
        pattern_a = PatternID("test-pattern-a")
        pattern_b = PatternID("test-pattern-b")
        pattern_c = PatternID("test-pattern-c")
        
        # Register patterns with windows
        for window in [high_freq_window, medium_freq_window, low_freq_window]:
            window.register_pattern_observer(pattern_a)
            window.register_pattern_observer(pattern_b)
            window.register_pattern_observer(pattern_c)
        
        # Activate windows
        high_freq_window.activate(stability_score=0.8)
        medium_freq_window.activate(stability_score=0.9)
        low_freq_window.activate(stability_score=0.7)
        
        # Generate some state changes
        for i in range(10):
            # High frequency changes
            high_freq_window.record_state_change(
                change_type=f"high_freq_change_{i}",
                old_value={"value": i},
                new_value={"value": i + 1},
                origin="high_freq",
                entity_id=f"entity_{i % 3}",
                tonic_value=0.6 + (i % 5) * 0.05,
                stability=0.7 + (i % 3) * 0.05
            )
            
            # Medium frequency changes (every other iteration)
            if i % 2 == 0:
                medium_freq_window.record_state_change(
                    change_type=f"medium_freq_change_{i}",
                    old_value={"value": i * 2},
                    new_value={"value": i * 2 + 2},
                    origin="medium_freq",
                    entity_id=f"entity_{i % 3}",
                    tonic_value=0.7 + (i % 4) * 0.05,
                    stability=0.8 + (i % 2) * 0.05
                )
            
            # Low frequency changes (every third iteration)
            if i % 3 == 0:
                low_freq_window.record_state_change(
                    change_type=f"low_freq_change_{i}",
                    old_value={"value": i * 3},
                    new_value={"value": i * 3 + 3},
                    origin="low_freq",
                    entity_id=f"entity_{i % 3}",
                    tonic_value=0.8 + (i % 3) * 0.05,
                    stability=0.9 + (i % 2) * 0.03
                )
        
        # Create topology manager
        manager = TopologyManager(persistence_mode=True)
        
        # Analyze patterns
        state = manager.analyze_patterns(
            patterns=[pattern_a, pattern_b, pattern_c],
            learning_windows=[high_freq_window, medium_freq_window, low_freq_window],
            time_period={
                "start": datetime.now() - timedelta(hours=1),
                "end": datetime.now()
            }
        )
        
        # Verify topology detection
        self.assertIsNotNone(state)
        self.assertGreaterEqual(len(state.frequency_domains), 1, 
                              "Should detect at least one frequency domain")
        
        # Verify frequency domains reflect window frequencies
        frequencies = [domain.dominant_frequency for domain in state.frequency_domains.values()]
        self.assertTrue(any(f > 0.5 for f in frequencies), 
                      "Should detect high frequency domain")
        self.assertTrue(any(f < 0.4 for f in frequencies), 
                      "Should detect low frequency domain")
        
        # Verify boundaries between frequency domains
        if len(state.frequency_domains) >= 2:
            self.assertGreaterEqual(len(state.boundaries), 1, 
                                  "Should detect boundaries between frequency domains")
        
        # Verify resonance points
        self.assertGreaterEqual(len(state.resonance_points), 0)
        
        # Verify field metrics
        self.assertGreaterEqual(state.field_metrics.coherence, 0.0)
        self.assertLessEqual(state.field_metrics.coherence, 1.0)


if __name__ == "__main__":
    unittest.main()
