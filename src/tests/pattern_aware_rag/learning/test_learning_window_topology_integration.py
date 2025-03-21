"""
Tests for the integration between learning windows and topology states.

These tests verify the bidirectional communication between learning windows and topology states, including:
1. Learning window state changes reflected in topology
2. Topology changes affecting learning window behavior
3. Tonic-harmonic properties synchronization
4. Wave interference patterns across learning windows
"""

import unittest
import json
import os
import sys
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
import random
import numpy as np
from unittest.mock import MagicMock, patch
import pathlib

# Add the src directory to the Python path
src_path = str(pathlib.Path(__file__).parent.parent.parent.parent.absolute())
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    print(f"Added {src_path} to Python path")

from neo4j import GraphDatabase, Driver, Session

# Import the modules we need to test
from habitat_evolution.pattern_aware_rag.topology.models import (
    FrequencyDomain, Boundary, ResonancePoint, FieldMetrics, TopologyState
)
from habitat_evolution.pattern_aware_rag.topology.manager import TopologyManager
from habitat_evolution.pattern_aware_rag.learning.learning_control import (
    LearningWindow, WindowState, EventCoordinator
)


class TestLearningWindowTopologyIntegration(unittest.TestCase):
    """Test case for integration between learning windows and topology states."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the entire test case."""
        # Check if Neo4j is available
        try:
            cls.driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "habitat123"))
            # Test connection
            with cls.driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) AS count")
                count = result.single()["count"]
                print(f"Connected to Neo4j, found {count} nodes")
            cls.neo4j_available = True
        except Exception as e:
            print(f"Neo4j not available: {e}")
            cls.neo4j_available = False
            cls.driver = None
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        if cls.driver:
            cls.driver.close()
    
    def setUp(self):
        """Set up test fixtures for each test."""
        if not self.neo4j_available:
            self.skipTest("Neo4j not available")
        
        # Create topology manager with real Neo4j connection
        self.manager = TopologyManager(neo4j_driver=self.driver, persistence_mode=True)
        
        # Clean up any existing topology data
        with self.driver.session() as session:
            session.run("""
                MATCH (n:TopologyState) DETACH DELETE n
            """)
            session.run("""
                MATCH (n:LearningWindow) DETACH DELETE n
            """)
            session.run("""
                MATCH (n:Pattern) DETACH DELETE n
            """)
        
        # Create mock learning windows
        self.learning_windows = self.create_mock_learning_windows()
        
        # Create mock event coordinator
        self.event_coordinator = self.create_mock_event_coordinator()
    
    def create_mock_learning_windows(self):
        """Create mock learning windows for testing."""
        windows = {}
        
        # Create windows in different states
        states = [WindowState.OPENING, WindowState.OPEN, WindowState.OPEN, WindowState.CLOSED]
        for i, state in enumerate(states):
            window_id = f"window-{i}"
            
            # Create a mock learning window
            window = MagicMock(spec=LearningWindow)
            window.id = window_id
            window.state = state
            window.stability_score = 0.7 + (i / 10.0)
            window.coherence_score = 0.8 + (i / 15.0)
            window.saturation = 0.3 + (i / 10.0)
            window.duration_minutes = 5 + (i * 2)
            window.is_active = state in [WindowState.OPENING, WindowState.OPEN]
            
            # Create window info dict for topology state
            window_info = {
                'state': state.name,
                'stability': window.stability_score,
                'coherence': window.coherence_score,
                'saturation': window.saturation,
                'duration_minutes': window.duration_minutes
            }
            
            windows[window_id] = {
                'window': window,
                'info': window_info
            }
        
        return windows
    
    def create_mock_event_coordinator(self):
        """Create a mock event coordinator for testing."""
        coordinator = MagicMock(spec=EventCoordinator)
        coordinator.learning_windows = [info['window'] for _, info in self.learning_windows.items()]
        coordinator.queue_event = MagicMock(return_value=True)
        coordinator.process_events = MagicMock(return_value=True)
        
        return coordinator
    
    def create_topology_state_with_windows(self):
        """Create a topology state with learning windows."""
        # Create pattern eigenspace properties
        pattern_eigenspace_properties = {}
        for i in range(5):
            pattern_id = f"pattern-{i}"
            pattern_eigenspace_properties[pattern_id] = {
                'dimensional_coordinates': [random.random() for _ in range(3)],
                'primary_dimensions': [i % 3],
                'eigenspace_centrality': 0.5 + (i / 10.0),
                'eigenspace_stability': 0.6 + (i / 10.0),
                'frequency': 0.1 + (i / 20.0),
                'phase_position': (i / 5.0) % 1.0,
                'tonic_value': 0.5 + 0.3 * math.sin(2 * math.pi * (i / 5.0)),
                'harmonic_value': 0.6 * (0.5 + 0.3 * math.sin(2 * math.pi * (i / 5.0)))
            }
        
        # Create field metrics
        field_metrics = FieldMetrics(
            coherence=0.75,
            energy_density={f"region-{i}": 0.5 + i * 0.1 for i in range(3)},
            adaptation_rate=0.45,
            entropy=0.35
        )
        
        # Create topology state
        state = TopologyState(
            id=f"ts-learning-{int(time.time())}",
            frequency_domains={},
            boundaries={},
            resonance_points={},
            field_metrics=field_metrics,
            timestamp=datetime.now(),
            pattern_eigenspace_properties=pattern_eigenspace_properties
        )
        
        # Add learning windows as an attribute after creation
        # This matches how TopologyManager.persist_to_neo4j checks for learning windows
        setattr(state, 'learning_windows', {window_id: info['info'] for window_id, info in self.learning_windows.items()})
        
        return state
    
    def test_learning_window_persistence(self):
        """Test persistence of learning windows in topology state."""
        # Create topology state with learning windows
        state = self.create_topology_state_with_windows()
        
        # Persist to Neo4j
        self.manager.persist_to_neo4j(state)
        
        # Verify learning windows were persisted
        with self.driver.session() as session:
            # Check LearningWindow nodes
            result = session.run("""
                MATCH (lw:LearningWindow)
                RETURN count(lw) as count
            """)
            
            count = result.single()["count"]
            self.assertEqual(count, 4)
            
            # Check relationships between topology state and learning windows
            result = session.run("""
                MATCH (ts:TopologyState)-[r:OBSERVED_IN]->(lw:LearningWindow)
                RETURN count(r) as count
            """)
            
            count = result.single()["count"]
            self.assertEqual(count, 4)
            
            # Check tonic-harmonic properties on learning windows
            result = session.run("""
                MATCH (lw:LearningWindow)
                WHERE lw.frequency IS NOT NULL 
                  AND lw.phase_position IS NOT NULL
                  AND lw.tonic_value IS NOT NULL
                  AND lw.harmonic_value IS NOT NULL
                RETURN count(lw) as count
            """)
            
            count = result.single()["count"]
            self.assertEqual(count, 4)
    
    def test_learning_window_state_changes(self):
        """Test learning window state changes reflected in topology."""
        # Create initial topology state with learning windows
        state = self.create_topology_state_with_windows()
        
        # Persist to Neo4j
        self.manager.persist_to_neo4j(state)
        
        # Change window states
        for window_id, info in self.learning_windows.items():
            if info['window'].state == WindowState.OPENING:
                info['window'].state = WindowState.OPEN
                info['info']['state'] = 'OPEN'
            elif info['window'].state == WindowState.OPEN:
                info['window'].state = WindowState.CLOSED
                info['info']['state'] = 'CLOSED'

            elif info['window'].state == WindowState.CLOSED:
                info['window'].state = WindowState.OPENING
                info['info']['state'] = 'OPENING'
        
        # Create updated topology state
        updated_state = TopologyState(
            id=f"ts-learning-updated-{int(time.time())}",
            frequency_domains={},
            boundaries={},
            resonance_points={},
            field_metrics=state.field_metrics,
            timestamp=datetime.now(),
            pattern_eigenspace_properties=state.pattern_eigenspace_properties
        )
        
        # Add learning windows as an attribute after creation
        setattr(updated_state, 'learning_windows', {window_id: info['info'] for window_id, info in self.learning_windows.items()})
        
        # Persist updated state
        self.manager.persist_to_neo4j(updated_state)
        
        # Verify learning window states were updated
        with self.driver.session() as session:
            for window_id, info in self.learning_windows.items():
                result = session.run("""
                    MATCH (lw:LearningWindow {id: $window_id})
                    RETURN lw.state as state
                """, window_id=window_id)
                
                record = result.single()
                self.assertEqual(record["state"], info['info']['state'])
    
    def test_bidirectional_communication(self):
        """Test bidirectional communication between learning windows and topology states."""
        # Create initial topology state with learning windows
        state = self.create_topology_state_with_windows()
        
        # Persist to Neo4j
        self.manager.persist_to_neo4j(state)
        
        # Simulate a learning window event
        active_window_id = next(window_id for window_id, info in self.learning_windows.items() 
                               if info['window'].is_active)
        active_window = self.learning_windows[active_window_id]['window']
        
        # Create a pattern event
        pattern_id = "pattern-new"
        pattern_event = {
            'pattern_id': pattern_id,
            'stability': 0.8,
            'coherence': 0.7,
            'timestamp': datetime.now().timestamp()
        }
        
        # Mock event coordination
        self.event_coordinator.queue_event.return_value = True
        
        # Update pattern eigenspace properties
        pattern_eigenspace_properties = dict(state.pattern_eigenspace_properties)
        pattern_eigenspace_properties[pattern_id] = {
            'dimensional_coordinates': [random.random() for _ in range(3)],
            'primary_dimensions': [0],
            'eigenspace_centrality': 0.6,
            'eigenspace_stability': 0.7,
            'frequency': 0.15,
            'phase_position': 0.3,
            'tonic_value': 0.6,
            'harmonic_value': 0.42
        }
        
        # Create updated topology state with new pattern
        updated_state = TopologyState(
            id=f"ts-learning-bidirectional-{int(time.time())}",
            frequency_domains={},
            boundaries={},
            resonance_points={},
            field_metrics=state.field_metrics,
            timestamp=datetime.now(),
            pattern_eigenspace_properties=pattern_eigenspace_properties
        )
        
        # Add learning windows as an attribute after creation
        setattr(updated_state, 'learning_windows', {window_id: info['info'] for window_id, info in self.learning_windows.items()})
        
        # Add resonance relationships to ensure pattern properties are persisted correctly
        resonance_relationships = {}
        for p_id in pattern_eigenspace_properties.keys():
            resonance_relationships[p_id] = []
        setattr(updated_state, 'resonance_relationships', resonance_relationships)
        
        # Persist updated state
        self.manager.persist_to_neo4j(updated_state)
        
        # Verify new pattern was persisted with tonic-harmonic properties
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Pattern {id: $pattern_id})
                RETURN p.frequency as frequency,
                       p.phase_position as phase_position,
                       p.tonic_value as tonic_value,
                       p.harmonic_value as harmonic_value
            """, pattern_id=pattern_id)
            
            record = result.single()
            self.assertIsNotNone(record)
            self.assertEqual(record["frequency"], 0.15)
            # Phase position is calculated dynamically based on current time
            # so we just check that it exists and is a float between 0 and 1
            self.assertIsNotNone(record["phase_position"])
            self.assertGreaterEqual(record["phase_position"], 0.0)
            self.assertLessEqual(record["phase_position"], 1.0)
            # Tonic and harmonic values are derived from phase position
            # so we just check that they exist and are in the expected range
            self.assertIsNotNone(record["tonic_value"])
            self.assertGreaterEqual(record["tonic_value"], 0.1)
            self.assertLessEqual(record["tonic_value"], 0.9)
            self.assertIsNotNone(record["harmonic_value"])
    
    def test_wave_interference_across_windows(self):
        """Test wave interference patterns across learning windows."""
        # Create initial topology state with learning windows
        state = self.create_topology_state_with_windows()
        
        # Add patterns with specific phase relationships
        pattern_eigenspace_properties = {}
        
        # Create patterns with constructive interference - identical phase positions
        # For constructive interference, phase_diff should be < 0.1 or > 0.9
        pattern_eigenspace_properties["pattern-constructive-1"] = {
            'dimensional_coordinates': [0.5, 0.3, 0.2],
            'primary_dimensions': [0],
            'eigenspace_centrality': 0.7,
            'eigenspace_stability': 0.8,
            'frequency': 0.2,
            'phase_position': 0.0,  # Identical phase for constructive interference
            'tonic_value': 0.7,
            'harmonic_value': 0.56
        }
        
        pattern_eigenspace_properties["pattern-constructive-2"] = {
            'dimensional_coordinates': [0.5, 0.3, 0.2],
            'primary_dimensions': [0],
            'eigenspace_centrality': 0.7,
            'eigenspace_stability': 0.8,
            'frequency': 0.2,
            'phase_position': 0.0,  # Identical phase for constructive interference
            'tonic_value': 0.7,
            'harmonic_value': 0.56
        }
        
        # Create patterns with destructive interference - opposite phase positions
        # For destructive interference, abs(phase_diff - 0.5) should be < 0.1
        pattern_eigenspace_properties["pattern-destructive-1"] = {
            'dimensional_coordinates': [0.2, 0.6, 0.1],
            'primary_dimensions': [1],
            'eigenspace_centrality': 0.6,
            'eigenspace_stability': 0.7,
            'frequency': 0.3,
            'phase_position': 0.0,  # Base phase
            'tonic_value': 0.6,
            'harmonic_value': 0.42
        }
        
        pattern_eigenspace_properties["pattern-destructive-2"] = {
            'dimensional_coordinates': [0.2, 0.6, 0.1],
            'primary_dimensions': [1],
            'eigenspace_centrality': 0.6,
            'eigenspace_stability': 0.7,
            'frequency': 0.3,
            'phase_position': 0.5,  # Exactly opposite phase (0.5 difference) for destructive interference
            'tonic_value': 0.6,
            'harmonic_value': 0.42
        }
        
        # Create topology state with interference patterns
        interference_state = TopologyState(
            id=f"ts-interference-{int(time.time())}",
            frequency_domains={},
            boundaries={},
            resonance_points={},
            field_metrics=state.field_metrics,
            timestamp=datetime.now(),
            pattern_eigenspace_properties=pattern_eigenspace_properties
        )
        
        # Add learning windows as an attribute after creation
        setattr(interference_state, 'learning_windows', {window_id: info['info'] for window_id, info in self.learning_windows.items()})
        
        # Add resonance relationships for wave interference detection
        resonance_relationships = {
            "pattern-constructive-1": [
                {
                    "pattern_id": "pattern-constructive-2",
                    "similarity": 0.9,
                    "resonance_types": ["direct", "harmonic"]
                }
            ],
            "pattern-destructive-1": [
                {
                    "pattern_id": "pattern-destructive-2",
                    "similarity": 0.8,
                    "resonance_types": ["direct", "harmonic"]
                }
            ]
        }
        setattr(interference_state, 'resonance_relationships', resonance_relationships)
        
        # Persist interference state
        self.manager.persist_to_neo4j(interference_state)
        
        # Verify wave interference relationships were created
        with self.driver.session() as session:
            # Check constructive interference
            result = session.run("""
                MATCH (p1:Pattern {id: 'pattern-constructive-1'})-[r:RESONATES_WITH]->(p2:Pattern {id: 'pattern-constructive-2'})
                RETURN r.wave_interference as interference_type, r.interference_strength as strength
            """)
            
            record = result.single()
            self.assertIsNotNone(record)
            self.assertEqual(record["interference_type"], "CONSTRUCTIVE")
            self.assertGreater(record["strength"], 0)
            
            # Check destructive interference
            result = session.run("""
                MATCH (p1:Pattern {id: 'pattern-destructive-1'})-[r:RESONATES_WITH]->(p2:Pattern {id: 'pattern-destructive-2'})
                RETURN r.wave_interference as interference_type, r.interference_strength as strength
            """)
            
            record = result.single()
            self.assertIsNotNone(record)
            self.assertEqual(record["interference_type"], "DESTRUCTIVE")
            self.assertLess(record["strength"], 0)


if __name__ == "__main__":
    unittest.main()
