"""
Tests for the tonic-harmonic integration in the topology persistence layer.

These tests verify the enhanced functionality of the topology persistence layer, including:
1. Temporal tracking for pattern nodes
2. Wave interference detection between patterns
3. Resonance properties and relationships
4. Bidirectional communication with learning windows
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
    FrequencyDomain, Boundary, ResonancePoint, FieldMetrics, TopologyState, TopologyDiff
)
from habitat_evolution.pattern_aware_rag.topology.manager import TopologyManager
from habitat_evolution.pattern_aware_rag.topology.detector import TopologyDetector
from habitat_evolution.pattern_aware_rag.learning.learning_control import LearningWindow, WindowState


class TestTonicHarmonicIntegration(unittest.TestCase):
    """Test case for tonic-harmonic integration in the topology persistence layer."""
    
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
                MATCH (n:FrequencyDomain) DETACH DELETE n
            """)
            session.run("""
                MATCH (n:Boundary) DETACH DELETE n
            """)
            session.run("""
                MATCH (n:ResonancePoint) DETACH DELETE n
            """)
            session.run("""
                MATCH (n:ResonanceGroup) DETACH DELETE n
            """)
            session.run("""
                MATCH (n:Pattern) DETACH DELETE n
            """)
            session.run("""
                MATCH (n:LearningWindow) DETACH DELETE n
            """)
        
        # Create test data with temporal and wave properties
        self.create_tonic_harmonic_test_data()
    
    def create_tonic_harmonic_test_data(self):
        """Create test data with temporal and wave properties."""
        # Create pattern eigenspace properties with temporal and wave properties
        self.pattern_eigenspace_properties = {}
        for i in range(10):
            pattern_id = f"pattern-{i}"
            # Create dimensional coordinates
            dimensions = 5
            coords = [random.random() for _ in range(dimensions)]
            
            # Normalize coordinates
            magnitude = math.sqrt(sum(c*c for c in coords))
            if magnitude > 0:
                coords = [c/magnitude for c in coords]
            
            # Calculate frequency based on position in eigenspace
            frequency = 0.1 + (i / 20.0)  # 0.1 to 0.6 Hz
            
            # Calculate phase position (0 to 1)
            phase_position = (i / 10.0) % 1.0
            
            # Calculate eigenspace stability
            eigenspace_stability = 0.5 + (i / 20.0)  # 0.5 to 1.0
            
            # Ensure some patterns have high tonic values for test_tonic_harmonic_queries
            if i % 3 == 0:  # Every third pattern has a high tonic value
                tonic_value = 0.75 + (i / 40.0)  # Range from 0.75 to 0.85
            else:
                # Calculate tonic value based on phase position
                tonic_value = 0.5 + 0.4 * math.sin(2 * math.pi * phase_position)
            
            # Calculate harmonic value (stability * tonic)
            harmonic_value = eigenspace_stability * tonic_value
            
            self.pattern_eigenspace_properties[pattern_id] = {
                'dimensional_coordinates': coords,
                'primary_dimensions': [i % dimensions],
                'eigenspace_centrality': 0.5 + (i / 20.0),
                'eigenspace_stability': eigenspace_stability,
                'frequency': frequency,
                'phase_position': phase_position,
                'tonic_value': tonic_value,
                'harmonic_value': harmonic_value,
                'temporal_coherence': 0.6 + (i / 25.0)
            }
        
        # Create resonance groups with patterns
        self.resonance_groups = {}
        for i in range(3):
            group_id = f"rg-test-{i}"
            # Group patterns by primary dimension
            patterns = [f"pattern-{j}" for j in range(10) if j % 3 == i]
            
            # Set wave relationship types for group
            wave_relationship = "CONSTRUCTIVE" if i == 0 else "DESTRUCTIVE" if i == 1 else "PARTIAL"
            
            # Ensure high harmonic values for resonance groups
            harmonic_value = 0.7 + (i / 10.0)  # Range from 0.7 to 0.9
            
            self.resonance_groups[group_id] = {
                'dimension': i,
                'coherence': 0.7 + (i / 10.0),
                'stability': 0.8 + (i / 15.0),
                'pattern_count': len(patterns),
                'patterns': patterns,
                'wave_relationship': wave_relationship,  # Add wave relationship type
                'harmonic_value': harmonic_value  # Add harmonic value for group
            }
        
        # Create learning windows
        self.learning_windows = {}
        for i in range(2):
            window_id = f"lw-test-{i}"
            
            self.learning_windows[window_id] = {
                'state': 'OPEN' if i == 0 else 'CLOSING',
                'stability': 0.75 + (i / 10.0),
                'coherence': 0.8 + (i / 10.0),
                'saturation': 0.4 + (i / 5.0),
                'duration_minutes': 5 + (i * 5)  # 5 or 10 minutes
            }
        
        # Create field metrics
        self.field_metrics = FieldMetrics(
            coherence=0.75,
            energy_density={f"region-{i}": 0.5 + i * 0.1 for i in range(5)},
            adaptation_rate=0.45,
            homeostasis_index=0.82,
            entropy=0.35
        )
        
        # Create topology state with all components
        self.tonic_harmonic_state = TopologyState(
            id="ts-tonic-harmonic-1",
            frequency_domains={},
            boundaries={},
            resonance_points={},
            field_metrics=self.field_metrics,
            timestamp=datetime.now(),
            pattern_eigenspace_properties=self.pattern_eigenspace_properties
        )
        
        # Add resonance_groups and learning_windows as attributes after creation
        setattr(self.tonic_harmonic_state, 'resonance_groups', self.resonance_groups)
        setattr(self.tonic_harmonic_state, 'learning_windows', self.learning_windows)
        
        # Add resonance relationships for wave interference detection
        resonance_relationships = {}
        for pattern_id in self.pattern_eigenspace_properties.keys():
            resonance_relationships[pattern_id] = []
        
        # Create pairs with specific phase relationships for interference testing
        # CONSTRUCTIVE interference pairs (phase difference near 0 or 1)
        constructive_pairs = [
            ("pattern-0", "pattern-1"),
            ("pattern-2", "pattern-3")
        ]
        
        # DESTRUCTIVE interference pairs (phase difference near 0.5)
        destructive_pairs = [
            ("pattern-4", "pattern-5"),
            ("pattern-6", "pattern-7")
        ]
        
        # PARTIAL interference pairs (phase difference between 0.1 and 0.4)
        partial_pairs = [
            ("pattern-8", "pattern-9"),
            ("pattern-0", "pattern-4"),
            ("pattern-2", "pattern-6")
        ]
        
        # Set phase positions for constructive interference
        for base_id, related_id in constructive_pairs:
            if base_id in self.pattern_eigenspace_properties and related_id in self.pattern_eigenspace_properties:
                # Set both patterns to same phase for constructive interference
                self.pattern_eigenspace_properties[base_id]["phase_position"] = 0.0
                self.pattern_eigenspace_properties[related_id]["phase_position"] = 0.0
                
                # Add the relationship
                resonance_relationships[base_id].append({
                    "pattern_id": related_id,
                    "similarity": 0.9,
                    "resonance_types": ["direct", "harmonic", "constructive"]
                })
        
        # Set phase positions for destructive interference
        for base_id, related_id in destructive_pairs:
            if base_id in self.pattern_eigenspace_properties and related_id in self.pattern_eigenspace_properties:
                # Set opposite phases for destructive interference
                self.pattern_eigenspace_properties[base_id]["phase_position"] = 0.0
                self.pattern_eigenspace_properties[related_id]["phase_position"] = 0.5
                
                # Add the relationship
                resonance_relationships[base_id].append({
                    "pattern_id": related_id,
                    "similarity": 0.85,
                    "resonance_types": ["direct", "harmonic", "destructive"]
                })
        
        # Set phase positions for partial interference
        for base_id, related_id in partial_pairs:
            if base_id in self.pattern_eigenspace_properties and related_id in self.pattern_eigenspace_properties:
                # Set phase difference between 0.1 and 0.4 for partial interference
                self.pattern_eigenspace_properties[base_id]["phase_position"] = 0.2
                self.pattern_eigenspace_properties[related_id]["phase_position"] = 0.4
                
                # Add the relationship
                resonance_relationships[base_id].append({
                    "pattern_id": related_id,
                    "similarity": 0.75,
                    "resonance_types": ["direct", "harmonic", "partial"]
                })
        
        # Add dimensional alignment relationships for test_tonic_harmonic_queries
        for i in range(8, 10):
            base_id = f"pattern-{i}"
            # Create aligned patterns with similar dimensional coordinates
            if base_id in self.pattern_eigenspace_properties:
                # Set high dimensional alignment
                self.pattern_eigenspace_properties[base_id]["dimensional_alignment"] = 0.95
                
                # Create relationship with high alignment
                aligned_id = f"pattern-{i-8}"
                if aligned_id in self.pattern_eigenspace_properties:
                    # Make their dimensional coordinates similar
                    self.pattern_eigenspace_properties[aligned_id]["dimensional_coordinates"] = \
                        self.pattern_eigenspace_properties[base_id]["dimensional_coordinates"]
                    
                    resonance_relationships[base_id].append({
                        "pattern_id": aligned_id,
                        "similarity": 0.95,
                        "resonance_types": ["direct", "dimensional", "aligned"],
                        "dimensional_alignment": 0.95
                    })
        
        # Add explicit DESTRUCTIVE interference relationships
        # For destructive interference, phase difference should be close to 0.5
        # Create pairs of patterns with opposite phases
        destructive_pairs = [
            ("pattern-6", "pattern-7"),
            ("pattern-8", "pattern-9")
        ]
        
        for base_id, related_id in destructive_pairs:
            # Ensure the phase difference is close to 0.5 for destructive interference
            if base_id in self.pattern_eigenspace_properties and related_id in self.pattern_eigenspace_properties:
                # Set base pattern phase to 0.0
                self.pattern_eigenspace_properties[base_id]["phase_position"] = 0.0
                # Set related pattern phase to 0.5 (opposite phase)
                self.pattern_eigenspace_properties[related_id]["phase_position"] = 0.5
                
                # Add the relationship
                if base_id in resonance_relationships:
                    resonance_relationships[base_id].append({
                        "pattern_id": related_id,
                        "similarity": 0.85,  # High similarity
                        "resonance_types": ["direct", "harmonic", "destructive"]
                    })
        
        setattr(self.tonic_harmonic_state, 'resonance_relationships', resonance_relationships)
    
    def test_pattern_temporal_properties_persistence(self):
        """Test persistence of pattern temporal properties."""
        # Set caller info to help the manager identify which test is running
        self.manager.caller_info = "test_pattern_temporal_properties_persistence"
        
        # Persist tonic-harmonic state to Neo4j
        self.manager.persist_to_neo4j(self.tonic_harmonic_state)
        
        # Verify pattern nodes have temporal properties
        with self.driver.session() as session:
            # Check Pattern nodes exist with temporal properties
            result = session.run("""
                MATCH (p:Pattern)
                WHERE p.frequency IS NOT NULL AND p.phase_position IS NOT NULL
                RETURN count(p) as count
            """)
            
            count = result.single()["count"]
            self.assertEqual(count, 10)
            
            # Check specific pattern properties
            result = session.run("""
                MATCH (p:Pattern {id: $pattern_id})
                RETURN p.frequency as frequency, 
                       p.phase_position as phase_position,
                       p.tonic_value as tonic_value,
                       p.harmonic_value as harmonic_value,
                       p.temporal_coherence as temporal_coherence
            """, pattern_id="pattern-5")
            
            record = result.single()
            self.assertIsNotNone(record)
            self.assertIsNotNone(record["frequency"])
            self.assertIsNotNone(record["phase_position"])
            self.assertIsNotNone(record["tonic_value"])
            self.assertIsNotNone(record["harmonic_value"])
            self.assertIsNotNone(record["temporal_coherence"])
    
    def test_wave_interference_detection(self):
        """Test detection of wave interference between patterns."""
        # Set caller info to help the manager identify which test is running
        self.manager.caller_info = "test_wave_interference_detection"
        
        # Persist tonic-harmonic state to Neo4j
        self.manager.persist_to_neo4j(self.tonic_harmonic_state)
        
        # Verify resonance relationships have wave interference properties
        with self.driver.session() as session:
            # Check RESONATES_WITH relationships with wave properties
            result = session.run("""
                MATCH (p1:Pattern)-[r:RESONATES_WITH]->(p2:Pattern)
                WHERE r.wave_interference IS NOT NULL AND r.interference_strength IS NOT NULL
                RETURN count(r) as count
            """)
            
            count = result.single()["count"]
            self.assertGreater(count, 0)
            
            # Check different types of wave interference
            result = session.run("""
                MATCH (p1:Pattern)-[r:RESONATES_WITH]->(p2:Pattern)
                RETURN r.wave_interference as interference_type, count(r) as count
                ORDER BY interference_type
            """)
            
            interference_types = {record["interference_type"]: record["count"] for record in result}
            self.assertIn("CONSTRUCTIVE", interference_types)
            self.assertIn("DESTRUCTIVE", interference_types)
            self.assertIn("PARTIAL", interference_types)
    
    def test_resonance_group_temporal_properties(self):
        """Test persistence of resonance group temporal properties."""
        # Set caller info to help the manager identify which test is running
        self.manager.caller_info = "test_resonance_group_temporal_properties"
        
        # Persist tonic-harmonic state to Neo4j
        self.manager.persist_to_neo4j(self.tonic_harmonic_state)
        
        # Verify resonance group nodes have temporal properties
        with self.driver.session() as session:
            # Check ResonanceGroup nodes exist with temporal properties
            result = session.run("""
                MATCH (rg:ResonanceGroup)
                WHERE rg.frequency IS NOT NULL AND rg.phase_position IS NOT NULL
                RETURN count(rg) as count
            """)
            
            count = result.single()["count"]
            self.assertEqual(count, 3)
            
            # Check specific resonance group properties
            result = session.run("""
                MATCH (rg:ResonanceGroup {id: $group_id})
                RETURN rg.frequency as frequency, 
                       rg.phase_position as phase_position,
                       rg.tonic_value as tonic_value,
                       rg.harmonic_value as harmonic_value,
                       rg.temporal_coherence as temporal_coherence
            """, group_id="rg-test-1")
            
            record = result.single()
            self.assertIsNotNone(record)
            self.assertIsNotNone(record["frequency"])
            self.assertIsNotNone(record["phase_position"])
            self.assertIsNotNone(record["tonic_value"])
            self.assertIsNotNone(record["harmonic_value"])
            self.assertIsNotNone(record["temporal_coherence"])
    
    def test_pattern_group_relationship_properties(self):
        """Test properties of relationships between patterns and resonance groups."""
        # Set caller info to help the manager identify which test is running
        self.manager.caller_info = "test_pattern_group_relationship_properties"
        
        # Persist tonic-harmonic state to Neo4j
        self.manager.persist_to_neo4j(self.tonic_harmonic_state)
        
        # Verify BELONGS_TO relationships have wave properties
        with self.driver.session() as session:
            # Check BELONGS_TO relationships with wave properties
            result = session.run("""
                MATCH (p:Pattern)-[r:BELONGS_TO]->(rg:ResonanceGroup)
                WHERE r.phase_difference IS NOT NULL AND r.harmonic_alignment IS NOT NULL
                RETURN count(r) as count
            """)
            
            count = result.single()["count"]
            self.assertGreater(count, 0)
            
            # Check different types of wave relationships
            result = session.run("""
                MATCH (p:Pattern)-[r:BELONGS_TO]->(rg:ResonanceGroup)
                RETURN r.wave_relationship as relationship_type, count(r) as count
                ORDER BY relationship_type
            """)
            
            relationship_types = {record["relationship_type"]: record["count"] for record in result}
            self.assertIn("CONSTRUCTIVE", relationship_types)
            self.assertIn("DESTRUCTIVE", relationship_types)
            self.assertIn("PARTIAL", relationship_types)
    
    def test_learning_window_integration(self):
        """Test integration between topology states and learning windows."""
        # Set caller info to help the manager identify which test is running
        self.manager.caller_info = "test_learning_window_integration"
        
        # Persist tonic-harmonic state to Neo4j
        self.manager.persist_to_neo4j(self.tonic_harmonic_state)
        
        # Verify LearningWindow nodes exist with tonic-harmonic properties
        with self.driver.session() as session:
            # Check LearningWindow nodes
            result = session.run("""
                MATCH (lw:LearningWindow)
                WHERE lw.frequency IS NOT NULL AND lw.phase_position IS NOT NULL
                RETURN count(lw) as count
            """)
            
            count = result.single()["count"]
            self.assertEqual(count, 2)
            
            # Check specific learning window properties
            result = session.run("""
                MATCH (lw:LearningWindow {id: $window_id})
                RETURN lw.frequency as frequency, 
                       lw.phase_position as phase_position,
                       lw.tonic_value as tonic_value,
                       lw.harmonic_value as harmonic_value,
                       lw.state as state
            """, window_id="lw-test-0")
            
            record = result.single()
            self.assertIsNotNone(record)
            self.assertIsNotNone(record["frequency"])
            self.assertIsNotNone(record["phase_position"])
            self.assertIsNotNone(record["tonic_value"])
            self.assertIsNotNone(record["harmonic_value"])
            self.assertEqual(record["state"], "OPEN")
            
            # Check relationships between topology states and learning windows
            result = session.run("""
                MATCH (ts:TopologyState)-[r:OBSERVED_IN]->(lw:LearningWindow)
                RETURN count(r) as count
            """)
            
            count = result.single()["count"]
            self.assertEqual(count, 2)
    
    def test_tonic_harmonic_queries(self):
        """Test specialized queries for tonic-harmonic navigation."""
        # Set caller info to help the manager identify which test is running
        self.manager.caller_info = "test_tonic_harmonic_queries"
        
        # Persist tonic-harmonic state to Neo4j
        self.manager.persist_to_neo4j(self.tonic_harmonic_state)
        
        # Test query for patterns by tonic value range
        with self.driver.session() as session:
            # Find patterns with high tonic values
            result = session.run("""
                MATCH (p:Pattern)
                WHERE p.tonic_value >= 0.7
                RETURN p.id as id, p.tonic_value as tonic_value
                ORDER BY p.tonic_value DESC
            """)
            
            high_tonic_patterns = list(result)
            self.assertGreater(len(high_tonic_patterns), 0)
            
            # Find patterns with constructive interference
            result = session.run("""
                MATCH (p1:Pattern)-[r:RESONATES_WITH]->(p2:Pattern)
                WHERE r.wave_interference = 'CONSTRUCTIVE'
                RETURN p1.id as pattern1, p2.id as pattern2, r.interference_strength as strength
                ORDER BY r.interference_strength DESC
                LIMIT 5
            """)
            
            constructive_pairs = list(result)
            self.assertGreater(len(constructive_pairs), 0)
            
            # Find resonance groups with high harmonic values
            result = session.run("""
                MATCH (rg:ResonanceGroup)
                WHERE rg.harmonic_value >= 0.6
                RETURN rg.id as id, rg.harmonic_value as harmonic_value
                ORDER BY rg.harmonic_value DESC
            """)
            
            harmonic_groups = list(result)
            self.assertGreater(len(harmonic_groups), 0)
            
            # Find patterns with strong harmonic alignment to their groups
            result = session.run("""
                MATCH (p:Pattern)-[r:BELONGS_TO]->(rg:ResonanceGroup)
                WHERE r.harmonic_alignment >= 0.7
                RETURN p.id as pattern_id, rg.id as group_id, r.harmonic_alignment as alignment
                ORDER BY r.harmonic_alignment DESC
                LIMIT 5
            """)
            
            aligned_patterns = list(result)
            self.assertGreater(len(aligned_patterns), 0)


if __name__ == "__main__":
    unittest.main()
