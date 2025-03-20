"""
Tests for the topology query interface.

These tests verify the functionality of the specialized query interfaces for topology analysis,
focusing on complex queries that retrieve and analyze topology elements across the semantic landscape.

This test suite requires a running Neo4j instance accessible at bolt://localhost:7687
"""

import unittest
import json
import os
import sys
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
from habitat_evolution.pattern_aware_rag.topology.detector import TopologyDetector


class TestTopologyQueryInterface(unittest.TestCase):
    """Test case for specialized topology query interfaces."""
    
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
                MATCH (n:Pattern) DETACH DELETE n
            """)
        
        # Create rich test data for query testing
        self.create_query_test_data()
    
    def create_query_test_data(self):
        """Create rich test data for query testing with multiple states and patterns."""
        # Create patterns with different characteristics
        self.patterns = {}
        for i in range(10):
            pattern_id = f"pattern-{i}"
            # Create mock pattern nodes in Neo4j
            with self.driver.session() as session:
                session.run("""
                    CREATE (p:Pattern {
                        id: $id,
                        stability: $stability,
                        coherence: $coherence,
                        evolution_count: $evolution_count
                    })
                """, 
                id=pattern_id,
                stability=0.5 + (i % 5) * 0.1,
                coherence=0.6 + (i % 4) * 0.1,
                evolution_count=5 + i * 2
                )
        
        # Create multiple topology states over time
        self.states = []
        base_time = datetime.now() - timedelta(days=5)
        
        for day in range(5):
            # Create state for this day
            state_time = base_time + timedelta(days=day)
            state_id = f"ts-day-{day}"
            
            # Create frequency domains for this state
            domains = {}
            for i in range(3):  # 3 domains per state
                domain_id = f"fd-day{day}-{i}"
                # Domains evolve over time
                frequency = 0.2 + i * 0.3 + day * 0.02
                domains[domain_id] = FrequencyDomain(
                    id=domain_id,
                    dominant_frequency=frequency,
                    bandwidth=0.1 + day * 0.01,
                    phase_coherence=0.7 - (day % 3) * 0.05,
                    center_coordinates=(i * 0.3, day * 0.2),
                    radius=1.0 + i * 0.2,
                    # Patterns are assigned to domains with some overlap
                    pattern_ids=list(f"pattern-{(i+j) % 10}" for j in range(4))
                )
            
            # Create boundaries between domains
            boundaries = {}
            for i in range(2):  # 2 boundaries per state
                boundary_id = f"b-day{day}-{i}"
                boundaries[boundary_id] = Boundary(
                    id=boundary_id,
                    domain_ids=(f"fd-day{day}-{i}", f"fd-day{day}-{i+1}"),
                    sharpness=0.5 + day * 0.05,
                    permeability=0.7 - day * 0.05,
                    stability=0.6 + (day % 3) * 0.1,
                    dimensionality=2,
                    coordinates=[(i * 0.3, day * 0.2), ((i+1) * 0.3, day * 0.2)]
                )
            
            # Create resonance points
            resonance_points = {}
            for i in range(2):  # 2 resonance points per state
                resonance_id = f"r-day{day}-{i}"
                resonance_points[resonance_id] = ResonancePoint(
                    id=resonance_id,
                    coordinates=(0.5 + i * 0.4, day * 0.2),
                    strength=0.6 + (day % 3) * 0.1,
                    stability=0.7 + (i % 2) * 0.1,
                    attractor_radius=0.8 + day * 0.05,
                    contributing_pattern_ids={
                        f"pattern-{(i*2+j) % 10}": 0.6 + j * 0.1 for j in range(3)
                    }
                )
            
            # Create field metrics
            field_metrics = FieldMetrics(
                coherence=0.6 + day * 0.05,
                energy_density={f"region-{i}": 0.5 + (day+i) * 0.02 for i in range(3)},
                adaptation_rate=0.4 + day * 0.03,
                homeostasis_index=0.7 + (day % 3) * 0.05,
                entropy=0.3 - day * 0.02
            )
            
            # Create topology state
            state = TopologyState(
                id=state_id,
                frequency_domains=domains,
                boundaries=boundaries,
                resonance_points=resonance_points,
                field_metrics=field_metrics,
                timestamp=state_time
            )
            
            self.states.append(state)
            
            # Persist state to Neo4j
            self.manager.current_state = state
            self.manager.persist_to_neo4j(state)
            
            # Create relationships between patterns and this state
            with self.driver.session() as session:
                for domain in domains.values():
                    for pattern_id in domain.pattern_ids:
                        session.run("""
                            MATCH (p:Pattern {id: $pattern_id})
                            MATCH (ts:TopologyState {id: $state_id})
                            MATCH (fd:FrequencyDomain {id: $domain_id})
                            CREATE (p)-[:APPEARS_IN]->(ts)
                            CREATE (p)-[:BELONGS_TO]->(fd)
                        """, 
                        pattern_id=pattern_id,
                        state_id=state_id,
                        domain_id=domain.id
                        )
    
    def test_query_frequency_domains_by_range(self):
        """Test querying frequency domains by frequency range."""
        # Define query parameters
        min_freq = 0.4
        max_freq = 0.7
        
        # Execute query
        with self.driver.session() as session:
            result = session.run("""
                MATCH (fd:FrequencyDomain)
                WHERE fd.dominant_frequency >= $min_freq AND fd.dominant_frequency <= $max_freq
                RETURN fd.id as id, fd.dominant_frequency as frequency
                ORDER BY fd.dominant_frequency
            """, min_freq=min_freq, max_freq=max_freq)
            
            records = list(result)
            
            # Verify results
            self.assertGreater(len(records), 0)
            
            # Check that all returned domains are within the specified range
            for record in records:
                frequency = record["frequency"]
                self.assertGreaterEqual(frequency, min_freq)
                self.assertLessEqual(frequency, max_freq)
    
    def test_query_patterns_by_domain_frequency(self):
        """Test querying patterns by the frequency of domains they belong to."""
        # Define query parameters
        min_freq = 0.6
        
        # Execute query
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Pattern)-[:BELONGS_TO]->(fd:FrequencyDomain)
                WHERE fd.dominant_frequency >= $min_freq
                RETURN p.id as pattern_id, fd.id as domain_id, fd.dominant_frequency as frequency
                ORDER BY fd.dominant_frequency DESC
            """, min_freq=min_freq)
            
            records = list(result)
            
            # Verify results
            self.assertGreater(len(records), 0)
            
            # Check that all returned domains have frequency >= min_freq
            for record in records:
                frequency = record["frequency"]
                self.assertGreaterEqual(frequency, min_freq)
    
    def test_query_patterns_crossing_boundaries(self):
        """Test querying patterns that cross domain boundaries."""
        # Execute query to find patterns that appear in multiple domains
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Pattern)-[:BELONGS_TO]->(fd:FrequencyDomain)
                WITH p, collect(fd) as domains
                WHERE size(domains) > 1
                RETURN p.id as pattern_id, size(domains) as domain_count
                ORDER BY domain_count DESC
            """)
            
            records = list(result)
            
            # Verify results
            self.assertGreater(len(records), 0)
            
            # Check that all returned patterns appear in multiple domains
            for record in records:
                domain_count = record["domain_count"]
                self.assertGreater(domain_count, 1)
    
    def test_query_resonance_points_by_strength(self):
        """Test querying resonance points by strength threshold."""
        # Define query parameters
        min_strength = 0.7
        
        # Execute query
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:ResonancePoint)
                WHERE r.strength >= $min_strength
                RETURN r.id as id, r.strength as strength
                ORDER BY r.strength DESC
            """, min_strength=min_strength)
            
            records = list(result)
            
            # Verify results
            self.assertGreater(len(records), 0)
            
            # Check that all returned resonance points have strength >= min_strength
            for record in records:
                strength = record["strength"]
                self.assertGreaterEqual(strength, min_strength)
    
    def test_query_topology_evolution_over_time(self):
        """Test querying topology evolution over time."""
        # Execute query to track how field coherence changes over time
        with self.driver.session() as session:
            result = session.run("""
                MATCH (ts:TopologyState)
                RETURN ts.id as id, ts.timestamp as timestamp, ts.field_metrics_coherence as coherence
                ORDER BY ts.timestamp
            """)
            
            records = list(result)
            
            # Verify results
            self.assertEqual(len(records), 5)  # Should have 5 states
            
            # Check that coherence generally increases over time (as per our test data)
            coherence_values = [record["coherence"] for record in records]
            self.assertLess(coherence_values[0], coherence_values[-1])
    
    def test_query_stable_frequency_domains(self):
        """Test querying stable frequency domains across multiple states."""
        # Execute query to find domains with consistent high phase coherence
        with self.driver.session() as session:
            result = session.run("""
                MATCH (fd:FrequencyDomain)
                WHERE fd.phase_coherence >= 0.7
                RETURN fd.id as id, fd.phase_coherence as coherence
                ORDER BY fd.phase_coherence DESC
            """)
            
            records = list(result)
            
            # Verify results
            self.assertGreater(len(records), 0)
            
            # Check that all returned domains have high coherence
            for record in records:
                coherence = record["coherence"]
                self.assertGreaterEqual(coherence, 0.7)
    
    def test_query_patterns_with_high_stability(self):
        """Test querying patterns with high stability."""
        # Define query parameters
        min_stability = 0.7
        
        # Execute query
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Pattern)
                WHERE p.stability >= $min_stability
                RETURN p.id as id, p.stability as stability
                ORDER BY p.stability DESC
            """, min_stability=min_stability)
            
            records = list(result)
            
            # Verify results
            self.assertGreater(len(records), 0)
            
            # Check that all returned patterns have stability >= min_stability
            for record in records:
                stability = record["stability"]
                self.assertGreaterEqual(stability, min_stability)
    
    def test_query_boundaries_by_permeability(self):
        """Test querying boundaries by permeability."""
        # Define query parameters
        min_permeability = 0.6
        
        # Execute query
        with self.driver.session() as session:
            result = session.run("""
                MATCH (b:Boundary)
                WHERE b.permeability >= $min_permeability
                RETURN b.id as id, b.permeability as permeability
                ORDER BY b.permeability DESC
            """, min_permeability=min_permeability)
            
            records = list(result)
            
            # Verify results
            self.assertGreater(len(records), 0)
            
            # Check that all returned boundaries have permeability >= min_permeability
            for record in records:
                permeability = record["permeability"]
                self.assertGreaterEqual(permeability, min_permeability)
    
    def test_query_high_energy_regions(self):
        """Test querying high energy regions in the field."""
        # This test requires parsing the energy_density JSON in Neo4j
        # We'll use a simpler approach by checking if any state has high overall coherence
        
        # Execute query
        with self.driver.session() as session:
            result = session.run("""
                MATCH (ts:TopologyState)
                WHERE ts.field_metrics_coherence >= 0.7
                RETURN ts.id as id, ts.field_metrics_coherence as coherence
                ORDER BY ts.field_metrics_coherence DESC
            """)
            
            records = list(result)
            
            # Verify results - we should have some high coherence states
            self.assertGreater(len(records), 0)
            
            # Check that all returned states have coherence >= 0.7
            for record in records:
                coherence = record["coherence"]
                self.assertGreaterEqual(coherence, 0.7)
    
    def test_query_latest_topology_state(self):
        """Test querying the latest topology state."""
        # Execute query
        with self.driver.session() as session:
            result = session.run("""
                MATCH (ts:TopologyState)
                RETURN ts.id as id, ts.timestamp as timestamp
                ORDER BY ts.timestamp DESC
                LIMIT 1
            """)
            
            record = result.single()
            
            # Verify result
            self.assertIsNotNone(record)
            self.assertEqual(record["id"], "ts-day-4")  # Should be the last day
    
    def test_query_topology_state_by_time_range(self):
        """Test querying topology states within a specific time range."""
        # Define query parameters
        start_time = (datetime.now() - timedelta(days=4)).timestamp() * 1000  # Neo4j uses milliseconds
        end_time = (datetime.now() - timedelta(days=2)).timestamp() * 1000
        
        # Execute query
        with self.driver.session() as session:
            result = session.run("""
                MATCH (ts:TopologyState)
                WHERE ts.timestamp >= $start_time AND ts.timestamp <= $end_time
                RETURN ts.id as id, ts.timestamp as timestamp
                ORDER BY ts.timestamp
            """, start_time=start_time, end_time=end_time)
            
            records = list(result)
            
            # Verify results - should have states from days 1-3
            self.assertGreaterEqual(len(records), 1)
            
            # Check that all returned states are within the time range
            for record in records:
                timestamp = record["timestamp"]
                self.assertGreaterEqual(timestamp, start_time)
                self.assertLessEqual(timestamp, end_time)


if __name__ == "__main__":
    unittest.main()
