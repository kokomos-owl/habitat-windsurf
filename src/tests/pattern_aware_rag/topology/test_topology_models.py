"""
Tests for the topology models.

These tests verify the functionality of the topology data structures, including
serialization, deserialization, and diff calculation.
"""

import unittest
import json
from datetime import datetime, timedelta

from habitat_evolution.pattern_aware_rag.topology.models import (
    FrequencyDomain, Boundary, ResonancePoint, FieldMetrics, TopologyState
)


class TestTopologyModels(unittest.TestCase):
    """Test case for topology models."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample frequency domains
        self.domain1 = FrequencyDomain(
            id="fd-test-1",
            dominant_frequency=0.25,
            bandwidth=0.15,
            phase_coherence=0.78,
            center_coordinates=(0.5, 0.3),
            radius=2.0,
            pattern_ids={"pattern-1", "pattern-2"}
        )
        
        self.domain2 = FrequencyDomain(
            id="fd-test-2",
            dominant_frequency=0.75,
            bandwidth=0.10,
            phase_coherence=0.85,
            center_coordinates=(0.8, 0.7),
            radius=1.5,
            pattern_ids={"pattern-3", "pattern-4"}
        )
        
        # Create sample boundaries
        self.boundary = Boundary(
            id="b-test-1",
            domain_ids=("fd-test-1", "fd-test-2"),
            sharpness=0.85,
            permeability=0.32,
            stability=0.91,
            dimensionality=2,
            coordinates=[(0.6, 0.4), (0.7, 0.5)]
        )
        
        # Create sample resonance points
        self.resonance = ResonancePoint(
            id="r-test-1",
            coordinates=(0.4, 0.6),
            strength=0.94,
            stability=0.87,
            attractor_radius=1.2,
            contributing_pattern_ids={"pattern-1": 0.8, "pattern-3": 0.6}
        )
        
        # Create sample field metrics
        self.metrics = FieldMetrics(
            coherence=0.76,
            energy_density={"region-1": 0.8, "region-2": 0.6},
            adaptation_rate=0.45,
            homeostasis_index=0.82,
            entropy=0.35
        )
        
        # Create sample topology state
        self.state = TopologyState(
            id="ts-test-1",
            frequency_domains={
                "fd-test-1": self.domain1,
                "fd-test-2": self.domain2
            },
            boundaries={
                "b-test-1": self.boundary
            },
            resonance_points={
                "r-test-1": self.resonance
            },
            field_metrics=self.metrics,
            timestamp=datetime.now()
        )
    
    def test_frequency_domain_creation(self):
        """Test creation of frequency domains."""
        domain = self.domain1
        self.assertEqual(domain.id, "fd-test-1")
        self.assertEqual(domain.dominant_frequency, 0.25)
        self.assertEqual(domain.bandwidth, 0.15)
        self.assertEqual(domain.phase_coherence, 0.78)
        self.assertEqual(domain.center_coordinates, (0.5, 0.3))
        self.assertEqual(domain.radius, 2.0)
        self.assertEqual(domain.pattern_ids, {"pattern-1", "pattern-2"})
    
    def test_boundary_creation(self):
        """Test creation of boundaries."""
        boundary = self.boundary
        self.assertEqual(boundary.id, "b-test-1")
        self.assertEqual(boundary.domain_ids, ("fd-test-1", "fd-test-2"))
        self.assertEqual(boundary.sharpness, 0.85)
        self.assertEqual(boundary.permeability, 0.32)
        self.assertEqual(boundary.stability, 0.91)
        self.assertEqual(boundary.dimensionality, 2)
        self.assertEqual(boundary.coordinates, [(0.6, 0.4), (0.7, 0.5)])
    
    def test_resonance_point_creation(self):
        """Test creation of resonance points."""
        resonance = self.resonance
        self.assertEqual(resonance.id, "r-test-1")
        self.assertEqual(resonance.coordinates, (0.4, 0.6))
        self.assertEqual(resonance.strength, 0.94)
        self.assertEqual(resonance.stability, 0.87)
        self.assertEqual(resonance.attractor_radius, 1.2)
        self.assertEqual(resonance.contributing_pattern_ids, {"pattern-1": 0.8, "pattern-3": 0.6})
    
    def test_field_metrics_creation(self):
        """Test creation of field metrics."""
        metrics = self.metrics
        self.assertEqual(metrics.coherence, 0.76)
        self.assertEqual(metrics.energy_density, {"region-1": 0.8, "region-2": 0.6})
        self.assertEqual(metrics.adaptation_rate, 0.45)
        self.assertEqual(metrics.homeostasis_index, 0.82)
        self.assertEqual(metrics.entropy, 0.35)
    
    def test_topology_state_creation(self):
        """Test creation of topology state."""
        state = self.state
        self.assertEqual(state.id, "ts-test-1")
        self.assertEqual(len(state.frequency_domains), 2)
        self.assertEqual(len(state.boundaries), 1)
        self.assertEqual(len(state.resonance_points), 1)
        self.assertEqual(state.field_metrics.coherence, 0.76)
    
    def test_topology_state_serialization(self):
        """Test serialization of topology state to JSON."""
        json_str = self.state.to_json()
        self.assertIsInstance(json_str, str)
        
        # Verify JSON can be parsed
        data = json.loads(json_str)
        self.assertEqual(data["id"], "ts-test-1")
        self.assertEqual(len(data["frequency_domains"]), 2)
        self.assertEqual(len(data["boundaries"]), 1)
        self.assertEqual(len(data["resonance_points"]), 1)
        self.assertEqual(data["field_metrics"]["coherence"], 0.76)
    
    def test_topology_state_deserialization(self):
        """Test deserialization of topology state from JSON."""
        json_str = self.state.to_json()
        loaded_state = TopologyState.from_json(json_str)
        
        self.assertEqual(loaded_state.id, self.state.id)
        self.assertEqual(len(loaded_state.frequency_domains), len(self.state.frequency_domains))
        self.assertEqual(len(loaded_state.boundaries), len(self.state.boundaries))
        self.assertEqual(len(loaded_state.resonance_points), len(self.state.resonance_points))
        
        # Check specific values
        domain_id = "fd-test-1"
        self.assertEqual(
            loaded_state.frequency_domains[domain_id].dominant_frequency,
            self.state.frequency_domains[domain_id].dominant_frequency
        )
        
        boundary_id = "b-test-1"
        self.assertEqual(
            loaded_state.boundaries[boundary_id].sharpness,
            self.state.boundaries[boundary_id].sharpness
        )
        
        resonance_id = "r-test-1"
        self.assertEqual(
            loaded_state.resonance_points[resonance_id].strength,
            self.state.resonance_points[resonance_id].strength
        )
        
        self.assertEqual(
            loaded_state.field_metrics.coherence,
            self.state.field_metrics.coherence
        )
    
    def test_topology_state_diff(self):
        """Test calculation of difference between topology states."""
        # Create a modified state
        modified_state = TopologyState(
            id="ts-test-2",
            frequency_domains={
                "fd-test-1": self.domain1,  # Same
                "fd-test-3": FrequencyDomain(  # New
                    id="fd-test-3",
                    dominant_frequency=0.5,
                    bandwidth=0.2,
                    phase_coherence=0.6
                )
                # fd-test-2 removed
            },
            boundaries={
                "b-test-1": Boundary(  # Modified
                    id="b-test-1",
                    sharpness=0.9,  # Changed
                    permeability=0.32,
                    stability=0.91,
                    dimensionality=2
                )
            },
            resonance_points={
                "r-test-1": self.resonance  # Same
            },
            field_metrics=FieldMetrics(  # Modified
                coherence=0.8,  # Changed
                energy_density={"region-1": 0.8, "region-2": 0.6},
                adaptation_rate=0.45,
                homeostasis_index=0.82,
                entropy=0.35
            ),
            timestamp=datetime.now()
        )
        
        # Calculate diff
        diff = modified_state.diff(self.state)
        
        # Verify diff contents
        self.assertIn("added_domains", diff)
        self.assertIn("removed_domains", diff)
        self.assertIn("modified_domains", diff)
        self.assertIn("modified_boundaries", diff)
        self.assertIn("field_metrics_changes", diff)
        
        # Check specific changes
        self.assertIn("fd-test-3", diff["added_domains"])
        self.assertIn("fd-test-2", diff["removed_domains"])
        self.assertIn("b-test-1", diff["modified_boundaries"])
        self.assertIn("coherence", diff["field_metrics_changes"])


if __name__ == "__main__":
    unittest.main()
