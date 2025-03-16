"""
Tests for the FlowDynamicsAnalyzer class.

This test suite validates the functionality of the FlowDynamicsAnalyzer,
which tracks energy flow and anticipates emergent patterns in the field.
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from typing import Dict, List, Any
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import the classes we need to test
from habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer
from habitat_evolution.field.field_navigator import FieldNavigator
from habitat_evolution.field.semantic_boundary_detector import SemanticBoundaryDetector
from habitat_evolution.field.flow_dynamics_analyzer import FlowDynamicsAnalyzer


class TestFlowDynamicsAnalyzer(unittest.TestCase):
    """Test suite for the FlowDynamicsAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock field analyzer
        self.mock_analyzer = MagicMock(spec=TopologicalFieldAnalyzer)
        
        # Create mock field navigator
        self.mock_navigator = MagicMock(spec=FieldNavigator)
        
        # Create mock semantic boundary detector
        self.mock_boundary_detector = MagicMock(spec=SemanticBoundaryDetector)
        
        # Initialize analyzer with mocks
        self.flow_analyzer = FlowDynamicsAnalyzer(
            field_analyzer=self.mock_analyzer,
            field_navigator=self.mock_navigator,
            boundary_detector=self.mock_boundary_detector
        )
        
        # Sample data for tests
        self.sample_vectors = np.random.rand(10, 5)  # 10 patterns with 5 features each
        self.sample_metadata = [
            {"id": f"pattern_{i}", "text": f"Sample pattern {i}", "timestamp": f"2025-03-{15+i}T10:00:00Z"} 
            for i in range(10)
        ]
        
        # Mock field data
        self.mock_field_data = {
            "communities": [0, 1, 2],
            "pattern_communities": [0, 0, 1, 0, 1, 2, 1, 2, 0, 2],
            "eigenvalues": [1.0, 0.8, 0.6, 0.4, 0.2],
            "eigenvectors": np.random.rand(5, 5),
            "resonance_matrix": np.random.rand(10, 10),
            "transition_zones": {
                "transition_zones": [
                    {"pattern_idx": 3, "uncertainty": 0.8, "source_community": 0, "neighboring_communities": [1]},
                    {"pattern_idx": 6, "uncertainty": 0.7, "source_community": 1, "neighboring_communities": [2]}
                ]
            }
        }
        
        # Set up mock returns
        self.mock_analyzer.analyze_field.return_value = self.mock_field_data
        self.mock_navigator.set_field.return_value = self.mock_field_data
        
        # Mock boundary detector returns
        self.mock_boundary_detector.detect_transition_patterns.return_value = [
            {"pattern_idx": 3, "uncertainty": 0.8, "source_community": 0, "neighboring_communities": [1]},
            {"pattern_idx": 6, "uncertainty": 0.7, "source_community": 1, "neighboring_communities": [2]}
        ]

    def test_initialization(self):
        """Test that the analyzer initializes correctly."""
        self.assertEqual(self.flow_analyzer.field_analyzer, self.mock_analyzer)
        self.assertEqual(self.flow_analyzer.field_navigator, self.mock_navigator)
        self.assertEqual(self.flow_analyzer.boundary_detector, self.mock_boundary_detector)
        self.assertIsInstance(self.flow_analyzer.flow_history, list)
        
    def test_analyze_flow_dynamics(self):
        """Test the analyze_flow_dynamics method."""
        # Call the method
        flow_data = self.flow_analyzer.analyze_flow_dynamics(
            self.sample_vectors, self.sample_metadata
        )
        
        # Verify analyzer was called with correct data
        self.mock_analyzer.analyze_field.assert_called_once_with(self.sample_vectors, self.sample_metadata)
        
        # Check flow data structure
        self.assertIsInstance(flow_data, dict)
        self.assertIn("energy_flow", flow_data)
        self.assertIn("density_centers", flow_data)
        self.assertIn("emergent_patterns", flow_data)
        self.assertIn("flow_metrics", flow_data)
        
    def test_calculate_energy_flow(self):
        """Test the _calculate_energy_flow method."""
        # Set up test data
        resonance_matrix = np.random.rand(10, 10)
        
        # Call the method
        energy_flow = self.flow_analyzer._calculate_energy_flow(
            resonance_matrix, self.sample_metadata
        )
        
        # Check results
        self.assertIsInstance(energy_flow, dict)
        self.assertIn("flow_vectors", energy_flow)
        self.assertIn("flow_magnitude", energy_flow)
        self.assertIn("source_patterns", energy_flow)
        self.assertIn("sink_patterns", energy_flow)
        
    def test_identify_density_centers(self):
        """Test the _identify_density_centers method."""
        # Set up test data
        energy_flow = {
            "flow_vectors": np.random.rand(10, 5),
            "flow_magnitude": np.random.rand(10),
            "source_patterns": [0, 3, 7],
            "sink_patterns": [2, 5, 9]
        }
        
        # Call the method
        density_centers = self.flow_analyzer._identify_density_centers(
            energy_flow, self.mock_field_data
        )
        
        # Check results
        self.assertIsInstance(density_centers, list)
        for center in density_centers:
            self.assertIn("center_idx", center)
            self.assertIn("density_score", center)
            self.assertIn("influence_radius", center)
            self.assertIn("contributing_patterns", center)
            
    def test_predict_emergent_patterns(self):
        """Test the _predict_emergent_patterns method."""
        # Set up test data
        energy_flow = {
            "flow_vectors": np.random.rand(10, 5),
            "flow_magnitude": np.random.rand(10),
            "source_patterns": [0, 3, 7],
            "sink_patterns": [2, 5, 9]
        }
        
        density_centers = [
            {"center_idx": 2, "density_score": 0.8, "influence_radius": 0.3, "contributing_patterns": [1, 2, 3]},
            {"center_idx": 5, "density_score": 0.7, "influence_radius": 0.4, "contributing_patterns": [4, 5, 6]}
        ]
        
        # Call the method
        emergent_patterns = self.flow_analyzer._predict_emergent_patterns(
            energy_flow, density_centers, self.sample_vectors, self.sample_metadata
        )
        
        # Check results
        self.assertIsInstance(emergent_patterns, list)
        for pattern in emergent_patterns:
            self.assertIn("emergence_point", pattern)
            self.assertIn("probability", pattern)
            self.assertIn("contributing_centers", pattern)
            self.assertIn("estimated_vector", pattern)
            
    def test_calculate_flow_metrics(self):
        """Test the _calculate_flow_metrics method."""
        # Set up test data
        energy_flow = {
            "flow_vectors": np.random.rand(10, 5),
            "flow_magnitude": np.random.rand(10),
            "source_patterns": [0, 3, 7],
            "sink_patterns": [2, 5, 9]
        }
        
        density_centers = [
            {"center_idx": 2, "density_score": 0.8, "influence_radius": 0.3, "contributing_patterns": [1, 2, 3]},
            {"center_idx": 5, "density_score": 0.7, "influence_radius": 0.4, "contributing_patterns": [4, 5, 6]}
        ]
        
        # Call the method
        metrics = self.flow_analyzer._calculate_flow_metrics(
            energy_flow, density_centers, self.mock_field_data
        )
        
        # Check metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn("flow_coherence", metrics)
        self.assertIn("flow_stability", metrics)
        self.assertIn("density_distribution", metrics)
        self.assertIn("emergence_potential", metrics)
        
    def test_integration_with_semantic_boundary_detector(self):
        """Test integration with SemanticBoundaryDetector."""
        # Set up mock boundary data
        transition_patterns = [
            {"pattern_idx": 3, "uncertainty": 0.8, "source_community": 0, "neighboring_communities": [1]},
            {"pattern_idx": 6, "uncertainty": 0.7, "source_community": 1, "neighboring_communities": [2]}
        ]
        
        self.mock_boundary_detector.detect_transition_patterns.return_value = transition_patterns
        
        # Call the method
        boundary_flow = self.flow_analyzer.analyze_boundary_flow_dynamics(
            self.sample_vectors, self.sample_metadata
        )
        
        # Check results
        self.assertIsInstance(boundary_flow, dict)
        self.assertIn("boundary_flow_vectors", boundary_flow)
        self.assertIn("boundary_density_centers", boundary_flow)
        self.assertIn("boundary_emergent_patterns", boundary_flow)


if __name__ == "__main__":
    unittest.main()
