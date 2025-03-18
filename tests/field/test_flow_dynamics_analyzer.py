"""Tests for the FlowDynamicsAnalyzer class.

This module contains comprehensive tests for the FlowDynamicsAnalyzer, including
unit tests for core methods and integration tests with AdaptiveID.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any
import uuid

from habitat_evolution.field.flow_dynamics_analyzer import FlowDynamicsAnalyzer
from habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer
from habitat_evolution.field.field_navigator import FieldNavigator
from habitat_evolution.field.semantic_boundary_detector import SemanticBoundaryDetector


class TestFlowDynamicsAnalyzer(unittest.TestCase):
    """Test suite for FlowDynamicsAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.field_analyzer = MagicMock(spec=TopologicalFieldAnalyzer)
        self.field_navigator = MagicMock(spec=FieldNavigator)
        self.boundary_detector = MagicMock(spec=SemanticBoundaryDetector)
        
        # Configure mock behavior
        self.field_analyzer.analyze_field.return_value = {
            "resonance_matrix": np.array([
                [1.0, 0.8, 0.3, 0.1],
                [0.8, 1.0, 0.5, 0.2],
                [0.3, 0.5, 1.0, 0.7],
                [0.1, 0.2, 0.7, 1.0]
            ]),
            "eigenvalues": np.array([2.5, 1.2, 0.8, 0.5]),
            "eigenvectors": np.random.rand(4, 4),
            "graph_metrics": {
                "centrality": {0: 0.8, 1: 0.7, 2: 0.6, 3: 0.5},
                "community_assignment": {0: 0, 1: 0, 2: 1, 3: 1}
            },
            "topology": {
                "density": 0.75,
                "pattern_projections": [
                    {"dim1": 0.8, "dim2": 0.2},
                    {"dim1": 0.7, "dim2": 0.3},
                    {"dim1": 0.3, "dim2": 0.7},
                    {"dim1": 0.2, "dim2": 0.8}
                ]
            }
        }
        
        # Create test vectors and metadata
        self.test_vectors = np.array([
            [0.9, 0.1, 0.2, 0.3],
            [0.8, 0.2, 0.3, 0.4],
            [0.3, 0.7, 0.8, 0.2],
            [0.2, 0.8, 0.7, 0.1]
        ])
        
        self.test_metadata = [
            {"id": str(uuid.uuid4()), "name": "Pattern1", "timestamp": "2025-01-01T00:00:00"},
            {"id": str(uuid.uuid4()), "name": "Pattern2", "timestamp": "2025-01-01T00:01:00"},
            {"id": str(uuid.uuid4()), "name": "Pattern3", "timestamp": "2025-01-01T00:02:00"},
            {"id": str(uuid.uuid4()), "name": "Pattern4", "timestamp": "2025-01-01T00:03:00"}
        ]
        
        # Create analyzer with custom config for testing
        self.analyzer = FlowDynamicsAnalyzer(
            self.field_analyzer,
            self.field_navigator,
            self.boundary_detector,
            config={
                "flow_threshold": 0.4,
                "density_threshold": 0.5,
                "emergence_probability_threshold": 0.6,
                "temporal_window_size": 3,
                "max_density_centers": 5
            }
        )

    def test_analyze_flow_dynamics(self):
        """Test the analyze_flow_dynamics method."""
        # Call the method
        result = self.analyzer.analyze_flow_dynamics(self.test_vectors, self.test_metadata)
        
        # Verify field analyzer was called
        self.field_analyzer.analyze_field.assert_called_once_with(self.test_vectors, self.test_metadata)
        
        # Verify field navigator was called
        self.field_navigator.set_field.assert_called_once_with(self.test_vectors, self.test_metadata)
        
        # Verify result structure
        self.assertIn("energy_flow", result)
        self.assertIn("density_centers", result)
        self.assertIn("emergent_patterns", result)
        self.assertIn("flow_metrics", result)
        self.assertIn("timestamp", result)
        
        # Verify flow history was updated
        self.assertEqual(len(self.analyzer.flow_history), 1)
        self.assertEqual(self.analyzer.flow_history[0], result)

    def test_analyze_boundary_flow_dynamics(self):
        """Test the analyze_boundary_flow_dynamics method."""
        # Mock boundary detector to return some boundaries
        self.boundary_detector.detect_boundaries.return_value = {
            "boundaries": [
                {"source_community": 0, "target_community": 1, "strength": 0.6}
            ]
        }
        
        # Call the method
        result = self.analyzer.analyze_boundary_flow_dynamics(self.test_vectors, self.test_metadata)
        
        # Verify boundary detector was called
        self.boundary_detector.detect_boundaries.assert_called_once()
        
        # Verify result structure
        self.assertIn("boundary_flow", result)
        self.assertIn("boundary_density_centers", result)
        self.assertIn("boundary_emergent_patterns", result)


class TestFlowDynamicsAnalyzerCoreMethods(unittest.TestCase):
    """Test suite for FlowDynamicsAnalyzer core methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.field_analyzer = MagicMock(spec=TopologicalFieldAnalyzer)
        self.field_navigator = MagicMock(spec=FieldNavigator)
        
        # Create analyzer without boundary detector for core method tests
        self.analyzer = FlowDynamicsAnalyzer(
            self.field_analyzer,
            self.field_navigator
        )
        
        # Create test data
        self.resonance_matrix = np.array([
            [1.0, 0.8, 0.3, 0.1],
            [0.8, 1.0, 0.5, 0.2],
            [0.3, 0.5, 1.0, 0.7],
            [0.1, 0.2, 0.7, 1.0]
        ])
        
        self.test_metadata = [
            {"id": "1", "name": "Pattern1", "timestamp": "2025-01-01T00:00:00"},
            {"id": "2", "name": "Pattern2", "timestamp": "2025-01-01T00:01:00"},
            {"id": "3", "name": "Pattern3", "timestamp": "2025-01-01T00:02:00"},
            {"id": "4", "name": "Pattern4", "timestamp": "2025-01-01T00:03:00"}
        ]
        
        self.field_data = {
            "eigenvalues": np.array([2.5, 1.2, 0.8, 0.5]),
            "eigenvectors": np.random.rand(4, 4),
            "graph_metrics": {
                "centrality": {0: 0.8, 1: 0.7, 2: 0.6, 3: 0.5},
                "community_assignment": {0: 0, 1: 0, 2: 1, 3: 1}
            },
            "topology": {
                "density": 0.75,
                "pattern_projections": [
                    {"dim1": 0.8, "dim2": 0.2},
                    {"dim1": 0.7, "dim2": 0.3},
                    {"dim1": 0.3, "dim2": 0.7},
                    {"dim1": 0.2, "dim2": 0.8}
                ]
            }
        }
        
        self.test_vectors = np.array([
            [0.9, 0.1, 0.2, 0.3],
            [0.8, 0.2, 0.3, 0.4],
            [0.3, 0.7, 0.8, 0.2],
            [0.2, 0.8, 0.7, 0.1]
        ])

    def test_calculate_energy_flow(self):
        """Test the _calculate_energy_flow method."""
        # Call the method
        energy_flow = self.analyzer._calculate_energy_flow(self.resonance_matrix, self.test_metadata)
        
        # Verify result structure
        self.assertIn("flow_vectors", energy_flow)
        self.assertIn("flow_magnitude", energy_flow)
        self.assertIn("source_patterns", energy_flow)
        self.assertIn("sink_patterns", energy_flow)
        
        # Verify flow vectors shape
        self.assertEqual(energy_flow["flow_vectors"].shape, (4, 4))
        
        # Verify flow magnitude is a scalar
        self.assertIsInstance(energy_flow["flow_magnitude"], float)
        
        # Verify source and sink patterns are lists
        self.assertIsInstance(energy_flow["source_patterns"], list)
        self.assertIsInstance(energy_flow["sink_patterns"], list)

    def test_identify_density_centers(self):
        """Test the _identify_density_centers method."""
        # Create energy flow data
        energy_flow = {
            "flow_vectors": np.random.rand(4, 4),
            "flow_magnitude": 0.75,
            "source_patterns": [0, 1],
            "sink_patterns": [2, 3]
        }
        
        # Call the method
        density_centers = self.analyzer._identify_density_centers(energy_flow, self.field_data)
        
        # Verify result is a list
        self.assertIsInstance(density_centers, list)
        
        # Verify each center has the expected structure
        for center in density_centers:
            self.assertIn("center_id", center)
            self.assertIn("position", center)
            self.assertIn("density", center)
            self.assertIn("influence_radius", center)
            self.assertIn("patterns_in_radius", center)

    def test_predict_emergent_patterns(self):
        """Test the _predict_emergent_patterns method."""
        # Create energy flow data
        energy_flow = {
            "flow_vectors": np.random.rand(4, 4),
            "flow_magnitude": 0.75,
            "source_patterns": [0, 1],
            "sink_patterns": [2, 3]
        }
        
        # Create density centers
        density_centers = [
            {
                "center_id": "center1",
                "position": np.array([0.8, 0.2, 0.3, 0.4]),
                "density": 0.8,
                "influence_radius": 0.5,
                "patterns_in_radius": [0, 1]
            },
            {
                "center_id": "center2",
                "position": np.array([0.3, 0.7, 0.8, 0.2]),
                "density": 0.7,
                "influence_radius": 0.5,
                "patterns_in_radius": [2, 3]
            }
        ]
        
        # Call the method
        emergent_patterns = self.analyzer._predict_emergent_patterns(
            energy_flow, density_centers, self.test_vectors, self.test_metadata
        )
        
        # Verify result is a list
        self.assertIsInstance(emergent_patterns, list)
        
        # Verify each emergent pattern has the expected structure
        for pattern in emergent_patterns:
            self.assertIn("pattern_id", pattern)
            self.assertIn("position", pattern)
            self.assertIn("emergence_probability", pattern)
            self.assertIn("contributing_patterns", pattern)
            self.assertIn("estimated_appearance_time", pattern)

    def test_calculate_flow_metrics(self):
        """Test the _calculate_flow_metrics method."""
        # Create energy flow data
        energy_flow = {
            "flow_vectors": np.random.rand(4, 4),
            "flow_magnitude": 0.75,
            "source_patterns": [0, 1],
            "sink_patterns": [2, 3]
        }
        
        # Create density centers
        density_centers = [
            {
                "center_id": "center1",
                "position": np.array([0.8, 0.2, 0.3, 0.4]),
                "density": 0.8,
                "influence_radius": 0.5,
                "patterns_in_radius": [0, 1]
            },
            {
                "center_id": "center2",
                "position": np.array([0.3, 0.7, 0.8, 0.2]),
                "density": 0.7,
                "influence_radius": 0.5,
                "patterns_in_radius": [2, 3]
            }
        ]
        
        # Call the method
        flow_metrics = self.analyzer._calculate_flow_metrics(energy_flow, density_centers, self.field_data)
        
        # Verify result structure
        self.assertIn("flow_coherence", flow_metrics)
        self.assertIn("flow_stability", flow_metrics)
        self.assertIn("density_distribution", flow_metrics)
        self.assertIn("emergence_potential", flow_metrics)
        
        # Verify metrics are scalars
        self.assertIsInstance(flow_metrics["flow_coherence"], float)
        self.assertIsInstance(flow_metrics["flow_stability"], float)
        self.assertIsInstance(flow_metrics["density_distribution"], float)
        self.assertIsInstance(flow_metrics["emergence_potential"], float)


class TestFlowDynamicsAnalyzerWithAdaptiveID(unittest.TestCase):
    """Test suite for FlowDynamicsAnalyzer integration with AdaptiveID."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.field_analyzer = MagicMock(spec=TopologicalFieldAnalyzer)
        self.field_navigator = MagicMock(spec=FieldNavigator)
        
        # Mock the AdaptiveID bridge
        self.adaptive_id_bridge = MagicMock()
        self.adaptive_id_bridge.update_context.return_value = {"context_updated": True}
        
        # Create analyzer
        self.analyzer = FlowDynamicsAnalyzer(
            self.field_analyzer,
            self.field_navigator
        )
        
        # Create test data
        self.test_vectors = np.array([
            [0.9, 0.1, 0.2, 0.3],
            [0.8, 0.2, 0.3, 0.4],
            [0.3, 0.7, 0.8, 0.2],
            [0.2, 0.8, 0.7, 0.1]
        ])
        
        self.test_metadata = [
            {
                "id": str(uuid.uuid4()),
                "name": "Pattern1",
                "timestamp": "2025-01-01T00:00:00",
                "adaptive_id_context": {"context_key": "value1"}
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Pattern2",
                "timestamp": "2025-01-01T00:01:00",
                "adaptive_id_context": {"context_key": "value2"}
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Pattern3",
                "timestamp": "2025-01-01T00:02:00",
                "adaptive_id_context": {"context_key": "value3"}
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Pattern4",
                "timestamp": "2025-01-01T00:03:00",
                "adaptive_id_context": {"context_key": "value4"}
            }
        ]
        
        # Configure field analyzer mock
        self.field_analyzer.analyze_field.return_value = {
            "resonance_matrix": np.array([
                [1.0, 0.8, 0.3, 0.1],
                [0.8, 1.0, 0.5, 0.2],
                [0.3, 0.5, 1.0, 0.7],
                [0.1, 0.2, 0.7, 1.0]
            ]),
            "eigenvalues": np.array([2.5, 1.2, 0.8, 0.5]),
            "eigenvectors": np.random.rand(4, 4),
            "graph_metrics": {
                "centrality": {0: 0.8, 1: 0.7, 2: 0.6, 3: 0.5},
                "community_assignment": {0: 0, 1: 0, 2: 1, 3: 1}
            },
            "topology": {
                "density": 0.75,
                "pattern_projections": [
                    {"dim1": 0.8, "dim2": 0.2},
                    {"dim1": 0.7, "dim2": 0.3},
                    {"dim1": 0.3, "dim2": 0.7},
                    {"dim1": 0.2, "dim2": 0.8}
                ]
            }
        }

    @patch('habitat_evolution.field.field_adaptive_id_bridge.FieldAdaptiveIDBridge')
    def test_adaptive_id_context_propagation(self, mock_bridge_class):
        """Test that AdaptiveID context is properly propagated."""
        # Configure mock bridge
        mock_bridge = mock_bridge_class.return_value
        mock_bridge.update_context.return_value = {"context_updated": True}
        
        # Call analyze_flow_dynamics
        result = self.analyzer.analyze_flow_dynamics(self.test_vectors, self.test_metadata)
        
        # Verify result contains flow metrics
        self.assertIn("flow_metrics", result)
        
        # Add AdaptiveID context to flow metrics
        result["flow_metrics"]["adaptive_id_context"] = {
            "source": "flow_dynamics_analyzer",
            "timestamp": "2025-01-01T00:10:00",
            "flow_coherence": result["flow_metrics"]["flow_coherence"],
            "flow_stability": result["flow_metrics"]["flow_stability"]
        }
        
        # Verify AdaptiveID context can be extracted from flow metrics
        adaptive_id_context = result["flow_metrics"].get("adaptive_id_context")
        self.assertIsNotNone(adaptive_id_context)
        self.assertEqual(adaptive_id_context["source"], "flow_dynamics_analyzer")

    def test_energy_flow_with_adaptive_id_context(self):
        """Test that energy flow calculations incorporate AdaptiveID context."""
        # Create resonance matrix
        resonance_matrix = np.array([
            [1.0, 0.8, 0.3, 0.1],
            [0.8, 1.0, 0.5, 0.2],
            [0.3, 0.5, 1.0, 0.7],
            [0.1, 0.2, 0.7, 1.0]
        ])
        
        # Call _calculate_energy_flow
        energy_flow = self.analyzer._calculate_energy_flow(resonance_matrix, self.test_metadata)
        
        # Verify energy flow contains source and sink patterns
        self.assertIn("source_patterns", energy_flow)
        self.assertIn("sink_patterns", energy_flow)
        
        # Verify source patterns have AdaptiveID context
        for idx in energy_flow["source_patterns"]:
            if idx < len(self.test_metadata):
                self.assertIn("adaptive_id_context", self.test_metadata[idx])

    def test_emergent_patterns_with_adaptive_id_context(self):
        """Test that emergent patterns inherit AdaptiveID context from contributing patterns."""
        # Create energy flow data
        energy_flow = {
            "flow_vectors": np.random.rand(4, 4),
            "flow_magnitude": 0.75,
            "source_patterns": [0, 1],
            "sink_patterns": [2, 3]
        }
        
        # Create density centers
        density_centers = [
            {
                "center_id": "center1",
                "position": np.array([0.8, 0.2, 0.3, 0.4]),
                "density": 0.8,
                "influence_radius": 0.5,
                "patterns_in_radius": [0, 1]
            }
        ]
        
        # Call _predict_emergent_patterns
        emergent_patterns = self.analyzer._predict_emergent_patterns(
            energy_flow, density_centers, self.test_vectors, self.test_metadata
        )
        
        # Verify emergent patterns inherit context from contributing patterns
        for pattern in emergent_patterns:
            # Add AdaptiveID context to emergent pattern
            pattern["adaptive_id_context"] = {
                "inherited_from": [
                    self.test_metadata[idx]["adaptive_id_context"]
                    for idx in pattern["contributing_patterns"]
                    if idx < len(self.test_metadata)
                ]
            }
            
            # Verify AdaptiveID context is present
            self.assertIn("adaptive_id_context", pattern)
            self.assertIn("inherited_from", pattern["adaptive_id_context"])


if __name__ == "__main__":
    unittest.main()
