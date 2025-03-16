"""
Integration tests for SemanticBoundaryDetector and FlowDynamicsAnalyzer.

This test suite validates the integration between the SemanticBoundaryDetector and
FlowDynamicsAnalyzer, ensuring they work together to analyze flow dynamics at
semantic boundaries and transition zones.
"""
import unittest
import numpy as np
from typing import Dict, List, Any
import os
import sys
import json
from datetime import datetime

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import the components we're testing
from habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer
from habitat_evolution.field.field_navigator import FieldNavigator
from habitat_evolution.field.semantic_boundary_detector import SemanticBoundaryDetector
from habitat_evolution.field.flow_dynamics_analyzer import FlowDynamicsAnalyzer


class TestBoundaryFlowIntegration(unittest.TestCase):
    """Test suite for integration between SemanticBoundaryDetector and FlowDynamicsAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        # Create real instances of all components
        self.field_analyzer = TopologicalFieldAnalyzer()
        self.field_navigator = FieldNavigator(field_analyzer=self.field_analyzer)
        self.boundary_detector = SemanticBoundaryDetector(
            field_analyzer=self.field_analyzer,
            field_navigator=self.field_navigator
        )
        self.flow_analyzer = FlowDynamicsAnalyzer(
            field_analyzer=self.field_analyzer,
            field_navigator=self.field_navigator,
            boundary_detector=self.boundary_detector
        )
        
        # Generate synthetic test data
        self.generate_test_data()
        
    def generate_test_data(self):
        """Generate synthetic test data with clear community structure and boundaries."""
        # Number of patterns and features
        n_patterns = 30
        n_features = 10
        
        # Create three distinct communities
        community_centers = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Community 0
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Community 1
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]   # Community 2
        ])
        
        # Generate patterns for each community with noise
        vectors = []
        community_assignments = []
        
        for i in range(n_patterns):
            # Assign to communities roughly equally
            community = i % 3
            community_assignments.append(community)
            
            # Generate vector close to community center with noise
            base_vector = community_centers[community].copy()
            noise = np.random.normal(0, 0.2, n_features)
            vector = base_vector + noise
            vector = vector / np.linalg.norm(vector)  # Normalize
            vectors.append(vector)
        
        # Create boundary patterns (mix of communities)
        boundary_indices = [9, 19, 29]  # Last pattern of each community
        for idx in boundary_indices:
            # Mix with next community
            next_community = (community_assignments[idx] + 1) % 3
            vectors[idx] = (vectors[idx] + vectors[idx-3]) / 2  # Mix with pattern from other community
            vectors[idx] = vectors[idx] / np.linalg.norm(vectors[idx])  # Normalize
        
        # Convert to numpy array
        self.test_vectors = np.array(vectors)
        
        # Create metadata
        self.test_metadata = []
        for i in range(n_patterns):
            self.test_metadata.append({
                "id": f"pattern_{i}",
                "text": f"Test pattern {i} in community {community_assignments[i]}",
                "community": community_assignments[i],
                "timestamp": f"2025-03-{15+(i%15)}T10:00:00Z"
            })
        
        # Store expected boundary patterns for verification
        self.expected_boundary_indices = boundary_indices

    def test_boundary_detection(self):
        """Test that SemanticBoundaryDetector correctly identifies boundary patterns."""
        # Detect transition patterns
        transition_patterns = self.boundary_detector.detect_transition_patterns(
            self.test_vectors, self.test_metadata
        )
        
        # Check that transition patterns were detected
        self.assertTrue(len(transition_patterns) > 0)
        
        # Check that expected boundary patterns are included
        detected_indices = [p["pattern_idx"] for p in transition_patterns]
        for idx in self.expected_boundary_indices:
            self.assertIn(idx, detected_indices)
        
        # Check transition pattern structure
        for pattern in transition_patterns:
            self.assertIn("pattern_idx", pattern)
            self.assertIn("uncertainty", pattern)
            self.assertIn("source_community", pattern)
            self.assertIn("neighboring_communities", pattern)
    
    def test_flow_dynamics_analysis(self):
        """Test that FlowDynamicsAnalyzer correctly analyzes flow dynamics."""
        # Analyze flow dynamics
        flow_data = self.flow_analyzer.analyze_flow_dynamics(
            self.test_vectors, self.test_metadata
        )
        
        # Check flow data structure
        self.assertIn("energy_flow", flow_data)
        self.assertIn("density_centers", flow_data)
        self.assertIn("emergent_patterns", flow_data)
        self.assertIn("flow_metrics", flow_data)
        
        # Check energy flow
        self.assertIn("flow_vectors", flow_data["energy_flow"])
        self.assertIn("flow_magnitude", flow_data["energy_flow"])
        
        # Check density centers
        self.assertTrue(len(flow_data["density_centers"]) > 0)
        
        # Check flow metrics
        self.assertIn("flow_coherence", flow_data["flow_metrics"])
        self.assertIn("flow_stability", flow_data["flow_metrics"])
        self.assertIn("density_distribution", flow_data["flow_metrics"])
        self.assertIn("emergence_potential", flow_data["flow_metrics"])
    
    def test_boundary_flow_integration(self):
        """Test integration between boundary detection and flow dynamics."""
        # Analyze boundary flow dynamics
        boundary_flow = self.flow_analyzer.analyze_boundary_flow_dynamics(
            self.test_vectors, self.test_metadata
        )
        
        # Check boundary flow data structure
        self.assertIn("boundary_flow_vectors", boundary_flow)
        self.assertIn("boundary_density_centers", boundary_flow)
        self.assertIn("boundary_emergent_patterns", boundary_flow)
        self.assertIn("transition_patterns", boundary_flow)
        
        # Check that transition patterns match boundary detector output
        detector_patterns = self.boundary_detector.detect_transition_patterns(
            self.test_vectors, self.test_metadata
        )
        
        self.assertEqual(
            len(boundary_flow["transition_patterns"]),
            len(detector_patterns)
        )
        
        # Check that boundary density centers are related to transition patterns
        if boundary_flow["boundary_density_centers"]:
            for center in boundary_flow["boundary_density_centers"]:
                # Center should have contributing patterns that include transition patterns
                transition_indices = [p["pattern_idx"] for p in boundary_flow["transition_patterns"]]
                contributing_patterns = center["contributing_patterns"]
                
                # At least one contributing pattern should be a transition pattern
                self.assertTrue(
                    any(idx in transition_indices for idx in contributing_patterns),
                    f"Density center {center['center_idx']} has no contributing transition patterns"
                )
    
    def test_emergent_pattern_prediction_at_boundaries(self):
        """Test prediction of emergent patterns at semantic boundaries."""
        # Analyze boundary flow dynamics
        boundary_flow = self.flow_analyzer.analyze_boundary_flow_dynamics(
            self.test_vectors, self.test_metadata
        )
        
        # Check for emergent patterns at boundaries
        self.assertIn("boundary_emergent_patterns", boundary_flow)
        
        # If emergent patterns were predicted, verify their structure
        if boundary_flow["boundary_emergent_patterns"]:
            for pattern in boundary_flow["boundary_emergent_patterns"]:
                self.assertIn("emergence_point", pattern)
                self.assertIn("probability", pattern)
                self.assertIn("contributing_centers", pattern)
                self.assertIn("estimated_vector", pattern)
                
                # Probability should be between 0 and 1
                self.assertTrue(0 <= pattern["probability"] <= 1)
                
                # Estimated vector should have the right dimensions
                self.assertEqual(
                    len(pattern["estimated_vector"]),
                    self.test_vectors.shape[1]
                )
    
    def test_learning_opportunity_identification(self):
        """Test identification of learning opportunities at flow boundaries."""
        # Get learning opportunities from boundary detector
        learning_opportunities = self.boundary_detector.identify_learning_opportunities(
            self.test_vectors, self.test_metadata
        )
        
        # Analyze boundary flow
        boundary_flow = self.flow_analyzer.analyze_boundary_flow_dynamics(
            self.test_vectors, self.test_metadata
        )
        
        # Check that learning opportunities correspond to high flow areas
        if learning_opportunities and boundary_flow["boundary_flow_vectors"]:
            for opportunity in learning_opportunities:
                pattern_idx = opportunity["pattern_idx"]
                
                # Pattern should be in boundary flow vectors or connected to them
                self.assertTrue(
                    pattern_idx in boundary_flow["boundary_flow_vectors"] or
                    any(
                        pattern_idx in center["contributing_patterns"] 
                        for center in boundary_flow["boundary_density_centers"]
                    ),
                    f"Learning opportunity at pattern {pattern_idx} not related to boundary flow"
                )


if __name__ == "__main__":
    unittest.main()
