"""Tests for eigenspace navigation in the Field Navigator."""

import unittest
import numpy as np
from typing import List, Dict, Any

from habitat_evolution.field.field_navigator import FieldNavigator
from habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer


class TestEigenspaceNavigation(unittest.TestCase):
    """Test cases for eigenspace navigation capabilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TopologicalFieldAnalyzer()
        self.navigator = FieldNavigator()
        
        # Create test vectors with known dimensional relationships
        # These vectors are designed to have specific resonance patterns:
        # - v0, v1, v2 form a harmonic pattern (similar projections on dimension 0)
        # - v3, v4 form a complementary pattern (opposite projections on dimension 1)
        # - v5, v6, v7, v8 form a sequential pattern (progressive sequence on dimension 2)
        # - v9, v10, v11 are boundary patterns between communities
        self.test_vectors = np.array([
            [1.0, 0.1, 0.1],    # v0: Strong on dim 0
            [0.9, 0.2, 0.1],    # v1: Strong on dim 0
            [0.8, 0.3, 0.2],    # v2: Strong on dim 0
            [0.2, 0.9, 0.1],    # v3: Strong positive on dim 1
            [0.2, -0.8, 0.1],   # v4: Strong negative on dim 1
            [0.3, 0.3, -0.8],   # v5: Sequential start on dim 2
            [0.3, 0.3, -0.4],   # v6: Sequential step 1 on dim 2
            [0.3, 0.3, 0.0],    # v7: Sequential step 2 on dim 2
            [0.3, 0.3, 0.4],    # v8: Sequential step 3 on dim 2
            [0.6, 0.6, 0.1],    # v9: Boundary pattern (between dim 0 and dim 1 communities)
            [0.4, 0.4, -0.6],   # v10: Boundary pattern (between dim 1 and dim 2 communities)
            [0.6, 0.1, -0.6]    # v11: Boundary pattern (between dim 0 and dim 2 communities)
        ])
        
        # Create test metadata
        self.test_metadata = [
            {"id": f"pattern_{i}", "type": "test_pattern"} for i in range(len(self.test_vectors))
        ]
        
        # Calculate resonance matrix (cosine similarity)
        self.resonance_matrix = np.zeros((len(self.test_vectors), len(self.test_vectors)))
        for i in range(len(self.test_vectors)):
            for j in range(len(self.test_vectors)):
                # Calculate cosine similarity
                dot_product = np.dot(self.test_vectors[i], self.test_vectors[j])
                norm_i = np.linalg.norm(self.test_vectors[i])
                norm_j = np.linalg.norm(self.test_vectors[j])
                if norm_i > 0 and norm_j > 0:
                    self.resonance_matrix[i, j] = dot_product / (norm_i * norm_j)
                else:
                    self.resonance_matrix[i, j] = 0.0
        
        # Analyze field and set up navigator
        self.field_analysis = self.analyzer.analyze_field(self.resonance_matrix, self.test_metadata)
        self.navigator.set_field(self.field_analysis, self.test_metadata)

    def test_dimensional_resonance_detection(self):
        """Test detection of dimensional resonance between patterns."""
        # Test resonance between patterns in the same harmonic group
        resonance = self.navigator._detect_dimensional_resonance(0, 1)
        self.assertIsNotNone(resonance)
        self.assertGreater(resonance["strength"], 0.5)
        
        # Test resonance between complementary patterns
        resonance = self.navigator._detect_dimensional_resonance(3, 4)
        self.assertIsNotNone(resonance)
        
        # Test resonance between sequential patterns
        resonance = self.navigator._detect_dimensional_resonance(5, 8)
        self.assertIsNotNone(resonance)
        
        # Test resonance between unrelated patterns (should be weak or None)
        resonance = self.navigator._detect_dimensional_resonance(0, 4)
        if resonance:
            self.assertLess(resonance["strength"], 0.3)

    def test_eigenspace_navigation(self):
        """Test navigation through eigenspace between patterns."""
        # Navigate between patterns in the same harmonic group
        path = self.navigator.navigate_eigenspace(0, 2)
        self.assertGreaterEqual(len(path), 2)
        self.assertEqual(path[0]["index"], 0)
        self.assertEqual(path[-1]["index"], 2)
        
        # Navigate between complementary patterns
        path = self.navigator.navigate_eigenspace(3, 4)
        self.assertGreaterEqual(len(path), 2)
        self.assertEqual(path[0]["index"], 3)
        self.assertEqual(path[-1]["index"], 4)
        
        # Navigate between sequential patterns
        path = self.navigator.navigate_eigenspace(5, 8)
        self.assertGreaterEqual(len(path), 2)
        self.assertEqual(path[0]["index"], 5)
        self.assertEqual(path[-1]["index"], 8)
        
        # Check intermediate steps in sequential navigation
        if len(path) > 2:
            # Verify that intermediate steps follow the sequence
            for i in range(1, len(path) - 1):
                self.assertIn(path[i]["index"], [6, 7])

    def test_fuzzy_boundary_detection(self):
        """Test detection of fuzzy boundaries between communities."""
        # Get boundary information
        boundary_info = self.navigator.detect_fuzzy_boundaries()
        
        # Check that boundaries were detected
        self.assertIsNotNone(boundary_info)
        self.assertIn("boundaries", boundary_info)
        
        # Verify that known boundary patterns are identified
        boundary_indices = [b["pattern_idx"] for b in boundary_info["boundaries"]]
        for idx in [9, 10, 11]:  # Known boundary patterns
            self.assertIn(idx, boundary_indices)

    def test_transition_zone_exploration(self):
        """Test exploration of transition zones around boundary patterns."""
        # Explore around a known boundary pattern
        boundary_idx = 9  # Boundary between dim 0 and dim 1 communities
        exploration_results = self.navigator.explore_transition_zone(boundary_idx)
        
        # Check that exploration found interesting patterns
        self.assertGreater(len(exploration_results), 0)
        
        # Verify that exploration results contain patterns from both communities
        if len(exploration_results) > 0:
            # Extract communities of exploration results
            communities = [result.get("community", -1) for result in exploration_results]
            self.assertGreater(len(set(communities)), 1)

    def test_boundary_crossing_path(self):
        """Test planning paths that cross community boundaries."""
        # Get boundary information
        boundary_info = self.navigator.detect_fuzzy_boundaries()
        boundaries = boundary_info.get("boundaries", [])
        
        # Plan a path between patterns in different communities
        path = self.navigator.plan_boundary_crossing_path(0, 4, boundaries)
        
        # Check that a path was found
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0]["index"], 0)
        self.assertEqual(path[-1]["index"], 4)
        
        # Verify that the path includes at least one boundary pattern
        boundary_indices = [b["pattern_idx"] for b in boundaries]
        path_indices = [step["index"] for step in path]
        
        # Check if any boundary pattern is in the path
        boundary_in_path = any(idx in boundary_indices for idx in path_indices)
        self.assertTrue(boundary_in_path)


if __name__ == "__main__":
    unittest.main()
