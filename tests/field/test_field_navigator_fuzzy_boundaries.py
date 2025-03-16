import unittest
import numpy as np
from typing import Dict, List, Any

from src.habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer
from src.habitat_evolution.field.field_navigator import FieldNavigator


class TestFieldNavigatorFuzzyBoundaries(unittest.TestCase):
    """Test the fuzzy boundary detection and navigation capabilities of the FieldNavigator."""

    def setUp(self):
        """Set up test fixtures."""
        self.field_analyzer = TopologicalFieldAnalyzer()
        self.navigator = FieldNavigator(self.field_analyzer)
        
        # Create a test resonance matrix with clear community structure and fuzzy boundaries
        # This matrix has 3 communities with patterns 0-3, 4-7, and 8-11
        # Patterns 3, 7, and 11 are at community boundaries
        size = 12
        self.resonance_matrix = np.zeros((size, size))
        
        # Create community 1 (patterns 0-3)
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.resonance_matrix[i, j] = 0.8 - 0.1 * abs(i - j)
        
        # Create community 2 (patterns 4-7)
        for i in range(4, 8):
            for j in range(4, 8):
                if i != j:
                    self.resonance_matrix[i, j] = 0.8 - 0.1 * abs(i - j)
        
        # Create community 3 (patterns 8-11)
        for i in range(8, 12):
            for j in range(8, 12):
                if i != j:
                    self.resonance_matrix[i, j] = 0.8 - 0.1 * abs(i - j)
        
        # Create fuzzy boundaries between communities
        # Between community 1 and 2
        self.resonance_matrix[3, 4] = 0.5
        self.resonance_matrix[4, 3] = 0.5
        
        # Between community 2 and 3
        self.resonance_matrix[7, 8] = 0.5
        self.resonance_matrix[8, 7] = 0.5
        
        # Between community 3 and 1 (creating a cycle)
        self.resonance_matrix[11, 0] = 0.4
        self.resonance_matrix[0, 11] = 0.4
        
        # Create pattern metadata
        self.pattern_metadata = []
        for i in range(size):
            community = i // 4  # Assign community based on index
            self.pattern_metadata.append({
                "id": f"pattern_{i}",
                "type": "test",
                "metrics": {
                    "coherence": 0.8,
                    "stability": 0.7
                },
                "community": community
            })
        
        # Initialize the field
        self.field = self.navigator.set_field(self.resonance_matrix, self.pattern_metadata)

    def test_transition_zones_detection(self):
        """Test that transition zones are correctly detected and processed."""
        # Verify transition zones exist in the field
        self.assertIn("transition_zones", self.field)
        transition_data = self.field["transition_zones"]
        
        # Verify transition zones were found
        self.assertIn("transition_zones", transition_data)
        zones = transition_data["transition_zones"]
        self.assertTrue(len(zones) > 0)
        
        # Verify boundary patterns (3, 7, 11) are identified as transition zones
        boundary_indices = [zone["pattern_idx"] for zone in zones]
        for idx in [3, 7]:  # These should definitely be boundaries
            self.assertIn(idx, boundary_indices, f"Pattern {idx} should be identified as a boundary")
            
        # Verify non-boundary patterns are not identified as transition zones
        for idx in [1, 5, 9]:  # These should definitely not be boundaries
            self.assertNotIn(idx, boundary_indices, f"Pattern {idx} should not be identified as a boundary")

    def test_fuzzy_boundary_path(self):
        """Test that paths through fuzzy boundaries are correctly identified."""
        # Test path from community 1 to community 2
        path = self.navigator.find_paths(1, 5, path_type="fuzzy_boundary")
        self.assertTrue(len(path) > 0)
        
        # Verify the path passes through the boundary pattern (3)
        self.assertIn(3, path, "Path should pass through boundary pattern 3")
        
        # Test path from community 2 to community 3
        path = self.navigator.find_paths(5, 9, path_type="fuzzy_boundary")
        self.assertTrue(len(path) > 0)
        
        # Verify the path passes through the boundary pattern (7)
        self.assertIn(7, path, "Path should pass through boundary pattern 7")

    def test_boundary_uncertainty_gradient(self):
        """Test that boundary uncertainty gradients are correctly calculated."""
        # Get a boundary pattern
        boundary_idx = 3  # Known boundary pattern
        
        # Calculate gradient
        gradient = self.navigator._calculate_uncertainty_gradient(boundary_idx)
        
        # Verify gradient is a 3D vector
        self.assertEqual(len(gradient), 3)
        
        # Verify gradient is not zero (should point in some direction)
        magnitude = np.sqrt(sum(g**2 for g in gradient))
        self.assertGreater(magnitude, 0, "Gradient magnitude should be greater than zero")

    def test_sliding_window_neighborhood(self):
        """Test that sliding window neighborhoods are correctly identified."""
        # Get neighborhood for a pattern
        pattern_idx = 5
        neighborhood = self.navigator._get_sliding_window_neighborhood(pattern_idx)
        
        # Verify neighborhood size matches sliding window size
        self.assertEqual(len(neighborhood), self.navigator.sliding_window_size)
        
        # Verify neighborhood contains nearby patterns (4, 6, 7)
        for idx in [4, 6]:
            self.assertIn(idx, neighborhood, f"Neighborhood should contain pattern {idx}")

    def test_explore_transition_zones(self):
        """Test that transition zone exploration candidates are correctly identified."""
        # Get exploration candidates from a non-boundary pattern in community 1
        candidates = self.navigator._explore_transition_zones(1)
        
        # Verify candidates exist
        self.assertTrue(len(candidates) > 0)
        
        # Verify candidates include boundary patterns
        candidate_indices = [c["index"] for c in candidates]
        self.assertIn(3, candidate_indices, "Candidates should include boundary pattern 3")
        
        # Verify candidates are sorted by relevance
        relevance_values = [c["relevance"] for c in candidates]
        self.assertEqual(relevance_values, sorted(relevance_values, reverse=True))

    def test_get_pattern_boundary_info(self):
        """Test that pattern boundary information is correctly retrieved."""
        # Get boundary info for a boundary pattern
        boundary_idx = 3
        boundary_info = self.navigator._get_pattern_boundary_info(boundary_idx)
        
        # Verify it's identified as a boundary
        self.assertTrue(boundary_info["is_boundary"])
        
        # Verify it has uncertainty value
        self.assertIn("uncertainty", boundary_info)
        
        # Verify it has source community
        self.assertIn("source_community", boundary_info)
        
        # Verify it has neighboring communities
        self.assertIn("neighboring_communities", boundary_info)
        
        # Get boundary info for a non-boundary pattern
        non_boundary_idx = 1
        non_boundary_info = self.navigator._get_pattern_boundary_info(non_boundary_idx)
        
        # Verify it's not identified as a boundary
        self.assertFalse(non_boundary_info["is_boundary"])

    def test_field_state_metrics(self):
        """Test that field state metrics include boundary information."""
        metrics = self.navigator.get_field_state_metrics()
        
        # Verify metrics include boundary section
        self.assertIn("boundary", metrics)
        
        # Verify boundary metrics include fuzziness
        self.assertIn("fuzziness", metrics["boundary"])
        
        # Verify boundary metrics include transition zone count
        self.assertIn("transition_zone_count", metrics["boundary"])
        
        # Verify boundary metrics include average uncertainty
        self.assertIn("average_uncertainty", metrics["boundary"])


if __name__ == "__main__":
    unittest.main()
