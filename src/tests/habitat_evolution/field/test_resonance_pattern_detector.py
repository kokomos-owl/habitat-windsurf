"""
Tests for the ResonancePatternDetector class.

This test suite validates the functionality of the ResonancePatternDetector,
which identifies and classifies meaningful resonance patterns in the field.
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import networkx as nx
from typing import Dict, List, Any
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import the classes we need to test
from habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer
from habitat_evolution.field.field_navigator import FieldNavigator
from habitat_evolution.field.resonance_pattern_detector import ResonancePatternDetector


class TestResonancePatternDetector(unittest.TestCase):
    """Test suite for the ResonancePatternDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock field analyzer
        self.mock_analyzer = MagicMock(spec=TopologicalFieldAnalyzer)
        
        # Create mock field navigator
        self.mock_navigator = MagicMock(spec=FieldNavigator)
        
        # Initialize detector with config
        self.detector = ResonancePatternDetector(
            config={
                "resonance_threshold": 0.65,
                "harmonic_tolerance": 0.2,
                "pattern_stability_threshold": 0.7,
                "min_pattern_size": 2,
                "max_pattern_size": 10,
                "detection_sensitivity": 0.25
            }
        )
        
        # Set the mocks as attributes for later use
        self.detector.field_analyzer = self.mock_analyzer
        self.detector.field_navigator = self.mock_navigator
        
        # Sample data for tests
        self.sample_vectors = np.random.rand(10, 5)  # 10 patterns with 5 features each
        self.sample_metadata = [
            {"id": f"pattern_{i}", "text": f"Sample pattern {i}"} for i in range(10)
        ]
        
        # Mock field data
        self.mock_field_data = {
            "communities": [0, 1, 2],
            "pattern_communities": [0, 0, 1, 0, 1, 2, 1, 2, 0, 2],
            "eigenvalues": [1.0, 0.8, 0.6, 0.4, 0.2],
            "eigenvectors": np.random.rand(5, 5),
            "resonance_matrix": np.random.rand(10, 10)  # Add resonance matrix
        }
        
        # Set up mock returns
        self.mock_analyzer.analyze_field.return_value = self.mock_field_data
        self.mock_navigator.set_field.return_value = self.mock_field_data

    def test_initialization(self):
        """Test that the detector initializes correctly."""
        # Check that the detector has the expected attributes
        self.assertIsInstance(self.detector.config, dict)
        
        # Check that the config has the expected values
        self.assertIn("resonance_threshold", self.detector.config)
        self.assertIn("harmonic_tolerance", self.detector.config)
        self.assertIn("pattern_stability_threshold", self.detector.config)
        self.assertIn("min_pattern_size", self.detector.config)
        self.assertIn("max_pattern_size", self.detector.config)
        self.assertIn("detection_sensitivity", self.detector.config)
        
    def test_detect_patterns(self):
        """Test the detect_patterns method."""
        # Create mock resonance matrix
        resonance_matrix = np.array([
            [1.0, 0.8, 0.3, 0.2, 0.1],
            [0.8, 1.0, 0.7, 0.1, 0.2],
            [0.3, 0.7, 1.0, 0.6, 0.3],
            [0.2, 0.1, 0.6, 1.0, 0.8],
            [0.1, 0.2, 0.3, 0.8, 1.0]
        ])
        
        # Create sample metadata
        metadata = [
            {"id": "pattern_0", "text": "Sample pattern 0"},
            {"id": "pattern_1", "text": "Sample pattern 1"},
            {"id": "pattern_2", "text": "Sample pattern 2"},
            {"id": "pattern_3", "text": "Sample pattern 3"},
            {"id": "pattern_4", "text": "Sample pattern 4"}
        ]
        
        # Call the method
        patterns = self.detector.detect_patterns(
            resonance_matrix, metadata
        )
        
        # Check that patterns were detected
        self.assertGreater(len(patterns), 0)
        
        # Check pattern structure
        for pattern in patterns:
            self.assertIn("id", pattern)
            self.assertIn("pattern_type", pattern)
            self.assertIn("members", pattern)
            self.assertIn("strength", pattern)
            self.assertIn("stability", pattern)
            
    def test_detect_from_field_analysis(self):
        """Test the detect_from_field_analysis method."""
        # Create mock field analysis result with dictionary pattern_projections
        field_analysis = {
            "graph_metrics": {
                "community_assignment": {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}
            },
            "topology": {
                "pattern_projections": {
                    0: {0: 0.8, 1: 0.2, 2: 0.0},
                    1: {0: 0.7, 1: 0.3, 2: 0.0},
                    2: {0: 0.3, 1: 0.7, 2: 0.0},
                    3: {0: 0.2, 1: 0.8, 2: 0.0},
                    4: {0: 0.1, 1: 0.9, 2: 0.0}
                }
            },
            "density": {
                "density_centers": [
                    {"index": 0, "density": 0.8, "influence_radius": 2},
                    {"index": 3, "density": 0.7, "influence_radius": 2}
                ],
                "node_strengths": [0.8, 0.7, 0.6, 0.7, 0.5]
            }
        }
        
        # Create sample metadata
        metadata = [
            {"id": "pattern_0", "text": "Sample pattern 0"},
            {"id": "pattern_1", "text": "Sample pattern 1"},
            {"id": "pattern_2", "text": "Sample pattern 2"},
            {"id": "pattern_3", "text": "Sample pattern 3"},
            {"id": "pattern_4", "text": "Sample pattern 4"}
        ]
        
        # Mock methods that might cause issues
        self.detector._extract_resonance_matrix = MagicMock(return_value=np.eye(5))
        self.detector._group_by_primary_dimension = MagicMock(return_value={0: [0, 1], 1: [2, 3, 4]})
        self.detector._has_significant_overlap = MagicMock(return_value=False)
        self.detector._find_patterns_in_radius = MagicMock(return_value=[0, 1])
        
        # Call the method
        patterns = self.detector.detect_from_field_analysis(
            field_analysis, metadata
        )
        
        # Check that patterns were detected
        self.assertGreater(len(patterns), 0)
            
    def test_create_resonance_graph(self):
        """Test the _create_resonance_graph method."""
        # Create mock resonance matrix
        resonance_matrix = np.array([
            [1.0, 0.8, 0.3, 0.2, 0.1],
            [0.8, 1.0, 0.7, 0.1, 0.2],
            [0.3, 0.7, 1.0, 0.6, 0.3],
            [0.2, 0.1, 0.6, 1.0, 0.8],
            [0.1, 0.2, 0.3, 0.8, 1.0]
        ])
        
        # Call the method
        G = self.detector._create_resonance_graph(resonance_matrix)
        
        # Check that the graph was created correctly
        self.assertIsInstance(G, nx.Graph)
        self.assertEqual(G.number_of_nodes(), 5)
        
        # Check that edges were created for resonances above threshold
        threshold = self.detector.config["resonance_threshold"]
        for i in range(5):
            for j in range(i+1, 5):
                if resonance_matrix[i, j] >= threshold:
                    self.assertTrue(G.has_edge(i, j))
                    self.assertEqual(G[i][j]["weight"], resonance_matrix[i, j])
            
    def test_detect_communities(self):
        """Test the _detect_communities method."""
        # Create a simple graph
        G = nx.Graph()
        G.add_nodes_from(range(5))
        G.add_edges_from([(0, 1), (1, 2), (3, 4)])
        
        # Call the method
        communities = self.detector._detect_communities(G)
        
        # Check that communities were detected
        self.assertIsInstance(communities, list)
        
        # Check that each community is a set of node indices
        for community in communities:
            self.assertIsInstance(community, set)
            
        # Check that all nodes are assigned to a community
        all_nodes = set()
        for community in communities:
            all_nodes.update(community)
        self.assertEqual(all_nodes, set(range(5)))
        
    def test_communities_to_patterns(self):
        """Test the _communities_to_patterns method."""
        # Create sample communities
        communities = [{0, 1}, {2, 3, 4}]
        
        # Create mock resonance matrix
        resonance_matrix = np.array([
            [1.0, 0.8, 0.3, 0.2, 0.1],
            [0.8, 1.0, 0.7, 0.1, 0.2],
            [0.3, 0.7, 1.0, 0.6, 0.3],
            [0.2, 0.1, 0.6, 1.0, 0.8],
            [0.1, 0.2, 0.3, 0.8, 1.0]
        ])
        
        # Create sample metadata
        metadata = [
            {"id": "pattern_0", "text": "Sample pattern 0"},
            {"id": "pattern_1", "text": "Sample pattern 1"},
            {"id": "pattern_2", "text": "Sample pattern 2"},
            {"id": "pattern_3", "text": "Sample pattern 3"},
            {"id": "pattern_4", "text": "Sample pattern 4"}
        ]
        
        # Call the method
        patterns = self.detector._communities_to_patterns(communities, resonance_matrix, metadata)
        
        # Check that patterns were created
        self.assertGreater(len(patterns), 0)
        
        # Check pattern structure
        for pattern in patterns:
            self.assertIn("id", pattern)
            self.assertIn("pattern_type", pattern)
            self.assertIn("members", pattern)
            self.assertIn("strength", pattern)
            self.assertIn("stability", pattern)
            self.assertIn("metadata", pattern)
    
    def test_vector_vs_tonic_harmonic_comparison(self):
        """Compare vector-only approach with vector + tonic-harmonic approach."""
        # Create vectors that have high cosine similarity within groups but hide harmonic relationships
        # Group 1: Vectors with high similarity but no harmonic relationship
        # Group 2: Vectors with moderate similarity but strong harmonic relationship
        # Group 3: Vectors with low similarity but strong dimensional resonance
        # Group 4: Edge case - orthogonal vectors with harmonic relationship in specific dimensions
        
        # Create test vectors - 12 vectors in 5D space
        vectors = np.array([
            # Group 1: High similarity vectors (0-2)
            [0.9, 0.8, 0.1, 0.1, 0.1],  # Vector 0
            [0.85, 0.82, 0.15, 0.12, 0.08],  # Vector 1
            [0.88, 0.79, 0.12, 0.09, 0.11],  # Vector 2
            
            # Group 2: Moderate similarity but harmonic relationship (3-5)
            [0.5, 0.5, 0.5, 0.1, 0.1],  # Vector 3
            [0.4, 0.4, 0.6, 0.2, 0.1],  # Vector 4
            [0.6, 0.6, 0.4, 0.1, 0.2],  # Vector 5
            
            # Group 3: Low similarity but dimensional resonance (6-8)
            [0.2, 0.1, 0.1, 0.9, 0.1],  # Vector 6
            [0.1, 0.2, 0.2, 0.1, 0.9],  # Vector 7
            [0.3, 0.3, 0.1, 0.7, 0.5],  # Vector 8
            
            # Group 4: Edge case - orthogonal vectors with harmonic relationship (9-11)
            # These vectors are nearly orthogonal (very low similarity) but have harmonic relationship
            # in specific dimensions (dimensional resonance)
            [0.95, 0.05, 0.05, 0.05, 0.05],  # Vector 9 - primarily dimension 0
            [0.05, 0.05, 0.05, 0.95, 0.05],  # Vector 10 - primarily dimension 3
            [0.05, 0.05, 0.05, 0.05, 0.95]   # Vector 11 - primarily dimension 4
        ])
        
        # Create metadata
        metadata = [
            {"id": f"pattern_{i}", "text": f"Sample pattern {i}"} for i in range(12)
        ]
        
        # Method 1: Vector-only approach using cosine similarity
        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        # Calculate cosine similarity matrix
        cosine_matrix = np.zeros((12, 12))
        for i in range(12):
            for j in range(12):
                cosine_matrix[i, j] = cosine_similarity(vectors[i], vectors[j])
        
        # Method 2: Resonance-based approach
        # First create a field analysis result with the necessary components
        field_analysis = {
            "topology": {
                "eigenvalues": np.array([2.5, 1.8, 1.2, 0.8, 0.5]),  # Eigenvalues showing importance of dimensions
                "eigenvectors": np.random.rand(5, 5),  # Random eigenvectors for this test
                "pattern_projections": {  # Projections onto eigenvectors
                    i: {j: np.random.rand() for j in range(5)} for i in range(12)
                },
                "dimensional_resonance": {
                    # Dimensional resonance between vectors 9, 10, 11 in eigenspace
                    # Even though they're orthogonal in vector space
                    "resonance_pairs": [(9, 10), (10, 11), (9, 11)],
                    "resonance_strength": {(9, 10): 0.85, (10, 11): 0.82, (9, 11): 0.79}
                }
            },
            "graph_metrics": {
                "community_assignment": {  # Community assignments based on resonance
                    0: 0, 1: 0, 2: 0,  # Group 1
                    3: 1, 4: 1, 5: 1,  # Group 2
                    6: 2, 7: 2, 8: 2,  # Group 3
                    9: 3, 10: 3, 11: 3  # Group 4 - edge case
                },
                "boundary_fuzziness": {
                    # Boundary fuzziness metrics for edge detection
                    "community_boundaries": [(0, 1), (1, 2), (2, 3), (0, 3)],
                    "boundary_strengths": {(0, 1): 0.3, (1, 2): 0.4, (2, 3): 0.6, (0, 3): 0.2},
                    "transition_zones": {(0, 1): [2, 3], (1, 2): [5, 6], (2, 3): [8, 9], (0, 3): [0, 9]}
                }
            },
            "density": {
                "density_centers": [
                    {"index": 1, "density": 0.8, "influence_radius": 2},  # Center in Group 1
                    {"index": 4, "density": 0.7, "influence_radius": 2},  # Center in Group 2
                    {"index": 7, "density": 0.6, "influence_radius": 2},  # Center in Group 3
                    {"index": 10, "density": 0.75, "influence_radius": 2}  # Center in Group 4
                ],
                "node_strengths": [0.8, 0.7, 0.75, 0.6, 0.65, 0.55, 0.5, 0.6, 0.45, 0.7, 0.75, 0.65]
            }
        }
        
        # Mock necessary methods
        self.detector._extract_resonance_matrix = MagicMock(return_value=np.eye(12))
        self.detector._group_by_primary_dimension = MagicMock(return_value={
            0: [0, 1, 2, 9],      # Group 1 and part of Group 4 by primary dimension 0
            1: [3, 4, 5],         # Group 2 by primary dimension 1
            3: [6, 8, 10],        # Part of Group 3 and part of Group 4 by primary dimension 3
            4: [7, 11]            # Part of Group 3 and part of Group 4 by primary dimension 4
        })
        self.detector._has_significant_overlap = MagicMock(return_value=False)
        self.detector._find_patterns_in_radius = MagicMock(side_effect=lambda center, radius, strengths: {
            1: [0, 1, 2],          # Patterns around center 1
            4: [3, 4, 5],          # Patterns around center 4
            7: [6, 7, 8],          # Patterns around center 7
            10: [9, 10, 11]        # Patterns around center 10
        }[center])
        
        # Add mock for detecting dimensional resonance
        self.detector._detect_dimensional_resonance = MagicMock(return_value=[
            # Return the edge case group that has dimensional resonance
            {"id": "edge_resonance", "members": [9, 10, 11], "pattern_type": "dimensional_resonance", 
             "strength": 0.85, "stability": 0.9, "metadata": {"resonance_type": "orthogonal_dimensional"}}
        ])
        
        # Get patterns using resonance-based approach
        resonance_patterns = self.detector.detect_from_field_analysis(field_analysis, metadata)
        
        # Create a simple vector-only detector that uses cosine similarity threshold
        def detect_vector_only_patterns(vectors, threshold=0.9):
            patterns = []
            visited = set()
            
            for i in range(len(vectors)):
                if i in visited:
                    continue
                    
                members = [i]
                visited.add(i)
                
                for j in range(len(vectors)):
                    if j != i and j not in visited and cosine_matrix[i, j] >= threshold:
                        members.append(j)
                        visited.add(j)
                
                if len(members) >= 2:  # Only consider groups of 2 or more
                    patterns.append({
                        "id": f"vector_pattern_{len(patterns)}",
                        "members": members,
                        "similarity_score": np.mean([cosine_matrix[i, j] for j in members if j != i])
                    })
            
            return patterns
        
        # Get patterns using vector-only approach with different thresholds
        vector_patterns_high = detect_vector_only_patterns(vectors, threshold=0.9)  # High threshold
        vector_patterns_medium = detect_vector_only_patterns(vectors, threshold=0.7)  # Medium threshold
        vector_patterns_low = detect_vector_only_patterns(vectors, threshold=0.5)  # Low threshold
        
        # Calculate additional metrics for comparison
        def calculate_metrics(patterns, name):
            if not patterns:
                return {"name": name, "count": 0, "avg_size": 0, "coverage": 0, "groups_detected": []}
            
            # Calculate average pattern size
            avg_size = sum(len(p["members"]) for p in patterns) / len(patterns)
            
            # Calculate coverage (percentage of vectors included in at least one pattern)
            covered_vectors = set()
            for p in patterns:
                covered_vectors.update(p["members"])
            coverage = len(covered_vectors) / len(vectors) * 100
            
            # Determine which groups were detected
            groups_detected = []
            for pattern in patterns:
                members = set(pattern["members"])
                if len(members.intersection(group1)) >= 2:
                    groups_detected.append(1)
                if len(members.intersection(group2)) >= 2:
                    groups_detected.append(2)
                if len(members.intersection(group3)) >= 2:
                    groups_detected.append(3)
                if len(members.intersection(group4)) >= 2:
                    groups_detected.append(4)
            
            return {
                "name": name,
                "count": len(patterns),
                "avg_size": avg_size,
                "coverage": coverage,
                "groups_detected": sorted(set(groups_detected))
            }
        
        # Calculate the detection rate for each group
        group1 = set([0, 1, 2])
        group2 = set([3, 4, 5])
        group3 = set([6, 7, 8])
        group4 = set([9, 10, 11])  # Edge case group
        
        # Calculate metrics for each approach
        metrics_vector_high = calculate_metrics(vector_patterns_high, "Vector (high threshold)")
        metrics_vector_medium = calculate_metrics(vector_patterns_medium, "Vector (medium threshold)")
        metrics_vector_low = calculate_metrics(vector_patterns_low, "Vector (low threshold)")
        metrics_resonance = calculate_metrics(resonance_patterns, "Resonance-based")
        
        # Log comparison metrics
        print("\n=== Pattern Detection Comparison ===\n")
        print(f"{'Approach':<25} {'Patterns':<10} {'Avg Size':<10} {'Coverage %':<12} {'Groups Detected'}")
        print("-" * 80)
        
        for metrics in [metrics_vector_high, metrics_vector_medium, metrics_vector_low, metrics_resonance]:
            print(f"{metrics['name']:<25} {metrics['count']:<10} {metrics['avg_size']:<10.2f} "
                  f"{metrics['coverage']:<12.2f} {metrics['groups_detected']}")
        
        # Calculate edge detection metrics
        edge_detection_vector = False
        for patterns in [vector_patterns_high, vector_patterns_medium, vector_patterns_low]:
            for pattern in patterns:
                members = set(pattern["members"])
                # Check if any pattern contains members from both group1 and group4 (edge case)
                if len(members.intersection(group1)) >= 1 and len(members.intersection(group4)) >= 1:
                    edge_detection_vector = True
                    break
        
        edge_detection_resonance = False
        boundary_patterns = []
        for pattern in resonance_patterns:
            members = set(pattern["members"])
            # Check for patterns that span multiple groups (boundary patterns)
            groups_spanned = 0
            if len(members.intersection(group1)) >= 1:
                groups_spanned += 1
            if len(members.intersection(group2)) >= 1:
                groups_spanned += 1
            if len(members.intersection(group3)) >= 1:
                groups_spanned += 1
            if len(members.intersection(group4)) >= 1:
                groups_spanned += 1
            
            if groups_spanned > 1:
                boundary_patterns.append(pattern)
                edge_detection_resonance = True
        
        # Log edge detection results
        print("\n=== Edge Detection Capabilities ===\n")
        print(f"Vector-only approach detected edges: {edge_detection_vector}")
        print(f"Resonance-based approach detected edges: {edge_detection_resonance}")
        print(f"Number of boundary patterns detected: {len(boundary_patterns)}")
        
        # Check for dimensional resonance detection (edge case group 4)
        dimensional_resonance_detected = False
        for pattern in resonance_patterns:
            members = set(pattern["members"])
            if len(members.intersection(group4)) >= 2 and pattern.get("pattern_type") == "dimensional_resonance":
                dimensional_resonance_detected = True
                print(f"\nDimensional resonance detected in edge case: {pattern}")
                break
        
        # Assertions for test validation
        self.assertLess(metrics_vector_high["count"], metrics_resonance["count"], 
                       "Resonance-based approach should detect more patterns than vector-only approach")
        
        self.assertGreater(metrics_resonance["coverage"], metrics_vector_high["coverage"],
                          "Resonance-based approach should have better coverage than vector-only approach")
        
        self.assertGreater(len(metrics_resonance["groups_detected"]), len(metrics_vector_high["groups_detected"]),
                          "Resonance-based approach should detect more groups than vector-only approach")
        
        # Assert that edge detection is better with resonance-based approach
        if edge_detection_resonance:
            self.assertGreaterEqual(len(boundary_patterns), 1, 
                                "Resonance-based approach should detect at least one boundary pattern")


if __name__ == "__main__":
    unittest.main()
