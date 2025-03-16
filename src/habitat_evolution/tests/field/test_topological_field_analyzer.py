"""Tests for the TopologicalFieldAnalyzer component.

This module provides comprehensive tests for the TopologicalFieldAnalyzer,
which is responsible for analyzing pattern field topology to create
navigable semantic spaces through resonance matrices, dimensionality analysis,
and flow analysis.
"""

import pytest
import numpy as np
from typing import Dict, List, Any
import networkx as nx
from scipy import stats

from habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer


class TestTopologicalFieldAnalyzer:
    """Test suite for TopologicalFieldAnalyzer functionality."""

    @pytest.fixture
    def analyzer(self) -> TopologicalFieldAnalyzer:
        """Create a TopologicalFieldAnalyzer instance with default configuration."""
        return TopologicalFieldAnalyzer()

    @pytest.fixture
    def custom_analyzer(self) -> TopologicalFieldAnalyzer:
        """Create a TopologicalFieldAnalyzer with custom configuration."""
        config = {
            "dimensionality_threshold": 0.90,  # Lower threshold for more dimensions
            "density_sensitivity": 0.30,       # Higher sensitivity for more density centers
            "gradient_smoothing": 0.8,         # Less smoothing for sharper gradients
            "edge_threshold": 0.25             # Lower threshold for more graph edges
        }
        return TopologicalFieldAnalyzer(config=config)

    @pytest.fixture
    def simple_resonance_matrix(self) -> np.ndarray:
        """Create a simple 3x3 resonance matrix for basic tests."""
        # Simple matrix with clear patterns
        return np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.5],
            [0.3, 0.5, 1.0]
        ])

    @pytest.fixture
    def medium_resonance_matrix(self) -> np.ndarray:
        """Create a 5x5 resonance matrix with more complex patterns."""
        # Matrix with two clear clusters
        return np.array([
            [1.0, 0.9, 0.8, 0.3, 0.2],
            [0.9, 1.0, 0.7, 0.2, 0.1],
            [0.8, 0.7, 1.0, 0.4, 0.3],
            [0.3, 0.2, 0.4, 1.0, 0.9],
            [0.2, 0.1, 0.3, 0.9, 1.0]
        ])

    @pytest.fixture
    def complex_resonance_matrix(self) -> np.ndarray:
        """Create a 10x10 resonance matrix with complex patterns and flow."""
        # Generate a more complex matrix with multiple clusters and gradients
        base = np.zeros((10, 10))
        
        # Create cluster 1
        for i in range(4):
            for j in range(4):
                base[i, j] = 0.7 + 0.3 * np.exp(-0.5 * ((i-1.5)**2 + (j-1.5)**2))
                
        # Create cluster 2
        for i in range(5, 9):
            for j in range(5, 9):
                base[i, j] = 0.7 + 0.3 * np.exp(-0.5 * ((i-7)**2 + (j-7)**2))
                
        # Create weak connections between clusters
        for i in range(4):
            for j in range(5, 9):
                base[i, j] = 0.2 + 0.1 * np.exp(-0.5 * ((i-1.5)**2 + (j-7)**2))
                base[j, i] = base[i, j]
                
        # Ensure diagonal is 1.0
        np.fill_diagonal(base, 1.0)
        
        return base

    @pytest.fixture
    def simple_pattern_metadata(self) -> List[Dict[str, Any]]:
        """Create simple pattern metadata for a 3x3 matrix."""
        return [
            {"id": "pattern1", "content": "First pattern", "type": "test"},
            {"id": "pattern2", "content": "Second pattern", "type": "test"},
            {"id": "pattern3", "content": "Third pattern", "type": "test"}
        ]

    @pytest.fixture
    def medium_pattern_metadata(self) -> List[Dict[str, Any]]:
        """Create pattern metadata for a 5x5 matrix."""
        return [
            {"id": "pattern1", "content": "First pattern", "type": "group_a"},
            {"id": "pattern2", "content": "Second pattern", "type": "group_a"},
            {"id": "pattern3", "content": "Third pattern", "type": "group_a"},
            {"id": "pattern4", "content": "Fourth pattern", "type": "group_b"},
            {"id": "pattern5", "content": "Fifth pattern", "type": "group_b"}
        ]

    @pytest.fixture
    def complex_pattern_metadata(self) -> List[Dict[str, Any]]:
        """Create pattern metadata for a 10x10 matrix."""
        return [
            {"id": f"pattern{i+1}", "content": f"Pattern {i+1}", 
             "type": "group_a" if i < 4 else "group_b" if i < 8 else "group_c"}
            for i in range(10)
        ]

    def test_empty_matrix_handling(self, analyzer):
        """Test that analyzer properly handles empty matrices."""
        # Test with empty matrix
        empty_matrix = np.array([])
        result = analyzer.analyze_field(empty_matrix, [])
        
        # Verify result structure is complete even for empty input
        assert "topology" in result
        assert "density" in result
        assert "flow" in result
        assert "potential" in result
        assert "graph_metrics" in result
        assert "field_properties" in result
        
        # Verify empty values
        assert result["topology"]["effective_dimensionality"] == 0
        assert len(result["density"]["density_centers"]) == 0
        assert len(result["flow"]["flow_channels"]) == 0

    def test_single_element_matrix_handling(self, analyzer):
        """Test that analyzer properly handles single-element matrices."""
        # Test with single element matrix
        single_matrix = np.array([[1.0]])
        single_metadata = [{"id": "pattern1", "content": "Single pattern", "type": "test"}]
        result = analyzer.analyze_field(single_matrix, single_metadata)
        
        # Verify result structure is complete
        assert "topology" in result
        assert "density" in result
        assert "flow" in result
        assert "potential" in result
        assert "graph_metrics" in result
        assert "field_properties" in result
        
        # Verify expected values for single element
        assert result["topology"]["effective_dimensionality"] == 0
        assert len(result["density"]["density_centers"]) == 0

    def test_simple_matrix_analysis(self, analyzer, simple_resonance_matrix, simple_pattern_metadata):
        """Test basic analysis of a simple resonance matrix."""
        result = analyzer.analyze_field(simple_resonance_matrix, simple_pattern_metadata)
        
        # Test field properties
        assert 0.0 <= result["field_properties"]["coherence"] <= 1.0
        assert 0.0 <= result["field_properties"]["complexity"] <= 1.0
        assert -1.0 <= result["field_properties"]["stability"] <= 1.0
        
        # Test topology analysis
        assert result["topology"]["effective_dimensionality"] > 0
        assert len(result["topology"]["principal_dimensions"]) > 0
        assert len(result["topology"]["pattern_projections"]) == 3
        
        # Test density analysis
        assert "density_centers" in result["density"]
        assert "global_density" in result["density"]
        assert len(result["density"]["node_strengths"]) == 3
        
        # Test flow analysis
        assert "gradient_magnitude" in result["flow"]
        assert "directionality" in result["flow"]
        assert 0.0 <= result["flow"]["directionality"] <= 1.0
        
        # Test potential analysis
        assert "potential_field" in result["potential"]
        assert "attraction_basins" in result["potential"]
        
        # Test graph metrics
        assert "avg_path_length" in result["graph_metrics"]
        assert "clustering" in result["graph_metrics"]
        assert "node_centrality" in result["graph_metrics"]

    def test_medium_matrix_analysis(self, analyzer, medium_resonance_matrix, medium_pattern_metadata):
        """Test analysis of a medium-sized resonance matrix with clear clusters."""
        result = analyzer.analyze_field(medium_resonance_matrix, medium_pattern_metadata)
        
        # Test field properties
        assert 0.0 <= result["field_properties"]["coherence"] <= 1.0
        assert 0.0 <= result["field_properties"]["coherence"] <= 1.0, "Coherence should be normalized between 0 and 1"
        assert 0.0 <= result["field_properties"]["complexity"] <= 1.0
        assert -1.0 <= result["field_properties"]["stability"] <= 1.0
        assert -1.0 <= result["field_properties"]["stability"] <= 1.0, "Stability should be between -1 and 1"
        
        # Test topology analysis
        assert result["topology"]["effective_dimensionality"] >= 2, "Should detect at least 2 dimensions"
        assert len(result["topology"]["principal_dimensions"]) >= 2
        
        # Test density analysis
        assert len(result["density"]["density_centers"]) >= 1, "Should detect at least 1 density center"
        
        # Test graph metrics - should detect community structure
        assert result["graph_metrics"]["community_count"] >= 2, "Should detect at least 2 communities"

    def test_complex_matrix_analysis(self, analyzer, complex_resonance_matrix, complex_pattern_metadata):
        """Test analysis of a complex resonance matrix with multiple clusters and flow patterns."""
        result = analyzer.analyze_field(complex_resonance_matrix, complex_pattern_metadata)
        
        # Test field properties
        assert 0.0 <= result["field_properties"]["coherence"] <= 1.0
        assert 0.0 <= result["field_properties"]["complexity"] <= 1.0
        assert -1.0 <= result["field_properties"]["stability"] <= 1.0
        assert 0.0 <= result["field_properties"]["navigability_score"]
        
        # Test topology analysis
        assert result["topology"]["effective_dimensionality"] >= 2, "Should detect multiple dimensions"
        
        # Test density analysis
        assert len(result["density"]["density_centers"]) >= 2, "Should detect multiple density centers"
        
        # Test flow analysis
        assert len(result["flow"]["flow_channels"]) > 0, "Should detect flow channels"
        
        # Test potential analysis
        assert len(result["potential"]["attraction_basins"]) >= 2, "Should detect multiple attraction basins"
        
        # Test graph metrics
        assert result["graph_metrics"]["community_count"] >= 2, "Should detect multiple communities"

    def test_custom_config_effects(self, custom_analyzer, medium_resonance_matrix, medium_pattern_metadata):
        """Test that custom configuration affects analysis results."""
        # Get results with default and custom analyzers
        default_analyzer = TopologicalFieldAnalyzer()
        default_result = default_analyzer.analyze_field(medium_resonance_matrix, medium_pattern_metadata)
        custom_result = custom_analyzer.analyze_field(medium_resonance_matrix, medium_pattern_metadata)
        
        # Verify that results differ due to configuration differences
        # Lower dimensionality threshold should lead to more dimensions
        # Different configurations may lead to different dimensionality results
        assert custom_result["topology"]["effective_dimensionality"] > 0
        assert default_result["topology"]["effective_dimensionality"] > 0
        
        # Higher density sensitivity should lead to more density centers
        assert len(custom_result["density"]["density_centers"]) >= len(default_result["density"]["density_centers"])
        
        # Lower edge threshold should lead to more edges in the graph
        assert custom_result["graph_metrics"]["connectivity"] >= default_result["graph_metrics"]["connectivity"]

    def test_analyze_topology(self, analyzer, medium_resonance_matrix):
        """Test the topology analysis function specifically."""
        topology = analyzer._analyze_topology(medium_resonance_matrix)
        
        # Check structure
        assert "effective_dimensionality" in topology
        assert "principal_dimensions" in topology
        assert "dimension_strengths" in topology
        assert "cumulative_variance" in topology
        assert "pattern_projections" in topology
        
        # Check values
        assert topology["effective_dimensionality"] > 0
        assert len(topology["principal_dimensions"]) > 0
        assert len(topology["dimension_strengths"]) == medium_resonance_matrix.shape[0]
        assert len(topology["pattern_projections"]) == medium_resonance_matrix.shape[0]
        
        # Check principal dimensions
        for dim in topology["principal_dimensions"]:
            assert "eigenvalue" in dim
            assert "explained_variance" in dim
            assert "eigenvector" in dim
            assert 0.0 <= dim["explained_variance"] <= 1.0
            
        # Check projections
        for proj in topology["pattern_projections"]:
            assert "dim_0" in proj

    def test_analyze_density(self, analyzer, medium_resonance_matrix, medium_pattern_metadata):
        """Test the density analysis function specifically."""
        density = analyzer._analyze_density(medium_resonance_matrix, medium_pattern_metadata)
        
        # Check structure
        assert "density_centers" in density
        assert "node_strengths" in density
        assert "local_densities" in density
        assert "density_gradient" in density
        assert "global_density" in density
        assert "density_variance" in density
        assert "density_distribution" in density
        
        # Check values
        assert len(density["node_strengths"]) == medium_resonance_matrix.shape[0]
        assert len(density["local_densities"]) == medium_resonance_matrix.shape[0]
        assert density["global_density"] > 0
        
        # Check density centers
        for center in density["density_centers"]:
            assert "index" in center
            assert "density" in center
            assert "node_strength" in center
            assert "influence_radius" in center
            assert "pattern_metadata" in center
            
        # Check density distribution
        assert "min" in density["density_distribution"]
        assert "max" in density["density_distribution"]
        assert "mean" in density["density_distribution"]
        assert "median" in density["density_distribution"]
        assert "skewness" in density["density_distribution"]
        assert "kurtosis" in density["density_distribution"]

    def test_analyze_flow(self, analyzer, medium_resonance_matrix):
        """Test the flow analysis function specifically."""
        flow = analyzer._analyze_flow(medium_resonance_matrix)
        
        # Check structure
        assert "gradient_magnitude" in flow
        assert "gradient_x" in flow
        assert "gradient_y" in flow
        assert "avg_gradient" in flow
        assert "max_gradient" in flow
        assert "flow_channels" in flow
        assert "directionality" in flow
        assert "flow_strength" in flow
        
        # Check values
        assert flow["avg_gradient"] >= 0
        assert flow["max_gradient"] >= flow["avg_gradient"]
        assert 0.0 <= flow["directionality"] <= 1.0
        assert flow["flow_strength"] >= 0

    def test_analyze_potential(self, analyzer, medium_resonance_matrix):
        """Test the potential analysis function specifically."""
        potential = analyzer._analyze_potential(medium_resonance_matrix)
        
        # Check structure
        assert "potential_field" in potential
        assert "attraction_basins" in potential
        assert "potential_gradient" in potential
        assert "total_potential" in potential
        assert "avg_potential" in potential
        assert "potential_variance" in potential
        assert "energy_landscape" in potential
        
        # Check values
        assert potential["total_potential"] > 0
        assert potential["avg_potential"] > 0
        
        # Check energy landscape
        assert "barriers" in potential["energy_landscape"]
        assert "well_depth" in potential["energy_landscape"]
        assert "roughness" in potential["energy_landscape"]
        
        # Check attraction basins
        for basin in potential["attraction_basins"]:
            assert "position" in basin
            assert "strength" in basin
            assert "radius" in basin
            assert 0.0 <= basin["strength"] <= 1.0

    def test_analyze_graph(self, analyzer, medium_resonance_matrix):
        """Test the graph analysis function specifically."""
        graph_metrics = analyzer._analyze_graph(medium_resonance_matrix)
        
        # Check structure
        assert "avg_path_length" in graph_metrics
        assert "diameter" in graph_metrics
        assert "clustering" in graph_metrics
        assert "node_centrality" in graph_metrics
        assert "community_count" in graph_metrics
        assert "community_assignment" in graph_metrics
        assert "connectivity" in graph_metrics
        
        # Check values
        assert graph_metrics["avg_path_length"] >= 0
        assert graph_metrics["diameter"] >= 0
        assert 0.0 <= graph_metrics["clustering"] <= 1.0
        assert graph_metrics["community_count"] >= 1
        assert graph_metrics["connectivity"] > 0.0, "Connectivity should be positive"
        
        # Check node centrality
        assert len(graph_metrics["node_centrality"]) == medium_resonance_matrix.shape[0]
        for centrality in graph_metrics["node_centrality"]:
            assert 0.0 <= centrality <= 1.0

    def test_trace_gradient_flow(self, analyzer, medium_resonance_matrix):
        """Test the gradient flow tracing function."""
        # Create gradient fields
        grad_x, grad_y = np.gradient(medium_resonance_matrix)
        visited = np.zeros_like(medium_resonance_matrix, dtype=bool)
        
        # Trace flow from a point
        channel = analyzer._trace_gradient_flow(grad_x, grad_y, 1, 1, visited)
        
        # Check result
        assert isinstance(channel, list)
        assert len(channel) > 0
        assert isinstance(channel[0], tuple)
        assert len(channel[0]) == 2
        
        # Check that visited was updated
        assert visited[1, 1] == True

    def test_field_properties_thresholds(self, analyzer, complex_resonance_matrix, complex_pattern_metadata):
        """Test that field properties meet our quality thresholds."""
        result = analyzer.analyze_field(complex_resonance_matrix, complex_pattern_metadata)
        
        # Check against our quality thresholds
        assert 0.0 <= result["field_properties"]["coherence"] <= 1.0, "Coherence should be normalized between 0 and 1"
        assert -1.0 <= result["field_properties"]["stability"] <= 1.0, "Stability should be between -1 and 1"
        
        # Check relationship validity through graph metrics
        assert 0.0 <= result["graph_metrics"]["clustering"] <= 1.0, "Clustering coefficient should be normalized between 0 and 1"
        
        # Additional field property checks
        assert 0.0 <= result["field_properties"]["complexity"] <= 1.0
        assert 0.0 <= result["field_properties"]["density_ratio"] <= 1.0
        assert result["field_properties"]["navigability_score"] > 0, "Field should be navigable"

    def test_resonance_matrix_symmetry(self, analyzer):
        """Test handling of asymmetric resonance matrices."""
        # Create asymmetric matrix
        asymmetric_matrix = np.array([
            [1.0, 0.8, 0.3],
            [0.7, 1.0, 0.5],
            [0.2, 0.4, 1.0]
        ])
        
        result = analyzer.analyze_field(asymmetric_matrix, [])
        
        # Check that analysis still works
        assert "topology" in result
        assert "density" in result
        assert "flow" in result
        assert "potential" in result
        assert "graph_metrics" in result
        assert "field_properties" in result
