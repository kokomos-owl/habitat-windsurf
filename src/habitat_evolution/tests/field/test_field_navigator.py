"""Tests for the FieldNavigator component.

This module provides comprehensive tests for the FieldNavigator,
which is responsible for navigating the topological field and providing
metrics for the system's state space.
"""

import pytest
import numpy as np
from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime

# Import the components
from habitat_evolution.field.field_navigator import FieldNavigator
from habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer


class TestFieldNavigator:
    """Test suite for FieldNavigator functionality."""

    @pytest.fixture
    def analyzer_config(self) -> Dict[str, Any]:
        """Create a configuration for the TopologicalFieldAnalyzer."""
        return {
            "min_eigenvalue": 0.1,
            "max_dimensions": 5,
            "density_threshold": 0.6,
            "flow_sensitivity": 0.3,
            "graph_weight_threshold": 0.5
        }

    @pytest.fixture
    def field_analyzer(self, analyzer_config) -> TopologicalFieldAnalyzer:
        """Create a TopologicalFieldAnalyzer instance."""
        return TopologicalFieldAnalyzer(analyzer_config)

    @pytest.fixture
    def navigator(self, field_analyzer) -> FieldNavigator:
        """Create a FieldNavigator instance."""
        return FieldNavigator(field_analyzer)

    @pytest.fixture
    def simple_resonance_matrix(self) -> np.ndarray:
        """Create a simple resonance matrix with clear patterns."""
        # Simple matrix with two clear patterns
        return np.array([
            [1.0, 0.9, 0.8, 0.2, 0.1],
            [0.9, 1.0, 0.7, 0.3, 0.2],
            [0.8, 0.7, 1.0, 0.3, 0.2],
            [0.2, 0.3, 0.3, 1.0, 0.9],
            [0.1, 0.2, 0.2, 0.9, 1.0]
        ])

    @pytest.fixture
    def complex_resonance_matrix(self) -> np.ndarray:
        """Create a complex resonance matrix with multiple patterns."""
        # 10x10 matrix with multiple patterns
        matrix = np.zeros((10, 10))
        
        # Pattern 1: Strong resonance (0-2)
        for i in range(3):
            for j in range(3):
                if i != j:
                    matrix[i, j] = 0.8 + 0.1 * np.random.random()
        
        # Pattern 2: Medium resonance (3-5)
        for i in range(3, 6):
            for j in range(3, 6):
                if i != j:
                    matrix[i, j] = 0.7 + 0.1 * np.random.random()
        
        # Pattern 3: Weak resonance (6-9)
        for i in range(6, 10):
            for j in range(6, 10):
                if i != j:
                    matrix[i, j] = 0.6 + 0.1 * np.random.random()
        
        # Some cross-pattern resonance
        matrix[2, 3] = matrix[3, 2] = 0.5
        matrix[5, 6] = matrix[6, 5] = 0.5
        
        # Set diagonal to 1.0
        np.fill_diagonal(matrix, 1.0)
        
        return matrix

    @pytest.fixture
    def pattern_metadata(self) -> List[Dict[str, Any]]:
        """Create pattern metadata for testing."""
        return [
            {
                "id": f"adaptive_id_{i}",
                "content": f"Pattern {i}",
                "type": "test",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "stability": 0.8 + 0.1 * np.random.random(),
                    "coherence": 0.7 + 0.2 * np.random.random(),
                    "confidence": 0.75 + 0.15 * np.random.random()
                }
            }
            for i in range(10)
        ]

    def test_navigator_initialization(self, navigator):
        """Test that the FieldNavigator initializes correctly."""
        assert navigator is not None
        assert navigator.field_analyzer is not None
        assert navigator.current_field is None
        assert navigator.pattern_metadata == []

    def test_set_field(self, navigator, complex_resonance_matrix, pattern_metadata):
        """Test setting and analyzing a field."""
        field = navigator.set_field(complex_resonance_matrix, pattern_metadata)
        
        # Check that the field was analyzed and set correctly
        assert navigator.current_field is not None
        assert "topology" in field
        assert "density" in field
        assert "flow" in field
        assert "potential" in field
        assert "graph" in field
        
        # Check that pattern metadata was set
        assert navigator.pattern_metadata == pattern_metadata

    def test_get_navigation_coordinates(self, navigator, complex_resonance_matrix, pattern_metadata):
        """Test getting navigation coordinates for patterns."""
        navigator.set_field(complex_resonance_matrix, pattern_metadata)
        
        # Test getting coordinates for a valid pattern
        coords = navigator.get_navigation_coordinates(0, dimensions=2)
        assert len(coords) == 2
        assert all(isinstance(c, float) for c in coords)
        
        # Test getting coordinates for a valid pattern with more dimensions
        coords = navigator.get_navigation_coordinates(0, dimensions=3)
        assert len(coords) == 3
        assert all(isinstance(c, float) for c in coords)
        
        # Test getting coordinates for an invalid pattern index
        coords = navigator.get_navigation_coordinates(100, dimensions=2)
        assert len(coords) == 2
        assert all(c == 0.0 for c in coords)
        
        # Test getting coordinates without setting a field
        navigator.current_field = None
        coords = navigator.get_navigation_coordinates(0, dimensions=2)
        assert len(coords) == 2
        assert all(c == 0.0 for c in coords)

    def test_find_nearest_density_center(self, navigator, complex_resonance_matrix, pattern_metadata):
        """Test finding the nearest density center to a pattern."""
        navigator.set_field(complex_resonance_matrix, pattern_metadata)
        
        # Test finding nearest center for a valid pattern
        center = navigator.find_nearest_density_center(0)
        assert center is not None
        assert "index" in center
        assert "density" in center
        assert "node_strength" in center
        assert "influence_radius" in center
        
        # Test finding nearest center for an invalid pattern index
        center = navigator.find_nearest_density_center(100)
        assert center is not None  # Should still find a center
        
        # Test finding nearest center without setting a field
        navigator.current_field = None
        center = navigator.find_nearest_density_center(0)
        assert center is None

    def test_find_paths(self, navigator, complex_resonance_matrix, pattern_metadata):
        """Test finding paths between patterns."""
        navigator.set_field(complex_resonance_matrix, pattern_metadata)
        
        # Test finding eigenvector path
        path = navigator.find_paths(0, 5, path_type="eigenvector")
        assert len(path) > 0
        assert path[0] == 0  # Start
        assert path[-1] == 5  # End
        
        # Test finding gradient path
        path = navigator.find_paths(0, 5, path_type="gradient")
        assert len(path) > 0
        assert path[0] == 0  # Start
        assert path[-1] == 5  # End
        
        # Test finding graph path
        path = navigator.find_paths(0, 5, path_type="graph")
        assert len(path) > 0
        assert path[0] == 0  # Start
        assert path[-1] == 5  # End
        
        # Test finding path with invalid path type
        path = navigator.find_paths(0, 5, path_type="invalid")
        assert path == []
        
        # Test finding path without setting a field
        navigator.current_field = None
        path = navigator.find_paths(0, 5)
        assert path == []

    def test_suggest_exploration_points(self, navigator, complex_resonance_matrix, pattern_metadata):
        """Test suggesting exploration points."""
        navigator.set_field(complex_resonance_matrix, pattern_metadata)
        
        # Test suggesting exploration points
        points = navigator.suggest_exploration_points(0, count=3)
        assert isinstance(points, list)
        
        # Test suggesting exploration points without setting a field
        navigator.current_field = None
        points = navigator.suggest_exploration_points(0, count=3)
        assert points == []

    def test_field_state_metrics(self, navigator, complex_resonance_matrix, pattern_metadata):
        """Test getting field state metrics."""
        navigator.set_field(complex_resonance_matrix, pattern_metadata)
        
        # Test getting field state metrics
        metrics = navigator.get_field_state_metrics()
        assert metrics is not None
        assert "dimensionality" in metrics
        assert "density" in metrics
        assert "flow" in metrics
        assert "stability" in metrics
        assert "coherence" in metrics
        
        # Test getting field state metrics without setting a field
        navigator.current_field = None
        metrics = navigator.get_field_state_metrics()
        assert metrics is None

    def test_get_navigable_field_representation(self, navigator, complex_resonance_matrix, pattern_metadata):
        """Test getting a navigable field representation."""
        navigator.set_field(complex_resonance_matrix, pattern_metadata)
        
        # Test getting a navigable field representation
        field_rep = navigator.get_navigable_field_representation()
        assert field_rep is not None
        assert "coordinates" in field_rep
        assert "connections" in field_rep
        assert "centers" in field_rep
        assert "metrics" in field_rep
        
        # Check that coordinates are provided for all patterns
        assert len(field_rep["coordinates"]) == len(pattern_metadata)
        
        # Test getting a navigable field representation without setting a field
        navigator.current_field = None
        field_rep = navigator.get_navigable_field_representation()
        assert field_rep is None

    def test_get_pattern_context(self, navigator, complex_resonance_matrix, pattern_metadata):
        """Test getting context for a pattern."""
        navigator.set_field(complex_resonance_matrix, pattern_metadata)
        
        # Test getting context for a valid pattern
        context = navigator.get_pattern_context(0)
        assert context is not None
        assert "pattern" in context
        assert "neighbors" in context
        assert "center" in context
        assert "position" in context
        
        # Test getting context for an invalid pattern index
        context = navigator.get_pattern_context(100)
        assert context is None
        
        # Test getting context without setting a field
        navigator.current_field = None
        context = navigator.get_pattern_context(0)
        assert context is None
