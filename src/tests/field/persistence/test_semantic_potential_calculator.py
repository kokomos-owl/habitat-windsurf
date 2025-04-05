"""
Tests for the SemanticPotentialCalculator.

This module tests the calculation of multi-dimensional potential metrics
for patterns, including evolutionary potential, constructive dissonance,
and topological energy.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.tests.adaptive_core.persistence.arangodb.test_state_models import PatternState
from src.habitat_evolution.adaptive_core.persistence.services.graph_service import GraphService
from src.habitat_evolution.field.persistence.semantic_potential_calculator import SemanticPotentialCalculator


class TestSemanticPotentialCalculator:
    """Tests for the SemanticPotentialCalculator class."""
    
    @pytest.fixture
    def mock_graph_service(self):
        """Create a mock graph service."""
        mock = MagicMock(spec=GraphService)
        mock.repository = MagicMock()
        mock.repository.find_node_by_id = MagicMock()
        mock.repository.find_quality_transitions_by_node_id = MagicMock()
        mock.repository.find_nodes_by_quality = MagicMock()
        mock.repository.find_relations_by_quality = MagicMock()
        return mock
    
    @pytest.fixture
    def calculator(self, mock_graph_service):
        """Create a SemanticPotentialCalculator instance."""
        return SemanticPotentialCalculator(mock_graph_service)
    
    @pytest.mark.asyncio
    async def test_calculate_pattern_potential(self, calculator, mock_graph_service):
        """Test calculating potential metrics for a pattern."""
        # Create a test pattern
        pattern = PatternState(
            id="test-pattern",
            content="Test pattern content",
            metadata={"type": "semantic"},
            timestamp=datetime.now(),
            confidence=0.8
        )
        
        # Mock the repository methods
        mock_graph_service.repository.find_node_by_id.return_value = pattern
        mock_graph_service.repository.find_quality_transitions_by_node_id.return_value = [
            type("Transition", (), {
                "from_quality": "uncertain",
                "to_quality": "good",
                "timestamp": datetime.now(),
                "context": {}
            })
        ]
        
        # Calculate pattern potential
        potential = await calculator.calculate_pattern_potential("test-pattern")
        
        # Check the results
        assert "evolutionary_potential" in potential
        assert "constructive_dissonance" in potential
        assert "stability_index" in potential
        assert "coherence_score" in potential
        assert "emergence_rate" in potential
        assert "pattern_id" in potential
        assert "timestamp" in potential
        
        # Verify method calls
        mock_graph_service.repository.find_node_by_id.assert_called_once_with("test-pattern")
        mock_graph_service.repository.find_quality_transitions_by_node_id.assert_called_once_with("test-pattern")
    
    @pytest.mark.asyncio
    async def test_calculate_field_potential(self, calculator, mock_graph_service):
        """Test calculating field potential."""
        # Create test patterns
        patterns = [
            PatternState(
                id=f"pattern-{i}",
                content=f"Test pattern {i}",
                metadata={"type": "semantic"},
                timestamp=datetime.now(),
                confidence=0.7 + (i * 0.1)
            )
            for i in range(3)
        ]
        
        # Mock the repository methods
        mock_graph_service.repository.find_nodes_by_quality.return_value = patterns
        
        # For each pattern, mock the transitions
        def mock_find_transitions(pattern_id):
            return [
                type("Transition", (), {
                    "from_quality": "uncertain",
                    "to_quality": "good",
                    "timestamp": datetime.now(),
                    "context": {}
                })
            ]
        
        mock_graph_service.repository.find_quality_transitions_by_node_id.side_effect = mock_find_transitions
        
        # Calculate field potential
        field_potential = await calculator.calculate_field_potential()
        
        # Check the results
        assert "avg_evolutionary_potential" in field_potential
        assert "avg_constructive_dissonance" in field_potential
        assert "gradient_field" in field_potential
        assert "pattern_count" in field_potential
        assert "window_id" in field_potential
        assert "timestamp" in field_potential
        
        # Check gradient field
        assert "magnitude" in field_potential["gradient_field"]
        assert "direction" in field_potential["gradient_field"]
        assert "uniformity" in field_potential["gradient_field"]
        
        # Verify method calls
        mock_graph_service.repository.find_nodes_by_quality.assert_called_once_with("good", node_type="pattern")
    
    @pytest.mark.asyncio
    async def test_calculate_topological_potential(self, calculator, mock_graph_service):
        """Test calculating topological potential."""
        # Create test patterns and relations
        patterns = [
            PatternState(
                id=f"pattern-{i}",
                content=f"Test pattern {i}",
                metadata={"type": "semantic"},
                timestamp=datetime.now(),
                confidence=0.7 + (i * 0.1)
            )
            for i in range(3)
        ]
        
        relations = [
            type("Relation", (), {
                "id": f"relation-{i}",
                "relation_type": "correlates_with",
                "source_id": f"pattern-{i}",
                "target_id": f"pattern-{(i+1) % 3}",
                "weight": 0.8,
                "attributes": {}
            })
            for i in range(3)
        ]
        
        # Mock the repository methods
        mock_graph_service.repository.find_nodes_by_quality.return_value = patterns
        mock_graph_service.repository.find_relations_by_quality.return_value = relations
        
        # Calculate topological potential
        topo_potential = await calculator.calculate_topological_potential()
        
        # Check the results
        assert "connectivity" in topo_potential
        assert "centrality" in topo_potential
        assert "temporal_stability" in topo_potential
        assert "manifold_curvature" in topo_potential
        assert "topological_energy" in topo_potential
        assert "window_id" in topo_potential
        assert "timestamp" in topo_potential
        
        # Check connectivity
        assert "density" in topo_potential["connectivity"]
        assert "clustering" in topo_potential["connectivity"]
        assert "path_efficiency" in topo_potential["connectivity"]
        
        # Check centrality
        assert "centralization" in topo_potential["centrality"]
        assert "heterogeneity" in topo_potential["centrality"]
        
        # Check temporal stability
        assert "persistence" in topo_potential["temporal_stability"]
        assert "evolution_rate" in topo_potential["temporal_stability"]
        assert "temporal_coherence" in topo_potential["temporal_stability"]
        
        # Check manifold curvature
        assert "average_curvature" in topo_potential["manifold_curvature"]
        assert "curvature_variance" in topo_potential["manifold_curvature"]
        assert "topological_depth" in topo_potential["manifold_curvature"]
        
        # Verify method calls
        mock_graph_service.repository.find_nodes_by_quality.assert_called_once_with("good", node_type="pattern")
        mock_graph_service.repository.find_relations_by_quality.assert_called_once_with(["good", "uncertain"])
