"""Test suite for Ghost toolbar integration."""

import pytest
from unittest.mock import Mock, patch
import json
from ...visualization.test_visualization import TestPatternVisualizer
from ...api.graph_service import process_text

class TestGhostToolbar:
    @pytest.fixture
    def mock_selection(self):
        """Mock text selection from Ghost editor."""
        return """
        As a result of climate change, the probability of the historical 100-
        year rainfall event, auseful indicator of flood risk, is expected to
        increase slightly by mid-century and be aboutfive times more
        likely by late-century. The likelihood of extreme drought events
        will also increase, from an 8.5% annual likelihood in today's
        climate to 13% and 26% by mid- andlate-century, respectively.
        Linked to this increase in drought stress, the annual
        averagenumber of high-danger wildfire days is expected to
        increase 44% by mid-century and 54%by late-century.
        """

    @pytest.fixture
    def mock_neo4j_session(self):
        """Mock Neo4j session for testing."""
        session = Mock()
        session.run = Mock(return_value=[{
            'n.hazard_type': 'drought',
            'n.temporal_horizon': 'current',
            'n.probability': 0.085,
            'n.spatial_context': json.dumps({'location': "Martha's Vineyard"})
        }])
        return session

    async def test_text_processing(self, mock_selection):
        """Test that selected text is properly processed into patterns."""
        response = await process_text({"text": mock_selection})
        
        # Verify response structure
        assert "graph_image" in response
        assert "nodes" in response
        assert "edges" in response
        
        # Verify pattern detection
        hazard_types = {node["hazard_type"] for node in response["nodes"]}
        expected_types = {"extreme_precipitation", "drought", "wildfire"}
        assert hazard_types == expected_types

    def test_pattern_visualization(self, mock_selection, mock_neo4j_session):
        """Test pattern visualization generation."""
        visualizer = TestPatternVisualizer()
        patterns = visualizer.discover_patterns(mock_selection)
        
        # Verify pattern creation
        assert len(patterns) > 0
        
        # Verify probability assignment
        drought_pattern = next(p for p in patterns if p.hazard_type == "drought")
        assert drought_pattern.probability == 0.085
        
        # Verify relationship creation
        for pattern in patterns:
            assert len(pattern.relationships) > 0

    @patch('networkx.spring_layout')
    def test_graph_layout(self, mock_spring_layout, mock_selection):
        """Test graph layout generation."""
        visualizer = TestPatternVisualizer()
        patterns = visualizer.discover_patterns(mock_selection)
        
        # Create test graph
        graph = visualizer.create_graph(patterns)
        
        # Verify graph structure
        assert len(graph.nodes()) == len(patterns)
        assert len(graph.edges()) > 0
        
        # Verify node attributes
        for node in graph.nodes():
            attrs = graph.nodes[node]
            assert "hazard_type" in attrs
            assert "probability" in attrs
            assert "temporal_horizon" in attrs

    def test_field_state_integration(self, mock_selection, mock_neo4j_session):
        """Test field state node creation and integration."""
        visualizer = TestPatternVisualizer()
        patterns = visualizer.discover_patterns(mock_selection)
        
        # Export to Neo4j
        visualizer.export_to_neo4j(patterns)
        
        # Verify field state creation
        calls = mock_neo4j_session.run.call_args_list
        field_state_calls = [
            call for call in calls 
            if "FieldState" in str(call)
        ]
        assert len(field_state_calls) > 0
        
        # Verify pattern relationships to field state
        pattern_field_calls = [
            call for call in calls 
            if "EXISTS_IN" in str(call)
        ]
        assert len(pattern_field_calls) == len(patterns)
