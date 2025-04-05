"""
Tests for the Concept-Predicate-Syntax Model.

This module tests the co-evolutionary model of language where concepts and
predicates co-evolve through their interactions, with syntax emerging as
momentary intentionality.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.habitat_evolution.field.emergence.concept_predicate_syntax_model import ConceptPredicateSyntaxModel
from src.tests.adaptive_core.persistence.arangodb.test_state_models import PatternState
from src.habitat_evolution.adaptive_core.persistence.services.graph_service import GraphService
from src.habitat_evolution.field.persistence.semantic_potential_calculator import SemanticPotentialCalculator


class TestConceptPredicateSyntaxModel:
    """Tests for the ConceptPredicateSyntaxModel class."""
    
    @pytest.fixture
    def mock_graph_service(self):
        """Create a mock graph service."""
        mock = MagicMock(spec=GraphService)
        mock.repository = MagicMock()
        mock.repository.find_nodes_by_quality = MagicMock()
        mock.repository.find_relations_by_quality = MagicMock()
        mock.repository.find_node_by_id = MagicMock()
        mock.repository.find_quality_transitions_by_node_id = MagicMock()
        return mock
    
    @pytest.fixture
    def mock_potential_calculator(self):
        """Create a mock semantic potential calculator."""
        mock = MagicMock(spec=SemanticPotentialCalculator)
        mock.calculate_field_potential = AsyncMock()
        mock.calculate_topological_potential = AsyncMock()
        return mock
    
    @pytest.fixture
    def model(self, mock_graph_service, mock_potential_calculator):
        """Create a ConceptPredicateSyntaxModel instance."""
        return ConceptPredicateSyntaxModel(mock_graph_service, mock_potential_calculator)
    
    @pytest.mark.asyncio
    async def test_map_co_resonance_field(self, model, mock_graph_service):
        """Test mapping the co-resonance field."""
        # Create test patterns
        concept1 = PatternState(
            id="concept1",
            content="Test Concept 1",
            metadata={"type": "entity"},
            timestamp=datetime.now(),
            confidence=0.8
        )
        
        concept2 = PatternState(
            id="concept2",
            content="Test Concept 2",
            metadata={"type": "property"},
            timestamp=datetime.now(),
            confidence=0.7
        )
        
        # Create test predicates
        predicate1 = {
            "id": "predicate1",
            "type": "has_property",
            "source_id": "concept1",
            "target_id": "concept2",
            "weight": 0.9,
            "attributes": {}
        }
        
        # Mock the methods
        async def mock_get_stabilized_patterns(window_id):
            return [concept1, concept2]
            
        async def mock_get_transformative_relationships(window_id):
            return [predicate1]
            
        async def mock_calculate_co_resonance(concept, predicate):
            return {
                "concept_id": concept.id,
                "predicate_id": predicate["id"],
                "strength": 0.8,
                "compatibility": 0.8,
                "mutual_information": 0.7,
                "concept_influence": 0.6,
                "predicate_influence": 0.5,
                "co_occurrences": 1
            }
        
        # Assign the mock methods
        model._get_stabilized_patterns = mock_get_stabilized_patterns
        model._get_transformative_relationships = mock_get_transformative_relationships
        model._calculate_co_resonance = mock_calculate_co_resonance
        
        # Call the method
        field = await model.map_co_resonance_field()
        
        # Check the results
        assert "concepts" in field
        assert "predicates" in field
        assert "resonances" in field
        assert len(field["resonances"]) == 2  # One for each concept-predicate pair
        assert field["field_strength"] == 0.8
        
    @pytest.mark.asyncio
    async def test_detect_intentionality_vectors(self, model, mock_potential_calculator):
        """Test detecting intentionality vectors."""
        # Mock the potential calculator methods
        mock_potential_calculator.calculate_field_potential.return_value = {
            "avg_evolutionary_potential": 0.7,
            "avg_constructive_dissonance": 0.6,
            "gradient_field": {
                "magnitude": 0.5,
                "direction": "increasing",
                "uniformity": 0.8
            },
            "pattern_count": 3,
            "window_id": "test-window",
            "timestamp": datetime.now().isoformat()
        }
        
        mock_potential_calculator.calculate_topological_potential.return_value = {
            "connectivity": {
                "density": 0.6,
                "clustering": 0.7,
                "path_efficiency": 0.8
            },
            "centrality": {
                "centralization": 0.4,
                "heterogeneity": 0.3
            },
            "temporal_stability": {
                "persistence": 0.8,
                "evolution_rate": 0.3,
                "temporal_coherence": 0.75
            },
            "manifold_curvature": {
                "average_curvature": 0.5,
                "curvature_variance": 0.2,
                "topological_depth": 0.6
            },
            "topological_energy": 0.65,
            "window_id": "test-window",
            "timestamp": datetime.now().isoformat()
        }
        
        # Call the method
        vectors = await model.detect_intentionality_vectors()
        
        # Check the results
        assert "primary_direction" in vectors
        assert vectors["primary_direction"] == "increasing"
        assert "magnitude" in vectors
        assert "focus" in vectors
        assert "stability" in vectors
        assert "composite_vector" in vectors
        
    @pytest.mark.asyncio
    async def test_generate_co_evolutionary_expression(self, model, mock_graph_service):
        """Test generating a co-evolutionary expression."""
        # Create test concepts
        concept1 = {
            "id": "concept1",
            "content": "Test Concept 1",
            "confidence": 0.8,
            "attributes": {"type": "entity"}
        }
        
        concept2 = {
            "id": "concept2",
            "content": "Test Concept 2",
            "confidence": 0.7,
            "attributes": {"type": "property"}
        }
        
        # Create test predicates
        predicate1 = {
            "id": "predicate1",
            "type": "has_property",
            "strength": 0.9,
            "connects": ["concept1", "concept2"]
        }
        
        # Create test intentionality
        intentionality = {
            "primary_direction": "increasing",
            "magnitude": 0.6,
            "focus": 0.8,
            "stability": 0.7,
            "composite_vector": {
                "direction": "increasing",
                "strength": 0.48,
                "stability": 0.7
            }
        }
        
        # Mock the methods
        async def mock_map_co_resonance_field(window_id):
            return {
                "concepts": {"concept1": {}, "concept2": {}},
                "predicates": {"predicate1": {}},
                "resonances": [
                    {
                        "concept_id": "concept1",
                        "predicate_id": "predicate1",
                        "strength": 0.8
                    },
                    {
                        "concept_id": "concept2",
                        "predicate_id": "predicate1",
                        "strength": 0.7
                    }
                ],
                "clusters": [
                    {
                        "concept_id": "concept1",
                        "predicate_ids": ["predicate1"],
                        "strength": 0.8
                    }
                ],
                "field_strength": 0.75
            }
            
        async def mock_select_concepts(seed_concepts, intentionality, field):
            return [concept1, concept2]
            
        async def mock_select_co_resonant_predicates(concepts, field):
            return [predicate1]
        
        # Assign the mock methods
        model.map_co_resonance_field = mock_map_co_resonance_field
        model._select_concepts = mock_select_concepts
        model._select_co_resonant_predicates = mock_select_co_resonant_predicates
        
        # Call the method
        expression = await model.generate_co_evolutionary_expression(
            seed_concepts=["concept1"], intentionality=intentionality
        )
        
        # Check the results
        assert "expression" in expression
        assert "components" in expression
        assert "intentionality" in expression
        assert "coherence" in expression
        assert expression["expression"] == "Test Concept 1 has_property Test Concept 2"
        assert len(expression["components"]) == 3  # Two concepts and one predicate
        
    @pytest.mark.asyncio
    async def test_calculate_co_resonance(self, model):
        """Test calculating co-resonance between concept and predicate."""
        # Create test concept and predicate
        concept = PatternState(
            id="concept1",
            content="Test Concept",
            metadata={"type": "entity"},
            timestamp=datetime.now(),
            confidence=0.8
        )
        
        predicate = {
            "id": "predicate1",
            "type": "has_property",
            "source_id": "concept1",
            "target_id": "concept2",
            "weight": 0.9,
            "attributes": {}
        }
        
        # Mock the methods
        async def mock_get_historical_co_occurrences(concept, predicate):
            return 1
            
        async def mock_calculate_evolutionary_influence(source, target):
            return 0.6
        
        # Assign the mock methods
        model._get_historical_co_occurrences = mock_get_historical_co_occurrences
        model._calculate_evolutionary_influence = mock_calculate_evolutionary_influence
        
        # Call the method
        resonance = await model._calculate_co_resonance(concept, predicate)
        
        # Check the results
        assert resonance["concept_id"] == "concept1"
        assert resonance["predicate_id"] == "predicate1"
        assert "strength" in resonance
        assert "compatibility" in resonance
        assert "mutual_information" in resonance
        assert "concept_influence" in resonance
        assert "predicate_influence" in resonance
        assert resonance["co_occurrences"] == 1
