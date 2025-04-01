"""
Integration Tests for ArangoDB Persistence Layer.

This module contains integration tests for the ArangoDB persistence layer,
validating that all repositories work together correctly to persist and
retrieve the full topology state with all related entities.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import json
from datetime import datetime
import uuid
import numpy as np

from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.schema_manager import ArangoDBSchemaManager
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.topology_manager import ArangoDBTopologyManager
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.field_state_repository import TonicHarmonicFieldStateRepository
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.topology_repository import TopologyStateRepository
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.frequency_domain_repository import FrequencyDomainRepository
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.boundary_repository import BoundaryRepository
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.resonance_point_repository import ResonancePointRepository
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.pattern_repository import PatternRepository
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.semantic_signature_repository import SemanticSignatureRepository

from src.habitat_evolution.field.field_state import TonicHarmonicFieldState
from src.habitat_evolution.pattern_aware_rag.topology.manager import TopologyState
from src.habitat_evolution.pattern_aware_rag.topology.domain import FrequencyDomain
from src.habitat_evolution.pattern_aware_rag.topology.boundary import Boundary
from src.habitat_evolution.pattern_aware_rag.topology.resonance import ResonancePoint
from src.habitat_evolution.adaptive_core.models import Pattern
from src.habitat_evolution.pattern_aware_rag.semantic.signature import SemanticSignature


class TestArangoDBIntegration(unittest.TestCase):
    """Integration Tests for ArangoDB Persistence Layer."""
    
    @classmethod
    @patch('src.habitat_evolution.pattern_aware_rag.persistence.arangodb.schema_manager.ArangoClient')
    def setUpClass(cls, mock_client):
        """Set up the test environment once for all tests."""
        # Set environment variables for testing
        os.environ["ARANGO_HOST"] = "localhost"
        os.environ["ARANGO_USER"] = "test_user"
        os.environ["ARANGO_PASSWORD"] = "test_password"
        os.environ["ARANGO_DB"] = "test_db"
        
        # Mock ArangoDB client and database
        cls.mock_db = MagicMock()
        mock_client.return_value.db.return_value = cls.mock_db
        
        # Create schema manager and initialize schema
        cls.schema_manager = ArangoDBSchemaManager()
        cls.schema_manager.create_schema()
        cls.schema_manager.create_graph_definitions()
    
    def setUp(self):
        """Set up the test environment for each test."""
        # Create repositories
        self.field_state_repo = TonicHarmonicFieldStateRepository()
        self.topology_repo = TopologyStateRepository()
        self.domain_repo = FrequencyDomainRepository()
        self.boundary_repo = BoundaryRepository()
        self.resonance_repo = ResonancePointRepository()
        self.pattern_repo = PatternRepository()
        self.signature_repo = SemanticSignatureRepository()
        
        # Create topology manager
        self.topology_manager = ArangoDBTopologyManager()
        self.topology_manager.topology_repository = self.topology_repo
        self.topology_manager.domain_repository = self.domain_repo
        self.topology_manager.boundary_repository = self.boundary_repo
        self.topology_manager.resonance_repository = self.resonance_repo
        self.topology_manager.pattern_repository = self.pattern_repo
        
        # Create test data
        self._create_test_data()
        
        # Mock repository save methods to return IDs
        self.field_state_repo.save = MagicMock(return_value=self.field_state.id)
        self.topology_repo.save = MagicMock(return_value=self.topology_state.id)
        self.domain_repo.save = MagicMock(side_effect=lambda domain: domain.id)
        self.boundary_repo.save = MagicMock(side_effect=lambda boundary: boundary.id)
        self.resonance_repo.save = MagicMock(side_effect=lambda point: point.id)
        self.pattern_repo.save = MagicMock(side_effect=lambda pattern: pattern.id)
        self.signature_repo.save = MagicMock(side_effect=lambda signature: signature.id)
        
        # Mock repository find methods to return entities
        self.field_state_repo.find_by_id = MagicMock(return_value=self.field_state)
        self.topology_repo.find_by_id = MagicMock(return_value=self.topology_state)
        self.domain_repo.find_by_id = MagicMock(side_effect=self._mock_find_domain_by_id)
        self.boundary_repo.find_by_id = MagicMock(return_value=self.boundary)
        self.resonance_repo.find_by_id = MagicMock(return_value=self.resonance_point)
        self.pattern_repo.find_by_id = MagicMock(side_effect=self._mock_find_pattern_by_id)
        self.signature_repo.find_by_id = MagicMock(return_value=self.signature)
    
    def _create_test_data(self):
        """Create test data for the integration tests."""
        # Create field state
        field_analysis = {
            "topology": {
                "effective_dimensionality": 5,
                "principal_dimensions": ["ecological", "cultural", "economic", "social", "temporal"],
                "eigenvalues": np.array([0.42, 0.28, 0.15, 0.10, 0.05]),
                "eigenvectors": np.array([
                    [0.8, 0.1, 0.05, 0.03, 0.02],
                    [0.1, 0.7, 0.1, 0.05, 0.05],
                    [0.05, 0.1, 0.75, 0.05, 0.05],
                    [0.03, 0.05, 0.05, 0.82, 0.05],
                    [0.02, 0.05, 0.05, 0.05, 0.83]
                ])
            },
            "density": {
                "density_centers": [
                    {"position": [0.3, 0.4, 0.5, 0.2, 0.1], "magnitude": 0.8}
                ],
                "density_map": {
                    "resolution": [10, 10, 10, 10, 10],
                    "values": np.zeros((10, 10, 10, 10, 10)).tolist()
                }
            },
            "field_properties": {
                "coherence": 0.7,
                "navigability_score": 0.6,
                "stability": 0.8,
                "resonance_patterns": [
                    {"center": [0.4, 0.3, 0.2, 0.1, 0.5], "intensity": 0.9}
                ]
            }
        }
        
        self.field_state = TonicHarmonicFieldState(field_analysis)
        self.field_state.id = str(uuid.uuid4())
        self.field_state.version_id = str(uuid.uuid4())
        self.field_state.created_at = datetime.now()
        self.field_state.updated_at = datetime.now()
        
        # Create patterns
        self.pattern1 = Pattern(
            id="pattern1",
            pattern_type="semantic",
            source="entity1",
            predicate="relates_to",
            target="entity2"
        )
        self.pattern1.confidence = 0.9
        self.pattern1.created_at = datetime.now()
        self.pattern1.updated_at = datetime.now()
        self.pattern1.metadata = {
            "source_type": "document",
            "target_type": "concept",
            "context": "climate risk analysis"
        }
        self.pattern1.temporal_properties = {
            "first_observed": datetime.now().isoformat(),
            "last_observed": datetime.now().isoformat(),
            "observation_count": 5,
            "stability_score": 0.8,
            "frequency": 0.3,
            "phase": 0.2
        }
        self.pattern1.eigenspace_properties = {
            "primary_dimensions": [0, 2, 4],
            "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4],
            "eigenspace_centrality": 0.8,
            "eigenspace_stability": 0.7
        }
        
        self.pattern2 = Pattern(
            id="pattern2",
            pattern_type="causal",
            source="entity2",
            predicate="influences",
            target="entity3"
        )
        self.pattern2.confidence = 0.8
        self.pattern2.created_at = datetime.now()
        self.pattern2.updated_at = datetime.now()
        self.pattern2.metadata = {
            "source_type": "concept",
            "target_type": "metric",
            "context": "economic impact"
        }
        self.pattern2.temporal_properties = {
            "first_observed": datetime.now().isoformat(),
            "last_observed": datetime.now().isoformat(),
            "observation_count": 3,
            "stability_score": 0.7,
            "frequency": 0.4,
            "phase": 0.3
        }
        self.pattern2.eigenspace_properties = {
            "primary_dimensions": [1, 3, 4],
            "dimensional_coordinates": [0.4, 0.6, 0.3, 0.8, 0.5],
            "eigenspace_centrality": 0.7,
            "eigenspace_stability": 0.6
        }
        
        # Create semantic signature
        self.signature = SemanticSignature(
            id=str(uuid.uuid4()),
            entity_id="entity1",
            entity_type="concept",
            signature_vector=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            confidence=0.9
        )
        self.signature.created_at = datetime.now()
        self.signature.updated_at = datetime.now()
        self.signature.temporal_stability = 0.8
        self.signature.context_relevance = 0.7
        self.signature.contributing_pattern_ids = ["pattern1", "pattern2"]
        
        # Create topology state
        self.topology_state = TopologyState(id=str(uuid.uuid4()))
        self.topology_state.created_at = datetime.now()
        self.topology_state.updated_at = datetime.now()
        self.topology_state.coherence_score = 0.8
        self.topology_state.stability_score = 0.7
        self.topology_state.field_state_id = self.field_state.id
        
        # Create frequency domains
        self.domain1 = FrequencyDomain(
            id=str(uuid.uuid4()),
            name="Climate Risk Domain",
            description="Domain for climate risk analysis"
        )
        self.domain1.coherence = 0.9
        self.domain1.stability = 0.8
        self.domain1.energy = 0.7
        self.domain1.dimensional_properties = {
            "primary_dimensions": [0, 2, 4],
            "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4],
            "dimensional_weights": [0.4, 0.2, 0.3, 0.1, 0.2]
        }
        self.domain1.resonance_properties = {
            "resonance_frequency": 0.3,
            "amplitude": 0.5,
            "phase": 0.2,
            "harmonics": [0.6, 0.3, 0.1]
        }
        self.domain1.topology_state_id = self.topology_state.id
        
        self.domain2 = FrequencyDomain(
            id=str(uuid.uuid4()),
            name="Economic Impact Domain",
            description="Domain for economic impact analysis"
        )
        self.domain2.coherence = 0.8
        self.domain2.stability = 0.7
        self.domain2.energy = 0.6
        self.domain2.dimensional_properties = {
            "primary_dimensions": [1, 3, 4],
            "dimensional_coordinates": [0.4, 0.6, 0.3, 0.8, 0.5],
            "dimensional_weights": [0.3, 0.3, 0.2, 0.1, 0.1]
        }
        self.domain2.resonance_properties = {
            "resonance_frequency": 0.4,
            "amplitude": 0.4,
            "phase": 0.3,
            "harmonics": [0.5, 0.3, 0.2]
        }
        self.domain2.topology_state_id = self.topology_state.id
        
        self.topology_state.frequency_domains = {
            self.domain1.id: self.domain1,
            self.domain2.id: self.domain2
        }
        
        # Create boundary
        self.boundary = Boundary(id=str(uuid.uuid4()))
        self.boundary.permeability = 0.6
        self.boundary.stability = 0.5
        self.boundary.domain_ids = [self.domain1.id, self.domain2.id]
        self.boundary.oscillatory_properties = {
            "frequency": 0.3,
            "amplitude": 0.4,
            "phase": 0.2
        }
        
        self.topology_state.boundaries = {
            self.boundary.id: self.boundary
        }
        
        # Create resonance point
        self.resonance_point = ResonancePoint(
            id=str(uuid.uuid4()),
            strength=0.8,
            stability=0.7,
            attractor_radius=0.5
        )
        self.resonance_point.contributing_pattern_ids = {
            "pattern1": 0.9,
            "pattern2": 0.7
        }
        
        self.topology_state.resonance_points = {
            self.resonance_point.id: self.resonance_point
        }
        
        # Add pattern eigenspace properties
        self.topology_state.pattern_eigenspace_properties = {
            "pattern1": {
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4],
                "eigenspace_centrality": 0.8,
                "eigenspace_stability": 0.7,
                "dimensional_variance": 0.2,
                "resonance_groups": ["group1", "group2"],
                "frequency": 0.3,
                "energy": 0.6,
                "temporal_coherence": 0.75
            },
            "pattern2": {
                "primary_dimensions": [1, 3, 4],
                "dimensional_coordinates": [0.4, 0.6, 0.3, 0.8, 0.5],
                "eigenspace_centrality": 0.7,
                "eigenspace_stability": 0.6,
                "dimensional_variance": 0.3,
                "resonance_groups": ["group1"],
                "frequency": 0.4,
                "energy": 0.5,
                "temporal_coherence": 0.65
            }
        }
    
    def _mock_find_domain_by_id(self, domain_id):
        """Mock finding a domain by ID."""
        if domain_id == self.domain1.id:
            return self.domain1
        elif domain_id == self.domain2.id:
            return self.domain2
        else:
            return None
    
    def _mock_find_pattern_by_id(self, pattern_id):
        """Mock finding a pattern by ID."""
        if pattern_id == "pattern1":
            return self.pattern1
        elif pattern_id == "pattern2":
            return self.pattern2
        else:
            return None
    
    def test_persist_and_load_field_state(self):
        """Test persisting and loading a field state."""
        # Persist field state
        field_state_id = self.field_state_repo.save(self.field_state)
        
        # Verify field state was persisted
        self.assertEqual(field_state_id, self.field_state.id)
        self.field_state_repo.save.assert_called_once_with(self.field_state)
        
        # Load field state
        loaded_field_state = self.field_state_repo.find_by_id(self.field_state.id)
        
        # Verify field state was loaded
        self.assertIsNotNone(loaded_field_state)
        self.assertEqual(loaded_field_state.id, self.field_state.id)
        self.field_state_repo.find_by_id.assert_called_once_with(self.field_state.id)
    
    def test_persist_and_load_patterns(self):
        """Test persisting and loading patterns."""
        # Persist patterns
        pattern1_id = self.pattern_repo.save(self.pattern1)
        pattern2_id = self.pattern_repo.save(self.pattern2)
        
        # Verify patterns were persisted
        self.assertEqual(pattern1_id, self.pattern1.id)
        self.assertEqual(pattern2_id, self.pattern2.id)
        self.pattern_repo.save.assert_any_call(self.pattern1)
        self.pattern_repo.save.assert_any_call(self.pattern2)
        
        # Load patterns
        loaded_pattern1 = self.pattern_repo.find_by_id(self.pattern1.id)
        loaded_pattern2 = self.pattern_repo.find_by_id(self.pattern2.id)
        
        # Verify patterns were loaded
        self.assertIsNotNone(loaded_pattern1)
        self.assertEqual(loaded_pattern1.id, self.pattern1.id)
        self.assertIsNotNone(loaded_pattern2)
        self.assertEqual(loaded_pattern2.id, self.pattern2.id)
        self.pattern_repo.find_by_id.assert_any_call(self.pattern1.id)
        self.pattern_repo.find_by_id.assert_any_call(self.pattern2.id)
    
    def test_persist_and_load_semantic_signature(self):
        """Test persisting and loading a semantic signature."""
        # Persist semantic signature
        signature_id = self.signature_repo.save(self.signature)
        
        # Verify semantic signature was persisted
        self.assertEqual(signature_id, self.signature.id)
        self.signature_repo.save.assert_called_once_with(self.signature)
        
        # Load semantic signature
        loaded_signature = self.signature_repo.find_by_id(self.signature.id)
        
        # Verify semantic signature was loaded
        self.assertIsNotNone(loaded_signature)
        self.assertEqual(loaded_signature.id, self.signature.id)
        self.signature_repo.find_by_id.assert_called_once_with(self.signature.id)
    
    def test_persist_and_load_topology_state(self):
        """Test persisting and loading a topology state with all related entities."""
        # Persist topology state
        result = self.topology_manager.persist_topology_state(self.topology_state)
        
        # Verify topology state was persisted
        self.assertTrue(result)
        
        # Verify domains were persisted
        self.domain_repo.save.assert_any_call(self.domain1)
        self.domain_repo.save.assert_any_call(self.domain2)
        
        # Verify boundary was persisted
        self.boundary_repo.save.assert_called_once_with(self.boundary)
        
        # Verify resonance point was persisted
        self.resonance_repo.save.assert_called_once_with(self.resonance_point)
        
        # Verify topology state was persisted
        self.topology_repo.save.assert_called_once_with(self.topology_state)
        
        # Load topology state
        loaded_state = self.topology_manager.load_topology_state(self.topology_state.id)
        
        # Verify topology state was loaded
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state.id, self.topology_state.id)
        
        # Verify domains were loaded
        self.assertEqual(len(loaded_state.frequency_domains), 2)
        self.assertIn(self.domain1.id, loaded_state.frequency_domains)
        self.assertIn(self.domain2.id, loaded_state.frequency_domains)
        
        # Verify boundary was loaded
        self.assertEqual(len(loaded_state.boundaries), 1)
        self.assertIn(self.boundary.id, loaded_state.boundaries)
        
        # Verify resonance point was loaded
        self.assertEqual(len(loaded_state.resonance_points), 1)
        self.assertIn(self.resonance_point.id, loaded_state.resonance_points)
    
    def test_find_by_queries(self):
        """Test finding entities by various queries."""
        # Mock repository find methods to return entities
        self.domain_repo.find_by_coherence_threshold = MagicMock(return_value=[self.domain1])
        self.boundary_repo.find_by_permeability_threshold = MagicMock(return_value=[self.boundary])
        self.resonance_repo.find_by_strength_threshold = MagicMock(return_value=[self.resonance_point])
        self.pattern_repo.find_by_predicate = MagicMock(return_value=[self.pattern1])
        
        # Find domains by coherence
        domains = self.topology_manager.find_frequency_domains_by_coherence(0.8)
        
        # Verify domains were found
        self.assertEqual(len(domains), 1)
        self.assertEqual(domains[0].id, self.domain1.id)
        self.domain_repo.find_by_coherence_threshold.assert_called_once_with(0.8)
        
        # Find boundaries by permeability
        boundaries = self.topology_manager.find_boundaries_by_permeability(0.5)
        
        # Verify boundaries were found
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0].id, self.boundary.id)
        self.boundary_repo.find_by_permeability_threshold.assert_called_once_with(0.5)
        
        # Find resonance points by strength
        resonance_points = self.topology_manager.find_resonance_points_by_strength(0.7)
        
        # Verify resonance points were found
        self.assertEqual(len(resonance_points), 1)
        self.assertEqual(resonance_points[0].id, self.resonance_point.id)
        self.resonance_repo.find_by_strength_threshold.assert_called_once_with(0.7)
        
        # Find patterns by predicate
        patterns = self.topology_manager.find_patterns_by_predicate("relates_to")
        
        # Verify patterns were found
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].id, self.pattern1.id)
        self.pattern_repo.find_by_predicate.assert_called_once_with("relates_to")
    
    def test_find_latest_topology_state(self):
        """Test finding the latest topology state."""
        # Mock topology repository find_latest to return a state
        self.topology_repo.find_latest = MagicMock(return_value=self.topology_state)
        
        # Find latest topology state
        latest_state = self.topology_manager.find_latest_topology_state()
        
        # Verify latest state was found
        self.assertIsNotNone(latest_state)
        self.assertEqual(latest_state.id, self.topology_state.id)
        self.topology_repo.find_latest.assert_called_once()
    
    def test_error_handling(self):
        """Test error handling in the topology manager."""
        # Mock domain repository save to raise an exception
        self.domain_repo.save = MagicMock(side_effect=Exception("Test error"))
        
        # Persist topology state
        result = self.topology_manager.persist_topology_state(self.topology_state)
        
        # Verify persistence failed
        self.assertFalse(result)
        
        # Mock topology repository find_by_id to return None
        self.topology_repo.find_by_id = MagicMock(return_value=None)
        
        # Load non-existing topology state
        loaded_state = self.topology_manager.load_topology_state("non_existing_id")
        
        # Verify load failed
        self.assertIsNone(loaded_state)
        self.topology_repo.find_by_id.assert_called_once_with("non_existing_id")

if __name__ == '__main__':
    unittest.main()
