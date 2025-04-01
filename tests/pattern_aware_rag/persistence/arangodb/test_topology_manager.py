"""
Tests for the ArangoDB Topology Manager.

This module contains tests for the ArangoDB Topology Manager, which is responsible
for persisting topology states, frequency domains, boundaries, and resonance points to ArangoDB.
"""

import unittest
from unittest.mock import MagicMock, patch
import json
from datetime import datetime
import uuid

from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.topology_manager import ArangoDBTopologyManager
from src.habitat_evolution.pattern_aware_rag.topology.manager import TopologyState
from src.habitat_evolution.pattern_aware_rag.topology.domain import FrequencyDomain
from src.habitat_evolution.pattern_aware_rag.topology.boundary import Boundary
from src.habitat_evolution.pattern_aware_rag.topology.resonance import ResonancePoint
from src.habitat_evolution.adaptive_core.models import Pattern

class TestArangoDBTopologyManager(unittest.TestCase):
    """Tests for the ArangoDB Topology Manager."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create mock repositories
        self.mock_topology_repo = MagicMock()
        self.mock_domain_repo = MagicMock()
        self.mock_boundary_repo = MagicMock()
        self.mock_resonance_repo = MagicMock()
        self.mock_pattern_repo = MagicMock()
        
        # Create topology manager with mock repositories
        self.manager = ArangoDBTopologyManager()
        self.manager.topology_repository = self.mock_topology_repo
        self.manager.domain_repository = self.mock_domain_repo
        self.manager.boundary_repository = self.mock_boundary_repo
        self.manager.resonance_repository = self.mock_resonance_repo
        self.manager.pattern_repository = self.mock_pattern_repo
        
        # Create a sample topology state
        self.topology_state = TopologyState(id=str(uuid.uuid4()))
        self.topology_state.created_at = datetime.now()
        self.topology_state.updated_at = datetime.now()
        self.topology_state.coherence_score = 0.8
        self.topology_state.stability_score = 0.7
        self.topology_state.field_state_id = str(uuid.uuid4())
        
        # Add frequency domains
        self.domain1 = FrequencyDomain(id=str(uuid.uuid4()), name="Domain 1")
        self.domain1.coherence = 0.9
        self.domain1.stability = 0.8
        self.domain1.energy = 0.7
        self.domain1.dimensional_properties = {
            "primary_dimensions": [0, 2, 4],
            "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4]
        }
        
        self.domain2 = FrequencyDomain(id=str(uuid.uuid4()), name="Domain 2")
        self.domain2.coherence = 0.8
        self.domain2.stability = 0.7
        self.domain2.energy = 0.6
        self.domain2.dimensional_properties = {
            "primary_dimensions": [1, 3, 4],
            "dimensional_coordinates": [0.4, 0.6, 0.3, 0.8, 0.5]
        }
        
        self.topology_state.frequency_domains = {
            self.domain1.id: self.domain1,
            self.domain2.id: self.domain2
        }
        
        # Add boundaries
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
        
        # Add resonance points
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
        
        # Add learning windows
        self.topology_state.learning_windows = {
            "window1": {
                "duration_minutes": 10,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat()
            },
            "window2": {
                "duration_minutes": 20,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat()
            }
        }
    
    def test_persist_topology_state(self):
        """Test persisting a topology state to ArangoDB."""
        # Mock repository save methods to return IDs
        self.mock_domain_repo.save.side_effect = lambda domain: domain.id
        self.mock_boundary_repo.save.side_effect = lambda boundary: boundary.id
        self.mock_resonance_repo.save.side_effect = lambda point: point.id
        self.mock_pattern_repo.save.side_effect = lambda pattern: pattern.id
        self.mock_topology_repo.save.side_effect = lambda state: state.id
        
        # Mock pattern repository find_by_id to return None (patterns don't exist yet)
        self.mock_pattern_repo.find_by_id.return_value = None
        
        # Call the method
        result = self.manager.persist_topology_state(self.topology_state)
        
        # Verify the result
        self.assertTrue(result)
        
        # Verify frequency domains were saved
        self.assertEqual(self.mock_domain_repo.save.call_count, 2)
        
        # Verify boundaries were saved
        self.mock_boundary_repo.save.assert_called_once()
        
        # Verify resonance points were saved
        self.mock_resonance_repo.save.assert_called_once()
        
        # Verify patterns were saved
        self.assertEqual(self.mock_pattern_repo.find_by_id.call_count, 2)
        self.assertEqual(self.mock_pattern_repo.save.call_count, 2)
        
        # Verify topology state was saved
        self.mock_topology_repo.save.assert_called_once_with(self.topology_state)
    
    def test_persist_to_arango(self):
        """Test the persist_to_arango alias method."""
        # Mock persist_topology_state to return True
        self.manager.persist_topology_state = MagicMock(return_value=True)
        
        # Call the method
        result = self.manager.persist_to_arango(self.topology_state)
        
        # Verify the result
        self.assertTrue(result)
        
        # Verify persist_topology_state was called
        self.manager.persist_topology_state.assert_called_once_with(self.topology_state)
    
    def test_load_topology_state(self):
        """Test loading a topology state from ArangoDB."""
        # Mock repository find_by_id methods to return entities
        self.mock_topology_repo.find_by_id.return_value = self.topology_state
        self.mock_domain_repo.find_by_id.side_effect = lambda domain_id: (
            self.domain1 if domain_id == self.domain1.id else self.domain2
        )
        self.mock_boundary_repo.find_by_id.return_value = self.boundary
        self.mock_resonance_repo.find_by_id.return_value = self.resonance_point
        
        # Mock topology state properties
        self.topology_state.get = MagicMock(side_effect=lambda key, default=None: {
            "domain_ids": json.dumps([self.domain1.id, self.domain2.id]),
            "boundary_ids": json.dumps([self.boundary.id]),
            "resonance_point_ids": json.dumps([self.resonance_point.id])
        }.get(key, default))
        
        # Call the method
        result = self.manager.load_topology_state(self.topology_state.id)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertEqual(result.id, self.topology_state.id)
        
        # Verify repository find_by_id methods were called
        self.mock_topology_repo.find_by_id.assert_called_once_with(self.topology_state.id)
        self.mock_domain_repo.find_by_id.assert_any_call(self.domain1.id)
        self.mock_domain_repo.find_by_id.assert_any_call(self.domain2.id)
        self.mock_boundary_repo.find_by_id.assert_called_once_with(self.boundary.id)
        self.mock_resonance_repo.find_by_id.assert_called_once_with(self.resonance_point.id)
    
    def test_find_latest_topology_state(self):
        """Test finding the latest topology state in ArangoDB."""
        # Mock topology repository find_latest to return a state
        self.mock_topology_repo.find_latest.return_value = self.topology_state
        
        # Mock load_topology_state to return the full state
        self.manager.load_topology_state = MagicMock(return_value=self.topology_state)
        
        # Call the method
        result = self.manager.find_latest_topology_state()
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertEqual(result.id, self.topology_state.id)
        
        # Verify repository find_latest was called
        self.mock_topology_repo.find_latest.assert_called_once()
        
        # Verify load_topology_state was called
        self.manager.load_topology_state.assert_called_once_with(self.topology_state.id)
    
    def test_find_frequency_domains_by_coherence(self):
        """Test finding frequency domains by coherence threshold."""
        # Mock domain repository find_by_coherence_threshold to return domains
        self.mock_domain_repo.find_by_coherence_threshold.return_value = [self.domain1, self.domain2]
        
        # Call the method
        result = self.manager.find_frequency_domains_by_coherence(0.7)
        
        # Verify the result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, self.domain1.id)
        self.assertEqual(result[1].id, self.domain2.id)
        
        # Verify repository find_by_coherence_threshold was called
        self.mock_domain_repo.find_by_coherence_threshold.assert_called_once_with(0.7)
    
    def test_find_boundaries_by_permeability(self):
        """Test finding boundaries by permeability threshold."""
        # Mock boundary repository find_by_permeability_threshold to return boundaries
        self.mock_boundary_repo.find_by_permeability_threshold.return_value = [self.boundary]
        
        # Call the method
        result = self.manager.find_boundaries_by_permeability(0.5)
        
        # Verify the result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, self.boundary.id)
        
        # Verify repository find_by_permeability_threshold was called
        self.mock_boundary_repo.find_by_permeability_threshold.assert_called_once_with(0.5)
    
    def test_find_resonance_points_by_strength(self):
        """Test finding resonance points by strength threshold."""
        # Mock resonance repository find_by_strength_threshold to return points
        self.mock_resonance_repo.find_by_strength_threshold.return_value = [self.resonance_point]
        
        # Call the method
        result = self.manager.find_resonance_points_by_strength(0.7)
        
        # Verify the result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, self.resonance_point.id)
        
        # Verify repository find_by_strength_threshold was called
        self.mock_resonance_repo.find_by_strength_threshold.assert_called_once_with(0.7)
    
    def test_find_patterns_by_predicate(self):
        """Test finding patterns by predicate."""
        # Create sample patterns
        pattern1 = Pattern(id="pattern1", pattern_type="test", source="entity1", predicate="relates_to", target="entity2")
        pattern2 = Pattern(id="pattern2", pattern_type="test", source="entity2", predicate="influences", target="entity3")
        
        # Mock pattern repository find_by_predicate to return patterns
        self.mock_pattern_repo.find_by_predicate.return_value = [pattern1]
        
        # Call the method
        result = self.manager.find_patterns_by_predicate("relates_to")
        
        # Verify the result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "pattern1")
        self.assertEqual(result[0].predicate, "relates_to")
        
        # Verify repository find_by_predicate was called
        self.mock_pattern_repo.find_by_predicate.assert_called_once_with("relates_to")
    
    def test_persist_topology_state_error(self):
        """Test persisting a topology state with an error."""
        # Mock domain repository save to raise an exception
        self.mock_domain_repo.save.side_effect = Exception("Test error")
        
        # Call the method
        result = self.manager.persist_topology_state(self.topology_state)
        
        # Verify the result
        self.assertFalse(result)
    
    def test_load_topology_state_not_found(self):
        """Test loading a non-existing topology state."""
        # Mock topology repository find_by_id to return None
        self.mock_topology_repo.find_by_id.return_value = None
        
        # Call the method
        result = self.manager.load_topology_state("non_existing_id")
        
        # Verify the result
        self.assertIsNone(result)
        
        # Verify repository find_by_id was called
        self.mock_topology_repo.find_by_id.assert_called_once_with("non_existing_id")
    
    def test_find_latest_topology_state_none(self):
        """Test finding the latest topology state when none exists."""
        # Mock topology repository find_latest to return None
        self.mock_topology_repo.find_latest.return_value = None
        
        # Call the method
        result = self.manager.find_latest_topology_state()
        
        # Verify the result
        self.assertIsNone(result)
        
        # Verify repository find_latest was called
        self.mock_topology_repo.find_latest.assert_called_once()

if __name__ == '__main__':
    unittest.main()
