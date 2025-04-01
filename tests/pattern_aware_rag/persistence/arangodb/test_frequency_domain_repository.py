"""
Tests for the Frequency Domain Repository.

This module contains tests for the Frequency Domain Repository, which is responsible
for persisting frequency domains to ArangoDB.
"""

import unittest
from unittest.mock import MagicMock, patch
import json
from datetime import datetime
import uuid

from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.frequency_domain_repository import FrequencyDomainRepository
from src.habitat_evolution.pattern_aware_rag.topology.domain import FrequencyDomain

class TestFrequencyDomainRepository(unittest.TestCase):
    """Tests for the Frequency Domain Repository."""
    
    @patch('src.habitat_evolution.pattern_aware_rag.persistence.arangodb.frequency_domain_repository.ArangoDBConnectionManager')
    def setUp(self, mock_connection_manager):
        """Set up the test environment."""
        # Mock ArangoDB connection manager
        self.mock_db = MagicMock()
        mock_connection_manager.return_value.get_db.return_value = self.mock_db
        
        # Create repository
        self.repository = FrequencyDomainRepository()
        
        # Mock collection
        self.mock_collection = MagicMock()
        self.mock_db.collection.return_value = self.mock_collection
        
        # Create sample frequency domains
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
        self.domain1.topology_state_id = str(uuid.uuid4())
        self.domain1.created_at = datetime.now()
        self.domain1.updated_at = datetime.now()
        self.domain1.boundary_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        self.domain1.resonance_point_ids = [str(uuid.uuid4())]
        
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
        self.domain2.topology_state_id = str(uuid.uuid4())
        self.domain2.created_at = datetime.now()
        self.domain2.updated_at = datetime.now()
        self.domain2.boundary_ids = [str(uuid.uuid4())]
        self.domain2.resonance_point_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    
    def test_to_document_properties(self):
        """Test converting a frequency domain to document properties."""
        # Call the method
        properties = self.repository._to_document_properties(self.domain1)
        
        # Verify core properties
        self.assertEqual(properties["id"], self.domain1.id)
        self.assertEqual(properties["name"], self.domain1.name)
        self.assertEqual(properties["description"], self.domain1.description)
        self.assertEqual(properties["topology_state_id"], self.domain1.topology_state_id)
        
        # Verify coherence, stability, and energy
        self.assertAlmostEqual(properties["coherence"], self.domain1.coherence)
        self.assertAlmostEqual(properties["stability"], self.domain1.stability)
        self.assertAlmostEqual(properties["energy"], self.domain1.energy)
        
        # Verify dimensional properties
        dimensional_properties = json.loads(properties["dimensional_properties"])
        self.assertEqual(dimensional_properties["primary_dimensions"], [0, 2, 4])
        self.assertEqual(len(dimensional_properties["dimensional_coordinates"]), 5)
        self.assertEqual(len(dimensional_properties["dimensional_weights"]), 5)
        
        # Verify resonance properties
        resonance_properties = json.loads(properties["resonance_properties"])
        self.assertAlmostEqual(resonance_properties["resonance_frequency"], 0.3)
        self.assertAlmostEqual(resonance_properties["amplitude"], 0.5)
        self.assertAlmostEqual(resonance_properties["phase"], 0.2)
        self.assertEqual(len(resonance_properties["harmonics"]), 3)
        
        # Verify boundary and resonance point IDs
        boundary_ids = json.loads(properties["boundary_ids"])
        self.assertEqual(len(boundary_ids), 2)
        
        resonance_point_ids = json.loads(properties["resonance_point_ids"])
        self.assertEqual(len(resonance_point_ids), 1)
    
    def test_dict_to_entity(self):
        """Test converting document properties to a frequency domain."""
        # Create document properties
        properties = {
            "id": "test_id",
            "name": "Test Domain",
            "description": "Test description",
            "topology_state_id": "topology_id",
            "coherence": 0.9,
            "stability": 0.8,
            "energy": 0.7,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "dimensional_properties": json.dumps({
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4],
                "dimensional_weights": [0.4, 0.2, 0.3, 0.1, 0.2]
            }),
            "resonance_properties": json.dumps({
                "resonance_frequency": 0.3,
                "amplitude": 0.5,
                "phase": 0.2,
                "harmonics": [0.6, 0.3, 0.1]
            }),
            "boundary_ids": json.dumps(["boundary1", "boundary2"]),
            "resonance_point_ids": json.dumps(["resonance1"])
        }
        
        # Call the method
        domain = self.repository._dict_to_entity(properties)
        
        # Verify core properties
        self.assertEqual(domain.id, "test_id")
        self.assertEqual(domain.name, "Test Domain")
        self.assertEqual(domain.description, "Test description")
        self.assertEqual(domain.topology_state_id, "topology_id")
        
        # Verify coherence, stability, and energy
        self.assertAlmostEqual(domain.coherence, 0.9)
        self.assertAlmostEqual(domain.stability, 0.8)
        self.assertAlmostEqual(domain.energy, 0.7)
        
        # Verify dimensional properties
        self.assertEqual(domain.dimensional_properties["primary_dimensions"], [0, 2, 4])
        self.assertEqual(len(domain.dimensional_properties["dimensional_coordinates"]), 5)
        self.assertEqual(len(domain.dimensional_properties["dimensional_weights"]), 5)
        
        # Verify resonance properties
        self.assertAlmostEqual(domain.resonance_properties["resonance_frequency"], 0.3)
        self.assertAlmostEqual(domain.resonance_properties["amplitude"], 0.5)
        self.assertAlmostEqual(domain.resonance_properties["phase"], 0.2)
        self.assertEqual(len(domain.resonance_properties["harmonics"]), 3)
        
        # Verify boundary and resonance point IDs
        self.assertEqual(len(domain.boundary_ids), 2)
        self.assertEqual(domain.boundary_ids[0], "boundary1")
        self.assertEqual(domain.boundary_ids[1], "boundary2")
        
        self.assertEqual(len(domain.resonance_point_ids), 1)
        self.assertEqual(domain.resonance_point_ids[0], "resonance1")
    
    def test_save_new_domain(self):
        """Test saving a new frequency domain."""
        # Mock collection.get to return None (document doesn't exist)
        self.mock_collection.get.side_effect = Exception("Document not found")
        
        # Call the method
        result = self.repository.save(self.domain1)
        
        # Verify the document was inserted
        self.mock_collection.insert.assert_called_once()
        self.assertEqual(result, self.domain1.id)
    
    def test_save_existing_domain(self):
        """Test saving an existing frequency domain."""
        # Mock collection.get to return a document (document exists)
        self.mock_collection.get.return_value = {"_key": self.domain1.id}
        
        # Call the method
        result = self.repository.save(self.domain1)
        
        # Verify the document was updated
        self.mock_collection.update.assert_called_once()
        self.assertEqual(result, self.domain1.id)
    
    def test_find_by_id_existing(self):
        """Test finding an existing frequency domain by ID."""
        # Mock collection.get to return a document
        self.mock_collection.get.return_value = {
            "id": "test_id",
            "name": "Test Domain",
            "description": "Test description",
            "topology_state_id": "topology_id",
            "coherence": 0.9,
            "stability": 0.8,
            "energy": 0.7,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "dimensional_properties": json.dumps({
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4],
                "dimensional_weights": [0.4, 0.2, 0.3, 0.1, 0.2]
            }),
            "resonance_properties": json.dumps({
                "resonance_frequency": 0.3,
                "amplitude": 0.5,
                "phase": 0.2,
                "harmonics": [0.6, 0.3, 0.1]
            }),
            "boundary_ids": json.dumps(["boundary1", "boundary2"]),
            "resonance_point_ids": json.dumps(["resonance1"])
        }
        
        # Call the method
        domain = self.repository.find_by_id("test_id")
        
        # Verify the result
        self.assertIsNotNone(domain)
        self.assertEqual(domain.id, "test_id")
        self.assertEqual(domain.name, "Test Domain")
        self.assertEqual(domain.description, "Test description")
    
    def test_find_by_id_not_existing(self):
        """Test finding a non-existing frequency domain by ID."""
        # Mock collection.get to return None
        self.mock_collection.get.return_value = None
        
        # Call the method
        domain = self.repository.find_by_id("non_existing_id")
        
        # Verify the result
        self.assertIsNone(domain)
    
    def test_find_by_topology_state_id(self):
        """Test finding frequency domains by topology state ID."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "test_id",
            "name": "Test Domain",
            "description": "Test description",
            "topology_state_id": "topology_id",
            "coherence": 0.9,
            "stability": 0.8,
            "energy": 0.7,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "dimensional_properties": json.dumps({
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4],
                "dimensional_weights": [0.4, 0.2, 0.3, 0.1, 0.2]
            }),
            "resonance_properties": json.dumps({
                "resonance_frequency": 0.3,
                "amplitude": 0.5,
                "phase": 0.2,
                "harmonics": [0.6, 0.3, 0.1]
            }),
            "boundary_ids": json.dumps(["boundary1", "boundary2"]),
            "resonance_point_ids": json.dumps(["resonance1"])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        domains = self.repository.find_by_topology_state_id("topology_id")
        
        # Verify the result
        self.assertEqual(len(domains), 1)
        self.assertEqual(domains[0].id, "test_id")
        self.assertEqual(domains[0].topology_state_id, "topology_id")
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_name(self):
        """Test finding frequency domains by name."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "test_id",
            "name": "Test Domain",
            "description": "Test description",
            "topology_state_id": "topology_id",
            "coherence": 0.9,
            "stability": 0.8,
            "energy": 0.7,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "dimensional_properties": json.dumps({
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4],
                "dimensional_weights": [0.4, 0.2, 0.3, 0.1, 0.2]
            }),
            "resonance_properties": json.dumps({
                "resonance_frequency": 0.3,
                "amplitude": 0.5,
                "phase": 0.2,
                "harmonics": [0.6, 0.3, 0.1]
            }),
            "boundary_ids": json.dumps(["boundary1", "boundary2"]),
            "resonance_point_ids": json.dumps(["resonance1"])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        domains = self.repository.find_by_name("Test Domain")
        
        # Verify the result
        self.assertEqual(len(domains), 1)
        self.assertEqual(domains[0].id, "test_id")
        self.assertEqual(domains[0].name, "Test Domain")
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_coherence_threshold(self):
        """Test finding frequency domains by coherence threshold."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "test_id",
            "name": "Test Domain",
            "description": "Test description",
            "topology_state_id": "topology_id",
            "coherence": 0.9,
            "stability": 0.8,
            "energy": 0.7,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "dimensional_properties": json.dumps({
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4],
                "dimensional_weights": [0.4, 0.2, 0.3, 0.1, 0.2]
            }),
            "resonance_properties": json.dumps({
                "resonance_frequency": 0.3,
                "amplitude": 0.5,
                "phase": 0.2,
                "harmonics": [0.6, 0.3, 0.1]
            }),
            "boundary_ids": json.dumps(["boundary1", "boundary2"]),
            "resonance_point_ids": json.dumps(["resonance1"])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        domains = self.repository.find_by_coherence_threshold(0.8)
        
        # Verify the result
        self.assertEqual(len(domains), 1)
        self.assertEqual(domains[0].id, "test_id")
        self.assertGreaterEqual(domains[0].coherence, 0.8)
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_resonance_frequency_range(self):
        """Test finding frequency domains by resonance frequency range."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "test_id",
            "name": "Test Domain",
            "description": "Test description",
            "topology_state_id": "topology_id",
            "coherence": 0.9,
            "stability": 0.8,
            "energy": 0.7,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "dimensional_properties": json.dumps({
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4],
                "dimensional_weights": [0.4, 0.2, 0.3, 0.1, 0.2]
            }),
            "resonance_properties": json.dumps({
                "resonance_frequency": 0.3,
                "amplitude": 0.5,
                "phase": 0.2,
                "harmonics": [0.6, 0.3, 0.1]
            }),
            "boundary_ids": json.dumps(["boundary1", "boundary2"]),
            "resonance_point_ids": json.dumps(["resonance1"])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        domains = self.repository.find_by_resonance_frequency_range(0.2, 0.4)
        
        # Verify the result
        self.assertEqual(len(domains), 1)
        self.assertEqual(domains[0].id, "test_id")
        self.assertGreaterEqual(domains[0].resonance_properties["resonance_frequency"], 0.2)
        self.assertLessEqual(domains[0].resonance_properties["resonance_frequency"], 0.4)
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_primary_dimension(self):
        """Test finding frequency domains by primary dimension."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "test_id",
            "name": "Test Domain",
            "description": "Test description",
            "topology_state_id": "topology_id",
            "coherence": 0.9,
            "stability": 0.8,
            "energy": 0.7,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "dimensional_properties": json.dumps({
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4],
                "dimensional_weights": [0.4, 0.2, 0.3, 0.1, 0.2]
            }),
            "resonance_properties": json.dumps({
                "resonance_frequency": 0.3,
                "amplitude": 0.5,
                "phase": 0.2,
                "harmonics": [0.6, 0.3, 0.1]
            }),
            "boundary_ids": json.dumps(["boundary1", "boundary2"]),
            "resonance_point_ids": json.dumps(["resonance1"])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        domains = self.repository.find_by_primary_dimension(2)
        
        # Verify the result
        self.assertEqual(len(domains), 1)
        self.assertEqual(domains[0].id, "test_id")
        self.assertIn(2, domains[0].dimensional_properties["primary_dimensions"])
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_domains_with_boundary(self):
        """Test finding frequency domains with a specific boundary."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "test_id",
            "name": "Test Domain",
            "description": "Test description",
            "topology_state_id": "topology_id",
            "coherence": 0.9,
            "stability": 0.8,
            "energy": 0.7,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "dimensional_properties": json.dumps({
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4],
                "dimensional_weights": [0.4, 0.2, 0.3, 0.1, 0.2]
            }),
            "resonance_properties": json.dumps({
                "resonance_frequency": 0.3,
                "amplitude": 0.5,
                "phase": 0.2,
                "harmonics": [0.6, 0.3, 0.1]
            }),
            "boundary_ids": json.dumps(["boundary1", "boundary2"]),
            "resonance_point_ids": json.dumps(["resonance1"])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        domains = self.repository.find_domains_with_boundary("boundary1")
        
        # Verify the result
        self.assertEqual(len(domains), 1)
        self.assertEqual(domains[0].id, "test_id")
        self.assertIn("boundary1", domains[0].boundary_ids)
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_domains_with_resonance_point(self):
        """Test finding frequency domains with a specific resonance point."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "test_id",
            "name": "Test Domain",
            "description": "Test description",
            "topology_state_id": "topology_id",
            "coherence": 0.9,
            "stability": 0.8,
            "energy": 0.7,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "dimensional_properties": json.dumps({
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4],
                "dimensional_weights": [0.4, 0.2, 0.3, 0.1, 0.2]
            }),
            "resonance_properties": json.dumps({
                "resonance_frequency": 0.3,
                "amplitude": 0.5,
                "phase": 0.2,
                "harmonics": [0.6, 0.3, 0.1]
            }),
            "boundary_ids": json.dumps(["boundary1", "boundary2"]),
            "resonance_point_ids": json.dumps(["resonance1"])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        domains = self.repository.find_domains_with_resonance_point("resonance1")
        
        # Verify the result
        self.assertEqual(len(domains), 1)
        self.assertEqual(domains[0].id, "test_id")
        self.assertIn("resonance1", domains[0].resonance_point_ids)
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()

if __name__ == '__main__':
    unittest.main()
