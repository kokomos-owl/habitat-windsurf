"""
Tests for the Predicate Relationship Repository.

This module contains tests for the Predicate Relationship Repository, which is responsible
for persisting and retrieving predicate relationships between actants in the Habitat Evolution system.
"""

import unittest
from unittest.mock import MagicMock, patch
import json
from datetime import datetime
import uuid

from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.predicate_relationship_repository import PredicateRelationshipRepository

class TestPredicateRelationshipRepository(unittest.TestCase):
    """Tests for the Predicate Relationship Repository."""
    
    @patch('src.habitat_evolution.pattern_aware_rag.persistence.arangodb.predicate_relationship_repository.ArangoDBConnectionManager')
    def setUp(self, mock_connection_manager):
        """Set up the test environment."""
        # Mock ArangoDB connection manager
        self.mock_db = MagicMock()
        mock_connection_manager.return_value.get_db.return_value = self.mock_db
        
        # Create repository
        self.repository = PredicateRelationshipRepository()
        
        # Mock collections
        self.mock_collections = {}
        for predicate in self.repository.specific_predicates + [self.repository.generic_predicate]:
            self.mock_collections[predicate] = MagicMock()
            self.mock_db.collection.side_effect = lambda name: self.mock_collections.get(name, MagicMock())
        
        # Sample data
        self.source_id = "Pattern/entity1"
        self.target_id = "Pattern/entity2"
        self.harmonic_properties = {
            "frequency": 0.35,
            "amplitude": 0.88,
            "phase": 0.12
        }
        self.vector_properties = {
            "x": 0.72, "y": 0.31, "z": 0.65, "w": 0.45, "v": 0.38
        }
        self.properties = {
            "confidence": 0.92,
            "harmonic_properties": self.harmonic_properties,
            "vector_properties": self.vector_properties,
            "observation_count": 1,
            "first_observed": datetime.now().isoformat(),
            "last_observed": datetime.now().isoformat()
        }
    
    def test_save_relationship_specific_predicate(self):
        """Test saving a relationship with a specific predicate."""
        # Mock insert to return an ID
        edge_id = f"Preserves/{uuid.uuid4()}"
        self.mock_collections["Preserves"].insert.return_value = {"_id": edge_id}
        
        # Call the method
        result = self.repository.save_relationship(
            self.source_id,
            "preserve",
            self.target_id,
            self.properties
        )
        
        # Verify the result
        self.assertEqual(result, edge_id)
        
        # Verify the correct collection was used
        self.mock_collections["Preserves"].insert.assert_called_once()
        
        # Verify the edge document
        args, kwargs = self.mock_collections["Preserves"].insert.call_args
        edge_doc = args[0]
        self.assertEqual(edge_doc["_from"], self.source_id)
        self.assertEqual(edge_doc["_to"], self.target_id)
        self.assertEqual(edge_doc["confidence"], 0.92)
        self.assertEqual(json.loads(edge_doc["harmonic_properties"]), self.harmonic_properties)
        self.assertEqual(json.loads(edge_doc["vector_properties"]), self.vector_properties)
    
    def test_save_relationship_generic_predicate(self):
        """Test saving a relationship with a generic predicate."""
        # Mock insert to return an ID
        edge_id = f"PredicateRelationship/{uuid.uuid4()}"
        self.mock_collections["PredicateRelationship"].insert.return_value = {"_id": edge_id}
        
        # Call the method
        result = self.repository.save_relationship(
            self.source_id,
            "reduces_dependence_on",
            self.target_id,
            self.properties
        )
        
        # Verify the result
        self.assertEqual(result, edge_id)
        
        # Verify the correct collection was used
        self.mock_collections["PredicateRelationship"].insert.assert_called_once()
        
        # Verify the edge document
        args, kwargs = self.mock_collections["PredicateRelationship"].insert.call_args
        edge_doc = args[0]
        self.assertEqual(edge_doc["_from"], self.source_id)
        self.assertEqual(edge_doc["_to"], self.target_id)
        self.assertEqual(edge_doc["predicate_type"], "reduces_dependence_on")
        self.assertEqual(edge_doc["confidence"], 0.92)
        self.assertEqual(json.loads(edge_doc["harmonic_properties"]), self.harmonic_properties)
        self.assertEqual(json.loads(edge_doc["vector_properties"]), self.vector_properties)
    
    def test_update_relationship(self):
        """Test updating a relationship."""
        # Mock get to return an existing edge
        self.mock_collections["Preserves"].get.return_value = {
            "_id": "Preserves/123",
            "_key": "123",
            "observation_count": 1
        }
        
        # Call the method
        result = self.repository.update_relationship(
            "Preserves/123",
            {
                "confidence": 0.95,
                "observation_count": 2,
                "harmonic_properties": {
                    "frequency": 0.36,
                    "amplitude": 0.89,
                    "phase": 0.13
                }
            }
        )
        
        # Verify the result
        self.assertTrue(result)
        
        # Verify the update was called
        self.mock_collections["Preserves"].update.assert_called_once()
        
        # Verify the update document
        args, kwargs = self.mock_collections["Preserves"].update.call_args
        key, update_doc = args
        self.assertEqual(key, "123")
        self.assertEqual(update_doc["confidence"], 0.95)
        self.assertEqual(update_doc["observation_count"], 2)
        self.assertEqual(
            json.loads(update_doc["harmonic_properties"]),
            {"frequency": 0.36, "amplitude": 0.89, "phase": 0.13}
        )
    
    def test_find_by_source_and_target_specific_predicate(self):
        """Test finding relationships by source and target with a specific predicate."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "_id": "Preserves/123",
            "_from": self.source_id,
            "_to": self.target_id,
            "confidence": 0.92,
            "harmonic_properties": json.dumps(self.harmonic_properties),
            "vector_properties": json.dumps(self.vector_properties)
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        results = self.repository.find_by_source_and_target(
            self.source_id,
            self.target_id,
            "preserve"
        )
        
        # Verify the result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["_id"], "Preserves/123")
        self.assertEqual(results[0]["_from"], self.source_id)
        self.assertEqual(results[0]["_to"], self.target_id)
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_source_and_target_generic_predicate(self):
        """Test finding relationships by source and target with a generic predicate."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "_id": "PredicateRelationship/123",
            "_from": self.source_id,
            "_to": self.target_id,
            "predicate_type": "reduces_dependence_on",
            "confidence": 0.92,
            "harmonic_properties": json.dumps(self.harmonic_properties),
            "vector_properties": json.dumps(self.vector_properties)
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        results = self.repository.find_by_source_and_target(
            self.source_id,
            self.target_id,
            "reduces_dependence_on"
        )
        
        # Verify the result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["_id"], "PredicateRelationship/123")
        self.assertEqual(results[0]["_from"], self.source_id)
        self.assertEqual(results[0]["_to"], self.target_id)
        self.assertEqual(results[0]["predicate_type"], "reduces_dependence_on")
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_predicate(self):
        """Test finding relationships by predicate."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "_id": "Preserves/123",
            "_from": self.source_id,
            "_to": self.target_id,
            "confidence": 0.92,
            "harmonic_properties": json.dumps(self.harmonic_properties),
            "vector_properties": json.dumps(self.vector_properties)
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        results = self.repository.find_by_predicate("preserve", 0.9)
        
        # Verify the result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["_id"], "Preserves/123")
        self.assertEqual(results[0]["confidence"], 0.92)
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_source(self):
        """Test finding relationships by source."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "_id": "Preserves/123",
            "_from": self.source_id,
            "_to": self.target_id,
            "confidence": 0.92,
            "harmonic_properties": json.dumps(self.harmonic_properties),
            "vector_properties": json.dumps(self.vector_properties)
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        results = self.repository.find_by_source(self.source_id, "preserve")
        
        # Verify the result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["_id"], "Preserves/123")
        self.assertEqual(results[0]["_from"], self.source_id)
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_target(self):
        """Test finding relationships by target."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "_id": "Preserves/123",
            "_from": self.source_id,
            "_to": self.target_id,
            "confidence": 0.92,
            "harmonic_properties": json.dumps(self.harmonic_properties),
            "vector_properties": json.dumps(self.vector_properties)
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        results = self.repository.find_by_target(self.target_id, "preserve")
        
        # Verify the result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["_id"], "Preserves/123")
        self.assertEqual(results[0]["_to"], self.target_id)
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_harmonic_properties(self):
        """Test finding relationships by harmonic properties."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "_id": "Preserves/123",
            "_from": self.source_id,
            "_to": self.target_id,
            "confidence": 0.92,
            "harmonic_properties": json.dumps(self.harmonic_properties),
            "vector_properties": json.dumps(self.vector_properties)
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        results = self.repository.find_by_harmonic_properties(
            frequency_range=(0.3, 0.4),
            amplitude_range=(0.8, 0.9)
        )
        
        # Verify the result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["_id"], "Preserves/123")
        self.assertEqual(
            json.loads(results[0]["harmonic_properties"]),
            self.harmonic_properties
        )
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_observation_count(self):
        """Test finding relationships by observation count."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "_id": "Preserves/123",
            "_from": self.source_id,
            "_to": self.target_id,
            "confidence": 0.92,
            "observation_count": 5,
            "harmonic_properties": json.dumps(self.harmonic_properties),
            "vector_properties": json.dumps(self.vector_properties)
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        results = self.repository.find_by_observation_count(3)
        
        # Verify the result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["_id"], "Preserves/123")
        self.assertEqual(results[0]["observation_count"], 5)
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_delete_relationship(self):
        """Test deleting a relationship."""
        # Call the method
        result = self.repository.delete_relationship("Preserves/123")
        
        # Verify the result
        self.assertTrue(result)
        
        # Verify the delete was called
        self.mock_collections["Preserves"].delete.assert_called_once_with("123")
    
    def test_normalize_predicate(self):
        """Test normalizing predicate strings."""
        test_cases = [
            ("preserve", "Preserve"),
            ("PRESERVE", "Preserve"),
            ("preserves", "Preserves"),
            ("reduces_dependence_on", "ReducesDependenceOn"),
            ("REDUCES_DEPENDENCE_ON", "ReducesDependenceOn")
        ]
        
        for input_str, expected in test_cases:
            result = self.repository._normalize_predicate(input_str)
            self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
