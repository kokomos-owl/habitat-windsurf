"""
Tests for the Semantic Signature Repository.

This module contains tests for the Semantic Signature Repository, which is responsible
for persisting semantic signatures to ArangoDB.
"""

import unittest
from unittest.mock import MagicMock, patch
import json
from datetime import datetime
import uuid
import numpy as np

from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.semantic_signature_repository import SemanticSignatureRepository
from src.habitat_evolution.pattern_aware_rag.semantic.signature import SemanticSignature

class TestSemanticSignatureRepository(unittest.TestCase):
    """Tests for the Semantic Signature Repository."""
    
    @patch('src.habitat_evolution.pattern_aware_rag.persistence.arangodb.semantic_signature_repository.ArangoDBConnectionManager')
    def setUp(self, mock_connection_manager):
        """Set up the test environment."""
        # Mock ArangoDB connection manager
        self.mock_db = MagicMock()
        mock_connection_manager.return_value.get_db.return_value = self.mock_db
        
        # Create repository
        self.repository = SemanticSignatureRepository()
        
        # Mock collection
        self.mock_collection = MagicMock()
        self.mock_db.collection.return_value = self.mock_collection
        
        # Create sample semantic signatures
        self.signature1 = SemanticSignature(
            id=str(uuid.uuid4()),
            entity_id="entity1",
            entity_type="concept",
            signature_vector=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            confidence=0.9
        )
        self.signature1.created_at = datetime.now()
        self.signature1.updated_at = datetime.now()
        self.signature1.temporal_stability = 0.8
        self.signature1.context_relevance = 0.7
        self.signature1.contributing_pattern_ids = ["pattern1", "pattern2"]
        self.signature1.dimensional_properties = {
            "primary_dimensions": [0, 2, 4],
            "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4]
        }
        self.signature1.version_history = [
            {"version_id": "v1", "timestamp": datetime.now().isoformat(), "changes": ["initial creation"]},
            {"version_id": "v2", "timestamp": datetime.now().isoformat(), "changes": ["updated vector"]}
        ]
        
        self.signature2 = SemanticSignature(
            id=str(uuid.uuid4()),
            entity_id="entity2",
            entity_type="document",
            signature_vector=np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
            confidence=0.8
        )
        self.signature2.created_at = datetime.now()
        self.signature2.updated_at = datetime.now()
        self.signature2.temporal_stability = 0.7
        self.signature2.context_relevance = 0.6
        self.signature2.contributing_pattern_ids = ["pattern2", "pattern3"]
        self.signature2.dimensional_properties = {
            "primary_dimensions": [1, 3, 4],
            "dimensional_coordinates": [0.4, 0.6, 0.3, 0.8, 0.5]
        }
        self.signature2.version_history = [
            {"version_id": "v1", "timestamp": datetime.now().isoformat(), "changes": ["initial creation"]}
        ]
    
    def test_to_document_properties(self):
        """Test converting a semantic signature to document properties."""
        # Call the method
        properties = self.repository._to_document_properties(self.signature1)
        
        # Verify core properties
        self.assertEqual(properties["id"], self.signature1.id)
        self.assertEqual(properties["entity_id"], self.signature1.entity_id)
        self.assertEqual(properties["entity_type"], self.signature1.entity_type)
        self.assertAlmostEqual(properties["confidence"], self.signature1.confidence)
        
        # Verify signature vector
        signature_vector = json.loads(properties["signature_vector"])
        self.assertEqual(len(signature_vector), 5)
        self.assertAlmostEqual(signature_vector[0], 0.1)
        self.assertAlmostEqual(signature_vector[1], 0.2)
        
        # Verify temporal stability and context relevance
        self.assertAlmostEqual(properties["temporal_stability"], self.signature1.temporal_stability)
        self.assertAlmostEqual(properties["context_relevance"], self.signature1.context_relevance)
        
        # Verify contributing pattern IDs
        contributing_pattern_ids = json.loads(properties["contributing_pattern_ids"])
        self.assertEqual(len(contributing_pattern_ids), 2)
        self.assertIn("pattern1", contributing_pattern_ids)
        self.assertIn("pattern2", contributing_pattern_ids)
        
        # Verify dimensional properties
        dimensional_properties = json.loads(properties["dimensional_properties"])
        self.assertEqual(dimensional_properties["primary_dimensions"], [0, 2, 4])
        self.assertEqual(len(dimensional_properties["dimensional_coordinates"]), 5)
        
        # Verify version history
        version_history = json.loads(properties["version_history"])
        self.assertEqual(len(version_history), 2)
        self.assertEqual(version_history[0]["version_id"], "v1")
        self.assertEqual(version_history[1]["version_id"], "v2")
    
    def test_dict_to_entity(self):
        """Test converting document properties to a semantic signature."""
        # Create document properties
        properties = {
            "id": "test_id",
            "entity_id": "entity1",
            "entity_type": "concept",
            "signature_vector": json.dumps([0.1, 0.2, 0.3, 0.4, 0.5]),
            "confidence": 0.9,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "temporal_stability": 0.8,
            "context_relevance": 0.7,
            "contributing_pattern_ids": json.dumps(["pattern1", "pattern2"]),
            "dimensional_properties": json.dumps({
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4]
            }),
            "version_history": json.dumps([
                {"version_id": "v1", "timestamp": datetime.now().isoformat(), "changes": ["initial creation"]},
                {"version_id": "v2", "timestamp": datetime.now().isoformat(), "changes": ["updated vector"]}
            ])
        }
        
        # Call the method
        signature = self.repository._dict_to_entity(properties)
        
        # Verify core properties
        self.assertEqual(signature.id, "test_id")
        self.assertEqual(signature.entity_id, "entity1")
        self.assertEqual(signature.entity_type, "concept")
        self.assertAlmostEqual(signature.confidence, 0.9)
        
        # Verify signature vector
        self.assertEqual(len(signature.signature_vector), 5)
        self.assertAlmostEqual(signature.signature_vector[0], 0.1)
        self.assertAlmostEqual(signature.signature_vector[1], 0.2)
        
        # Verify temporal stability and context relevance
        self.assertAlmostEqual(signature.temporal_stability, 0.8)
        self.assertAlmostEqual(signature.context_relevance, 0.7)
        
        # Verify contributing pattern IDs
        self.assertEqual(len(signature.contributing_pattern_ids), 2)
        self.assertIn("pattern1", signature.contributing_pattern_ids)
        self.assertIn("pattern2", signature.contributing_pattern_ids)
        
        # Verify dimensional properties
        self.assertEqual(signature.dimensional_properties["primary_dimensions"], [0, 2, 4])
        self.assertEqual(len(signature.dimensional_properties["dimensional_coordinates"]), 5)
        
        # Verify version history
        self.assertEqual(len(signature.version_history), 2)
        self.assertEqual(signature.version_history[0]["version_id"], "v1")
        self.assertEqual(signature.version_history[1]["version_id"], "v2")
    
    def test_save_new_signature(self):
        """Test saving a new semantic signature."""
        # Mock collection.get to return None (document doesn't exist)
        self.mock_collection.get.side_effect = Exception("Document not found")
        
        # Call the method
        result = self.repository.save(self.signature1)
        
        # Verify the document was inserted
        self.mock_collection.insert.assert_called_once()
        self.assertEqual(result, self.signature1.id)
    
    def test_save_existing_signature(self):
        """Test saving an existing semantic signature."""
        # Mock collection.get to return a document (document exists)
        self.mock_collection.get.return_value = {"_key": self.signature1.id}
        
        # Call the method
        result = self.repository.save(self.signature1)
        
        # Verify the document was updated
        self.mock_collection.update.assert_called_once()
        self.assertEqual(result, self.signature1.id)
    
    def test_find_by_id_existing(self):
        """Test finding an existing semantic signature by ID."""
        # Mock collection.get to return a document
        self.mock_collection.get.return_value = {
            "id": "test_id",
            "entity_id": "entity1",
            "entity_type": "concept",
            "signature_vector": json.dumps([0.1, 0.2, 0.3, 0.4, 0.5]),
            "confidence": 0.9,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "temporal_stability": 0.8,
            "context_relevance": 0.7,
            "contributing_pattern_ids": json.dumps(["pattern1", "pattern2"]),
            "dimensional_properties": json.dumps({
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4]
            }),
            "version_history": json.dumps([])
        }
        
        # Call the method
        signature = self.repository.find_by_id("test_id")
        
        # Verify the result
        self.assertIsNotNone(signature)
        self.assertEqual(signature.id, "test_id")
        self.assertEqual(signature.entity_id, "entity1")
        self.assertEqual(signature.entity_type, "concept")
    
    def test_find_by_id_not_existing(self):
        """Test finding a non-existing semantic signature by ID."""
        # Mock collection.get to return None
        self.mock_collection.get.return_value = None
        
        # Call the method
        signature = self.repository.find_by_id("non_existing_id")
        
        # Verify the result
        self.assertIsNone(signature)
    
    def test_find_by_entity_id(self):
        """Test finding semantic signatures by entity ID."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "test_id",
            "entity_id": "entity1",
            "entity_type": "concept",
            "signature_vector": json.dumps([0.1, 0.2, 0.3, 0.4, 0.5]),
            "confidence": 0.9,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "temporal_stability": 0.8,
            "context_relevance": 0.7,
            "contributing_pattern_ids": json.dumps(["pattern1", "pattern2"]),
            "dimensional_properties": json.dumps({
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4]
            }),
            "version_history": json.dumps([])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        signatures = self.repository.find_by_entity_id("entity1")
        
        # Verify the result
        self.assertEqual(len(signatures), 1)
        self.assertEqual(signatures[0].id, "test_id")
        self.assertEqual(signatures[0].entity_id, "entity1")
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_entity_type(self):
        """Test finding semantic signatures by entity type."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "test_id",
            "entity_id": "entity1",
            "entity_type": "concept",
            "signature_vector": json.dumps([0.1, 0.2, 0.3, 0.4, 0.5]),
            "confidence": 0.9,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "temporal_stability": 0.8,
            "context_relevance": 0.7,
            "contributing_pattern_ids": json.dumps(["pattern1", "pattern2"]),
            "dimensional_properties": json.dumps({
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4]
            }),
            "version_history": json.dumps([])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        signatures = self.repository.find_by_entity_type("concept")
        
        # Verify the result
        self.assertEqual(len(signatures), 1)
        self.assertEqual(signatures[0].id, "test_id")
        self.assertEqual(signatures[0].entity_type, "concept")
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_contributing_pattern(self):
        """Test finding semantic signatures by contributing pattern."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "test_id",
            "entity_id": "entity1",
            "entity_type": "concept",
            "signature_vector": json.dumps([0.1, 0.2, 0.3, 0.4, 0.5]),
            "confidence": 0.9,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "temporal_stability": 0.8,
            "context_relevance": 0.7,
            "contributing_pattern_ids": json.dumps(["pattern1", "pattern2"]),
            "dimensional_properties": json.dumps({
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4]
            }),
            "version_history": json.dumps([])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        signatures = self.repository.find_by_contributing_pattern("pattern1")
        
        # Verify the result
        self.assertEqual(len(signatures), 1)
        self.assertEqual(signatures[0].id, "test_id")
        self.assertIn("pattern1", signatures[0].contributing_pattern_ids)
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_temporal_stability(self):
        """Test finding semantic signatures by temporal stability."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "test_id",
            "entity_id": "entity1",
            "entity_type": "concept",
            "signature_vector": json.dumps([0.1, 0.2, 0.3, 0.4, 0.5]),
            "confidence": 0.9,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "temporal_stability": 0.8,
            "context_relevance": 0.7,
            "contributing_pattern_ids": json.dumps(["pattern1", "pattern2"]),
            "dimensional_properties": json.dumps({
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4]
            }),
            "version_history": json.dumps([])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        signatures = self.repository.find_by_temporal_stability(0.7)
        
        # Verify the result
        self.assertEqual(len(signatures), 1)
        self.assertEqual(signatures[0].id, "test_id")
        self.assertGreaterEqual(signatures[0].temporal_stability, 0.7)
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_similar_signatures(self):
        """Test finding similar semantic signatures."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "test_id",
            "entity_id": "entity1",
            "entity_type": "concept",
            "signature_vector": json.dumps([0.1, 0.2, 0.3, 0.4, 0.5]),
            "confidence": 0.9,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "temporal_stability": 0.8,
            "context_relevance": 0.7,
            "contributing_pattern_ids": json.dumps(["pattern1", "pattern2"]),
            "dimensional_properties": json.dumps({
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4]
            }),
            "version_history": json.dumps([]),
            "similarity": 0.95
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Create a query vector
        query_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Call the method
        signatures = self.repository.find_similar_signatures(query_vector, 0.9)
        
        # Verify the result
        self.assertEqual(len(signatures), 1)
        self.assertEqual(signatures[0].id, "test_id")
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_version_history(self):
        """Test finding the version history of a semantic signature."""
        # Mock collection.get to return a document
        self.mock_collection.get.return_value = {
            "version_history": json.dumps([
                {"version_id": "v1", "timestamp": datetime.now().isoformat(), "changes": ["initial creation"]},
                {"version_id": "v2", "timestamp": datetime.now().isoformat(), "changes": ["updated vector"]}
            ])
        }
        
        # Call the method
        version_history = self.repository.find_version_history("test_id")
        
        # Verify the result
        self.assertEqual(len(version_history), 2)
        self.assertEqual(version_history[0]["version_id"], "v1")
        self.assertEqual(version_history[1]["version_id"], "v2")
        
        # Verify collection.get was called
        self.mock_collection.get.assert_called_once_with("test_id")

if __name__ == '__main__':
    unittest.main()
