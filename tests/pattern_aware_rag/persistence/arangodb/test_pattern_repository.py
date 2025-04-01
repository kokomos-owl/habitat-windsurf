"""
Tests for the Pattern Repository.

This module contains tests for the Pattern Repository, which is responsible
for persisting patterns to ArangoDB.
"""

import unittest
from unittest.mock import MagicMock, patch
import json
from datetime import datetime
import uuid

from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.pattern_repository import PatternRepository
from src.habitat_evolution.adaptive_core.models import Pattern

class TestPatternRepository(unittest.TestCase):
    """Tests for the Pattern Repository."""
    
    @patch('src.habitat_evolution.pattern_aware_rag.persistence.arangodb.pattern_repository.ArangoDBConnectionManager')
    def setUp(self, mock_connection_manager):
        """Set up the test environment."""
        # Mock ArangoDB connection manager
        self.mock_db = MagicMock()
        mock_connection_manager.return_value.get_db.return_value = self.mock_db
        
        # Create repository
        self.repository = PatternRepository()
        
        # Mock collection
        self.mock_collection = MagicMock()
        self.mock_db.collection.return_value = self.mock_collection
        
        # Create sample patterns
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
        self.pattern1.version_history = [
            {"version_id": "v1", "timestamp": datetime.now().isoformat(), "changes": ["initial creation"]},
            {"version_id": "v2", "timestamp": datetime.now().isoformat(), "changes": ["updated confidence"]}
        ]
        
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
        self.pattern2.version_history = [
            {"version_id": "v1", "timestamp": datetime.now().isoformat(), "changes": ["initial creation"]}
        ]
    
    def test_to_document_properties(self):
        """Test converting a pattern to document properties."""
        # Call the method
        properties = self.repository._to_document_properties(self.pattern1)
        
        # Verify core properties
        self.assertEqual(properties["id"], self.pattern1.id)
        self.assertEqual(properties["pattern_type"], self.pattern1.pattern_type)
        self.assertEqual(properties["source"], self.pattern1.source)
        self.assertEqual(properties["predicate"], self.pattern1.predicate)
        self.assertEqual(properties["target"], self.pattern1.target)
        self.assertAlmostEqual(properties["confidence"], self.pattern1.confidence)
        
        # Verify metadata
        metadata = json.loads(properties["metadata"])
        self.assertEqual(metadata["source_type"], "document")
        self.assertEqual(metadata["target_type"], "concept")
        self.assertEqual(metadata["context"], "climate risk analysis")
        
        # Verify temporal properties
        temporal_properties = json.loads(properties["temporal_properties"])
        self.assertEqual(temporal_properties["observation_count"], 5)
        self.assertAlmostEqual(temporal_properties["stability_score"], 0.8)
        self.assertAlmostEqual(temporal_properties["frequency"], 0.3)
        self.assertAlmostEqual(temporal_properties["phase"], 0.2)
        
        # Verify eigenspace properties
        eigenspace_properties = json.loads(properties["eigenspace_properties"])
        self.assertEqual(eigenspace_properties["primary_dimensions"], [0, 2, 4])
        self.assertEqual(len(eigenspace_properties["dimensional_coordinates"]), 5)
        self.assertAlmostEqual(eigenspace_properties["eigenspace_centrality"], 0.8)
        self.assertAlmostEqual(eigenspace_properties["eigenspace_stability"], 0.7)
        
        # Verify version history
        version_history = json.loads(properties["version_history"])
        self.assertEqual(len(version_history), 2)
        self.assertEqual(version_history[0]["version_id"], "v1")
        self.assertEqual(version_history[1]["version_id"], "v2")
    
    def test_dict_to_entity(self):
        """Test converting document properties to a pattern."""
        # Create document properties
        properties = {
            "id": "test_id",
            "pattern_type": "semantic",
            "source": "entity1",
            "predicate": "relates_to",
            "target": "entity2",
            "confidence": 0.9,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": json.dumps({
                "source_type": "document",
                "target_type": "concept",
                "context": "climate risk analysis"
            }),
            "temporal_properties": json.dumps({
                "first_observed": datetime.now().isoformat(),
                "last_observed": datetime.now().isoformat(),
                "observation_count": 5,
                "stability_score": 0.8,
                "frequency": 0.3,
                "phase": 0.2
            }),
            "eigenspace_properties": json.dumps({
                "primary_dimensions": [0, 2, 4],
                "dimensional_coordinates": [0.3, 0.5, 0.7, 0.2, 0.4],
                "eigenspace_centrality": 0.8,
                "eigenspace_stability": 0.7
            }),
            "version_history": json.dumps([
                {"version_id": "v1", "timestamp": datetime.now().isoformat(), "changes": ["initial creation"]},
                {"version_id": "v2", "timestamp": datetime.now().isoformat(), "changes": ["updated confidence"]}
            ])
        }
        
        # Call the method
        pattern = self.repository._dict_to_entity(properties)
        
        # Verify core properties
        self.assertEqual(pattern.id, "test_id")
        self.assertEqual(pattern.pattern_type, "semantic")
        self.assertEqual(pattern.source, "entity1")
        self.assertEqual(pattern.predicate, "relates_to")
        self.assertEqual(pattern.target, "entity2")
        self.assertAlmostEqual(pattern.confidence, 0.9)
        
        # Verify metadata
        self.assertEqual(pattern.metadata["source_type"], "document")
        self.assertEqual(pattern.metadata["target_type"], "concept")
        self.assertEqual(pattern.metadata["context"], "climate risk analysis")
        
        # Verify temporal properties
        self.assertEqual(pattern.temporal_properties["observation_count"], 5)
        self.assertAlmostEqual(pattern.temporal_properties["stability_score"], 0.8)
        self.assertAlmostEqual(pattern.temporal_properties["frequency"], 0.3)
        self.assertAlmostEqual(pattern.temporal_properties["phase"], 0.2)
        
        # Verify eigenspace properties
        self.assertEqual(pattern.eigenspace_properties["primary_dimensions"], [0, 2, 4])
        self.assertEqual(len(pattern.eigenspace_properties["dimensional_coordinates"]), 5)
        self.assertAlmostEqual(pattern.eigenspace_properties["eigenspace_centrality"], 0.8)
        self.assertAlmostEqual(pattern.eigenspace_properties["eigenspace_stability"], 0.7)
        
        # Verify version history
        self.assertEqual(len(pattern.version_history), 2)
        self.assertEqual(pattern.version_history[0]["version_id"], "v1")
        self.assertEqual(pattern.version_history[1]["version_id"], "v2")
    
    def test_save_new_pattern(self):
        """Test saving a new pattern."""
        # Mock collection.get to return None (document doesn't exist)
        self.mock_collection.get.side_effect = Exception("Document not found")
        
        # Call the method
        result = self.repository.save(self.pattern1)
        
        # Verify the document was inserted
        self.mock_collection.insert.assert_called_once()
        self.assertEqual(result, self.pattern1.id)
    
    def test_save_existing_pattern(self):
        """Test saving an existing pattern."""
        # Mock collection.get to return a document (document exists)
        self.mock_collection.get.return_value = {"_key": self.pattern1.id}
        
        # Call the method
        result = self.repository.save(self.pattern1)
        
        # Verify the document was updated
        self.mock_collection.update.assert_called_once()
        self.assertEqual(result, self.pattern1.id)
    
    def test_find_by_id_existing(self):
        """Test finding an existing pattern by ID."""
        # Mock collection.get to return a document
        self.mock_collection.get.return_value = {
            "id": "test_id",
            "pattern_type": "semantic",
            "source": "entity1",
            "predicate": "relates_to",
            "target": "entity2",
            "confidence": 0.9,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": json.dumps({}),
            "temporal_properties": json.dumps({}),
            "eigenspace_properties": json.dumps({}),
            "version_history": json.dumps([])
        }
        
        # Call the method
        pattern = self.repository.find_by_id("test_id")
        
        # Verify the result
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.id, "test_id")
        self.assertEqual(pattern.pattern_type, "semantic")
        self.assertEqual(pattern.source, "entity1")
        self.assertEqual(pattern.predicate, "relates_to")
        self.assertEqual(pattern.target, "entity2")
    
    def test_find_by_id_not_existing(self):
        """Test finding a non-existing pattern by ID."""
        # Mock collection.get to return None
        self.mock_collection.get.return_value = None
        
        # Call the method
        pattern = self.repository.find_by_id("non_existing_id")
        
        # Verify the result
        self.assertIsNone(pattern)
    
    def test_find_by_predicate(self):
        """Test finding patterns by predicate."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "pattern1",
            "pattern_type": "semantic",
            "source": "entity1",
            "predicate": "relates_to",
            "target": "entity2",
            "confidence": 0.9,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": json.dumps({}),
            "temporal_properties": json.dumps({}),
            "eigenspace_properties": json.dumps({}),
            "version_history": json.dumps([])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        patterns = self.repository.find_by_predicate("relates_to")
        
        # Verify the result
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].id, "pattern1")
        self.assertEqual(patterns[0].predicate, "relates_to")
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_source_and_target(self):
        """Test finding patterns by source and target."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "pattern1",
            "pattern_type": "semantic",
            "source": "entity1",
            "predicate": "relates_to",
            "target": "entity2",
            "confidence": 0.9,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": json.dumps({}),
            "temporal_properties": json.dumps({}),
            "eigenspace_properties": json.dumps({}),
            "version_history": json.dumps([])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        patterns = self.repository.find_by_source_and_target("entity1", "entity2")
        
        # Verify the result
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].id, "pattern1")
        self.assertEqual(patterns[0].source, "entity1")
        self.assertEqual(patterns[0].target, "entity2")
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_pattern_type(self):
        """Test finding patterns by pattern type."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "pattern1",
            "pattern_type": "semantic",
            "source": "entity1",
            "predicate": "relates_to",
            "target": "entity2",
            "confidence": 0.9,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": json.dumps({}),
            "temporal_properties": json.dumps({}),
            "eigenspace_properties": json.dumps({}),
            "version_history": json.dumps([])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        patterns = self.repository.find_by_pattern_type("semantic")
        
        # Verify the result
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].id, "pattern1")
        self.assertEqual(patterns[0].pattern_type, "semantic")
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_confidence_threshold(self):
        """Test finding patterns by confidence threshold."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "pattern1",
            "pattern_type": "semantic",
            "source": "entity1",
            "predicate": "relates_to",
            "target": "entity2",
            "confidence": 0.9,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": json.dumps({}),
            "temporal_properties": json.dumps({}),
            "eigenspace_properties": json.dumps({}),
            "version_history": json.dumps([])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        patterns = self.repository.find_by_confidence_threshold(0.8)
        
        # Verify the result
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].id, "pattern1")
        self.assertGreaterEqual(patterns[0].confidence, 0.8)
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_temporal_stability(self):
        """Test finding patterns by temporal stability."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "pattern1",
            "pattern_type": "semantic",
            "source": "entity1",
            "predicate": "relates_to",
            "target": "entity2",
            "confidence": 0.9,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": json.dumps({}),
            "temporal_properties": json.dumps({
                "stability_score": 0.8
            }),
            "eigenspace_properties": json.dumps({}),
            "version_history": json.dumps([])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        patterns = self.repository.find_by_temporal_stability(0.7)
        
        # Verify the result
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].id, "pattern1")
        self.assertGreaterEqual(patterns[0].temporal_properties["stability_score"], 0.7)
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_by_eigenspace_centrality(self):
        """Test finding patterns by eigenspace centrality."""
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "pattern1",
            "pattern_type": "semantic",
            "source": "entity1",
            "predicate": "relates_to",
            "target": "entity2",
            "confidence": 0.9,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": json.dumps({}),
            "temporal_properties": json.dumps({}),
            "eigenspace_properties": json.dumps({
                "eigenspace_centrality": 0.8
            }),
            "version_history": json.dumps([])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        patterns = self.repository.find_by_eigenspace_centrality(0.7)
        
        # Verify the result
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].id, "pattern1")
        self.assertGreaterEqual(patterns[0].eigenspace_properties["eigenspace_centrality"], 0.7)
        
        # Verify AQL was executed
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_version_history(self):
        """Test finding the version history of a pattern."""
        # Mock collection.get to return a document
        self.mock_collection.get.return_value = {
            "version_history": json.dumps([
                {"version_id": "v1", "timestamp": datetime.now().isoformat(), "changes": ["initial creation"]},
                {"version_id": "v2", "timestamp": datetime.now().isoformat(), "changes": ["updated confidence"]}
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
