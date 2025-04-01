"""
Tests for the ArangoDB Schema Manager.

This module contains tests for the ArangoDB Schema Manager, which is responsible
for creating and managing the ArangoDB schema for pattern-aware RAG.
"""

import unittest
from unittest.mock import MagicMock, patch
import json
import os

from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.schema_manager import ArangoDBSchemaManager

class TestArangoDBSchemaManager(unittest.TestCase):
    """Tests for the ArangoDB Schema Manager."""
    
    @patch('src.habitat_evolution.pattern_aware_rag.persistence.arangodb.schema_manager.ArangoClient')
    def setUp(self, mock_client):
        """Set up the test environment."""
        # Mock ArangoDB client and database
        self.mock_db = MagicMock()
        mock_client.return_value.db.return_value = self.mock_db
        
        # Create schema manager
        self.schema_manager = ArangoDBSchemaManager()
        
        # Mock collections
        self.mock_collections = {}
        self.mock_db.collections.return_value = self.mock_collections
        
        # Mock collection creation
        def mock_create_collection(name, edge=False):
            self.mock_collections[name] = MagicMock()
            return self.mock_collections[name]
        
        self.mock_db.create_collection.side_effect = mock_create_collection
        
        # Mock collection retrieval
        def mock_collection(name):
            if name not in self.mock_collections:
                self.mock_collections[name] = MagicMock()
            return self.mock_collections[name]
        
        self.mock_db.collection.side_effect = mock_collection
        
        # Mock has_collection
        self.mock_db.has_collection.return_value = False
        
        # Mock has_graph
        self.mock_db.has_graph.return_value = False
        
        # Mock server_info
        self.mock_db.server_info.return_value = {"time": 123456789}
    
    def test_create_schema(self):
        """Test creating the ArangoDB schema."""
        # Call the method
        self.schema_manager.create_schema()
        
        # Verify document collections were created
        expected_document_collections = [
            "TonicHarmonicFieldState",
            "TopologyState",
            "FrequencyDomain",
            "Boundary",
            "ResonancePoint",
            "Pattern",
            "ResonanceGroup",
            "SemanticSignature"
        ]
        
        for collection_name in expected_document_collections:
            self.mock_db.create_collection.assert_any_call(collection_name, edge=False)
        
        # Verify edge collections were created
        expected_edge_collections = [
            "HasDomain",
            "HasBoundary",
            "HasResonance",
            "HasPattern",
            "HasResonanceGroup",
            "Connects",
            "BelongsTo",
            "ContributesTo",
            "StatisticallyCorrelatedWith",
            "LocatedAt",
            "DiffersFromControlBy"
        ]
        
        for collection_name in expected_edge_collections:
            self.mock_db.create_collection.assert_any_call(collection_name, edge=True)
    
    def test_create_indexes(self):
        """Test creating indexes for the collections."""
        # Set up collections to exist
        self.mock_db.has_collection.return_value = True
        self.mock_db.collections.return_value = [
            "TonicHarmonicFieldState",
            "TopologyState",
            "FrequencyDomain",
            "Boundary",
            "ResonancePoint",
            "Pattern",
            "ResonanceGroup",
            "SemanticSignature"
        ]
        
        # Call the method
        self.schema_manager._create_indexes()
        
        # Verify indexes were created
        for collection_name in self.mock_db.collections():
            mock_collection = self.mock_db.collection(collection_name)
            self.assertTrue(mock_collection.add_hash_index.called or mock_collection.add_skiplist_index.called)
    
    def test_create_graph_definitions(self):
        """Test creating graph definitions."""
        # Call the method
        self.schema_manager.create_graph_definitions()
        
        # Verify graph was created
        self.mock_db.create_graph.assert_called_once()
        
        # Verify edge definitions were included
        args, kwargs = self.mock_db.create_graph.call_args
        self.assertEqual(args[0], "PatternAwareRAG")
        self.assertIsInstance(args[1], list)
        self.assertGreater(len(args[1]), 0)
        
        # Verify edge definitions have correct structure
        for edge_def in args[1]:
            self.assertIn("edge_collection", edge_def)
            self.assertIn("from_vertex_collections", edge_def)
            self.assertIn("to_vertex_collections", edge_def)
            self.assertIsInstance(edge_def["from_vertex_collections"], list)
            self.assertIsInstance(edge_def["to_vertex_collections"], list)

if __name__ == '__main__':
    unittest.main()
