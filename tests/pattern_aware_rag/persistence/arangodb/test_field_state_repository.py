"""
Tests for the TonicHarmonicFieldState Repository.

This module contains tests for the TonicHarmonicFieldState Repository, which is responsible
for persisting field states to ArangoDB.
"""

import unittest
from unittest.mock import MagicMock, patch
import json
from datetime import datetime
import numpy as np

from src.habitat_evolution.field.field_state import TonicHarmonicFieldState
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.field_state_repository import TonicHarmonicFieldStateRepository

class TestTonicHarmonicFieldStateRepository(unittest.TestCase):
    """Tests for the TonicHarmonicFieldState Repository."""
    
    @patch('src.habitat_evolution.pattern_aware_rag.persistence.arangodb.field_state_repository.ArangoDBConnectionManager')
    def setUp(self, mock_connection_manager):
        """Set up the test environment."""
        # Mock ArangoDB connection manager
        self.mock_db = MagicMock()
        mock_connection_manager.return_value.get_db.return_value = self.mock_db
        
        # Create repository
        self.repository = TonicHarmonicFieldStateRepository()
        
        # Mock collection
        self.mock_collection = MagicMock()
        self.mock_db.collection.return_value = self.mock_collection
        
        # Create a sample field state
        self.field_analysis = {
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
        
        self.field_state = TonicHarmonicFieldState(self.field_analysis)
        
        # Add some patterns and resonance relationships
        self.field_state.patterns = {
            "pattern1": {"source": "entity1", "predicate": "relates_to", "target": "entity2"},
            "pattern2": {"source": "entity2", "predicate": "influences", "target": "entity3"}
        }
        
        self.field_state.resonance_relationships = {
            "pattern1": {"pattern2": 0.8, "pattern3": 0.6},
            "pattern2": {"pattern1": 0.8, "pattern4": 0.7}
        }
    
    def test_to_document_properties(self):
        """Test converting a field state to document properties."""
        # Call the method
        properties = self.repository._to_document_properties(self.field_state)
        
        # Verify core properties
        self.assertEqual(properties["id"], self.field_state.id)
        self.assertEqual(properties["version_id"], self.field_state.version_id)
        
        # Verify field analysis properties
        self.assertEqual(properties["effective_dimensionality"], 5)
        self.assertEqual(json.loads(properties["principal_dimensions"]), ["ecological", "cultural", "economic", "social", "temporal"])
        
        # Verify eigenvalues and eigenvectors are serialized correctly
        eigenvalues = json.loads(properties["eigenvalues"])
        self.assertEqual(len(eigenvalues), 5)
        self.assertAlmostEqual(eigenvalues[0], 0.42)
        
        eigenvectors = json.loads(properties["eigenvectors"])
        self.assertEqual(len(eigenvectors), 5)
        self.assertEqual(len(eigenvectors[0]), 5)
        self.assertAlmostEqual(eigenvectors[0][0], 0.8)
        
        # Verify density centers
        density_centers = json.loads(properties["density_centers"])
        self.assertEqual(len(density_centers), 1)
        self.assertEqual(len(density_centers[0]["position"]), 5)
        self.assertAlmostEqual(density_centers[0]["magnitude"], 0.8)
        
        # Verify field properties
        self.assertAlmostEqual(properties["coherence"], 0.7)
        self.assertAlmostEqual(properties["navigability_score"], 0.6)
        self.assertAlmostEqual(properties["stability"], 0.8)
        
        # Verify patterns and resonance relationships
        pattern_ids = json.loads(properties["pattern_ids"])
        self.assertEqual(len(pattern_ids), 2)
        self.assertIn("pattern1", pattern_ids)
        self.assertIn("pattern2", pattern_ids)
        
        resonance_relationships = json.loads(properties["resonance_relationships"])
        self.assertEqual(len(resonance_relationships), 2)
        self.assertIn("pattern1", resonance_relationships)
        self.assertIn("pattern2", resonance_relationships)
        self.assertEqual(len(resonance_relationships["pattern1"]), 2)
        self.assertEqual(len(resonance_relationships["pattern2"]), 2)
    
    def test_dict_to_entity(self):
        """Test converting document properties to a field state."""
        # Create document properties
        properties = {
            "id": "test_id",
            "version_id": "test_version_id",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "effective_dimensionality": 3,
            "principal_dimensions": json.dumps(["dim1", "dim2", "dim3"]),
            "eigenvalues": json.dumps([0.5, 0.3, 0.2]),
            "eigenvectors": json.dumps([
                [0.8, 0.1, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7]
            ]),
            "density_centers": json.dumps([
                {"position": [0.3, 0.4, 0.3], "magnitude": 0.7}
            ]),
            "density_map_metadata": json.dumps({
                "resolution": [5, 5, 5],
                "dimensions": 3,
                "has_values": True
            }),
            "coherence": 0.6,
            "navigability_score": 0.5,
            "stability": 0.7,
            "resonance_patterns": json.dumps([
                {"center": [0.4, 0.3, 0.3], "intensity": 0.8}
            ]),
            "version_history": json.dumps([
                {"version_id": "v1", "timestamp": "2023-01-01T00:00:00"},
                {"version_id": "v2", "timestamp": "2023-01-02T00:00:00"}
            ]),
            "field_metrics": json.dumps({
                "metric1": 0.5,
                "metric2": 0.7
            })
        }
        
        # Call the method
        field_state = self.repository._dict_to_entity(properties)
        
        # Verify core properties
        self.assertEqual(field_state.id, "test_id")
        self.assertEqual(field_state.version_id, "test_version_id")
        
        # Verify field analysis properties
        self.assertEqual(field_state.field_analysis["topology"]["effective_dimensionality"], 3)
        self.assertEqual(field_state.field_analysis["topology"]["principal_dimensions"], ["dim1", "dim2", "dim3"])
        
        # Verify eigenvalues and eigenvectors
        self.assertEqual(len(field_state.field_analysis["topology"]["eigenvalues"]), 3)
        self.assertAlmostEqual(field_state.field_analysis["topology"]["eigenvalues"][0], 0.5)
        
        self.assertEqual(len(field_state.field_analysis["topology"]["eigenvectors"]), 3)
        self.assertEqual(len(field_state.field_analysis["topology"]["eigenvectors"][0]), 3)
        self.assertAlmostEqual(field_state.field_analysis["topology"]["eigenvectors"][0][0], 0.8)
        
        # Verify density centers
        self.assertEqual(len(field_state.field_analysis["density"]["density_centers"]), 1)
        self.assertEqual(len(field_state.field_analysis["density"]["density_centers"][0]["position"]), 3)
        self.assertAlmostEqual(field_state.field_analysis["density"]["density_centers"][0]["magnitude"], 0.7)
        
        # Verify field properties
        self.assertAlmostEqual(field_state.field_analysis["field_properties"]["coherence"], 0.6)
        self.assertAlmostEqual(field_state.field_analysis["field_properties"]["navigability_score"], 0.5)
        self.assertAlmostEqual(field_state.field_analysis["field_properties"]["stability"], 0.7)
        
        # Verify version history and field metrics
        self.assertEqual(len(field_state.version_history), 2)
        self.assertEqual(field_state.version_history[0]["version_id"], "v1")
        
        self.assertEqual(len(field_state.field_metrics), 2)
        self.assertAlmostEqual(field_state.field_metrics["metric1"], 0.5)
        self.assertAlmostEqual(field_state.field_metrics["metric2"], 0.7)
    
    def test_save_new_field_state(self):
        """Test saving a new field state."""
        # Mock collection.get to return None (document doesn't exist)
        self.mock_collection.get.side_effect = Exception("Document not found")
        
        # Call the method
        result = self.repository.save(self.field_state)
        
        # Verify the document was inserted
        self.mock_collection.insert.assert_called_once()
        self.assertEqual(result, self.field_state.id)
    
    def test_save_existing_field_state(self):
        """Test saving an existing field state."""
        # Mock collection.get to return a document (document exists)
        self.mock_collection.get.return_value = {"_key": self.field_state.id}
        
        # Call the method
        result = self.repository.save(self.field_state)
        
        # Verify the document was updated
        self.mock_collection.update.assert_called_once()
        self.assertEqual(result, self.field_state.id)
    
    def test_find_by_id_existing(self):
        """Test finding an existing field state by ID."""
        # Mock collection.get to return a document
        self.mock_collection.get.return_value = {
            "id": "test_id",
            "version_id": "test_version_id",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "effective_dimensionality": 3,
            "principal_dimensions": json.dumps(["dim1", "dim2", "dim3"]),
            "eigenvalues": json.dumps([0.5, 0.3, 0.2]),
            "eigenvectors": json.dumps([
                [0.8, 0.1, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7]
            ]),
            "density_centers": json.dumps([]),
            "density_map_metadata": json.dumps({"has_values": False}),
            "coherence": 0.6,
            "navigability_score": 0.5,
            "stability": 0.7,
            "resonance_patterns": json.dumps([])
        }
        
        # Call the method
        field_state = self.repository.find_by_id("test_id")
        
        # Verify the result
        self.assertIsNotNone(field_state)
        self.assertEqual(field_state.id, "test_id")
        self.assertEqual(field_state.version_id, "test_version_id")
    
    def test_find_by_id_not_existing(self):
        """Test finding a non-existing field state by ID."""
        # Mock collection.get to return None
        self.mock_collection.get.return_value = None
        
        # Call the method
        field_state = self.repository.find_by_id("non_existing_id")
        
        # Verify the result
        self.assertIsNone(field_state)
    
    def test_find_latest(self):
        """Test finding the latest field state."""
        # Mock AQL execution to return a document
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "latest_id",
            "version_id": "latest_version_id",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "effective_dimensionality": 3,
            "principal_dimensions": json.dumps(["dim1", "dim2", "dim3"]),
            "eigenvalues": json.dumps([0.5, 0.3, 0.2]),
            "eigenvectors": json.dumps([
                [0.8, 0.1, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7]
            ]),
            "density_centers": json.dumps([]),
            "density_map_metadata": json.dumps({"has_values": False}),
            "coherence": 0.6,
            "navigability_score": 0.5,
            "stability": 0.7,
            "resonance_patterns": json.dumps([])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        field_state = self.repository.find_latest()
        
        # Verify the result
        self.assertIsNotNone(field_state)
        self.assertEqual(field_state.id, "latest_id")
        self.assertEqual(field_state.version_id, "latest_version_id")
    
    def test_find_version_history(self):
        """Test finding the version history of a field state."""
        # Mock collection.get to return a document
        self.mock_collection.get.return_value = {
            "version_history": json.dumps([
                {"version_id": "v1", "timestamp": "2023-01-01T00:00:00"},
                {"version_id": "v2", "timestamp": "2023-01-02T00:00:00"}
            ])
        }
        
        # Call the method
        version_history = self.repository.find_version_history("test_id")
        
        # Verify the result
        self.assertEqual(len(version_history), 2)
        self.assertEqual(version_history[0]["version_id"], "v1")
        self.assertEqual(version_history[1]["version_id"], "v2")
    
    def test_find_by_version_id(self):
        """Test finding a field state by version ID."""
        # Mock AQL execution to return a document
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "id": "test_id",
            "version_id": "test_version_id",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "effective_dimensionality": 3,
            "principal_dimensions": json.dumps(["dim1", "dim2", "dim3"]),
            "eigenvalues": json.dumps([0.5, 0.3, 0.2]),
            "eigenvectors": json.dumps([
                [0.8, 0.1, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7]
            ]),
            "density_centers": json.dumps([]),
            "density_map_metadata": json.dumps({"has_values": False}),
            "coherence": 0.6,
            "navigability_score": 0.5,
            "stability": 0.7,
            "resonance_patterns": json.dumps([])
        }]
        
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Call the method
        field_state = self.repository.find_by_version_id("test_version_id")
        
        # Verify the result
        self.assertIsNotNone(field_state)
        self.assertEqual(field_state.id, "test_id")
        self.assertEqual(field_state.version_id, "test_version_id")

if __name__ == '__main__':
    unittest.main()
