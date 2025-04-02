"""
Test suite for Vector-Tonic Persistence Integration.

This test suite defines the expected behavior of the integration between
the vector-tonic-window system and the ArangoDB persistence layer.
"""

import unittest
import logging
import os
import sys
from unittest.mock import MagicMock, patch
from datetime import datetime
import uuid

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from src.habitat_evolution.adaptive_core.persistence.arangodb.connection import ArangoDBConnectionManager
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.pattern_repository import PatternRepository
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.field_state_repository import TonicHarmonicFieldStateRepository
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.predicate_relationship_repository import PredicateRelationshipRepository

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] %(message)s')

logger = logging.getLogger(__name__)

# Import the modules we want to test
from src.habitat_evolution.adaptive_core.emergence.persistence_integration import (
    AdaptiveIDRepository,
    PatternPersistenceService,
    FieldStatePersistenceService,
    RelationshipPersistenceService,
    VectorTonicPersistenceIntegration
)


class TestAdaptiveIDRepository(unittest.TestCase):
    """Test the AdaptiveIDRepository class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the ArangoDB connection
        self.mock_db = MagicMock()
        self.mock_collection = MagicMock()
        self.mock_version_collection = MagicMock()
        
        # Configure mock returns
        self.mock_db.has_collection.return_value = False
        self.mock_db.collection.side_effect = lambda name: self.mock_collection if name == "AdaptiveID" else self.mock_version_collection
        
        # Create repository with mock DB
        self.repo = AdaptiveIDRepository(self.mock_db)
    
    def test_initialize(self):
        """Test that initialize creates collections if they don't exist."""
        # Call initialize
        self.repo.initialize()
        
        # Verify collections were created
        self.mock_db.has_collection.assert_any_call("AdaptiveID")
        self.mock_db.has_collection.assert_any_call("AdaptiveIDVersion")
        self.mock_db.create_collection.assert_any_call("AdaptiveID")
        self.mock_db.create_collection.assert_any_call("AdaptiveIDVersion")
    
    def test_save_new_adaptive_id(self):
        """Test saving a new AdaptiveID."""
        # Create an AdaptiveID to save
        adaptive_id = AdaptiveID(
            base_concept="test_concept",
            creator_id="test_creator"
        )
        
        # Configure mock for new document
        self.mock_collection.has.return_value = False
        self.mock_collection.insert.return_value = {"_id": f"AdaptiveID/{adaptive_id.id}"}
        
        # Save the AdaptiveID
        result = self.repo.save(adaptive_id)
        
        # Verify the result
        self.assertEqual(result, f"AdaptiveID/{adaptive_id.id}")
        
        # Verify insert was called with correct document
        self.mock_collection.insert.assert_called_once()
        args, _ = self.mock_collection.insert.call_args
        doc = args[0]
        self.assertEqual(doc["_key"], adaptive_id.id)
        self.assertEqual(doc["base_concept"], "test_concept")
        self.assertEqual(doc["creator_id"], "test_creator")
    
    def test_save_existing_adaptive_id(self):
        """Test updating an existing AdaptiveID."""
        # Create an AdaptiveID to save
        adaptive_id = AdaptiveID(
            base_concept="test_concept",
            creator_id="test_creator"
        )
        
        # Configure mock for existing document
        self.mock_collection.has.return_value = True
        
        # Save the AdaptiveID
        result = self.repo.save(adaptive_id)
        
        # Verify the result
        self.assertEqual(result, f"AdaptiveID/{adaptive_id.id}")
        
        # Verify replace was called with correct document
        self.mock_collection.replace.assert_called_once()
        args, _ = self.mock_collection.replace.call_args
        self.assertEqual(args[0], adaptive_id.id)
    
    def test_save_versions(self):
        """Test saving versions of an AdaptiveID."""
        # Create an AdaptiveID with versions
        adaptive_id = AdaptiveID(
            base_concept="test_concept",
            creator_id="test_creator"
        )
        
        # Mock has method for versions
        self.mock_version_collection.has.return_value = False
        
        # Save the versions
        self.repo._save_versions(adaptive_id)
        
        # Verify version was inserted
        self.mock_version_collection.insert.assert_called()
        
        # Verify version data
        args, _ = self.mock_version_collection.insert.call_args
        version_doc = args[0]
        self.assertEqual(version_doc["adaptive_id"], adaptive_id.id)
    
    def test_find_by_id(self):
        """Test finding an AdaptiveID by ID."""
        # Mock the get method
        self.mock_collection.get.return_value = {"_id": "AdaptiveID/test_id", "base_concept": "test_concept"}
        
        # Find by ID
        result = self.repo.find_by_id("test_id")
        
        # Verify the result
        self.assertEqual(result["base_concept"], "test_concept")
        self.mock_collection.get.assert_called_with("test_id")
    
    def test_find_by_base_concept(self):
        """Test finding AdaptiveIDs by base concept."""
        # Mock the AQL execution
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{"_id": "AdaptiveID/test_id", "base_concept": "test_concept"}]
        self.mock_db.aql.execute.return_value = mock_cursor
        
        # Find by base concept
        results = self.repo.find_by_base_concept("test_concept")
        
        # Verify the results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["base_concept"], "test_concept")
        
        # Verify AQL execution
        self.mock_db.aql.execute.assert_called_once()
        args, kwargs = self.mock_db.aql.execute.call_args
        self.assertIn("test_concept", str(kwargs))


class TestPatternPersistenceService(unittest.TestCase):
    """Test the PatternPersistenceService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_event_bus = MagicMock(spec=LocalEventBus)
        self.mock_db = MagicMock()
        self.mock_adaptive_id_repo = MagicMock()
        self.mock_pattern_repo = MagicMock()
        
        # Create service with mocks
        self.service = PatternPersistenceService(self.mock_event_bus, self.mock_db)
        self.service.adaptive_id_repo = self.mock_adaptive_id_repo
        self.service.pattern_repo = self.mock_pattern_repo
    
    def test_initialize(self):
        """Test that initialize subscribes to events."""
        # Call initialize
        self.service.initialize()
        
        # Verify event subscriptions
        self.mock_event_bus.subscribe.assert_any_call("pattern.detected", self.service._on_pattern_detected)
        self.mock_event_bus.subscribe.assert_any_call("pattern.evolved", self.service._on_pattern_evolved)
        self.mock_event_bus.subscribe.assert_any_call("pattern.semantic_boundary", self.service._on_semantic_boundary)
    
    def test_on_pattern_detected(self):
        """Test handling pattern detected events."""
        # Create a mock event
        mock_event = MagicMock()
        mock_event.data = {
            "pattern_id": "pattern_16_Climate Change_impacts_Food Security",
            "pattern_data": {
                "description": "Climate change negatively impacts food security",
                "confidence": 0.85,
                "supporting_evidence": ["temperature increases", "crop yields", "water scarcity"]
            },
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat()
        }
        
        # Configure mock adaptive_id_repo
        self.mock_adaptive_id_repo.save.return_value = "AdaptiveID/test_id"
        
        # Call the event handler
        self.service._on_pattern_detected(mock_event)
        
        # Verify AdaptiveID was created and saved
        self.mock_adaptive_id_repo.save.assert_called_once()
        
        # Verify AdaptiveID properties
        args, _ = self.mock_adaptive_id_repo.save.call_args
        adaptive_id = args[0]
        self.assertEqual(adaptive_id.base_concept, "16_Climate Change_impacts_Food Security")
        self.assertEqual(adaptive_id.confidence, 0.85)
    
    def test_on_pattern_evolved(self):
        """Test handling pattern evolved events."""
        # Create a mock event
        mock_event = MagicMock()
        mock_event.data = {
            "pattern_id": "pattern_16_Climate Change_impacts_Food Security",
            "old_state": {"confidence": 0.75},
            "new_state": {"confidence": 0.85}
        }
        
        # Configure mock adaptive_id_repo
        self.mock_adaptive_id_repo.find_by_base_concept.return_value = [
            {
                "_id": "AdaptiveID/test_id",
                "_key": "test_id",
                "metadata": {"version_count": 1}
            }
        ]
        
        # Call the event handler
        self.service._on_pattern_evolved(mock_event)
        
        # Verify version was created
        self.mock_db.collection.assert_any_call("AdaptiveIDVersion")
        version_collection = self.mock_db.collection.return_value
        version_collection.insert.assert_called_once()
        
        # Verify AdaptiveID was updated
        self.mock_db.collection.assert_any_call("AdaptiveID")
        adaptive_id_collection = self.mock_db.collection.return_value
        adaptive_id_collection.update.assert_called_once()
    
    def test_on_semantic_boundary(self):
        """Test handling semantic boundary events."""
        # Create a mock event
        mock_event = MagicMock()
        mock_event.data = {
            "pattern_id": "pattern_16_Climate Change_impacts_Food Security",
            "from_state": {"confidence": 0.75},
            "to_state": {"confidence": 0.85},
            "boundary_type": "confidence_threshold",
            "field_state_id": "field_state_1"
        }
        
        # Configure mock adaptive_id_repo
        self.mock_adaptive_id_repo.find_by_base_concept.return_value = [
            {
                "_id": "AdaptiveID/test_id",
                "_key": "test_id"
            }
        ]
        
        # Mock boundary repository
        mock_boundary_repo = MagicMock()
        with patch('src.habitat_evolution.pattern_aware_rag.persistence.arangodb.boundary_repository.BoundaryRepository', return_value=mock_boundary_repo):
            # Call the event handler
            self.service._on_semantic_boundary(mock_event)
            
            # Verify boundary was saved
            mock_boundary_repo.save.assert_called_once()
            
            # Verify boundary data
            args, _ = mock_boundary_repo.save.call_args
            boundary_doc = args[0]
            self.assertEqual(boundary_doc["pattern_id"], "pattern_16_Climate Change_impacts_Food Security")
            self.assertEqual(boundary_doc["boundary_type"], "confidence_threshold")


class TestFieldStatePersistenceService(unittest.TestCase):
    """Test the FieldStatePersistenceService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_event_bus = MagicMock(spec=LocalEventBus)
        self.mock_db = MagicMock()
        self.mock_field_state_repo = MagicMock()
        
        # Create service with mocks
        self.service = FieldStatePersistenceService(self.mock_event_bus, self.mock_db)
        self.service.field_state_repo = self.mock_field_state_repo
    
    def test_initialize(self):
        """Test that initialize subscribes to events."""
        # Call initialize
        self.service.initialize()
        
        # Verify event subscriptions
        self.mock_event_bus.subscribe.assert_called_with("field.state.updated", self.service._on_field_state_updated)
    
    def test_on_field_state_updated(self):
        """Test handling field state updated events."""
        # Create a mock event
        mock_event = MagicMock()
        mock_event.data = {
            "field_state": {
                "id": "field_state_1",
                "density": 0.65,
                "turbulence": 0.35,
                "coherence": 0.75,
                "stability": 0.8,
                "metrics": {
                    "pattern_count": 12,
                    "meta_pattern_count": 3,
                    "resonance_density": 0.45
                }
            }
        }
        
        # Call the event handler
        self.service._on_field_state_updated(mock_event)
        
        # Verify field state was saved
        self.mock_field_state_repo.save.assert_called_once()
        
        # Verify field state data
        args, _ = self.mock_field_state_repo.save.call_args
        field_state_doc = args[0]
        self.assertEqual(field_state_doc["id"], "field_state_1")
        self.assertEqual(field_state_doc["density"], 0.65)
        self.assertEqual(field_state_doc["pattern_count"], 12)


class TestRelationshipPersistenceService(unittest.TestCase):
    """Test the RelationshipPersistenceService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_event_bus = MagicMock(spec=LocalEventBus)
        self.mock_db = MagicMock()
        self.mock_adaptive_id_repo = MagicMock()
        self.mock_predicate_repo = MagicMock()
        
        # Create service with mocks
        self.service = RelationshipPersistenceService(self.mock_event_bus, self.mock_db)
        self.service.adaptive_id_repo = self.mock_adaptive_id_repo
        self.service.predicate_repo = self.mock_predicate_repo
    
    def test_initialize(self):
        """Test that initialize subscribes to events."""
        # Call initialize
        self.service.initialize()
        
        # Verify event subscriptions
        self.mock_event_bus.subscribe.assert_called_with("relationship.detected", self.service._on_relationship_detected)
    
    def test_on_relationship_detected(self):
        """Test handling relationship detected events."""
        # Create a mock event
        mock_event = MagicMock()
        mock_event.data = {
            "source": "Climate Change",
            "predicate": "impacts",
            "target": "Food Security",
            "confidence": 0.75,
            "harmonic_properties": {
                "resonance": 0.65,
                "coherence": 0.7
            }
        }
        
        # Configure mock adaptive_id_repo
        self.mock_adaptive_id_repo.find_by_base_concept.side_effect = [
            [{"_id": "AdaptiveID/source_id"}],  # For source
            [{"_id": "AdaptiveID/target_id"}]   # For target
        ]
        
        # Call the event handler
        self.service._on_relationship_detected(mock_event)
        
        # Verify relationship was saved
        self.mock_predicate_repo.save_relationship.assert_called_once()
        
        # Verify relationship data
        args, _ = self.mock_predicate_repo.save_relationship.call_args
        source_id, predicate, target_id, properties = args
        self.assertEqual(source_id, "AdaptiveID/source_id")
        self.assertEqual(predicate, "impacts")
        self.assertEqual(target_id, "AdaptiveID/target_id")
        self.assertEqual(properties["confidence"], 0.75)
        self.assertEqual(properties["harmonic_properties"]["resonance"], 0.65)
    
    def test_find_or_create_adaptive_id_existing(self):
        """Test finding an existing AdaptiveID."""
        # Configure mock adaptive_id_repo
        self.mock_adaptive_id_repo.find_by_base_concept.return_value = [
            {"_id": "AdaptiveID/existing_id"}
        ]
        
        # Find or create
        result = self.service._find_or_create_adaptive_id("existing_concept")
        
        # Verify result
        self.assertEqual(result, "existing_id")
        
        # Verify find was called
        self.mock_adaptive_id_repo.find_by_base_concept.assert_called_with("existing_concept")
        
        # Verify save was not called
        self.mock_adaptive_id_repo.save.assert_not_called()
    
    def test_find_or_create_adaptive_id_new(self):
        """Test creating a new AdaptiveID."""
        # Configure mock adaptive_id_repo
        self.mock_adaptive_id_repo.find_by_base_concept.return_value = []
        self.mock_adaptive_id_repo.save.return_value = "AdaptiveID/new_id"
        
        # Find or create
        result = self.service._find_or_create_adaptive_id("new_concept")
        
        # Verify result
        self.assertEqual(result, "new_id")
        
        # Verify find was called
        self.mock_adaptive_id_repo.find_by_base_concept.assert_called_with("new_concept")
        
        # Verify save was called
        self.mock_adaptive_id_repo.save.assert_called_once()
        
        # Verify AdaptiveID properties
        args, _ = self.mock_adaptive_id_repo.save.call_args
        adaptive_id = args[0]
        self.assertEqual(adaptive_id.base_concept, "new_concept")
        self.assertEqual(adaptive_id.creator_id, "relationship_detector")


class TestVectorTonicPersistenceIntegration(unittest.TestCase):
    """Test the VectorTonicPersistenceIntegration class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_event_bus = MagicMock(spec=LocalEventBus)
        self.mock_db = MagicMock()
        
        # Mock services
        self.mock_pattern_service = MagicMock()
        self.mock_relationship_service = MagicMock()
        self.mock_field_state_service = MagicMock()
        
        # Create integration with mocks
        self.integration = VectorTonicPersistenceIntegration(self.mock_event_bus, self.mock_db)
        self.integration.pattern_service = self.mock_pattern_service
        self.integration.relationship_service = self.mock_relationship_service
        self.integration.field_state_service = self.mock_field_state_service
    
    def test_initialize(self):
        """Test that initialize initializes all services."""
        # Call initialize
        self.integration.initialize()
        
        # Verify services were initialized
        self.mock_pattern_service.initialize.assert_called_once()
        self.mock_relationship_service.initialize.assert_called_once()
        self.mock_field_state_service.initialize.assert_called_once()
    
    def test_process_document(self):
        """Test processing a document."""
        # Create a test document
        document = {
            "id": "test_doc",
            "content": "This is a test document about climate change."
        }
        
        # Mock adaptive_id_repo
        mock_adaptive_id_repo = MagicMock()
        mock_adaptive_id_repo.save.return_value = "AdaptiveID/doc_id"
        
        # Patch the AdaptiveIDRepository constructor
        with patch('src.habitat_evolution.adaptive_core.emergence.persistence_integration.AdaptiveIDRepository', return_value=mock_adaptive_id_repo):
            # Process the document
            result = self.integration.process_document(document)
            
            # Verify AdaptiveID was created and saved
            mock_adaptive_id_repo.initialize.assert_called_once()
            mock_adaptive_id_repo.save.assert_called_once()
            
            # Verify event was published
            self.mock_event_bus.publish.assert_called_once()
            
            # Verify event data
            args, _ = self.mock_event_bus.publish.call_args
            event = args[0]
            self.assertEqual(event.type, "document.processing")
            self.assertEqual(event.data["content"], "This is a test document about climate change.")


if __name__ == "__main__":
    unittest.main()
