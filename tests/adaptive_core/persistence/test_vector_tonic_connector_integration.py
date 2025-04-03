"""
Integration test for Vector-Tonic Persistence Connector with refactored persistence layer.

This test suite verifies that the VectorTonicPersistenceConnector properly integrates
with the refactored persistence layer, ensuring that events from the vector-tonic-window
system are correctly persisted using the new repository structure.
"""

import unittest
import logging
import os
import sys
from unittest.mock import MagicMock, patch, call
from datetime import datetime
import uuid

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s')

logger = logging.getLogger(__name__)

# Import core components
from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

# Import the connector
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector import (
    VectorTonicPersistenceConnector,
    create_connector
)

# Import the refactored persistence layer
from src.habitat_evolution.adaptive_core.persistence.factory import create_repositories
from src.habitat_evolution.adaptive_core.persistence.interfaces.field_state_repository import FieldStateRepositoryInterface
from src.habitat_evolution.adaptive_core.persistence.interfaces.pattern_repository import PatternRepositoryInterface
from src.habitat_evolution.adaptive_core.persistence.interfaces.relationship_repository import RelationshipRepositoryInterface
from src.habitat_evolution.adaptive_core.persistence.interfaces.topology_repository import TopologyRepositoryInterface


class TestVectorTonicConnectorWithRefactoredPersistence(unittest.TestCase):
    """
    Test the integration between VectorTonicPersistenceConnector and the refactored persistence layer.
    
    This test suite verifies that:
    1. The connector correctly uses the refactored repository interfaces
    2. Events are properly persisted using the new repository structure
    3. The factory methods correctly create and wire up all components
    """
    
    def setUp(self):
        """Set up test fixtures."""
        logger.debug("Setting up test fixtures")
        
        # Mock the event bus
        self.mock_event_bus = MagicMock(spec=LocalEventBus)
        
        # Mock the database connection
        self.mock_db = MagicMock()
        
        # Create mock repositories
        self.mock_field_state_repo = MagicMock(spec=FieldStateRepositoryInterface)
        self.mock_pattern_repo = MagicMock(spec=PatternRepositoryInterface)
        self.mock_relationship_repo = MagicMock(spec=RelationshipRepositoryInterface)
        self.mock_topology_repo = MagicMock(spec=TopologyRepositoryInterface)
        
        # Set up the repositories dictionary
        self.repositories = {
            "field_state_repository": self.mock_field_state_repo,
            "pattern_repository": self.mock_pattern_repo,
            "relationship_repository": self.mock_relationship_repo,
            "topology_repository": self.mock_topology_repo
        }
        
        logger.debug("Test fixtures set up complete")
    
    @patch('src.habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector.create_repositories')
    def test_connector_uses_refactored_repositories(self, mock_create_repositories):
        """Test that the connector correctly uses the refactored repositories."""
        logger.debug("Starting test_connector_uses_refactored_repositories")
        
        # Configure the mock to return our repositories
        mock_create_repositories.return_value = self.repositories
        logger.debug(f"Configured mock_create_repositories to return: {self.repositories}")
        
        # Create the connector
        connector = create_connector(
            event_bus=self.mock_event_bus,
            db=self.mock_db
        )
        logger.debug(f"Created connector: {connector}")
        
        # Verify that create_repositories was called with the correct arguments
        mock_create_repositories.assert_called_once_with(self.mock_db)
        logger.debug("Verified create_repositories was called correctly")
        
        # Verify that the connector has the correct repositories
        self.assertEqual(connector.field_state_repository, self.mock_field_state_repo)
        self.assertEqual(connector.pattern_repository, self.mock_pattern_repo)
        self.assertEqual(connector.relationship_repository, self.mock_relationship_repo)
        self.assertEqual(connector.topology_repository, self.mock_topology_repo)
        logger.debug("Verified connector has correct repositories")
        
        logger.debug("test_connector_uses_refactored_repositories completed successfully")
    
    @patch('src.habitat_evolution.adaptive_core.persistence.factory.create_repositories')
    @patch('src.habitat_evolution.core.services.event_bus.Event')
    def test_pattern_detected_event_persistence(self, mock_event, mock_create_repositories):
        """Test that pattern detected events are correctly persisted using the refactored repositories."""
        logger.debug("Starting test_pattern_detected_event_persistence")
        
        # Configure the mock to return our repositories
        mock_create_repositories.return_value = self.repositories
        
        # Create the connector
        connector = create_connector(
            event_bus=self.mock_event_bus,
            db=self.mock_db
        )
        
        # Create a test pattern
        pattern_id = str(uuid.uuid4())
        pattern_data = {
            "id": pattern_id,
            "name": "Test Pattern",
            "vector": [0.1, 0.2, 0.3],
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat()
        }
        
        # Call the on_pattern_detected method
        connector.on_pattern_detected(
            pattern_id=pattern_id,
            pattern_data=pattern_data,
            metadata={"source": "test"}
        )
        
        # Verify that the pattern repository's save method was called
        self.mock_pattern_repo.save.assert_called_once()
        logger.debug("Verified pattern_repository.save was called")
        
        # Get the argument that was passed to save
        saved_pattern = self.mock_pattern_repo.save.call_args[0][0]
        
        # Verify the pattern properties
        self.assertEqual(saved_pattern.id, pattern_id)
        logger.debug("Verified saved pattern has correct ID")
        
        logger.debug("test_pattern_detected_event_persistence completed successfully")
    
    @patch('src.habitat_evolution.adaptive_core.persistence.factory.create_repositories')
    @patch('src.habitat_evolution.core.services.event_bus.Event')
    def test_field_state_updated_event_persistence(self, mock_event, mock_create_repositories):
        """Test that field state updated events are correctly persisted using the refactored repositories."""
        logger.debug("Starting test_field_state_updated_event_persistence")
        
        # Configure the mock to return our repositories
        mock_create_repositories.return_value = self.repositories
        
        # Create the connector
        connector = create_connector(
            event_bus=self.mock_event_bus,
            db=self.mock_db
        )
        
        # Create a test field state
        field_id = str(uuid.uuid4())
        field_state = {
            "id": field_id,
            "name": "Test Field",
            "state_vector": [0.4, 0.5, 0.6],
            "stability": 0.75,
            "timestamp": datetime.now().isoformat()
        }
        
        # Call the on_field_state_updated method
        connector.on_field_state_updated(
            field_id=field_id,
            field_state=field_state,
            metadata={"source": "test"}
        )
        
        # Verify that the field state repository's save method was called
        self.mock_field_state_repo.save.assert_called_once()
        logger.debug("Verified field_state_repository.save was called")
        
        # Get the argument that was passed to save
        saved_field_state = self.mock_field_state_repo.save.call_args[0][0]
        
        # Verify the field state properties
        self.assertEqual(saved_field_state.id, field_id)
        logger.debug("Verified saved field state has correct ID")
        
        logger.debug("test_field_state_updated_event_persistence completed successfully")
    
    @patch('src.habitat_evolution.adaptive_core.persistence.factory.create_repositories')
    @patch('src.habitat_evolution.core.services.event_bus.Event')
    def test_pattern_relationship_detected_event_persistence(self, mock_event, mock_create_repositories):
        """Test that pattern relationship detected events are correctly persisted using the refactored repositories."""
        logger.debug("Starting test_pattern_relationship_detected_event_persistence")
        
        # Configure the mock to return our repositories
        mock_create_repositories.return_value = self.repositories
        
        # Create the connector
        connector = create_connector(
            event_bus=self.mock_event_bus,
            db=self.mock_db
        )
        
        # Create test pattern IDs
        source_pattern_id = str(uuid.uuid4())
        target_pattern_id = str(uuid.uuid4())
        relationship_id = str(uuid.uuid4())
        
        # Create relationship data
        relationship_data = {
            "id": relationship_id,
            "source_id": source_pattern_id,
            "target_id": target_pattern_id,
            "type": "resonance",
            "strength": 0.8,
            "properties": {"direction": "bidirectional"},
            "timestamp": datetime.now().isoformat()
        }
        
        # Call the on_pattern_relationship_detected method
        connector.on_pattern_relationship_detected(
            source_id=source_pattern_id,
            target_id=target_pattern_id,
            relationship_type="resonance",
            relationship_data=relationship_data,
            metadata={"source": "test"}
        )
        
        # Verify that the relationship repository's save method was called
        self.mock_relationship_repo.save.assert_called_once()
        logger.debug("Verified relationship_repository.save was called")
        
        # Get the argument that was passed to save
        saved_relationship = self.mock_relationship_repo.save.call_args[0][0]
        
        # Verify the relationship properties
        self.assertEqual(saved_relationship.id, relationship_id)
        self.assertEqual(saved_relationship.source_id, source_pattern_id)
        self.assertEqual(saved_relationship.target_id, target_pattern_id)
        logger.debug("Verified saved relationship has correct properties")
        
        logger.debug("test_pattern_relationship_detected_event_persistence completed successfully")
    
    @patch('src.habitat_evolution.adaptive_core.persistence.factory.create_repositories')
    @patch('src.habitat_evolution.core.services.event_bus.Event')
    def test_topology_change_event_persistence(self, mock_event, mock_create_repositories):
        """Test that topology change events are correctly persisted using the refactored repositories."""
        logger.debug("Starting test_topology_change_event_persistence")
        
        # Configure the mock to return our repositories
        mock_create_repositories.return_value = self.repositories
        
        # Create the connector
        connector = create_connector(
            event_bus=self.mock_event_bus,
            db=self.mock_db
        )
        
        # Create test field ID
        field_id = str(uuid.uuid4())
        
        # Create topology data
        previous_topology = {
            "nodes": [{"id": "node1"}, {"id": "node2"}],
            "edges": [{"source": "node1", "target": "node2"}]
        }
        
        new_topology = {
            "nodes": [{"id": "node1"}, {"id": "node2"}, {"id": "node3"}],
            "edges": [
                {"source": "node1", "target": "node2"},
                {"source": "node2", "target": "node3"}
            ]
        }
        
        # Call the on_topology_change method
        connector.on_topology_change(
            field_id=field_id,
            previous_topology=previous_topology,
            new_topology=new_topology,
            metadata={"source": "test"}
        )
        
        # Verify that the topology repository's save method was called
        self.mock_topology_repo.save.assert_called_once()
        logger.debug("Verified topology_repository.save was called")
        
        # Get the argument that was passed to save
        saved_topology = self.mock_topology_repo.save.call_args[0][0]
        
        # Verify the topology properties
        self.assertEqual(saved_topology.field_id, field_id)
        logger.debug("Verified saved topology has correct field ID")
        
        logger.debug("test_topology_change_event_persistence completed successfully")


if __name__ == "__main__":
    # Print some system information
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Python path: {sys.path}")
    logger.debug(f"Current working directory: {os.getcwd()}")
    
    # Run the tests
    unittest.main()
