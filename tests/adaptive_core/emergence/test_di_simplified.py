"""
Simplified test for Vector-Tonic Persistence Connector dependency injection.

This test suite focuses on testing the dependency injection functionality
without relying on the full application stack.
"""

import unittest
import logging
import os
import sys
from unittest.mock import MagicMock, patch
from datetime import datetime
import uuid

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] %(message)s')

logger = logging.getLogger(__name__)

# Import mock repositories
from tests.adaptive_core.emergence.mocks import (
    MockFieldStateRepository,
    MockPatternRepository,
    MockRelationshipRepository,
    MockTopologyRepository,
    MockVectorTonicPersistenceIntegration
)


# Mock the problematic imports
sys.modules['src.habitat_evolution.adaptive_core.emergence.persistence_integration'] = MagicMock()
sys.modules['src.habitat_evolution.core.services.event_bus'] = MagicMock()


# Now import the connector class
from src.habitat_evolution.adaptive_core.emergence.interfaces.learning_window_observer import LearningWindowState
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector import VectorTonicPersistenceConnector


class TestVectorTonicDISimplified(unittest.TestCase):
    """Test the dependency injection functionality of VectorTonicPersistenceConnector."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock event bus
        self.mock_event_bus = MagicMock()
        self.mock_event_bus.subscribe = MagicMock()
        self.mock_event_bus.publish = MagicMock()
        
        # Mock database
        self.mock_db = MagicMock()
        
        # Create mock repositories
        self.field_state_repository = MockFieldStateRepository()
        self.pattern_repository = MockPatternRepository()
        self.relationship_repository = MockRelationshipRepository()
        self.topology_repository = MockTopologyRepository()
        
        # Create connector with mock dependencies
        self.connector = VectorTonicPersistenceConnector(
            event_bus=self.mock_event_bus,
            db=self.mock_db,
            field_state_repository=self.field_state_repository,
            pattern_repository=self.pattern_repository,
            relationship_repository=self.relationship_repository,
            topology_repository=self.topology_repository
        )
        
        # Mock the persistence_integration
        self.mock_integration = MockVectorTonicPersistenceIntegration(self.mock_db)
        self.connector.persistence_integration = self.mock_integration
    
    def test_initialization_with_repositories(self):
        """Test that the connector correctly initializes with injected repositories."""
        # Verify repositories were correctly assigned
        self.assertEqual(self.connector.field_state_repository, self.field_state_repository)
        self.assertEqual(self.connector.pattern_repository, self.pattern_repository)
        self.assertEqual(self.connector.relationship_repository, self.relationship_repository)
        self.assertEqual(self.connector.topology_repository, self.topology_repository)
    
    def test_pattern_detected_uses_repository(self):
        """Test that the pattern_detected method uses the injected pattern repository."""
        # Create test data
        pattern_id = str(uuid.uuid4())
        pattern_data = {
            "id": pattern_id,
            "name": "Test Pattern",
            "confidence": 0.9
        }
        
        # Call the method
        self.connector.on_pattern_detected(pattern_id, pattern_data)
        
        # Verify pattern repository was used
        self.pattern_repository.save.assert_called_once()
        args, _ = self.pattern_repository.save.call_args
        saved_pattern = args[0]
        self.assertEqual(saved_pattern["id"], pattern_id)
    
    def test_field_state_change_uses_repository(self):
        """Test that the field_state_change method uses the injected field state repository."""
        # Create test data
        field_id = str(uuid.uuid4())
        previous_state = {"id": field_id, "coherence": 0.7}
        new_state = {"id": field_id, "coherence": 0.8}
        
        # Call the method
        self.connector.on_field_state_change(field_id, previous_state, new_state)
        
        # Verify field state repository was used
        self.field_state_repository.save.assert_called_once()
        args, _ = self.field_state_repository.save.call_args
        saved_state = args[0]
        self.assertEqual(saved_state["id"], field_id)
    
    def test_relationship_detected_uses_repository(self):
        """Test that the relationship_detected method uses the injected relationship repository."""
        # Create test data
        source_id = str(uuid.uuid4())
        target_id = str(uuid.uuid4())
        relationship_data = {"type": "SIMILAR_TO", "strength": 0.85}
        
        # Call the method
        self.connector.on_pattern_relationship_detected(source_id, target_id, relationship_data)
        
        # Verify relationship repository was used
        self.relationship_repository.save.assert_called_once()
        args, _ = self.relationship_repository.save.call_args
        saved_relationship = args[0]
        self.assertEqual(saved_relationship["source_id"], source_id)
        self.assertEqual(saved_relationship["target_id"], target_id)
    
    def test_topology_change_uses_repository(self):
        """Test that the topology_change method uses the injected topology repository."""
        # Create test data
        field_id = str(uuid.uuid4())
        previous_topology = {"clusters": 3, "connectivity": 0.6}
        new_topology = {"clusters": 4, "connectivity": 0.7}
        
        # Call the method
        self.connector.on_topology_change(field_id, previous_topology, new_topology)
        
        # Verify topology repository was used
        self.topology_repository.save.assert_called_once()
        args, _ = self.topology_repository.save.call_args
        saved_topology = args[0]
        self.assertEqual(saved_topology["field_id"], field_id)
        self.assertEqual(saved_topology["topology"], new_topology)


if __name__ == "__main__":
    unittest.main()
