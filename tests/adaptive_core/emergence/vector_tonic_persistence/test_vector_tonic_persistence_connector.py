"""
Test suite for Vector-Tonic Persistence Connector.

This test suite defines the expected behavior of the connector between
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
from src.habitat_evolution.adaptive_core.emergence.persistence_integration import VectorTonicPersistenceIntegration
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import VectorTonicWindowIntegrator

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] %(message)s')

logger = logging.getLogger(__name__)

# Import the modules we want to test
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector import (
    VectorTonicPersistenceConnector,
    create_connector
)


class TestVectorTonicPersistenceConnector(unittest.TestCase):
    """Test the VectorTonicPersistenceConnector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_event_bus = MagicMock(spec=LocalEventBus)
        self.mock_db = MagicMock()
        self.mock_persistence_integration = MagicMock()
        
        # Create connector with mocks
        self.connector = VectorTonicPersistenceConnector(self.mock_event_bus, self.mock_db)
        self.connector.persistence_integration = self.mock_persistence_integration
        self.connector.pattern_service = self.mock_persistence_integration.pattern_service
        self.connector.field_state_service = self.mock_persistence_integration.field_state_service
        self.connector.relationship_service = self.mock_persistence_integration.relationship_service
    
    def test_initialize(self):
        """Test that initialize initializes the persistence integration and subscribes to events."""
        # Call initialize
        self.connector.initialize()
        
        # Verify persistence integration was initialized
        self.mock_persistence_integration.initialize.assert_called_once()
        
        # Verify event subscriptions
        self.mock_event_bus.subscribe.assert_any_call("document.processed", self.connector._on_document_processed)
        self.mock_event_bus.subscribe.assert_any_call("vector.gradient.updated", self.connector._on_vector_gradient_updated)
        self.mock_event_bus.subscribe.assert_any_call("learning.window.closed", self.connector._on_learning_window_closed)
        
        # Verify initialization state
        self.assertTrue(self.connector.initialized)
    
    def test_connect_to_integrator(self):
        """Test connecting to a VectorTonicWindowIntegrator."""
        # Create a mock integrator
        mock_integrator = MagicMock(spec=VectorTonicWindowIntegrator)
        mock_integrator.initialized = False
        
        # Connect to the integrator
        self.connector.connect_to_integrator(mock_integrator)
        
        # Verify integrator was initialized if not already
        mock_integrator.initialize.assert_called_once()
        
        # Verify connection event was published
        self.mock_event_bus.publish.assert_called_once()
        
        # Verify event data
        args, _ = self.mock_event_bus.publish.call_args
        event = args[0]
        self.assertEqual(event.type, "persistence.connected")
        self.assertEqual(event.data["integrator_id"], id(mock_integrator))
    
    def test_process_document(self):
        """Test processing a document."""
        # Create a test document
        document = {
            "id": "test_doc",
            "content": "This is a test document about climate change."
        }
        
        # Configure mock persistence integration
        self.mock_persistence_integration.process_document.return_value = "doc_id"
        
        # Process the document
        result = self.connector.process_document(document)
        
        # Verify document was processed
        self.mock_persistence_integration.process_document.assert_called_with(document)
        
        # Verify result
        self.assertEqual(result, "doc_id")
    
    def test_on_document_processed(self):
        """Test handling document processed events."""
        # Create a mock event
        mock_event = MagicMock()
        mock_event.data = {
            "document_id": "doc_id",
            "entities": [
                {
                    "id": "entity_1",
                    "type": "CONCEPT",
                    "text": "Climate Change",
                    "confidence": 0.9
                },
                {
                    "id": "entity_2",
                    "type": "CONCEPT",
                    "text": "Food Security",
                    "confidence": 0.85
                }
            ],
            "relationships": [
                {
                    "source": "Climate Change",
                    "predicate": "impacts",
                    "target": "Food Security",
                    "confidence": 0.8
                }
            ]
        }
        
        # Call the event handler
        self.connector._on_document_processed(mock_event)
        
        # Verify entity events were published
        self.assertEqual(self.mock_event_bus.publish.call_count, 3)  # 2 entities + 1 relationship
        
        # Verify entity event data
        entity_calls = [
            call for call in self.mock_event_bus.publish.call_args_list 
            if call[0][0].type == "entity.detected"
        ]
        self.assertEqual(len(entity_calls), 2)
        
        # Verify relationship event data
        rel_calls = [
            call for call in self.mock_event_bus.publish.call_args_list 
            if call[0][0].type == "relationship.detected"
        ]
        self.assertEqual(len(rel_calls), 1)
        rel_event = rel_calls[0][0][0]
        self.assertEqual(rel_event.data["source"], "Climate Change")
        self.assertEqual(rel_event.data["predicate"], "impacts")
        self.assertEqual(rel_event.data["target"], "Food Security")
    
    def test_on_vector_gradient_updated(self):
        """Test handling vector gradient updated events."""
        # Create a mock event
        mock_event = MagicMock()
        mock_event.data = {
            "gradient": {
                "field_state_id": "field_state_1",
                "metrics": {
                    "density": 0.65,
                    "turbulence": 0.35,
                    "coherence": 0.75,
                    "stability": 0.8,
                    "pattern_count": 12
                }
            }
        }
        
        # Call the event handler
        self.connector._on_vector_gradient_updated(mock_event)
        
        # Verify field state event was published
        self.mock_event_bus.publish.assert_called_once()
        
        # Verify event data
        args, _ = self.mock_event_bus.publish.call_args
        event = args[0]
        self.assertEqual(event.type, "field.state.updated")
        self.assertEqual(event.data["field_state"]["id"], "field_state_1")
        self.assertEqual(event.data["field_state"]["metrics"]["pattern_count"], 12)
    
    def test_on_learning_window_closed(self):
        """Test handling learning window closed events."""
        # Create a mock event
        mock_event = MagicMock()
        mock_event.data = {
            "window_id": "window_1",
            "patterns": [
                {
                    "id": "pattern_1",
                    "confidence": 0.85,
                    "description": "Climate change pattern"
                },
                {
                    "id": "pattern_2",
                    "confidence": 0.75,
                    "description": "Food security pattern"
                }
            ],
            "field_state": {
                "id": "field_state_1",
                "density": 0.65,
                "turbulence": 0.35
            }
        }
        
        # Call the event handler
        self.connector._on_learning_window_closed(mock_event)
        
        # Verify events were published
        self.assertEqual(self.mock_event_bus.publish.call_count, 3)  # 2 patterns + 1 field state
        
        # Verify pattern event data
        pattern_calls = [
            call for call in self.mock_event_bus.publish.call_args_list 
            if call[0][0].type == "pattern.detected"
        ]
        self.assertEqual(len(pattern_calls), 2)
        
        # Verify field state event data
        field_calls = [
            call for call in self.mock_event_bus.publish.call_args_list 
            if call[0][0].type == "field.state.updated"
        ]
        self.assertEqual(len(field_calls), 1)
        field_event = field_calls[0][0][0]
        self.assertEqual(field_event.data["field_state"]["id"], "field_state_1")
        self.assertEqual(field_event.data["window_id"], "window_1")


class TestVectorTonicPersistenceConnectorFactory(unittest.TestCase):
    """Test the factory function for creating VectorTonicPersistenceConnector instances."""
    
    def test_create_connector(self):
        """Test creating a connector with the factory function."""
        # Mock dependencies
        mock_event_bus = MagicMock(spec=LocalEventBus)
        mock_db = MagicMock()
        
        # Mock the connector class
        mock_connector = MagicMock()
        
        # Patch the connector constructor
        with patch('src.habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector.VectorTonicPersistenceConnector', return_value=mock_connector):
            # Import the factory function
            from src.habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector import create_connector
            
            # Create a connector
            connector = create_connector(mock_event_bus, mock_db)
            
            # Verify connector was created with correct arguments
            from src.habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector import VectorTonicPersistenceConnector
            VectorTonicPersistenceConnector.assert_called_with(mock_event_bus, mock_db)
            
            # Verify connector was initialized
            mock_connector.initialize.assert_called_once()
            
            # Verify the result
            self.assertEqual(connector, mock_connector)


if __name__ == "__main__":
    unittest.main()
