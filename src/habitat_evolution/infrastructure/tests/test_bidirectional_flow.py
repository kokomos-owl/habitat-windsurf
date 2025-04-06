"""
Test for the bidirectional flow service in Habitat Evolution.

This module tests the bidirectional flow service, verifying that it correctly
integrates with the DI system and enables communication between components.
"""

import unittest
import logging
from typing import Dict, Any

from src.habitat_evolution.infrastructure.di.service_locator import ServiceLocator
from src.habitat_evolution.infrastructure.interfaces.services.bidirectional_flow_interface import BidirectionalFlowInterface
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.pattern_aware_rag_interface import PatternAwareRAGInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class TestBidirectionalFlow(unittest.TestCase):
    """Test case for the bidirectional flow service."""
    
    def setUp(self):
        """Set up the test case."""
        # Initialize the service locator
        self.service_locator = ServiceLocator()
        
        # Get the services
        self.bidirectional_flow = self.service_locator.get_service(BidirectionalFlowInterface)
        self.event_service = self.service_locator.get_service(EventServiceInterface)
        self.pattern_aware_rag = self.service_locator.get_service(PatternAwareRAGInterface)
        
        # Track events
        self.pattern_events = []
        self.field_state_events = []
        self.relationship_events = []
        
        # Register handlers
        self.bidirectional_flow.register_pattern_handler(self._handle_pattern)
        self.bidirectional_flow.register_field_state_handler(self._handle_field_state)
        self.bidirectional_flow.register_relationship_handler(self._handle_relationship)
        
        # Start the bidirectional flow service
        self.bidirectional_flow.start()
        
    def _handle_pattern(self, event_data: Dict[str, Any]):
        """Handle a pattern event."""
        self.pattern_events.append(event_data)
        
    def _handle_field_state(self, event_data: Dict[str, Any]):
        """Handle a field state event."""
        self.field_state_events.append(event_data)
        
    def _handle_relationship(self, event_data: Dict[str, Any]):
        """Handle a relationship event."""
        self.relationship_events.append(event_data)
        
    def test_service_resolution(self):
        """Test that all services can be resolved."""
        self.assertIsNotNone(self.bidirectional_flow)
        self.assertIsNotNone(self.event_service)
        self.assertIsNotNone(self.pattern_aware_rag)
        
    def test_bidirectional_flow(self):
        """Test the bidirectional flow service."""
        # Verify that the service is running
        self.assertTrue(self.bidirectional_flow.is_running())
        
        # Publish a pattern
        pattern = {
            "id": "test-pattern-1",
            "type": "test",
            "content": "This is a test pattern",
            "metadata": {
                "coherence": 0.8,
                "stability": 0.7
            }
        }
        self.bidirectional_flow.publish_pattern(pattern)
        
        # Verify that the pattern event was received
        self.assertEqual(len(self.pattern_events), 1)
        self.assertEqual(self.pattern_events[0]["id"], "test-pattern-1")
        
        # Publish a field state
        field_state = {
            "id": "test-field-1",
            "state": "active",
            "metrics": {
                "coherence": 0.9,
                "stability": 0.8
            }
        }
        self.bidirectional_flow.publish_field_state(field_state)
        
        # Verify that the field state event was received
        self.assertEqual(len(self.field_state_events), 1)
        self.assertEqual(self.field_state_events[0]["id"], "test-field-1")
        
        # Publish a relationship
        relationship = {
            "source_id": "test-pattern-1",
            "target_id": "test-pattern-2",
            "type": "related",
            "properties": {
                "strength": 0.7
            }
        }
        self.bidirectional_flow.publish_relationship(relationship)
        
        # Verify that the relationship event was received
        self.assertEqual(len(self.relationship_events), 1)
        self.assertEqual(self.relationship_events[0]["source_id"], "test-pattern-1")
        self.assertEqual(self.relationship_events[0]["target_id"], "test-pattern-2")
        
    def test_document_processing(self):
        """Test document processing through the bidirectional flow."""
        # Create a test document
        document = {
            "id": "test-doc-1",
            "content": "This is a test document about climate change and its effects on coastal regions.",
            "metadata": {
                "source": "test",
                "timestamp": "2025-04-06T09:00:00Z"
            }
        }
        
        # Process the document
        result = self.bidirectional_flow.process_document(document)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertIn("bidirectional_flow", result)
        
    def test_query(self):
        """Test querying through the bidirectional flow."""
        # Create a test query
        query = "What are the effects of climate change on coastal regions?"
        context = {
            "user_id": "test-user",
            "session_id": "test-session"
        }
        
        # Query the system
        result = self.bidirectional_flow.query(query, context)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertIn("bidirectional_flow", result)
        
    def tearDown(self):
        """Clean up after the test."""
        # Stop the bidirectional flow service
        self.bidirectional_flow.stop()


if __name__ == "__main__":
    unittest.main()
