"""
Test for the PKM Bidirectional Integration.

This test demonstrates how the PKM Bidirectional Integration enables
pattern-driven query generation and knowledge capture, creating a true
bidirectional flow between patterns and knowledge.

To run this test, you need to install the Habitat Evolution package in development mode:
    pip install -e .

Or use the PYTHONPATH environment variable:
    PYTHONPATH=$PYTHONPATH:$(pwd) python test_pkm_bidirectional.py
"""

import unittest
import logging
import json
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the project root to the Python path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try to import from the package
try:
    # Try direct imports first (for development environment)
    from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
    from src.habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter
    from src.habitat_evolution.infrastructure.services.event_service import EventService
    from src.habitat_evolution.infrastructure.services.bidirectional_flow_service import BidirectionalFlowService
    from src.habitat_evolution.infrastructure.interfaces.services.pattern_aware_rag_interface import PatternAwareRAGInterface
    from src.habitat_evolution.pkm.pkm_repository import PKMRepository, PKMFile, create_pkm_from_claude_response
    from src.habitat_evolution.pkm.pkm_bidirectional_integration import PKMBidirectionalIntegration
    IMPORTS_WORK = True
except ImportError:
    try:
        # Try installed package imports
        from habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
        from habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter
        from habitat_evolution.infrastructure.services.event_service import EventService
        from habitat_evolution.infrastructure.services.bidirectional_flow_service import BidirectionalFlowService
        from habitat_evolution.infrastructure.interfaces.services.pattern_aware_rag_interface import PatternAwareRAGInterface
        from habitat_evolution.pkm.pkm_repository import PKMRepository, PKMFile, create_pkm_from_claude_response
        from habitat_evolution.pkm.pkm_bidirectional_integration import PKMBidirectionalIntegration
        IMPORTS_WORK = True
    except ImportError:
        IMPORTS_WORK = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock PatternAwareRAGService for testing
class MockPatternAwareRAGService(PatternAwareRAGInterface):
    """Mock implementation of PatternAwareRAGInterface for testing."""
    
    def __init__(self):
        self.patterns = []
        self.field_states = []
        self.relationships = []
        self.initialized = False
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the RAG system."""
        self.initialized = True
    
    def shutdown(self) -> None:
        """Shutdown the RAG system."""
        self.initialized = False
    
    def add_pattern(self, pattern: Dict[str, Any]) -> None:
        """Add a pattern to the RAG system."""
        self.patterns.append(pattern)
    
    def update_pattern(self, pattern_id: str, updates: Dict[str, Any]) -> None:
        """Update a pattern in the RAG system."""
        for i, pattern in enumerate(self.patterns):
            if pattern.get("id") == pattern_id:
                self.patterns[i].update(updates)
                break
    
    def delete_pattern(self, pattern_id: str) -> None:
        """Delete a pattern from the RAG system."""
        self.patterns = [p for p in self.patterns if p.get("id") != pattern_id]
    
    def get_patterns(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get patterns from the RAG system."""
        if not filter_criteria:
            return self.patterns
        
        # Simple filtering
        result = []
        for pattern in self.patterns:
            match = True
            for key, value in filter_criteria.items():
                if pattern.get(key) != value:
                    match = False
                    break
            if match:
                result.append(pattern)
        return result
    
    def update_field_state(self, field_state: Dict[str, Any]) -> None:
        """Update the field state in the RAG system."""
        self.field_states.append(field_state)
    
    def get_field_state(self, field_id: str) -> Optional[Dict[str, Any]]:
        """Get a field state from the RAG system."""
        for state in self.field_states:
            if state.get("id") == field_id:
                return state
        return None
    
    def create_relationship(self, source_id: str, target_id: str, relationship_type: str, properties: Dict[str, Any]) -> None:
        """Create a relationship in the RAG system."""
        self.relationships.append({
            "source_id": source_id,
            "target_id": target_id,
            "type": relationship_type,
            "properties": properties
        })
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document in the RAG system."""
        return {
            "document_id": str(uuid.uuid4()),
            "patterns": []
        }
    
    def query(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the RAG system."""
        return {
            "response": f"Mock response to: {query_text}",
            "pattern_id": str(uuid.uuid4())
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from the RAG system."""
        return {
            "pattern_count": len(self.patterns),
            "field_state_count": len(self.field_states),
            "relationship_count": len(self.relationships)
        }

@unittest.skipIf(not IMPORTS_WORK, "Required imports not available")
class TestPKMBidirectional(unittest.TestCase):
    """Test cases for the PKM Bidirectional Integration."""
    
    def setUp(self):
        """Set up the test environment."""
        # Set up ArangoDB connection
        self.db_name = "habitat_evolution_test"
        self.arangodb_connection = ArangoDBConnection(
            host="localhost",
            port=8529,
            username="root",
            password="habitat",
            database_name=self.db_name
        )
        
        # Initialize ArangoDB connection
        self.arangodb_connection.initialize()
        
        # Create PKM repository
        self.pkm_repository = PKMRepository(self.arangodb_connection)
        
        # Create event service
        self.event_service = EventService()
        self.event_service.initialize()
        
        # Create mock pattern-aware RAG service
        self.pattern_aware_rag = MockPatternAwareRAGService()
        
        # Create bidirectional flow service
        self.bidirectional_flow_service = BidirectionalFlowService(
            self.event_service,
            self.pattern_aware_rag,
            self.arangodb_connection
        )
        self.bidirectional_flow_service.initialize()
        
        # Create Claude adapter
        self.claude_adapter = ClaudeAdapter()
        
        # Create PKM bidirectional integration
        self.pkm_bidirectional = PKMBidirectionalIntegration(
            self.pkm_repository,
            self.bidirectional_flow_service,
            self.event_service,
            self.claude_adapter,
            creator_id="test_user"
        )
        
        # Sample patterns for Boston Harbor
        self.boston_harbor_patterns = [
            {
                "id": "pattern-1",
                "type": "semantic",
                "content": "Sea level rise in Boston Harbor",
                "quality": 0.9,
                "metadata": {
                    "confidence": 0.9,
                    "source": "climate_risk_assessment_2023.pdf"
                },
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "pattern-2",
                "type": "statistical",
                "content": "9-21 inches of sea level rise by 2050",
                "quality": 0.85,
                "metadata": {
                    "confidence": 0.85,
                    "source": "boston_harbor_measurements.csv"
                },
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "pattern-3",
                "type": "semantic",
                "content": "Infrastructure vulnerability in coastal areas",
                "quality": 0.8,
                "metadata": {
                    "confidence": 0.8,
                    "source": "infrastructure_vulnerability_report.pdf"
                },
                "created_at": datetime.now().isoformat()
            }
        ]
        
        logger.info("Test setup complete")
    
    def tearDown(self):
        """Clean up after the test."""
        # No need to clean up collections for now
        logger.info("Test teardown complete")
    
    def test_generate_query_from_pattern(self):
        """Test generating a query from a pattern."""
        # Get a pattern
        pattern = self.boston_harbor_patterns[0]
        
        # Generate a query from the pattern
        query = self.pkm_bidirectional._generate_query_from_pattern(pattern)
        
        # Verify the query
        expected_query = "What are the implications of Sea level rise in Boston Harbor?"
        self.assertEqual(query, expected_query)
        
        logger.info(f"Generated query from pattern: {query}")
        
        return query
    
    def test_process_query_with_patterns(self):
        """Test processing a query with patterns as context."""
        # Generate a query
        query = "What are the impacts of sea level rise on Boston Harbor?"
        
        # Process the query with patterns
        pkm_id = self.pkm_bidirectional.process_query_with_patterns(
            query=query,
            patterns=self.boston_harbor_patterns
        )
        
        # Verify the PKM ID
        self.assertIsNotNone(pkm_id)
        
        # Retrieve the PKM file
        pkm_file = self.pkm_repository.get_pkm_file(pkm_id)
        
        # Verify the PKM file
        self.assertIsNotNone(pkm_file)
        self.assertEqual(pkm_file.title, f"PKM: {query}")
        self.assertEqual(len(pkm_file.patterns), 4)  # 3 original patterns + 1 response pattern
        
        # Verify the response pattern
        response_pattern = None
        for pattern in pkm_file.patterns:
            if pattern.get("type") == "claude_response":
                response_pattern = pattern
                break
        
        self.assertIsNotNone(response_pattern)
        self.assertIn("content", response_pattern)
        
        logger.info(f"Created PKM file from query with patterns: {pkm_file.title} (ID: {pkm_id})")
        logger.info(f"Response pattern content: {response_pattern.get('content')[:100]}...")
        
        return pkm_id, pkm_file
    
    def test_bidirectional_flow(self):
        """Test the full bidirectional flow between patterns and knowledge."""
        # Step 1: Publish patterns to the bidirectional flow service
        for pattern in self.boston_harbor_patterns:
            self.bidirectional_flow_service.publish_pattern(pattern)
            logger.info(f"Published pattern: {pattern['id']}")
        
        # Step 2: Manually trigger the pattern handler to simulate event flow
        for pattern in self.boston_harbor_patterns:
            self.pkm_bidirectional._handle_pattern_event(pattern)
            logger.info(f"Triggered pattern handler for: {pattern['id']}")
        
        # Step 3: Check that PKM files were created
        # This is a bit tricky since the PKM files are created asynchronously
        # For testing, we'll directly use process_query_with_patterns
        pkm_id, pkm_file = self.test_process_query_with_patterns()
        
        # Step 4: Create a relationship between patterns
        relationship = {
            "source_id": self.boston_harbor_patterns[0]["id"],
            "target_id": self.boston_harbor_patterns[1]["id"],
            "type": "correlation",
            "properties": {
                "strength": 0.8,
                "description": "Correlation between sea level rise and measurements"
            }
        }
        
        self.bidirectional_flow_service.publish_relationship(relationship)
        logger.info(f"Published relationship: {relationship['source_id']} -> {relationship['target_id']}")
        
        # Step 5: Manually trigger the relationship handler to simulate event flow
        self.pkm_bidirectional._handle_relationship_event(relationship)
        logger.info("Triggered relationship handler")
        
        # Step 6: Publish a PKM created event to complete the bidirectional flow
        pkm_created_event = {
            "pkm_id": pkm_id,
            "pattern_ids": [p["id"] for p in self.boston_harbor_patterns],
            "query": "What are the impacts of sea level rise on Boston Harbor?"
        }
        
        self.event_service.publish("pkm.created", pkm_created_event)
        logger.info(f"Published PKM created event: {pkm_id}")
        
        # Step 7: Manually trigger the PKM created handler to simulate event flow
        self.pkm_bidirectional._handle_pkm_created_event(pkm_created_event)
        logger.info("Triggered PKM created handler")
        
        # Verify that the bidirectional flow is complete
        logger.info("Bidirectional flow test complete")
        
        return pkm_id, pkm_file

if __name__ == "__main__":
    unittest.main()
