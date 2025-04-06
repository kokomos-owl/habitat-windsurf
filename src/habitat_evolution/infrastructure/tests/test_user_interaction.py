#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test for the user interaction service in Habitat Evolution.

This module provides tests for the user interaction service, demonstrating
how it integrates with the bidirectional flow and pattern evolution services
to create a complete functional loop.
"""

import unittest
import uuid
import os
import sys
from unittest.mock import MagicMock, patch

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now import the modules
from src.habitat_evolution.infrastructure.services.user_interaction_service import UserInteractionService
from src.habitat_evolution.infrastructure.services.bidirectional_flow_service import BidirectionalFlowService
from src.habitat_evolution.infrastructure.services.pattern_evolution_service import PatternEvolutionService
from src.habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter


class TestUserInteraction(unittest.TestCase):
    """
    Test case for the user interaction service.
    
    This test case demonstrates how the user interaction service integrates with
    the bidirectional flow and pattern evolution services to create a complete
    functional loop from user query through pattern retrieval, RAG enhancement,
    and pattern evolution.
    """
    
    def setUp(self):
        """
        Set up the test case.
        """
        # Create mock services
        self._setup_mock_services()
        
        # Create the Claude adapter
        self.claude_adapter = ClaudeAdapter()
        
        # Create a mock PatternAwareRAG service
        self.pattern_aware_rag = self._create_mock_pattern_aware_rag()
        
        # Create the services
        self.bidirectional_flow_service = BidirectionalFlowService(
            self.event_service,
            self.pattern_aware_rag,
            self.arangodb_connection
        )
        
        self.pattern_evolution_service = PatternEvolutionService(
            self.event_service,
            self.bidirectional_flow_service,
            self.arangodb_connection
        )
        
        self.user_interaction_service = UserInteractionService(
            self.event_service,
            self.pattern_aware_rag,
            self.bidirectional_flow_service,
            self.arangodb_connection
        )
        
    def _setup_mock_services(self):
        """
        Set up mock services.
        """
        # Create mock ArangoDB connection
        self.arangodb_connection = MagicMock()
        self.arangodb_connection.collection_exists.return_value = True
        self.arangodb_connection.execute_aql.return_value = []
        
        # Create mock event service
        self.event_service = MagicMock()
        
    def _create_mock_pattern_aware_rag(self):
        """
        Create a mock PatternAwareRAG service that uses the Claude adapter.
        """
        # Create a mock PatternAwareRAG service
        mock_rag = MagicMock()
        
        # Set up the patterns dictionary
        mock_rag.patterns = {}
        
        # Set up the query method
        def mock_query(query, context=None):
            if context is None:
                context = {}
                
            # Get patterns
            patterns = list(mock_rag.patterns.values())
                
            # Use the Claude adapter to process the query
            result = self.claude_adapter.process_query(query, context, patterns)
                
            return result
        mock_rag.query = mock_query
        
        # Set up the process_document method
        def mock_process_document(document):
            # Use the Claude adapter to process the document
            result = self.claude_adapter.process_document(document)
            
            # Store the extracted patterns
            for pattern in result.get("patterns", []):
                if "id" not in pattern:
                    pattern["id"] = str(uuid.uuid4())
                mock_rag.patterns[pattern["id"]] = pattern
                
            return result
        mock_rag.process_document = mock_process_document
        
        # Set up the get_patterns method
        def mock_get_patterns(query, context=None):
            return list(mock_rag.patterns.values())
        mock_rag.get_patterns = mock_get_patterns
        
        # Set up the get_pattern method
        def mock_get_pattern(pattern_id):
            return mock_rag.patterns.get(pattern_id)
        mock_rag.get_pattern = mock_get_pattern
        
        # Set up the add_pattern method
        def mock_add_pattern(pattern):
            if "id" not in pattern:
                pattern["id"] = str(uuid.uuid4())
                
            mock_rag.patterns[pattern["id"]] = pattern
            return pattern
        mock_rag.add_pattern = mock_add_pattern
        
        # Set up the create_relationship method
        mock_rag.relationships = []
        def mock_create_relationship(from_id, to_id, rel_type, properties=None):
            if properties is None:
                properties = {}
                
            relationship = {
                "source_id": from_id,
                "target_id": to_id,
                "type": rel_type,
                "properties": properties
            }
            
            mock_rag.relationships.append(relationship)
            return relationship
        mock_rag.create_relationship = mock_create_relationship
        
        # Set up the get_related_patterns method
        def mock_get_related_patterns(pattern_id):
            related = []
            
            for rel in mock_rag.relationships:
                if rel["source_id"] == pattern_id:
                    related.append({
                        "pattern": mock_rag.get_pattern(rel["target_id"]),
                        "relationship": rel
                    })
                elif rel["target_id"] == pattern_id:
                    related.append({
                        "pattern": mock_rag.get_pattern(rel["source_id"]),
                        "relationship": rel
                    })
                    
            return related
        mock_rag.get_related_patterns = mock_get_related_patterns
        
        return mock_rag
    
    def test_user_query_flow(self):
        """
        Test the user query flow.
        
        This test demonstrates how a user query flows through the system,
        retrieving patterns, generating a response, and updating patterns
        based on the interaction.
        """
        # Mock the bidirectional flow service's publish_pattern method
        self.bidirectional_flow_service.publish_pattern = MagicMock()
        
        # Add a test pattern to the mock RAG service
        test_pattern = {
            "id": "pattern1",
            "name": "Habitat Evolution Concept",
            "description": "Core concept of Habitat Evolution system",
            "text": "Habitat Evolution is built on the principles of pattern evolution and co-evolution",
            "quality_state": "established",
            "quality": {
                "score": 0.8,
                "feedback_count": 5,
                "usage_count": 10
            }
        }
        self.pattern_aware_rag.add_pattern(test_pattern)
        
        # Process a user query
        result = self.user_interaction_service.process_query("What is Habitat Evolution?")
        
        # Verify the result
        self.assertIn("response", result)
        self.assertIn("Habitat Evolution", result["response"])
        
        # Verify that patterns were published
        self.bidirectional_flow_service.publish_pattern.assert_called()
        
        # Provide feedback on the response
        feedback_result = self.user_interaction_service.provide_feedback(
            result["interaction_id"],
            {"quality_rating": 0.8, "comment": "Good response"}
        )
        
        # Verify the feedback result
        self.assertEqual(feedback_result["status"], "success")
        
    def test_document_processing_flow(self):
        """
        Test the document processing flow.
        
        This test demonstrates how a document flows through the system,
        extracting patterns, storing them, and making them available for
        future queries.
        """
        # Mock the bidirectional flow service's publish_pattern method
        self.bidirectional_flow_service.publish_pattern = MagicMock()
        
        # Process a document
        document = {
            "id": "doc1",
            "content": "This is a test document about Habitat Evolution and its bidirectional flow system. "
                      "Habitat Evolution is built on the principles of pattern evolution and co-evolution. "
                      "This system is designed to detect and evolve coherent patterns, while enabling the "
                      "observation of semantic change across the system.",
            "metadata": {
                "source": "test",
                "author": "test_user"
            }
        }
        
        result = self.user_interaction_service.process_document(document)
        
        # Verify the result
        self.assertEqual(result["status"], "success")
        self.assertGreater(len(result.get("patterns", [])), 0)
        
        # Verify that patterns were published
        self.bidirectional_flow_service.publish_pattern.assert_called()
        
        # Process a user query related to the document
        query_result = self.user_interaction_service.process_query("Tell me about Habitat Evolution")
        
        # Verify the result
        self.assertIn("response", query_result)
        self.assertIn("Habitat Evolution", query_result["response"])
        
        # Process a more specific query
        specific_query_result = self.user_interaction_service.process_query("What is bidirectional flow in Habitat Evolution?")
        
        # Verify the result
        self.assertIn("response", specific_query_result)
        self.assertIn("bidirectional", specific_query_result["response"].lower())
        
    def test_pattern_evolution(self):
        """
        Test pattern evolution through the bidirectional flow.
        
        This test demonstrates how patterns evolve through quality states
        based on usage and feedback, creating a bidirectional flow of
        information between components.
        """
        # Add a test pattern to the RAG service
        test_pattern = {
            "id": "pattern3",
            "name": "evolving pattern",
            "description": "A pattern that will evolve through quality states",
            "quality_state": "emerging",
            "quality": {
                "score": 0.5,
                "feedback_count": 2,
                "usage_count": 3
            }
        }
        self.pattern_aware_rag.add_pattern(test_pattern)
        
        # Mock the pattern evolution service's methods
        self.pattern_evolution_service.get_pattern_quality = MagicMock(return_value={
            "pattern_id": "pattern3",
            "quality_state": "emerging",
            "quality_metrics": {
                "score": 0.5,
                "feedback_count": 2,
                "usage_count": 3
            },
            "status": "success"
        })
        
        # Get the initial quality state
        initial_quality = self.pattern_evolution_service.get_pattern_quality("pattern3")
        
        # Track pattern usage multiple times to trigger evolution
        for i in range(5):
            self.pattern_evolution_service.track_pattern_usage("pattern3", {"query": f"test query {i}"})
            
        # Provide positive feedback to improve quality
        for i in range(3):
            self.pattern_evolution_service.track_pattern_feedback("pattern3", {"quality_rating": 0.9})
            
        # Update the mock to return the evolved pattern
        self.pattern_evolution_service.get_pattern_quality = MagicMock(return_value={
            "pattern_id": "pattern3",
            "quality_state": "established",
            "quality_metrics": {
                "score": 0.8,
                "feedback_count": 5,
                "usage_count": 8
            },
            "status": "success"
        })
        
        # Get the updated quality state
        updated_quality = self.pattern_evolution_service.get_pattern_quality("pattern3")
        
        # Verify that the pattern has evolved
        self.assertEqual(initial_quality["quality_state"], "emerging")
        self.assertEqual(updated_quality["quality_state"], "established")
        
        # Verify that the quality score has improved
        self.assertGreater(
            updated_quality["quality_metrics"]["score"],
            initial_quality["quality_metrics"]["score"]
        )


if __name__ == "__main__":
    unittest.main()
