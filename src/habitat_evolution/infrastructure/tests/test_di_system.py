"""
Test for the Dependency Injection system in Habitat Evolution.

This module tests the end-to-end functionality of the DI system,
ensuring that all components are properly registered and can be resolved.
"""

import logging
import unittest
from typing import Dict, Any, List

from src.habitat_evolution.infrastructure.di.service_locator import ServiceLocator
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.vector_tonic_service_interface import VectorTonicServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.pattern_aware_rag_interface import PatternAwareRAGInterface
from src.habitat_evolution.infrastructure.interfaces.services.document_service_interface import DocumentServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.unified_graph_service_interface import UnifiedGraphServiceInterface
from src.habitat_evolution.infrastructure.interfaces.persistence.arangodb_connection_interface import ArangoDBConnectionInterface
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_pattern_repository import ArangoDBPatternRepository
from src.habitat_evolution.infrastructure.adapters.pattern_bridge import PatternBridge
from src.habitat_evolution.adaptive_core.models.pattern import Pattern as AdaptiveCorePattern


class TestDISystem(unittest.TestCase):
    """Test case for the Dependency Injection system."""
    
    def setUp(self):
        """Set up the test case."""
        # Reset the service locator
        ServiceLocator._instance = None
        self.service_locator = ServiceLocator.instance()
        self.service_locator.initialize()
        
    def test_service_resolution(self):
        """Test that all services can be resolved."""
        # Test resolving core services
        event_service = self.service_locator.event_service
        self.assertIsNotNone(event_service)
        
        # Test resolving infrastructure services
        db_connection = self.service_locator.get_service(ArangoDBConnectionInterface)
        self.assertIsNotNone(db_connection)
        
        # Test resolving repositories
        pattern_repository = self.service_locator.pattern_repository
        self.assertIsNotNone(pattern_repository)
        
        # Test resolving adapters
        pattern_bridge = self.service_locator.pattern_bridge
        self.assertIsNotNone(pattern_bridge)
        
        # Test resolving RAG services
        pattern_aware_rag = self.service_locator.pattern_aware_rag
        self.assertIsNotNone(pattern_aware_rag)
        
        # Test resolving vector tonic service
        vector_tonic_service = self.service_locator.vector_tonic_service
        self.assertIsNotNone(vector_tonic_service)
        
        # Test resolving document service
        document_service = self.service_locator.document_service
        self.assertIsNotNone(document_service)
        
        # Test resolving unified graph service
        unified_graph_service = self.service_locator.unified_graph_service
        self.assertIsNotNone(unified_graph_service)
        
    def test_pattern_bridge(self):
        """Test that the PatternBridge works correctly."""
        # Get the pattern bridge
        pattern_bridge = self.service_locator.pattern_bridge
        
        # Create a test pattern
        test_pattern = AdaptiveCorePattern(
            id="test-pattern-1",
            base_concept="test pattern",
            creator_id="test-system",
            coherence=0.8,
            confidence=0.9
        )
        
        # Enhance the pattern with metadata
        enhanced_pattern = pattern_bridge.enhance_pattern(test_pattern)
        
        # Verify that the pattern has metadata
        self.assertTrue(hasattr(enhanced_pattern, 'metadata'))
        self.assertIsNotNone(enhanced_pattern.metadata)
        
        # Verify that the metadata contains the expected values
        self.assertEqual(enhanced_pattern.metadata.get("coherence"), 0.8)
        self.assertEqual(enhanced_pattern.metadata.get("quality"), 0.9)
        
        # Verify that the pattern has text
        self.assertTrue(hasattr(enhanced_pattern, 'text'))
        self.assertEqual(enhanced_pattern.text, "test pattern")
        
    def test_pattern_aware_rag(self):
        """Test that the PatternAwareRAG service works with the PatternBridge."""
        # Get the pattern aware RAG service
        pattern_aware_rag = self.service_locator.pattern_aware_rag
        
        # Verify that it has the pattern bridge
        self.assertTrue(hasattr(pattern_aware_rag, '_pattern_bridge'))
        self.assertIsNotNone(pattern_aware_rag._pattern_bridge)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
