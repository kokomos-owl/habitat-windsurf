"""
Test Climate Risk Document Processing

This module tests the climate risk document processing implementation,
verifying that it can correctly extract patterns from climate risk documents
and store them in ArangoDB with proper versioning and relationship tracking.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add the project root to the Python path to ensure imports work correctly
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.habitat_evolution.climate_risk.document_processing_service import DocumentProcessingService
from src.habitat_evolution.infrastructure.services.pattern_evolution_service import PatternEvolutionService
from src.habitat_evolution.infrastructure.services.claude_pattern_extraction_service import ClaudePatternExtractionService
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection


class TestClimateRiskDocumentProcessing(unittest.TestCase):
    """
    Test the climate risk document processing implementation.
    """
    
    def setUp(self):
        """
        Set up the test environment.
        """
        # Mock ArangoDB connection
        self.mock_arangodb = MagicMock(spec=ArangoDBConnection)
        
        # Mock pattern evolution service
        self.mock_pattern_evolution_service = MagicMock(spec=PatternEvolutionService)
        self.mock_pattern_evolution_service.store_pattern.return_value = "test-pattern-id"
        
        # Create document processing service with mocked dependencies
        self.document_processing_service = DocumentProcessingService(
            pattern_evolution_service=self.mock_pattern_evolution_service,
            arangodb_connection=self.mock_arangodb
        )
        
        # Path to test document
        self.test_document_path = os.path.join(
            project_root, 
            "data", 
            "climate_risk", 
            "climate_risk_marthas_vineyard.txt"
        )
        
    def test_document_exists(self):
        """
        Test that the test document exists.
        """
        self.assertTrue(os.path.exists(self.test_document_path), 
                       f"Test document not found: {self.test_document_path}")
        
    def test_process_document(self):
        """
        Test processing a climate risk document.
        """
        # Process the document
        patterns = self.document_processing_service.process_document(self.test_document_path)
        
        # Verify that patterns were extracted
        self.assertGreater(len(patterns), 0, "No patterns were extracted from the document")
        
        # Verify that the pattern evolution service was called to store patterns
        self.mock_pattern_evolution_service.store_pattern.assert_called()
        
        # Verify that each pattern has the required fields
        for pattern in patterns:
            self.assertIn("id", pattern, "Pattern missing 'id' field")
            self.assertIn("base_concept", pattern, "Pattern missing 'base_concept' field")
            self.assertIn("properties", pattern, "Pattern missing 'properties' field")
            self.assertIn("quality_state", pattern, "Pattern missing 'quality_state' field")
            
            # Verify that the pattern properties include location and source document
            self.assertIn("location", pattern["properties"], "Pattern properties missing 'location'")
            self.assertIn("source_document", pattern["properties"], "Pattern properties missing 'source_document'")
            
    def test_claude_integration(self):
        """
        Test integration with Claude pattern extraction service.
        """
        # Mock Claude pattern extraction service
        mock_claude_service = MagicMock(spec=ClaudePatternExtractionService)
        mock_claude_service.extract_patterns.return_value = [
            {
                "id": "test-claude-pattern",
                "base_concept": "sea_level_rise",
                "creator_id": "claude_test",
                "weight": 1.0,
                "confidence": 0.9,
                "uncertainty": 0.1,
                "coherence": 0.85,
                "phase_stability": 0.8,
                "signal_strength": 0.95,
                "quality_state": "hypothetical",
                "properties": {
                    "location": "Martha's Vineyard",
                    "risk_type": "flooding",
                    "timeframe": "2050",
                    "source_document": "climate_risk_marthas_vineyard.txt"
                }
            }
        ]
        
        # Replace the Claude extraction service in the document processing service
        self.document_processing_service.claude_extraction_service = mock_claude_service
        
        # Process the document
        patterns = self.document_processing_service.process_document(self.test_document_path)
        
        # Verify that the Claude extraction service was called
        mock_claude_service.extract_patterns.assert_called_once()
        
        # Verify that patterns were extracted
        self.assertEqual(len(patterns), 1, "Expected exactly one pattern from Claude extraction")
        
        # Verify that the pattern has the expected values
        pattern = patterns[0]
        self.assertEqual(pattern["base_concept"], "sea_level_rise")
        self.assertEqual(pattern["creator_id"], "claude_test")
        self.assertEqual(pattern["properties"]["location"], "Martha's Vineyard")
        
    def test_pattern_evolution_integration(self):
        """
        Test integration with pattern evolution service.
        """
        # Mock pattern evolution service to return different IDs for each pattern
        pattern_ids = ["pattern-1", "pattern-2", "pattern-3", "pattern-4"]
        self.mock_pattern_evolution_service.store_pattern.side_effect = pattern_ids
        
        # Process the document
        patterns = self.document_processing_service.process_document(self.test_document_path)
        
        # Verify that the pattern evolution service was called for each pattern
        self.assertEqual(
            self.mock_pattern_evolution_service.store_pattern.call_count,
            len(patterns),
            "Pattern evolution service not called for each pattern"
        )
        
        # Verify that each pattern has a unique ID
        pattern_ids = [pattern["id"] for pattern in patterns]
        self.assertEqual(len(pattern_ids), len(set(pattern_ids)), "Pattern IDs are not unique")
        
    def test_arangodb_query_integration(self):
        """
        Test integration with ArangoDB for querying patterns.
        """
        # Mock ArangoDB to return test patterns
        self.mock_arangodb.execute_aql.return_value = [
            {
                "id": "test-db-pattern-1",
                "base_concept": "extreme_drought",
                "quality_state": "hypothetical"
            },
            {
                "id": "test-db-pattern-2",
                "base_concept": "extreme_drought",
                "quality_state": "emergent"
            }
        ]
        
        # Query patterns
        patterns = self.document_processing_service.query_patterns(base_concept="extreme_drought")
        
        # Verify that ArangoDB was queried
        self.mock_arangodb.execute_aql.assert_called_once()
        
        # Verify that patterns were returned
        self.assertEqual(len(patterns), 2, "Expected exactly two patterns from ArangoDB query")
        
        # Verify that the patterns have the expected values
        self.assertEqual(patterns[0]["base_concept"], "extreme_drought")
        self.assertEqual(patterns[1]["quality_state"], "emergent")


if __name__ == "__main__":
    unittest.main()
