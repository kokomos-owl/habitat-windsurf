"""
Integration tests for the Claude API with dependent services.

This module tests the integration of the Claude API with dependent services,
including the SignificanceAccretionService and AccretivePatternRAG.
"""

import asyncio
import json
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[4]))

from src.habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter
from src.habitat_evolution.pattern_aware_rag.services.claude_baseline_service import ClaudeBaselineService
from src.habitat_evolution.pattern_aware_rag.services.enhanced_claude_baseline_service import EnhancedClaudeBaselineService
from src.habitat_evolution.pattern_aware_rag.services.significance_accretion_service import SignificanceAccretionService
from src.habitat_evolution.pattern_aware_rag.accretive_pattern_rag import AccretivePatternRAG


class TestClaudeServiceIntegration(unittest.TestCase):
    """Test the integration of Claude API with dependent services."""

    def setUp(self):
        """Set up the test environment."""
        # Check if the API key is available
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            self.skipTest("ANTHROPIC_API_KEY not set. Skipping integration tests.")
        
        # Initialize the Claude adapter
        self.claude_adapter = ClaudeAdapter(api_key=self.api_key)
        
        # Initialize the services
        self.claude_baseline_service = ClaudeBaselineService(claude_adapter=self.claude_adapter)
        self.enhanced_claude_baseline_service = EnhancedClaudeBaselineService(claude_adapter=self.claude_adapter)
        
        # Initialize the SignificanceAccretionService
        self.significance_accretion_service = SignificanceAccretionService()
        
        # Initialize the AccretivePatternRAG
        self.accretive_pattern_rag = AccretivePatternRAG(
            claude_baseline_service=self.claude_baseline_service,
            enhanced_claude_baseline_service=self.enhanced_claude_baseline_service,
            significance_accretion_service=self.significance_accretion_service
        )

    async def _test_significance_accretion(self):
        """Test the SignificanceAccretionService with Claude integration."""
        # Create a test query
        query_id = "test_accretion_" + datetime.now().strftime("%Y%m%d%H%M%S")
        query_text = "How will sea level rise affect coastal infrastructure on Martha's Vineyard?"
        
        # Track initial significance
        initial_significance = await self.significance_accretion_service.get_significance(query_id)
        
        # Process the query through the accretion service
        result = await self.significance_accretion_service.process_query(
            query_id=query_id,
            query_text=query_text,
            claude_baseline_service=self.claude_baseline_service
        )
        
        # Check if significance has accreted
        updated_significance = await self.significance_accretion_service.get_significance(query_id)
        
        print(f"Initial significance: {initial_significance}")
        print(f"Updated significance: {updated_significance}")
        
        # Validate the result
        self.assertIsNotNone(result)
        self.assertIn("enhanced_query", result)
        
        # Validate significance accretion
        self.assertGreater(updated_significance, initial_significance)
        
        return result

    async def _test_accretive_pattern_rag(self):
        """Test the AccretivePatternRAG with Claude integration."""
        # Create a test query
        query_id = "test_rag_" + datetime.now().strftime("%Y%m%d%H%M%S")
        query_text = "What adaptation strategies should Martha's Vineyard implement for sea level rise?"
        
        # Process the query through the RAG system
        result = await self.accretive_pattern_rag.process_query(
            query_id=query_id,
            query_text=query_text
        )
        
        # Validate the result
        self.assertIsNotNone(result)
        self.assertIn("response", result)
        self.assertIn("patterns", result)
        
        print(f"RAG response: {result['response'][:100]}...")
        print(f"Patterns used: {len(result['patterns'])}")
        
        return result

    async def _test_real_and_mock_responses(self):
        """Test that services handle both real and mock responses correctly."""
        # Save the original API key
        original_api_key = self.claude_adapter.api_key
        
        # Test with real API
        print("Testing with real API...")
        real_query = "What are the impacts of extreme weather events on Martha's Vineyard?"
        real_result = await self.claude_baseline_service.enhance_query_baseline(real_query)
        
        # Temporarily set the API key to None to force mock mode
        self.claude_adapter.api_key = None
        self.claude_adapter.use_mock = True
        
        # Test with mock API
        print("Testing with mock API...")
        mock_query = "What are the impacts of extreme weather events on Martha's Vineyard?"
        mock_result = await self.claude_baseline_service.enhance_query_baseline(mock_query)
        
        # Restore the API key
        self.claude_adapter.api_key = original_api_key
        self.claude_adapter.use_mock = not original_api_key
        
        # Validate that both responses have the same structure
        self.assertEqual(set(real_result.keys()), set(mock_result.keys()))
        
        print("Real API response structure:", list(real_result.keys()))
        print("Mock API response structure:", list(mock_result.keys()))
        
        return {
            "real": real_result,
            "mock": mock_result
        }

    async def _run_all_tests(self):
        """Run all tests in sequence."""
        # Test significance accretion
        accretion_result = await self._test_significance_accretion()
        print("\nSignificance accretion test completed successfully")
        
        # Test accretive pattern RAG
        rag_result = await self._test_accretive_pattern_rag()
        print("\nAccretive pattern RAG test completed successfully")
        
        # Test real and mock responses
        api_results = await self._test_real_and_mock_responses()
        print("\nReal and mock API response test completed successfully")

    def test_claude_service_integration(self):
        """Test the complete Claude service integration."""
        asyncio.run(self._run_all_tests())


if __name__ == "__main__":
    unittest.main()
