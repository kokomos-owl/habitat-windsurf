"""
Test for the ClaudeAdapter query method.

This test validates that the query method in ClaudeAdapter works correctly,
which is essential for the PatternAwareRAG integration.
"""

import unittest
import logging
import asyncio
from typing import Dict, Any, Optional

from src.habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestClaudeAdapterQuery(unittest.TestCase):
    """Test cases for the ClaudeAdapter query method."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a ClaudeAdapter instance without an API key (will use mock responses)
        self.claude_adapter = ClaudeAdapter()
        logger.info("Created ClaudeAdapter instance for testing")
    
    def test_query_method_exists(self):
        """Test that the query method exists on the ClaudeAdapter class."""
        self.assertTrue(hasattr(self.claude_adapter, 'query'), 
                        "ClaudeAdapter should have a 'query' method")
        logger.info("Verified that query method exists on ClaudeAdapter")
    
    def test_query_returns_response(self):
        """Test that the query method returns a response dictionary."""
        query_text = "What are the impacts of sea level rise on Boston Harbor?"
        response = self.claude_adapter.query(query_text)
        
        # Verify that the response is a dictionary
        self.assertIsInstance(response, dict, "Response should be a dictionary")
        
        # Verify that the response contains the expected keys
        self.assertIn("response", response, "Response should contain a 'response' key")
        
        # Log the response for debugging
        logger.info(f"Query response: {response}")
        
        # Since we're using mock responses, we can't verify the exact content
        # but we can verify that it's not empty
        self.assertTrue(response["response"], "Response text should not be empty")
    
    def test_query_with_context(self):
        """Test that the query method accepts and uses context."""
        query_text = "What are the impacts of sea level rise?"
        context = {
            "location": "Boston Harbor",
            "time_period": "2050"
        }
        
        response = self.claude_adapter.query(query_text, context)
        
        # Verify that the response is a dictionary
        self.assertIsInstance(response, dict, "Response should be a dictionary")
        
        # Verify that the response contains the expected keys
        self.assertIn("response", response, "Response should contain a 'response' key")
        
        # Log the response for debugging
        logger.info(f"Query with context response: {response}")
    
    def test_query_error_handling(self):
        """Test that the query method handles errors gracefully."""
        # Create a ClaudeAdapter with a broken process_query method
        class BrokenClaudeAdapter(ClaudeAdapter):
            async def process_query(self, *args, **kwargs):
                raise Exception("Test error")
        
        broken_adapter = BrokenClaudeAdapter()
        
        # Call the query method and verify it handles the error
        response = broken_adapter.query("Test query")
        
        # Verify that the response contains error information
        self.assertIn("error", response, "Response should contain an 'error' key")
        self.assertEqual(response["error"], "Test error", 
                         "Error message should match the raised exception")

if __name__ == "__main__":
    unittest.main()
