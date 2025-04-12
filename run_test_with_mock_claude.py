#!/usr/bin/env python
"""
Script to run the integrated climate e2e test with a mock Claude adapter.

This script ensures the test doesn't hang during cross-domain relationship detection
by using a mock Claude adapter when the API key is not available.
"""

import logging
import sys
import os
import json
import uuid
from datetime import datetime
import pytest
from unittest.mock import patch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class MockClaudeAdapter:
    """Mock Claude adapter for testing purposes."""
    
    def __init__(self, api_key=None, model=None):
        """Initialize the mock Claude adapter."""
        self.api_key = api_key
        self.model = model or "claude-3-opus-20240229"
        logger.info(f"Initialized MockClaudeAdapter (model: {self.model})")
    
    def query(self, prompt):
        """
        Mock query method that returns a predefined response.
        
        Args:
            prompt: The prompt to send to Claude
            
        Returns:
            A mock response
        """
        logger.info("Using MockClaudeAdapter.query")
        
        # Extract pattern information from the prompt
        import re
        semantic_patterns_match = re.search(r'SEMANTIC PATTERNS:\s*(\[.*?\])', prompt, re.DOTALL)
        statistical_patterns_match = re.search(r'STATISTICAL PATTERNS:\s*(\[.*?\])', prompt, re.DOTALL)
        
        semantic_patterns = []
        statistical_patterns = []
        
        if semantic_patterns_match:
            try:
                semantic_patterns = json.loads(semantic_patterns_match.group(1))
            except json.JSONDecodeError:
                logger.error("Failed to parse semantic patterns from prompt")
        
        if statistical_patterns_match:
            try:
                statistical_patterns = json.loads(statistical_patterns_match.group(1))
            except json.JSONDecodeError:
                logger.error("Failed to parse statistical patterns from prompt")
        
        # Create mock relationships
        relationships = []
        
        # Limit the number of patterns to process
        max_semantic = min(10, len(semantic_patterns) if semantic_patterns else 5)
        max_statistical = min(5, len(statistical_patterns) if statistical_patterns else 3)
        
        # Create mock relationships
        for i in range(max_semantic):
            for j in range(max_statistical):
                if i % 2 == 0 and j % 2 == 0:  # Create relationships for some pattern pairs
                    relationship = {
                        "semantic_index": i,
                        "statistical_index": j,
                        "related": True,
                        "relationship_type": "temporal_correlation",
                        "strength": "moderate",
                        "description": "Both patterns show similar temporal trends in the same region."
                    }
                    relationships.append(relationship)
        
        # Return mock response
        return json.dumps(relationships)

def run_test_with_mock_claude():
    """Run the integrated climate e2e test with a mock Claude adapter."""
    try:
        # Check if Claude API key is set
        claude_api_key = os.environ.get("CLAUDE_API_KEY")
        if not claude_api_key:
            logger.info("Claude API key not set, using mock Claude adapter")
            
            # Import the Claude adapter
            from src.habitat_evolution.infrastructure.services.claude_adapter import ClaudeAdapter
            
            # Patch the Claude adapter
            with patch("src.habitat_evolution.infrastructure.services.claude_adapter.ClaudeAdapter", MockClaudeAdapter):
                # Run the integrated test
                logger.info("Running integrated climate e2e test with mock Claude adapter...")
                result = pytest.main(["-xvs", "tests/integration/climate_e2e/test_climate_e2e.py::test_integrated_climate_e2e"])
                
                if result != 0:
                    logger.error(f"Test failed with exit code: {result}")
                    return False
        else:
            logger.info("Claude API key is set, using real Claude adapter")
            
            # Run the integrated test
            logger.info("Running integrated climate e2e test with real Claude adapter...")
            result = pytest.main(["-xvs", "tests/integration/climate_e2e/test_climate_e2e.py::test_integrated_climate_e2e"])
            
            if result != 0:
                logger.error(f"Test failed with exit code: {result}")
                return False
        
        logger.info("Test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error running test: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting test with mock Claude adapter")
    success = run_test_with_mock_claude()
    
    if success:
        logger.info("Test with mock Claude adapter completed successfully")
        sys.exit(0)
    else:
        logger.error("Test with mock Claude adapter failed")
        sys.exit(1)
