#!/usr/bin/env python
"""
Script to fix the hanging issue in cross-domain relationship detection.

This script addresses the issue where the test hangs during cross-domain relationship
detection due to Claude API calls failing when the API key is not set.
"""

import logging
import sys
import os
import pytest
from unittest.mock import patch
import json
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def create_mock_relationships(semantic_patterns, statistical_patterns):
    """
    Create mock relationships between semantic and statistical patterns.
    
    Args:
        semantic_patterns: List of semantic patterns
        statistical_patterns: List of statistical patterns
        
    Returns:
        List of mock relationships
    """
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
                    "source_id": f"semantic-pattern-{uuid.uuid4()}",
                    "target_id": f"statistical-pattern-{uuid.uuid4()}",
                    "related": True,
                    "type": "temporal_correlation",
                    "strength": 0.75,
                    "metadata": {
                        "relationship_type": "temporal_correlation",
                        "strength_label": "moderate",
                        "description": "Both patterns show similar temporal trends in the same region."
                    },
                    "created_at": datetime.now().isoformat()
                }
                relationships.append(relationship)
    
    return relationships

def patch_claude_adapter():
    """
    Patch the Claude adapter to return mock data when the API key is not set.
    
    This function ensures that the test doesn't hang when trying to call the Claude API
    without an API key.
    """
    try:
        # Import the Claude adapter
        from src.habitat_evolution.infrastructure.services.claude_adapter import ClaudeAdapter
        
        # Store the original query method
        original_query = ClaudeAdapter.query
        
        # Define a patched query method
        def patched_query(self, prompt):
            # Check if API key is set
            if not os.environ.get("CLAUDE_API_KEY"):
                logger.warning("CLAUDE_API_KEY not set, returning mock response")
                
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
                relationships = create_mock_relationships(semantic_patterns, statistical_patterns)
                
                # Return mock response
                return json.dumps(relationships)
            else:
                # Call original method if API key is set
                return original_query(self, prompt)
        
        # Apply the patch
        ClaudeAdapter.query = patched_query
        logger.info("Successfully patched ClaudeAdapter.query method")
        return True
    
    except Exception as e:
        logger.error(f"Error patching ClaudeAdapter: {e}")
        return False

def run_test():
    """Run the integrated climate e2e test with patched ClaudeAdapter."""
    try:
        # Patch ClaudeAdapter
        success = patch_claude_adapter()
        if not success:
            logger.error("Failed to patch ClaudeAdapter")
            return False
        
        # Set environment variable to indicate we're using the patched version
        os.environ["USING_PATCHED_CLAUDE_ADAPTER"] = "true"
        
        # Run the integrated test
        logger.info("Running integrated climate e2e test...")
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
    logger.info("Starting cross-domain relationship detection fix")
    success = run_test()
    
    if success:
        logger.info("Cross-domain relationship detection fix completed successfully")
        sys.exit(0)
    else:
        logger.error("Cross-domain relationship detection fix failed")
        sys.exit(1)
