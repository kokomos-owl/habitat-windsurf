"""
Test script for the ClaudeAdapter.

This script tests the ClaudeAdapter's ability to connect to the Anthropic API
and process a simple query.
"""

import asyncio
import os
import json
import sys

# Add the project root to the Python path
sys.path.append('/Users/prphillips/Documents/GitHub/habitat_alpha')

# Now import the ClaudeAdapter
from src.habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter

async def test_claude_adapter():
    """Test the ClaudeAdapter with a simple query."""
    print(f"ANTHROPIC_API_KEY set: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")
    
    # Initialize the ClaudeAdapter
    adapter = ClaudeAdapter()
    print(f"ClaudeAdapter initialized (use_mock: {adapter.use_mock})")
    
    # Process a simple query
    query = "What is Habitat Evolution and how does it use pattern evolution?"
    context = {"system": "Habitat Evolution"}
    patterns = [
        {
            "name": "Pattern Evolution",
            "description": "Patterns evolve through quality states based on usage and feedback.",
            "quality_state": "stable"
        }
    ]
    
    print("\nProcessing query...")
    result = await adapter.process_query(query, context, patterns)
    
    # Print the result
    print("\nResult:")
    print(f"Model: {result.get('model')}")
    print(f"Query ID: {result.get('query_id')}")
    print(f"Tokens Used: {result.get('tokens_used', 'N/A')}")
    print("\nResponse:")
    print(result.get('response'))

if __name__ == "__main__":
    asyncio.run(test_claude_adapter())
