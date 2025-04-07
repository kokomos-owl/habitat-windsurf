"""
Test script for the Claude Adapter with Anthropic API integration.

This script tests the ClaudeAdapter's ability to connect to the Anthropic API
and process queries and documents.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

async def test_process_query():
    """Test processing a query with the Claude adapter."""
    # Initialize the Claude adapter
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    claude_adapter = ClaudeAdapter(api_key=api_key)
    
    # Test query
    query = "What are the key principles of Habitat Evolution?"
    
    # Test context
    context = {
        "task": "information_retrieval",
        "user_id": "test_user"
    }
    
    # Test patterns
    patterns = [
        {
            "id": "pattern-1",
            "name": "Pattern Evolution",
            "description": "The concept of patterns evolving through quality states based on usage and feedback",
            "quality_state": "stable"
        },
        {
            "id": "pattern-2",
            "name": "Bidirectional Flow",
            "description": "The concept of bidirectional communication between components",
            "quality_state": "emergent"
        }
    ]
    
    # Process the query
    logger.info(f"Processing query: {query}")
    response = await claude_adapter.process_query(query, context, patterns)
    
    # Log the response
    logger.info(f"Response: {response}")
    
    return response

async def test_process_document():
    """Test processing a document with the Claude adapter."""
    # Initialize the Claude adapter
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    claude_adapter = ClaudeAdapter(api_key=api_key)
    
    # Test document
    document = {
        "id": "doc-1",
        "title": "Habitat Evolution Overview",
        "content": """
        Habitat Evolution is a system designed to detect and evolve coherent patterns, 
        while enabling the observation of semantic change across the system. It's built 
        on the principles of pattern evolution and co-evolution.
        
        The system includes several key components:
        1. A bidirectional flow system that enables communication between components
        2. Pattern-aware RAG for enhanced retrieval and generation
        3. ArangoDB for pattern persistence and relationship management
        4. Pattern evolution tracking to improve pattern quality over time
        
        This creates a complete functional loop from document ingestion through processing, 
        persistence, and retrieval, with user interactions driving pattern evolution.
        """,
        "metadata": {
            "source": "internal",
            "author": "test_author",
            "date": "2025-04-06"
        }
    }
    
    # Process the document
    logger.info(f"Processing document: {document['id']}")
    response = await claude_adapter.process_document(document)
    
    # Log the response
    logger.info(f"Response: {response}")
    
    return response

async def main():
    """Run all tests."""
    # Test query processing
    query_response = await test_process_query()
    
    # Test document processing
    document_response = await test_process_document()
    
    # Log test results
    logger.info("Tests completed.")
    logger.info(f"Query processing: {'Success' if query_response else 'Failed'}")
    logger.info(f"Document processing: {'Success' if document_response else 'Failed'}")

if __name__ == "__main__":
    asyncio.run(main())
