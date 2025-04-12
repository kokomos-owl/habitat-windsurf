#!/usr/bin/env python
"""
Script to retrieve PKM collections from ArangoDB.

This script connects to the ArangoDB database and retrieves PKM collections,
displaying the contents in a readable format.
"""

import logging
import json
from typing import Dict, List, Any

from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Retrieve PKM collections from ArangoDB."""
    # Initialize ArangoDB connection
    arangodb_connection = ArangoDBConnection(
        host="localhost",
        port=8529,
        username="root",
        password="habitat",
        database_name="habitat_evolution"
    )
    
    # Initialize the connection
    arangodb_connection.initialize()
    logger.info("ArangoDB connection initialized")
    
    # Get the ArangoDB database object
    db = arangodb_connection._db
    
    # List all collections
    collections = [c['name'] for c in db.collections()]
    logger.info(f"Collections in database: {collections}")
    
    # Look for PKM-related collections
    pkm_collections = [c for c in collections if 'pkm' in c.lower()]
    if not pkm_collections:
        logger.info("No PKM-specific collections found. Looking for pattern-related collections...")
        pkm_collections = [c for c in collections if 'pattern' in c.lower()]
        
    if not pkm_collections:
        logger.info("No PKM or pattern-specific collections found. Checking all collections...")
        pkm_collections = collections
    
    # Retrieve documents from each collection
    for collection_name in pkm_collections:
        try:
            logger.info(f"\nRetrieving documents from collection: {collection_name}")
            
            # Get the collection and retrieve documents
            collection = db.collection(collection_name)
            # Use AQL query to get documents
            query = f"FOR doc IN {collection_name} LIMIT 10 RETURN doc"
            documents = list(db.aql.execute(query))
            
            if not documents:
                logger.info(f"No documents found in collection: {collection_name}")
                continue
                
            logger.info(f"Found {len(documents)} documents in collection: {collection_name}")
            
            # Display document information
            for i, doc in enumerate(documents):
                # Check if this looks like a PKM file
                if 'title' in doc and 'patterns' in doc:
                    logger.info(f"\nPKM File {i+1}:")
                    logger.info(f"  ID: {doc.get('_id', 'Unknown')}")
                    logger.info(f"  Title: {doc.get('title', 'Untitled')}")
                    logger.info(f"  Creator: {doc.get('creator_id', 'Unknown')}")
                    logger.info(f"  Created: {doc.get('created_at', 'Unknown')}")
                    
                    # Display patterns
                    patterns = doc.get('patterns', [])
                    logger.info(f"  Patterns: {len(patterns)}")
                    for j, pattern in enumerate(patterns[:3]):  # Show first 3 patterns
                        logger.info(f"    Pattern {j+1}: {pattern.get('type', 'Unknown')} - {pattern.get('id', 'Unknown')}")
                        content = pattern.get('content', '')
                        if content and len(content) > 100:
                            content = content[:100] + "..."
                        logger.info(f"      Content: {content}")
                    
                    if len(patterns) > 3:
                        logger.info(f"      ... and {len(patterns) - 3} more patterns")
                else:
                    # Just show basic document info
                    doc_id = doc.get('_id', 'Unknown')
                    logger.info(f"\nDocument {i+1} (ID: {doc_id}):")
                    
                    # Try to identify what kind of document this is
                    if 'type' in doc:
                        logger.info(f"  Type: {doc.get('type')}")
                    
                    # Show a few key-value pairs as a sample
                    sample_keys = list(doc.keys())[:5]  # First 5 keys
                    for key in sample_keys:
                        value = doc.get(key)
                        if isinstance(value, (dict, list)):
                            value = f"{type(value).__name__} with {len(value)} items"
                        elif isinstance(value, str) and len(value) > 50:
                            value = value[:50] + "..."
                        logger.info(f"  {key}: {value}")
                    
                    if len(doc) > 5:
                        logger.info(f"  ... and {len(doc) - 5} more fields")
        
        except Exception as e:
            logger.error(f"Error retrieving documents from collection {collection_name}: {e}")

if __name__ == "__main__":
    main()
