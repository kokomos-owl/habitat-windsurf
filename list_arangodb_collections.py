#!/usr/bin/env python
"""
Script to list all collections in the ArangoDB database used by Habitat Evolution.
"""

import logging
import sys
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def list_arangodb_collections():
    """List all collections in the ArangoDB database."""
    try:
        # Initialize ArangoDB connection with default parameters
        logger.info("Initializing ArangoDB connection...")
        db_connection = ArangoDBConnection(
            host="localhost",
            port=8529,
            username="root",
            password="habitat",
            database_name="habitat_evolution"
        )
        db_connection.initialize()
        logger.info("ArangoDB connection initialized")
        
        # Get the database
        db = db_connection.get_database()
        
        # Get all collections
        collections = db.collections()
        
        # Filter out system collections
        user_collections = [c for c in collections if not c['name'].startswith('_')]
        
        logger.info(f"Found {len(user_collections)} user collections")
        
        # Print collection details
        print("\n=== ARANGODB COLLECTIONS ===")
        for i, collection in enumerate(user_collections, 1):
            collection_name = collection['name']
            collection_type = collection['type']
            collection_id = collection['id']
            
            type_str = "Document" if collection_type == 2 else "Edge" if collection_type == 3 else "Unknown"
            
            print(f"{i}. {collection_name} (Type: {type_str}, ID: {collection_id})")
            
            # Get collection count
            count = db.collection(collection_name).count()
            print(f"   Documents: {count}")
            
            # If collection has documents, show a sample
            if count > 0:
                sample_query = f"FOR doc IN {collection_name} LIMIT 1 RETURN doc"
                sample = db_connection.execute_query(sample_query)
                if sample:
                    print(f"   Sample keys: {', '.join(sample[0].keys())}")
            
            print("")  # Empty line between collections
        
        return user_collections
    except Exception as e:
        logger.error(f"Error listing ArangoDB collections: {e}")
        return None

if __name__ == "__main__":
    logger.info("Starting ArangoDB collection listing")
    collections = list_arangodb_collections()
    
    if collections:
        logger.info("Collection listing completed successfully")
        sys.exit(0)
    else:
        logger.error("Collection listing failed")
        sys.exit(1)
