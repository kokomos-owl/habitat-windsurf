#!/usr/bin/env python
"""
Script to clear all collections in the ArangoDB database used by Habitat Evolution.
"""

import logging
import sys
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def clear_arangodb():
    """Clear all collections in the ArangoDB database."""
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
        
        # Get the database handle
        # The ArangoDBConnection class might expose the database through a different attribute
        # Let's check what attributes are available
        logger.info(f"Available attributes: {dir(db_connection)}")
        
        # Try to get the database using common attribute names
        if hasattr(db_connection, 'database'):
            db = db_connection.database
        elif hasattr(db_connection, 'db'):
            db = db_connection.db
        elif hasattr(db_connection, 'client') and hasattr(db_connection.client, 'db'):
            db = db_connection.client.db('habitat_evolution')
        elif hasattr(db_connection, 'get_database'):
            db = db_connection.get_database()
        else:
            # If we can't find the database attribute, let's try to access the ArangoDB client directly
            from arango import ArangoClient
            client = ArangoClient(hosts=f"http://localhost:8529")
            sys_db = client.db('_system', username='root', password='habitat')
            if sys_db.has_database('habitat_evolution'):
                db = client.db('habitat_evolution', username='root', password='habitat')
            else:
                raise ValueError("Could not find habitat_evolution database")
        
        # Get all collections
        collections = db.collections()
        
        # Filter out system collections
        user_collections = [c for c in collections if not c['name'].startswith('_')]
        
        logger.info(f"Found {len(user_collections)} user collections")
        
        # Truncate each collection
        for collection in user_collections:
            collection_name = collection['name']
            logger.info(f"Truncating collection: {collection_name}")
            db.collection(collection_name).truncate()
        
        logger.info("All collections have been cleared")
        
        return True
    except Exception as e:
        logger.error(f"Error clearing ArangoDB: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting ArangoDB clear operation")
    success = clear_arangodb()
    if success:
        logger.info("ArangoDB clear operation completed successfully")
        sys.exit(0)
    else:
        logger.error("ArangoDB clear operation failed")
        sys.exit(1)
