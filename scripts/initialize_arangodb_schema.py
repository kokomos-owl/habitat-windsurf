"""
Script to initialize ArangoDB schema for Habitat Evolution.

This script creates the necessary collections and indexes for the Habitat Evolution system.
"""

import os
import logging
import sys
import getpass

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.schema_manager import ArangoDBSchemaManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Initialize the ArangoDB schema."""
    # Get credentials from user input
    host = input("ArangoDB host (default: http://localhost:8529): ") or "http://localhost:8529"
    username = input("ArangoDB username (default: root): ") or "root"
    password = getpass.getpass(f"ArangoDB password for {username}: ")
    db_name = input("ArangoDB database name (default: habitat_evolution): ") or "habitat_evolution"
    
    # Set environment variables for the connection manager
    os.environ["ARANGO_HOST"] = host
    os.environ["ARANGO_USER"] = username
    os.environ["ARANGO_PASSWORD"] = password
    os.environ["ARANGO_DB"] = db_name
    
    logger.info(f"Initializing ArangoDB schema for {db_name} at {host}")
    
    try:
        # Create schema manager
        schema_manager = ArangoDBSchemaManager()
        
        # Initialize schema
        schema_manager.initialize_schema()
        
        logger.info("Schema initialization completed successfully")
        
        # Print summary of created collections
        db = schema_manager.db
        collections = db.collections()
        
        # Print document collections
        logger.info("Document Collections:")
        for collection in collections:
            if not collection['name'].startswith('_') and not collection['edge']:
                logger.info(f"  - {collection['name']}")
        
        # Print edge collections
        logger.info("Edge Collections:")
        for collection in collections:
            if not collection['name'].startswith('_') and collection['edge']:
                logger.info(f"  - {collection['name']}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error initializing schema: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("Schema initialization completed successfully")
    else:
        logger.error("Schema initialization failed")
