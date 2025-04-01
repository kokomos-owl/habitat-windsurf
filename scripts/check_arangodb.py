"""
Script to check ArangoDB connection and list collections.
"""

import os
import logging
import getpass
from arango import ArangoClient
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()
    
    # Get connection details from environment variables or prompt user
    host = os.getenv("ARANGO_HOST", "http://localhost:8529")
    username = os.getenv("ARANGO_USER")
    password = os.getenv("ARANGO_PASSWORD")
    db_name = os.getenv("ARANGO_DB", "habitat_evolution")
    
    # If credentials not in environment, prompt user
    if not username:
        username = input("ArangoDB username (default: root): ") or "root"
    if not password:
        password = getpass.getpass(f"ArangoDB password for {username}: ")
    
    logger.info(f"Connecting to ArangoDB at {host} with username: {username}")
    
    try:
        # Initialize the ArangoDB client
        client = ArangoClient(hosts=host)
        
        # Connect to the system database
        sys_db = client.db("_system", username=username, password=password)
        
        # Check if our database exists, create it if it doesn't
        if not sys_db.has_database(db_name):
            logger.info(f"Database '{db_name}' does not exist, creating it...")
            sys_db.create_database(db_name)
            logger.info(f"Database '{db_name}' created successfully")
        else:
            logger.info(f"Database '{db_name}' already exists")
        
        # Connect to our database
        db = client.db(db_name, username=username, password=password)
        
        # List all collections
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
        
        # Print some database stats
        logger.info(f"Total collections: {len([c for c in collections if not c['name'].startswith('_')])}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error connecting to ArangoDB: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("ArangoDB connection check completed successfully")
    else:
        logger.error("ArangoDB connection check failed")
