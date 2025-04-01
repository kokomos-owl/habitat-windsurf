"""
Simple script to check ArangoDB connection with hardcoded credentials.
For testing purposes only.
"""

import logging
from arango import ArangoClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Hardcoded connection details for testing
    host = "http://localhost:8529"
    username = "root"  # Default ArangoDB username
    password = ""      # Default empty password for root user
    db_name = "habitat_evolution"
    
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
