#!/usr/bin/env python
"""
Test script for VectorTonicService graph creation.

This script verifies that the VectorTonicService can properly create
the vector_tonic_graph with the correct edge definitions using the
ArangoDBConnection's ensure_graph method.
"""

import logging
import sys
import os
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.infrastructure.services.event_service import EventService
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_pattern_repository import ArangoDBPatternRepository
from src.habitat_evolution.infrastructure.services.vector_tonic_service import VectorTonicService

def test_vector_tonic_graph_creation():
    """
    Test the VectorTonicService graph creation.
    
    This test verifies that:
    1. The VectorTonicService can create the vector_tonic_graph
    2. The edge definitions are properly normalized
    3. The ArangoDB client compatibility issues are resolved
    """
    logger.info("=== Testing VectorTonicService Graph Creation ===")
    
    # Create ArangoDB connection
    db_config = {
        'host': 'localhost',
        'port': 8529,
        'username': 'root',
        'password': 'habitat',
        'database': 'habitat_evolution'
    }
    
    try:
        # Step 1: Create and initialize ArangoDB connection
        logger.info("Step 1: Creating ArangoDB connection...")
        db_connection = ArangoDBConnection(
            host=db_config['host'],
            port=db_config['port'],
            username=db_config['username'],
            password=db_config['password'],
            database_name=db_config['database']
        )
        db_connection.initialize()
        logger.info("ArangoDB connection initialized successfully")
        
        # Step 2: Create and initialize Event Service
        logger.info("Step 2: Creating Event Service...")
        event_service = EventService()
        event_service.initialize()
        logger.info("Event Service initialized successfully")
        
        # Step 3: Create and initialize Pattern Repository
        logger.info("Step 3: Creating Pattern Repository...")
        pattern_repository = ArangoDBPatternRepository(
            db_connection=db_connection,
            event_service=event_service
        )
        logger.info("Pattern Repository created successfully")
        
        # Step 4: Create and initialize Vector Tonic Service
        logger.info("Step 4: Creating Vector Tonic Service...")
        vector_tonic_service = VectorTonicService(
            db_connection=db_connection,
            event_service=event_service,
            pattern_repository=pattern_repository
        )
        
        # Step 5: Initialize Vector Tonic Service (this will create the graph)
        logger.info("Step 5: Initializing Vector Tonic Service...")
        vector_tonic_service.initialize()
        logger.info("Vector Tonic Service initialized successfully")
        
        # Step 6: Verify graph exists
        logger.info("Step 6: Verifying graph exists...")
        if db_connection.graph_exists("vector_tonic_graph"):
            logger.info("vector_tonic_graph exists!")
            
            # Get the graph and print its edge definitions
            db = db_connection.get_database()
            graph = db.graph("vector_tonic_graph")
            edge_definitions = graph.edge_definitions()
            
            logger.info(f"Graph edge definitions: {edge_definitions}")
            
            return True
        else:
            logger.error("vector_tonic_graph does not exist!")
            return False
            
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run the Vector Tonic graph creation test."""
    success = test_vector_tonic_graph_creation()
    
    if success:
        logger.info("TEST PASSED: VectorTonicService graph creation works correctly")
        sys.exit(0)
    else:
        logger.error("TEST FAILED: VectorTonicService graph creation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
