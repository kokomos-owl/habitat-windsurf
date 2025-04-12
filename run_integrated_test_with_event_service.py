#!/usr/bin/env python
"""
Script to run the integrated climate e2e test with a properly initialized EventService.
"""

import logging
import sys
import json
import uuid
from datetime import datetime
import pytest
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def run_integrated_test():
    """Run the integrated climate e2e test with a properly initialized EventService."""
    try:
        # First, ensure EventService is properly initialized
        logger.info("Ensuring EventService is properly initialized...")
        from ensure_event_service import ensure_event_service
        event_service = ensure_event_service()
        
        if not event_service:
            logger.error("Failed to initialize EventService")
            return False
        
        logger.info("EventService properly initialized")
        
        # Set environment variable to indicate EventService is initialized
        os.environ["EVENT_SERVICE_INITIALIZED"] = "true"
        
        # Run the integrated test
        logger.info("Running integrated climate e2e test...")
        result = pytest.main(["-xvs", "tests/integration/climate_e2e/test_climate_e2e.py::test_integrated_climate_e2e"])
        
        if result != 0:
            logger.error(f"Test failed with exit code: {result}")
            return False
        
        logger.info("Test completed successfully")
        
        # Now retrieve the persisted query-reply results
        logger.info("Retrieving persisted query-reply results...")
        from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
        
        # Initialize ArangoDB connection
        db_connection = ArangoDBConnection(
            host="localhost",
            port=8529,
            username="root",
            password="habitat",
            database_name="habitat_evolution"
        )
        db_connection.initialize()
        
        # Query for patterns
        query = """
        FOR p IN patterns
            FILTER p.type == 'query' OR p.type == 'response' OR p.type == 'sea_level_rise' OR p.type == 'temperature_trend'
            RETURN p
        """
        
        patterns = db_connection.execute_query(query)
        logger.info(f"Found {len(patterns)} patterns")
        
        # Query for relationships
        relationships_query = """
        FOR r IN pattern_relationships
            RETURN r
        """
        
        relationships = db_connection.execute_query(relationships_query)
        logger.info(f"Found {len(relationships)} relationships")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"integrated_test_results_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump({
                "patterns": patterns,
                "relationships": relationships
            }, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        
        # Print summary
        print("\n=== TEST RESULTS SUMMARY ===")
        print(f"Patterns found: {len(patterns)}")
        print(f"Relationships found: {len(relationships)}")
        print(f"Results saved to: {results_file}")
        print("===========================\n")
        
        return True
    
    except Exception as e:
        logger.error(f"Error running integrated test: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting integrated test with initialized EventService")
    success = run_integrated_test()
    
    if success:
        logger.info("Integrated test completed successfully")
        sys.exit(0)
    else:
        logger.error("Integrated test failed")
        sys.exit(1)
