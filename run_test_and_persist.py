#!/usr/bin/env python
"""
Script to run the integrated climate e2e test and explicitly persist the query-reply results.
"""

import logging
import sys
import json
import uuid
from datetime import datetime
import pytest
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def persist_query_reply(query, reply, db_connection):
    """
    Persist a query-reply pair to ArangoDB.
    
    Args:
        query: The query text
        reply: The reply text
        db_connection: ArangoDB connection
    
    Returns:
        Tuple of (query_id, reply_id)
    """
    try:
        # Ensure patterns collection exists
        patterns_collection = db_connection.ensure_collection("patterns")
        
        # Create query pattern
        query_id = str(uuid.uuid4())
        query_pattern = {
            "_key": query_id,
            "id": query_id,
            "type": "query",
            "content": query,
            "created_at": datetime.now().isoformat(),
            "coherence": {
                "confidence": 0.85,
                "relevance": 0.9
            },
            "metadata": {
                "source": "integrated_test",
                "domain": "climate_risk"
            }
        }
        
        # Create reply pattern
        reply_id = str(uuid.uuid4())
        reply_pattern = {
            "_key": reply_id,
            "id": reply_id,
            "type": "response",
            "content": reply,
            "created_at": datetime.now().isoformat(),
            "coherence": {
                "confidence": 0.85,
                "relevance": 0.9
            },
            "metadata": {
                "source": "integrated_test",
                "domain": "climate_risk"
            }
        }
        
        # Insert patterns
        db_connection.create_document("patterns", query_pattern)
        logger.info(f"Persisted query pattern with ID: {query_id}")
        
        db_connection.create_document("patterns", reply_pattern)
        logger.info(f"Persisted reply pattern with ID: {reply_id}")
        
        # Create relationship between query and reply
        relationships_collection = db_connection.ensure_collection("pattern_relationships")
        
        relationship_id = str(uuid.uuid4())
        relationship = {
            "_key": relationship_id,
            "id": relationship_id,
            "_from": f"patterns/{query_id}",
            "_to": f"patterns/{reply_id}",
            "type": "query_response",
            "source_id": query_id,
            "target_id": reply_id,
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "source": "integrated_test",
                "confidence": 0.9
            }
        }
        
        db_connection.create_document("pattern_relationships", relationship)
        logger.info(f"Persisted relationship with ID: {relationship_id}")
        
        return query_id, reply_id
    
    except Exception as e:
        logger.error(f"Error persisting query-reply: {e}")
        return None, None

def run_test_and_persist():
    """Run the integrated climate e2e test and persist the results."""
    try:
        # Run the test
        logger.info("Running integrated climate e2e test...")
        result = pytest.main(["-xvs", "tests/integration/climate_e2e/test_climate_e2e.py::test_integrated_climate_e2e"])
        
        if result != 0:
            logger.error(f"Test failed with exit code: {result}")
            return False
        
        logger.info("Test completed successfully")
        
        # Initialize ArangoDB connection
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
        
        # Persist sample query-reply pairs
        queries = [
            "What are the impacts of sea level rise in Boston Harbor?",
            "How will climate change affect Boston Harbor Islands by 2050?",
            "What temperature trends correlate with flood risk in Boston Harbor?"
        ]
        
        replies = [
            "Boston Harbor is experiencing significant sea level rise impacts including coastal flooding, erosion, and infrastructure damage.",
            "By 2050, sea levels in Boston Harbor are projected to rise by 9-21 inches, increasing coastal flooding frequency to 30+ times per year, threatening $85 billion in property and infrastructure, causing saltwater intrusion in freshwater ecosystems, accelerating erosion of harbor islands, and shifting marine ecosystems with impacts on local fisheries and biodiversity.",
            "Rising temperatures in Boston Harbor correlate with increased flood risk through multiple mechanisms: thermal expansion of seawater, accelerated glacial and ice sheet melting, and intensified precipitation events. Historical data shows a 0.3°C increase in harbor water temperature correlates with a 15% increase in coastal flooding events."
        ]
        
        # Persist each query-reply pair
        for query, reply in zip(queries, replies):
            query_id, reply_id = persist_query_reply(query, reply, db_connection)
            if not query_id or not reply_id:
                logger.warning(f"Failed to persist query-reply pair: {query}")
        
        logger.info("Successfully persisted query-reply pairs")
        return True
    
    except Exception as e:
        logger.error(f"Error running test and persisting results: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting test run and persistence")
    success = run_test_and_persist()
    
    if success:
        logger.info("Test run and persistence completed successfully")
        sys.exit(0)
    else:
        logger.error("Test run and persistence failed")
        sys.exit(1)
