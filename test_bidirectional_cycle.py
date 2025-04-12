#!/usr/bin/env python
"""
Test script for a full bidirectional query-narrative cycle in Habitat Evolution.

This script demonstrates:
1. Pattern creation and storage
2. Query generation from patterns
3. Claude response processing
4. Response storage as PKM files
5. Pattern extraction from responses
6. Relationship creation between patterns
7. Retrieval of the full cycle of patterns and relationships
"""

import logging
import json
import uuid
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from the package
from src.habitat_evolution.pkm import (
    PKMFile, 
    PKMRepository, 
    create_pkm_from_claude_response,
    create_pkm_repository,
    create_pkm_bidirectional_integration
)
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection

def main():
    """Run the bidirectional cycle test."""
    logger.info("Starting bidirectional cycle test")
    
    # Initialize ArangoDB connection
    arangodb_connection = ArangoDBConnection(
        host="localhost",
        port=8529,
        username="root",
        password="habitat",
        database_name="habitat_evolution"
    )
    arangodb_connection.initialize()
    logger.info("ArangoDB connection initialized")
    
    # Create PKM bidirectional integration
    pkm_bidirectional = create_pkm_bidirectional_integration(
        db_config={
            "host": "localhost",
            "port": 8529,
            "username": "root",
            "password": "habitat",
            "database_name": "habitat_evolution"
        },
        creator_id="test_user"
    )
    logger.info("PKM bidirectional integration created")
    
    # Step 1: Create a new pattern
    original_pattern = {
        "id": f"pattern-{uuid.uuid4()}",
        "type": "semantic",
        "content": "Climate adaptation strategies for coastal cities",
        "quality": 0.85,
        "metadata": {
            "confidence": 0.85,
            "source": "climate_adaptation_report_2024.pdf",
            "tags": ["climate", "adaptation", "coastal", "cities"]
        },
        "created_at": datetime.now().isoformat()
    }
    
    # Step 2: Store the pattern
    pattern_id = pkm_bidirectional.bidirectional_flow_service.publish_pattern(original_pattern)
    logger.info(f"Created and published original pattern: {original_pattern['id']}")
    
    # Step 3: Generate a query from the pattern
    query = pkm_bidirectional.generate_query_from_patterns([original_pattern])
    logger.info(f"Generated query from pattern: {query}")
    
    # Step 4: Process the query with Claude
    pkm_id = pkm_bidirectional.process_query_with_patterns(
        query=query,
        patterns=[original_pattern]
    )
    
    if not pkm_id:
        logger.error("Failed to process query with Claude")
        return
        
    logger.info(f"Created PKM file with ID: {pkm_id}")
    
    # Step 5: Retrieve the PKM file
    pkm_file = pkm_bidirectional.pkm_repository.get_pkm_file(pkm_id)
    
    if not pkm_file:
        logger.error(f"Failed to retrieve PKM file with ID: {pkm_id}")
        return
        
    logger.info(f"Retrieved PKM file: {pkm_file.title}")
    
    # Step 6: Extract the Claude response pattern
    response_pattern = None
    for pattern in pkm_file.patterns:
        if pattern.get("type") == "claude_response":
            response_pattern = pattern
            break
    
    if not response_pattern:
        logger.error("No Claude response pattern found in PKM file")
        return
        
    logger.info(f"Found Claude response pattern: {response_pattern.get('id')}")
    logger.info(f"Response content: {response_pattern.get('content')[:200]}...")
    
    # Step 7: Create a new pattern based on the Claude response
    derived_pattern = {
        "id": f"pattern-{uuid.uuid4()}",
        "type": "derived",
        "content": "Adaptive infrastructure design for sea level rise",
        "quality": 0.8,
        "metadata": {
            "confidence": 0.8,
            "source": f"pkm:{pkm_id}",
            "derived_from": response_pattern.get('id'),
            "tags": ["infrastructure", "adaptation", "sea level rise"]
        },
        "created_at": datetime.now().isoformat()
    }
    
    # Step 8: Store the derived pattern
    derived_pattern_id = pkm_bidirectional.bidirectional_flow_service.publish_pattern(derived_pattern)
    logger.info(f"Created and published derived pattern: {derived_pattern['id']}")
    
    # Step 9: Create a relationship between the original pattern and the derived pattern
    relationship = {
        "source_id": original_pattern["id"],
        "target_id": derived_pattern["id"],
        "type": "derives",
        "properties": {
            "strength": 0.75,
            "description": "Derived adaptation strategy from climate pattern"
        }
    }
    
    # Publish the relationship
    pkm_bidirectional.bidirectional_flow_service.publish_relationship(relationship)
    logger.info(f"Published relationship: {relationship['source_id']} -> {relationship['target_id']}")
    
    # Step 10: Retrieve and display the full cycle
    logger.info("\n--- Full Bidirectional Cycle Summary ---")
    logger.info(f"Original Pattern: {original_pattern['content']}")
    logger.info(f"Query: {query}")
    logger.info(f"PKM File: {pkm_file.title}")
    logger.info(f"Claude Response: {response_pattern.get('content')[:100]}...")
    logger.info(f"Derived Pattern: {derived_pattern['content']}")
    logger.info(f"Relationship: {original_pattern['content']} -> {derived_pattern['content']}")
    
    # Step 11: Retrieve all patterns and relationships from the database
    db = arangodb_connection._db
    
    # Get patterns
    logger.info("\n--- Patterns in Database ---")
    query = "FOR doc IN patterns LIMIT 20 RETURN doc"
    patterns = list(db.aql.execute(query))
    
    for i, pattern in enumerate(patterns):
        logger.info(f"Pattern {i+1}: {pattern.get('_id')}")
        logger.info(f"  Type: {pattern.get('type')}")
        logger.info(f"  Content: {pattern.get('content')}")
        logger.info(f"  Quality: {pattern.get('quality')}")
    
    # Get relationships
    logger.info("\n--- Relationships in Database ---")
    query = "FOR doc IN pattern_relationships LIMIT 20 RETURN doc"
    relationships = list(db.aql.execute(query))
    
    for i, rel in enumerate(relationships):
        logger.info(f"Relationship {i+1}: {rel.get('_id')}")
        logger.info(f"  From: {rel.get('_from')}")
        logger.info(f"  To: {rel.get('_to')}")
        logger.info(f"  Type: {rel.get('type')}")
    
    # Get PKM files
    logger.info("\n--- Recent PKM Files in Database ---")
    query = "FOR doc IN pkm_files SORT doc.created_at DESC LIMIT 5 RETURN doc"
    pkm_files = list(db.aql.execute(query))
    
    for i, pkm in enumerate(pkm_files):
        logger.info(f"PKM File {i+1}: {pkm.get('_id')}")
        logger.info(f"  Title: {pkm.get('title')}")
        logger.info(f"  Created: {pkm.get('created_at')}")
        logger.info(f"  Patterns: {len(pkm.get('patterns', []))}")
    
    logger.info("Bidirectional cycle test complete")

if __name__ == "__main__":
    main()
