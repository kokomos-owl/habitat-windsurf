"""
Helper module for setting up ArangoDB structures in test environments.

This module provides functions to ensure proper setup of ArangoDB collections,
edge collections, and graphs for testing the Habitat Evolution system.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def ensure_pattern_graph_structure(db_connection):
    """
    Ensure that the pattern graph structure exists in ArangoDB.
    
    This function creates the necessary collections and graph structure
    for pattern relationships, with proper error handling and logging.
    
    Args:
        db_connection: The ArangoDB connection to use
        
    Returns:
        bool: True if the graph structure was successfully created, False otherwise
    """
    try:
        # Get direct database access
        db = db_connection.get_database()
        
        # Step 1: First create the vertex collection (patterns) if it doesn't exist
        if not db.has_collection("patterns"):
            logger.info("Creating patterns vertex collection")
            db.create_collection("patterns")
            logger.debug("Created patterns vertex collection successfully")
        else:
            logger.debug("Patterns vertex collection already exists")
        
        # Step 2: Create the edge collection separately before creating the graph
        if not db.has_collection("pattern_relationships"):
            logger.info("Creating pattern_relationships edge collection")
            db.create_collection("pattern_relationships", edge=True)
            logger.debug("Created pattern_relationships edge collection successfully")
        else:
            logger.debug("Pattern_relationships edge collection already exists")
        
        # Step 3: Create the graph with proper edge definitions
        # The key is to ensure the collections exist BEFORE creating the graph
        if not db.has_graph("pattern_graph"):
            logger.info("Creating pattern_graph")
            try:
                # Create the graph with proper edge definitions
                graph = db.create_graph(
                    "pattern_graph",
                    edge_definitions=[
                        {
                            "edge_collection": "pattern_relationships",
                            "from_vertex_collections": ["patterns"],
                            "to_vertex_collections": ["patterns"]
                        }
                    ]
                )
                logger.info(f"Created pattern_graph successfully: {graph}")
            except Exception as graph_error:
                logger.error(f"Error creating graph: {graph_error}")
                
                # Alternative approach if the standard approach fails
                try:
                    logger.info("Trying alternative graph creation approach")
                    # Create the graph without edge definitions first
                    graph = db.create_graph("pattern_graph")
                    
                    # Then add the edge definition
                    graph.create_edge_definition(
                        edge_collection="pattern_relationships",
                        from_vertex_collections=["patterns"],
                        to_vertex_collections=["patterns"]
                    )
                    logger.info("Created pattern_graph using alternative approach")
                except Exception as alt_error:
                    logger.error(f"Alternative graph creation also failed: {alt_error}")
                    return False
        else:
            logger.debug("Pattern_graph already exists")
        
        # Verify the graph structure is correct
        graph = db.graph("pattern_graph")
        edge_definitions = graph.edge_definitions()
        logger.debug(f"Graph edge definitions: {edge_definitions}")
        
        logger.info("Pattern graph structure verified successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating pattern graph structure: {e}")
        return False

def create_test_patterns(db_connection, count=5):
    """
    Create test patterns in the database.
    
    Args:
        db_connection: The ArangoDB connection to use
        count: The number of test patterns to create
        
    Returns:
        List[str]: The IDs of the created patterns
    """
    pattern_ids = []
    
    try:
        # Ensure the pattern collection exists
        patterns_collection = db_connection.ensure_collection("patterns")
        
        # Create test patterns
        for i in range(count):
            pattern = {
                "name": f"Test Pattern {i+1}",
                "description": f"Description for test pattern {i+1}",
                "type": "semantic" if i % 2 == 0 else "statistical",
                "confidence": 0.7 + (i / 10),
                "created_at": "2025-04-12T00:00:00Z",
                "metadata": {
                    "test": True,
                    "index": i
                }
            }
            
            result = db_connection.insert("patterns", pattern)
            pattern_ids.append(result["_id"])
            
        logger.info(f"Created {count} test patterns")
        return pattern_ids
    except Exception as e:
        logger.error(f"Error creating test patterns: {e}")
        return pattern_ids

def create_test_relationships(db_connection, pattern_ids):
    """
    Create test relationships between patterns.
    
    Args:
        db_connection: The ArangoDB connection to use
        pattern_ids: The IDs of the patterns to create relationships between
        
    Returns:
        List[str]: The IDs of the created relationships
    """
    relationship_ids = []
    
    try:
        # Ensure the graph structure exists
        if not ensure_pattern_graph_structure(db_connection):
            logger.error("Graph structure not properly set up, cannot create relationships")
            return relationship_ids
        
        # Get the database object
        db = db_connection.get_database()
        
        # Get the graph object
        graph = db.graph("pattern_graph")
        
        # Get the edge collection
        edge_collection = db.collection("pattern_relationships")
        
        # Create relationships between patterns
        for i in range(len(pattern_ids) - 1):
            from_id = pattern_ids[i]
            to_id = pattern_ids[i + 1]
            
            # Extract just the ID part from the full document ID
            from_key = from_id.split("/")[1] if "/" in from_id else from_id
            to_key = to_id.split("/")[1] if "/" in to_id else to_id
            
            # Prepare the edge document
            edge = {
                "_from": f"patterns/{from_key}",
                "_to": f"patterns/{to_key}",
                "type": "related_to" if i % 2 == 0 else "derives_from",
                "strength": 0.6 + (i / 10),
                "metadata": {
                    "test": True,
                    "index": i
                }
            }
            
            try:
                # First try using the graph API
                logger.debug(f"Creating edge with graph API: {edge}")
                result = graph.edge_collection("pattern_relationships").insert(edge)
                logger.debug(f"Created edge with graph API: {result}")
                relationship_ids.append(result["_id"])
            except Exception as graph_error:
                logger.warning(f"Error creating edge with graph API: {graph_error}")
                
                try:
                    # Try direct edge collection insertion as fallback
                    logger.debug(f"Creating edge with direct edge collection: {edge}")
                    result = edge_collection.insert(edge)
                    logger.debug(f"Created edge with direct edge collection: {result}")
                    relationship_ids.append(result["_id"])
                except Exception as edge_error:
                    logger.error(f"Error creating edge with direct edge collection: {edge_error}")
        
        logger.info(f"Created {len(relationship_ids)} test relationships")
        return relationship_ids
    except Exception as e:
        logger.error(f"Error creating test relationships: {e}")
        return relationship_ids
