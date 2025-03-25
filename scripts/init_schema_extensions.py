#!/usr/bin/env python3
"""
Initialize ArangoDB Schema Extensions

This script initializes the schema extensions for ArangoDB to support cross-domain topology,
temporal pattern tracking, and actant journey mapping.

It creates the necessary document and edge collections defined in the schema_extensions module.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from arango import ArangoClient

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from habitat_evolution.adaptive_core.persistence.arangodb.schema_extensions import (
    DOCUMENT_COLLECTIONS,
    EDGE_COLLECTIONS,
    GRAPH_DEFINITIONS
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_db():
    """Initialize the ArangoDB database with schema extensions."""
    # Load environment variables
    load_dotenv()
    
    # Get database connection parameters from environment variables
    host = os.getenv("ARANGO_HOST", "http://localhost:8529")
    username = os.getenv("ARANGO_USER", "root")
    password = os.getenv("ARANGO_PASSWORD", "habitat")
    db_name = os.getenv("ARANGO_DB", "habitat")
    
    # Connect to ArangoDB
    client = ArangoClient(hosts=host)
    sys_db = client.db("_system", username=username, password=password)
    
    # Create database if it doesn't exist
    if not sys_db.has_database(db_name):
        sys_db.create_database(db_name)
        logger.info(f"Created database: {db_name}")
    
    # Connect to the database
    db = client.db(db_name, username=username, password=password)
    
    # Create document collections
    for collection_name in DOCUMENT_COLLECTIONS:
        if not db.has_collection(collection_name):
            db.create_collection(collection_name)
            logger.info(f"Created document collection: {collection_name}")
    
    # Create edge collections
    for collection_name in EDGE_COLLECTIONS:
        if not db.has_collection(collection_name):
            db.create_collection(collection_name, edge=True)
            logger.info(f"Created edge collection: {collection_name}")
    
    # Create graphs
    for graph_def in GRAPH_DEFINITIONS:
        graph_name = graph_def["name"]
        edge_definitions = graph_def["edge_definitions"]
        
        if not db.has_graph(graph_name):
            graph = db.create_graph(graph_name)
            logger.info(f"Created graph: {graph_name}")
            
            # Add edge definitions to the graph
            for edge_def in edge_definitions:
                edge_collection = edge_def["edge_collection"]
                from_collections = edge_def["from_collections"]
                to_collections = edge_def["to_collections"]
                
                graph.create_edge_definition(
                    edge_collection=edge_collection,
                    from_vertex_collections=from_collections,
                    to_vertex_collections=to_collections
                )
                logger.info(f"Added edge definition: {edge_collection} to graph {graph_name}")
    
    # Create AQL traversal functions for complex queries
    create_aql_traversals(db)
    
    logger.info("Schema extensions initialization complete")

def create_aql_traversals(db):
    """Create AQL traversal functions for complex graph queries."""
    # Create AQLTraversals collection if it doesn't exist
    if not db.has_collection("AQLTraversals"):
        db.create_collection("AQLTraversals")
        logger.info("Created AQLTraversals collection")
    
    # Define the ACTANT_JOURNEY traversal function
    actant_journey_code = """
    function (actant_name) {
        // This function returns the complete journey of an actant across domains
        let actant = (FOR a IN Actant FILTER a.name == actant_name RETURN a)[0];
        if (!actant) {
            return [];
        }
        
        // Get the actant's journey
        let journey = (
            FOR j IN OUTBOUND actant._id ActantHasJourney
            RETURN j
        )[0];
        
        if (!journey) {
            return [];
        }
        
        // Get domain transitions
        let transitions = (
            FOR t IN OUTBOUND journey._id JourneyContainsTransition
            SORT t.timestamp
            RETURN {
                event_type: "transition",
                actant_name: t.actant_name,
                source_domain_id: t.source_domain_id,
                target_domain_id: t.target_domain_id,
                source_predicate_id: t.source_predicate_id,
                target_predicate_id: t.target_predicate_id,
                source_role: t.source_role,
                target_role: t.target_role,
                timestamp: t.timestamp,
                transformation_id: t.transformation_id
            }
        );
        
        // Get role shifts
        let role_shifts = (
            FOR rs IN OUTBOUND journey._id JourneyContainsRoleShift
            SORT rs.timestamp
            RETURN {
                event_type: "role_shift",
                actant_name: rs.actant_name,
                source_role: rs.source_role,
                target_role: rs.target_role,
                source_predicate_id: rs.source_predicate_id,
                target_predicate_id: rs.target_predicate_id,
                timestamp: rs.timestamp
            }
        );
        
        // Combine and sort all events
        let all_events = APPEND(transitions, role_shifts);
        all_events = SORT(all_events, "timestamp");
        
        return all_events;
    }
    """
    
    # Define the PATTERN_EVOLUTION traversal function
    pattern_evolution_code = """
    function (start_time, end_time) {
        // This function returns pattern evolution chains within a time range
        let patterns = (
            FOR p IN TemporalPattern
            FILTER p.created_at >= start_time && p.created_at <= end_time
            RETURN p
        );
        
        let evolution_chains = [];
        
        // For each pattern, find its evolution chain
        FOR pattern IN patterns
            // Skip patterns that are already part of a chain
            FILTER NOT pattern._id IN FLATTEN(
                FOR chain IN evolution_chains
                RETURN APPEND(
                    [chain.source._id],
                    chain.evolved_patterns[*]._id
                )
            )
            
            // Get patterns that evolved from this one
            LET evolved_patterns = (
                FOR v, e IN 1..10 OUTBOUND pattern PatternEvolvesTo
                SORT e.timestamp
                RETURN {
                    pattern: v,
                    similarity: e.similarity,
                    evolution_type: e.evolution_type,
                    timestamp: e.timestamp
                }
            )
            
            // Only include chains with at least one evolution
            FILTER LENGTH(evolved_patterns) > 0
            
            // Add the chain to our results
            evolution_chains.PUSH({
                source: pattern,
                evolved_patterns: evolved_patterns,
                length: LENGTH(evolved_patterns) + 1,
                start_time: pattern.created_at,
                end_time: evolved_patterns[-1].timestamp,
                stability_trend: AVG(evolved_patterns[*].pattern.stability) - pattern.stability
            });
        
        RETURN evolution_chains;
    }
    """
    
    # Create or update the traversal functions
    actant_journey = {
        "_key": "ACTANT_JOURNEY",
        "description": "Traversal function to get the complete journey of an actant across domains",
        "code": actant_journey_code
    }
    
    pattern_evolution = {
        "_key": "PATTERN_EVOLUTION",
        "description": "Traversal function to get pattern evolution chains within a time range",
        "code": pattern_evolution_code
    }
    
    # Upsert the traversal functions
    for func in [actant_journey, pattern_evolution]:
        if db.collection("AQLTraversals").has(func["_key"]):
            db.collection("AQLTraversals").update(func)
            logger.info(f"Updated AQL traversal function: {func['_key']}")
        else:
            db.collection("AQLTraversals").insert(func)
            logger.info(f"Created AQL traversal function: {func['_key']}")

if __name__ == "__main__":
    try:
        init_db()
    except Exception as e:
        logger.error(f"Error initializing schema extensions: {e}")
        sys.exit(1)
