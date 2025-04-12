#!/usr/bin/env python
"""
Script to retrieve persisted query-reply results from ArangoDB.
"""

import logging
import sys
import json
from datetime import datetime
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def retrieve_query_results():
    """Retrieve persisted query-reply results from ArangoDB."""
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
        
        # Get patterns collection
        patterns = db_connection.get_collection("patterns")
        if not patterns:
            logger.error("Patterns collection not found")
            return None
        
        # Query for patterns related to queries or responses
        query = """
        FOR p IN patterns
            FILTER p.type == 'query' OR p.type == 'response' OR p.type == 'sea_level_rise' OR p.type == 'temperature_trend'
            RETURN p
        """
        
        logger.info("Executing query for patterns...")
        results = db_connection.execute_query(query)
        
        if not results:
            logger.info("No query-reply patterns found")
            return []
        
        logger.info(f"Found {len(results)} patterns")
        
        # Get pattern relationships
        relationships_query = """
        FOR r IN pattern_relationships
            RETURN r
        """
        
        logger.info("Executing query for relationships...")
        relationships = db_connection.execute_query(relationships_query)
        
        logger.info(f"Found {len(relationships)} relationships")
        
        # Combine patterns with their relationships
        enriched_patterns = []
        for pattern in results:
            pattern_id = pattern.get("_key") or pattern.get("id")
            
            # Find relationships for this pattern
            pattern_relationships = []
            for rel in relationships:
                if (rel.get("_from", "").endswith(pattern_id) or 
                    rel.get("_to", "").endswith(pattern_id) or
                    rel.get("source_id") == pattern_id or 
                    rel.get("target_id") == pattern_id):
                    pattern_relationships.append(rel)
            
            # Add relationships to pattern
            pattern["relationships"] = pattern_relationships
            enriched_patterns.append(pattern)
        
        return enriched_patterns
        
    except Exception as e:
        logger.error(f"Error retrieving query results: {e}")
        return None

def format_patterns(patterns):
    """Format patterns for display."""
    if not patterns:
        return "No patterns found."
    
    output = []
    output.append("=== QUERY-REPLY PATTERNS ===\n")
    
    for i, pattern in enumerate(patterns, 1):
        pattern_id = pattern.get("_key") or pattern.get("id")
        pattern_type = pattern.get("type", "unknown")
        content = pattern.get("content", "No content")
        created_at = pattern.get("created_at", "Unknown time")
        coherence = pattern.get("coherence", {})
        
        output.append(f"Pattern {i}: {pattern_type.upper()}")
        output.append(f"ID: {pattern_id}")
        output.append(f"Created: {created_at}")
        output.append(f"Content: {content[:200]}..." if len(content) > 200 else f"Content: {content}")
        
        if isinstance(coherence, dict):
            coherence_str = ", ".join([f"{k}: {v}" for k, v in coherence.items()])
            output.append(f"Coherence: {coherence_str}")
        elif coherence:
            output.append(f"Coherence: {coherence}")
        
        # Add relationships
        relationships = pattern.get("relationships", [])
        if relationships:
            output.append(f"Relationships ({len(relationships)}):")
            for j, rel in enumerate(relationships, 1):
                rel_type = rel.get("type", "unknown")
                source = rel.get("_from", rel.get("source_id", "unknown"))
                target = rel.get("_to", rel.get("target_id", "unknown"))
                
                # Clean up IDs (remove collection prefix if present)
                if "/" in source:
                    source = source.split("/")[-1]
                if "/" in target:
                    target = target.split("/")[-1]
                
                output.append(f"  {j}. {rel_type}: {source} → {target}")
        
        output.append("")  # Empty line between patterns
    
    return "\n".join(output)

if __name__ == "__main__":
    logger.info("Starting retrieval of query results")
    patterns = retrieve_query_results()
    
    if patterns:
        formatted_output = format_patterns(patterns)
        print("\n" + formatted_output)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_results_{timestamp}.txt"
        with open(filename, "w") as f:
            f.write(formatted_output)
        
        logger.info(f"Results saved to {filename}")
        
        # Also save raw JSON for further analysis
        json_filename = f"query_results_{timestamp}.json"
        with open(json_filename, "w") as f:
            json.dump(patterns, f, indent=2, default=str)
        
        logger.info(f"Raw results saved to {json_filename}")
        
        sys.exit(0)
    else:
        logger.error("Failed to retrieve query results")
        sys.exit(1)
