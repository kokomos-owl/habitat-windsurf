"""
Utility functions for climate end-to-end integration tests.

This module provides helper functions for setting up test environments,
processing data, and validating results for the climate end-to-end tests.
"""

import os
import json
import uuid
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """
    Get the absolute path to the project root directory.
    
    Returns:
        Path to the project root
    """
    # The tests/integration/climate_e2e directory is 3 levels deep from project root
    return Path(__file__).parents[3]

def setup_arangodb(db_connection):
    """
    Set up ArangoDB collections required for the tests.
    
    Args:
        db_connection: ArangoDB connection object
        
    Returns:
        The initialized ArangoDB connection
    """
    logger.info("Setting up ArangoDB collections...")
    
    # Initialize the connection
    db_connection.initialize()
    
    # Collections for patterns
    db_connection.ensure_collection("semantic_patterns")
    db_connection.ensure_collection("statistical_patterns")
    db_connection.ensure_edge_collection("pattern_relationships")
    
    # Collections for actant-predicate model
    db_connection.ensure_collection("actants")
    db_connection.ensure_collection("predicates")
    db_connection.ensure_edge_collection("actant_relationships")
    
    # Collections for NER and lexicon
    db_connection.ensure_collection("ner_entities")
    db_connection.ensure_collection("climate_lexicon")
    
    # Collections for temporal-topological analysis
    db_connection.ensure_collection("temporal_slices")
    db_connection.ensure_collection("topological_fields")
    
    # Collections for events and queries
    db_connection.ensure_collection("events")
    db_connection.ensure_collection("queries")
    db_connection.ensure_edge_collection("query_pattern_interactions")
    
    # Collections for adaptive evolution
    db_connection.ensure_collection("pattern_versions")
    db_connection.ensure_collection("pattern_evolution_history")
    
    logger.info("ArangoDB collections set up successfully")
    return db_connection

def load_time_series_data(file_path: str) -> pd.DataFrame:
    """
    Load time series data from a JSON file and convert to DataFrame.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        DataFrame with time series data
    """
    logger.info(f"Loading time series data from {file_path}")
    
    with open(file_path, 'r') as f:
        data_json = json.load(f)
    
    # Convert JSON to DataFrame
    data_list = []
    for timestamp, value in data_json.items():
        if isinstance(value, (int, float)):
            data_list.append({"date": timestamp, "temperature": value})
    
    df = pd.DataFrame(data_list)
    
    # Convert date strings to datetime objects if possible
    if 'date' in df.columns:
        try:
            # Try to convert YYYYMM format
            df['date'] = pd.to_datetime(df['date'], format='%Y%m')
        except:
            # If that fails, try standard datetime parsing
            try:
                df['date'] = pd.to_datetime(df['date'])
            except:
                logger.warning("Could not convert date column to datetime")
    
    logger.info(f"Loaded {len(df)} time series data points")
    return df

def process_document_files(document_processing_service, directory_path: str) -> List[Dict[str, Any]]:
    """
    Process all document files in a directory.
    
    Args:
        document_processing_service: Document processing service
        directory_path: Path to directory containing documents
        
    Returns:
        List of extracted patterns
    """
    logger.info(f"Processing documents in {directory_path}")
    
    patterns = []
    directory = Path(directory_path)
    
    for doc_path in directory.glob("*.txt"):
        logger.info(f"Processing document: {doc_path.name}")
        
        # Process the document
        result = document_processing_service.process_document(document_path=str(doc_path))
        
        # Extract patterns
        if "patterns" in result:
            for pattern in result["patterns"]:
                pattern["source"] = "semantic"
                pattern["source_file"] = doc_path.name
                patterns.append(pattern)
    
    logger.info(f"Extracted {len(patterns)} patterns from documents")
    return patterns

def validate_results(arangodb_connection):
    """
    Validate test results by checking ArangoDB collections.
    
    Args:
        arangodb_connection: ArangoDB connection object
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating test results...")
    
    # Check if semantic patterns were stored
    semantic_count = arangodb_connection.execute_query(
        "RETURN LENGTH(FOR doc IN semantic_patterns RETURN doc)"
    )[0]
    
    # Check if statistical patterns were stored
    statistical_count = arangodb_connection.execute_query(
        "RETURN LENGTH(FOR doc IN statistical_patterns RETURN doc)"
    )[0]
    
    # Check if relationships were stored
    relationship_count = arangodb_connection.execute_query(
        "RETURN LENGTH(FOR doc IN pattern_relationships RETURN doc)"
    )[0]
    
    # Check if events were stored
    event_count = arangodb_connection.execute_query(
        "RETURN LENGTH(FOR doc IN events RETURN doc)"
    )[0]
    
    # Check if queries were stored
    query_count = arangodb_connection.execute_query(
        "RETURN LENGTH(FOR doc IN queries RETURN doc)"
    )[0]
    
    # Validation results
    results = {
        "semantic_patterns": semantic_count,
        "statistical_patterns": statistical_count,
        "relationships": relationship_count,
        "events": event_count,
        "queries": query_count,
        "success": semantic_count > 0 and statistical_count > 0 and relationship_count > 0
    }
    
    logger.info(f"Validation results: {results}")
    return results

def generate_report(arangodb_connection, semantic_patterns, statistical_patterns, relationships):
    """
    Generate a report of the test results.
    
    Args:
        arangodb_connection: ArangoDB connection object
        semantic_patterns: List of semantic patterns
        statistical_patterns: List of statistical patterns
        relationships: List of relationships
        
    Returns:
        Report as a string
    """
    logger.info("Generating test report...")
    
    report = f"""# Climate Data Integration Test Report
Generated: {datetime.now().isoformat()}

## Overview

This report summarizes the results of the climate data integration test,
which processes both semantic and statistical data, detects patterns and
relationships, and stores them in ArangoDB.

## Pattern Statistics

- **Semantic Patterns**: {len(semantic_patterns)}
- **Statistical Patterns**: {len(statistical_patterns)}
- **Cross-Domain Relationships**: {len(relationships)}

## Pattern Sources

### Semantic Pattern Sources:
{chr(10).join([f"- {pattern.get('source_file', 'unknown')}" for pattern in semantic_patterns[:5]])}
{f"... and {len(semantic_patterns) - 5} more" if len(semantic_patterns) > 5 else ""}

### Statistical Pattern Sources:
{chr(10).join([f"- {pattern.get('source_file', 'unknown')}" for pattern in statistical_patterns[:5]])}
{f"... and {len(statistical_patterns) - 5} more" if len(statistical_patterns) > 5 else ""}

## Relationship Examples

{chr(10).join([f"- {rel.get('type', 'unknown')}: {rel.get('description', 'No description')[:100]}..." for rel in relationships[:3]])}
{f"... and {len(relationships) - 3} more" if len(relationships) > 3 else ""}

## Database Status

The following collections were populated:
- semantic_patterns: {arangodb_connection.execute_query("RETURN LENGTH(FOR doc IN semantic_patterns RETURN doc)")[0]} documents
- statistical_patterns: {arangodb_connection.execute_query("RETURN LENGTH(FOR doc IN statistical_patterns RETURN doc)")[0]} documents
- pattern_relationships: {arangodb_connection.execute_query("RETURN LENGTH(FOR doc IN pattern_relationships RETURN doc)")[0]} documents
- events: {arangodb_connection.execute_query("RETURN LENGTH(FOR doc IN events RETURN doc)")[0]} documents
- queries: {arangodb_connection.execute_query("RETURN LENGTH(FOR doc IN queries RETURN doc)")[0]} documents

## Conclusion

The climate data integration test has successfully demonstrated the ability of
Habitat Evolution to process both semantic and statistical data, detect patterns
and relationships, and store them in a coherent knowledge graph.
"""
    
    # Save the report to a file
    report_dir = get_project_root() / "data" / "analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "climate_integration_test_report.md"
    
    with open(report_path, "w") as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")
    return report
