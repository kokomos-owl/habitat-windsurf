"""
Utility functions for climate e2e tests.

This module provides utility functions for the climate e2e tests,
including document processing, pattern extraction, and relationship detection.
"""

import os
import json
import logging
import uuid
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter
from src.habitat_evolution.climate_risk.document_processing_service import DocumentProcessingService
from src.habitat_evolution.pattern_aware_rag.pattern_aware_rag import PatternAwareRAG
from src.habitat_evolution.infrastructure.services.bidirectional_flow_service import BidirectionalFlowService

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
    
    # Collections for documents and fields
    db_connection.ensure_collection("documents")
    db_connection.ensure_collection("fields")
    
    # Collections for queries and responses
    db_connection.ensure_collection("queries")
    db_connection.ensure_collection("responses")
    db_connection.ensure_edge_collection("query_response_relationships")
    
    logger.info("ArangoDB collections set up successfully")
    return db_connection

def load_test_data(file_path: str) -> Dict[str, Any]:
    """
    Load test data from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary containing test data
    """
    with open(file_path, "r") as f:
        return json.load(f)

def save_test_data(data: Dict[str, Any], file_path: str) -> None:
    """
    Save test data to a JSON file.
    
    Args:
        data: Dictionary containing test data
        file_path: Path to JSON file
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

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
        
        # Process the document using the correct parameter name
        doc_path_str = str(doc_path)
        try:
            # Use keyword argument to ensure we're passing the parameter correctly
            result = document_processing_service.process_document(document_path=doc_path_str)
            
            # Extract patterns
            if "patterns" in result:
                for pattern in result["patterns"]:
                    pattern["source"] = "semantic"
                    pattern["source_file"] = doc_path.name
                    patterns.append(pattern)
        except Exception as e:
            logger.error(f"Error processing document {doc_path.name}: {e}")
            # Try with a fallback approach if the regular approach fails
            try:
                with open(doc_path_str, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                result = document_processing_service.process_document(
                    document_id=doc_path.name,
                    content=content,
                    metadata={"source": "file", "path": doc_path_str}
                )
                
                # Extract patterns
                if "patterns" in result:
                    for pattern in result["patterns"]:
                        pattern["source"] = "semantic"
                        pattern["source_file"] = doc_path.name
                        patterns.append(pattern)
            except Exception as inner_e:
                logger.error(f"Fallback processing failed for {doc_path.name}: {inner_e}")
    
    logger.info(f"Extracted {len(patterns)} patterns from documents")
    return patterns

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

def validate_results(arangodb_connection):
    """
    Validate test results by checking ArangoDB collections.
    
    Args:
        arangodb_connection: ArangoDB connection
        
    Returns:
        Dictionary containing validation results
    """
    # Check patterns collection
    patterns_query = "FOR p IN patterns RETURN p"
    patterns = arangodb_connection.execute_query(patterns_query)
    
    # Check relationships collection
    relationships_query = "FOR r IN pattern_relationships RETURN r"
    relationships = arangodb_connection.execute_query(relationships_query)
    
    return {
        "patterns_count": len(patterns),
        "relationships_count": len(relationships)
    }

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

The test demonstrates the system's ability to process both semantic and statistical data,
identify patterns, detect cross-domain relationships, and store the results in ArangoDB.
This provides a foundation for further development of the climate risk analysis system.
"""
    
    # Save report to file
    report_path = Path(get_project_root()) / "reports" / f"climate_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")
    return report

def create_test_pattern(pattern_type: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a test pattern for testing purposes.
    
    Args:
        pattern_type: Type of pattern
        content: Pattern content
        metadata: Optional metadata
        
    Returns:
        Dictionary representing a pattern
    """
    return {
        "id": str(uuid.uuid4()),
        "type": pattern_type,
        "content": content,
        "metadata": metadata or {},
        "created_at": datetime.now().isoformat()
    }
