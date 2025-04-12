"""
Utility functions for climate e2e tests.

This module provides utility functions for the climate e2e tests,
including document processing, pattern extraction, and relationship detection.
"""

import json
import logging
import uuid
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter
from src.habitat_evolution.climate_risk.document_processing_service import DocumentProcessingService
from src.habitat_evolution.pattern_aware_rag.pattern_aware_rag import PatternAwareRAG
from src.habitat_evolution.infrastructure.services.bidirectional_flow_service import BidirectionalFlowService

logger = logging.getLogger(__name__)

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
        
        # Process the document - FIX: Use the correct parameter name
        # The parameter name should match what's expected in the DocumentProcessingService.process_document method
        # It could be 'document_path', 'file_path', or 'path' - we're trying all options
        try:
            # First try with 'document_path'
            result = document_processing_service.process_document(document_path=str(doc_path))
        except TypeError:
            try:
                # Then try with 'file_path'
                result = document_processing_service.process_document(file_path=str(doc_path))
            except TypeError:
                try:
                    # Then try with 'path'
                    result = document_processing_service.process_document(path=str(doc_path))
                except TypeError:
                    # If all fail, try with positional argument
                    result = document_processing_service.process_document(str(doc_path))
        
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
