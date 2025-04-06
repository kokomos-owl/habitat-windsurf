"""
Process Climate Risk Document

This script processes a climate risk document, extracts patterns, and stores them in ArangoDB.
It demonstrates the complete workflow of document processing, pattern extraction, and
pattern evolution tracking using the PatternEvolutionService with AdaptiveID integration.
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add the project root to the Python path to ensure imports work correctly
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.infrastructure.services.pattern_evolution_service import PatternEvolutionService
from src.habitat_evolution.climate_risk.document_processing_service import DocumentProcessingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('climate_risk_processing.log')
    ]
)

logger = logging.getLogger(__name__)

def setup_arangodb() -> ArangoDBConnection:
    """
    Set up the ArangoDB connection.
    
    Returns:
        ArangoDBConnection instance
    """
    logger.info("Setting up ArangoDB connection")
    
    # Create ArangoDB connection
    arangodb_connection = ArangoDBConnection(
        host="localhost",
        port=8529,
        username="root",
        password="",  # Empty password for default setup
        database_name="habitat_evolution"
    )
    
    # Initialize connection
    arangodb_connection.initialize()
    
    # Ensure required collections exist
    arangodb_connection.ensure_collection("patterns")
    arangodb_connection.ensure_collection("pattern_quality_transitions")
    arangodb_connection.ensure_collection("pattern_usage")
    arangodb_connection.ensure_collection("pattern_feedback")
    arangodb_connection.ensure_edge_collection("pattern_relationships")
    
    logger.info("ArangoDB setup complete")
    return arangodb_connection

def process_document(document_path: str, arangodb_connection: ArangoDBConnection) -> List[Dict[str, Any]]:
    """
    Process a climate risk document.
    
    Args:
        document_path: Path to the document
        arangodb_connection: ArangoDB connection
        
    Returns:
        List of extracted patterns
    """
    logger.info(f"Processing document: {document_path}")
    
    # Create PatternEvolutionService
    pattern_evolution_service = PatternEvolutionService(
        arangodb_connection=arangodb_connection,
        verbose=True
    )
    
    # Create DocumentProcessingService
    document_processing_service = DocumentProcessingService(
        pattern_evolution_service=pattern_evolution_service,
        arangodb_connection=arangodb_connection
    )
    
    # Process the document
    patterns = document_processing_service.process_document(document_path)
    
    return patterns

def query_pattern_evolution(pattern_id: str, arangodb_connection: ArangoDBConnection) -> Dict[str, Any]:
    """
    Query the evolution history of a pattern.
    
    Args:
        pattern_id: ID of the pattern
        arangodb_connection: ArangoDB connection
        
    Returns:
        Pattern evolution history
    """
    logger.info(f"Querying evolution history for pattern: {pattern_id}")
    
    # Create PatternEvolutionService
    pattern_evolution_service = PatternEvolutionService(
        arangodb_connection=arangodb_connection,
        verbose=True
    )
    
    # Get pattern evolution
    evolution = pattern_evolution_service.get_pattern_evolution(pattern_id)
    
    return evolution

def main():
    """
    Main entry point for the script.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process climate risk documents")
    parser.add_argument("--document", type=str, required=True, help="Path to the climate risk document")
    parser.add_argument("--query-pattern", type=str, help="Query evolution history for a specific pattern ID")
    args = parser.parse_args()
    
    try:
        # Set up ArangoDB connection
        arangodb_connection = setup_arangodb()
        
        if args.query_pattern:
            # Query pattern evolution
            evolution = query_pattern_evolution(args.query_pattern, arangodb_connection)
            logger.info(f"Pattern evolution: {evolution}")
        else:
            # Process document
            document_path = args.document
            if not os.path.exists(document_path):
                logger.error(f"Document not found: {document_path}")
                sys.exit(1)
                
            patterns = process_document(document_path, arangodb_connection)
            
            # Display results
            logger.info(f"Processed document: {document_path}")
            logger.info(f"Extracted {len(patterns)} patterns")
            
            # Print pattern IDs for reference
            logger.info("Pattern IDs:")
            for pattern in patterns:
                logger.info(f"  - {pattern['id']}: {pattern['base_concept']}")
                
            logger.info("To query pattern evolution, run:")
            logger.info(f"python {__file__} --query-pattern <pattern_id>")
    except Exception as e:
        logger.error(f"Error processing document: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up resources
        if 'arangodb_connection' in locals():
            arangodb_connection.shutdown()

if __name__ == "__main__":
    main()
