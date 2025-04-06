"""
Bidirectional Flow Test

This script demonstrates the complete bidirectional flow from document processing
through pattern extraction, persistence, RAG, and back to pattern evolution.
"""

import os
import sys
import logging
import time
from datetime import datetime
import json
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.habitat_evolution.climate_risk.document_processing_service import DocumentProcessingService
from src.habitat_evolution.climate_risk.climate_risk_query_service import ClimateRiskQueryService
from src.habitat_evolution.infrastructure.services.pattern_evolution_service import PatternEvolutionService
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.infrastructure.services.event_service import EventService
from src.habitat_evolution.pattern_aware_rag.pattern_aware_rag import PatternAwareRAG
from src.habitat_evolution.pattern_aware_rag.services.claude_integration_service import ClaudeRAGService
from src.habitat_evolution.core.services.field.field_state_service import FieldStateService
from src.habitat_evolution.core.services.field.gradient_service import GradientService
from src.habitat_evolution.core.services.field.flow_dynamics_service import FlowDynamicsService
from src.habitat_evolution.adaptive_core.services.metrics_service import MetricsService
from src.habitat_evolution.adaptive_core.services.quality_metrics_service import QualityMetricsService
from src.habitat_evolution.pattern_aware_rag.interfaces.pattern_emergence import PatternEmergenceFlow
from src.habitat_evolution.pattern_aware_rag.services.graph_service import GraphService
from src.habitat_evolution.pattern_aware_rag.core.coherence_analyzer import CoherenceAnalyzer

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "../pattern_aware_rag/test_env.example"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_services():
    """Set up the services for the bidirectional flow test."""
    # Initialize services
    event_service = EventService()
    arangodb_connection = ArangoDBConnection(
        uri=os.getenv("ARANGODB_URI", "bolt://localhost:8529"),
        username=os.getenv("ARANGODB_USER", "root"),
        password=os.getenv("ARANGODB_PASSWORD", "habitat")
    )
    
    # Initialize pattern evolution service
    pattern_evolution_service = PatternEvolutionService(
        db_connection=arangodb_connection,
        event_service=event_service
    )
    
    # Initialize field services
    field_state_service = FieldStateService()
    gradient_service = GradientService()
    flow_dynamics_service = FlowDynamicsService()
    
    # Initialize metrics services
    metrics_service = MetricsService()
    quality_metrics_service = QualityMetricsService()
    
    # Initialize coherence analyzer
    coherence_analyzer = CoherenceAnalyzer()
    
    # Initialize emergence flow
    emergence_flow = PatternEmergenceFlow()
    
    # Initialize graph service
    graph_service = GraphService()
    
    # Settings for PatternAwareRAG
    settings = type('Settings', (), {
        'VECTOR_STORE_DIR': os.getenv("TEST_PERSIST_DIR", "./.habitat/test_data"),
        'CACHE_DIR': os.getenv("TEST_CACHE_DIR", "./.habitat/test_cache"),
        'TIMEOUT': int(os.getenv("TEST_TIMEOUT", "30")),
        'WINDOW_DURATION': int(os.getenv("TEST_WINDOW_DURATION", "5")),
        'MAX_CHANGES': int(os.getenv("TEST_MAX_CHANGES", "10")),
        'STABILITY_THRESHOLD': float(os.getenv("TEST_STABILITY_THRESHOLD", "0.7")),
        'COHERENCE_THRESHOLD': float(os.getenv("TEST_COHERENCE_THRESHOLD", "0.6")),
        'BASE_DELAY': float(os.getenv("TEST_BASE_DELAY", "0.1")),
        'MAX_DELAY': float(os.getenv("TEST_MAX_DELAY", "2.0")),
        'PRESSURE_THRESHOLD': float(os.getenv("TEST_PRESSURE_THRESHOLD", "0.8"))
    })()
    
    # Initialize PatternAwareRAG
    pattern_aware_rag = PatternAwareRAG(
        pattern_evolution_service=pattern_evolution_service,
        field_state_service=field_state_service,
        gradient_service=gradient_service,
        flow_dynamics_service=flow_dynamics_service,
        metrics_service=metrics_service,
        quality_metrics_service=quality_metrics_service,
        event_service=event_service,
        coherence_analyzer=coherence_analyzer,
        emergence_flow=emergence_flow,
        settings=settings,
        graph_service=graph_service,
        claude_api_key=os.getenv("CLAUDE_API_KEY")
    )
    
    # Initialize document processing service
    document_processing_service = DocumentProcessingService(
        pattern_evolution_service=pattern_evolution_service,
        arangodb_connection=arangodb_connection,
        claude_api_key=os.getenv("CLAUDE_API_KEY"),
        pattern_aware_rag_service=pattern_aware_rag,
        event_service=event_service
    )
    
    # Initialize climate risk query service
    climate_risk_query_service = ClimateRiskQueryService(
        pattern_aware_rag_service=pattern_aware_rag,
        pattern_evolution_service=pattern_evolution_service,
        event_service=event_service
    )
    
    return {
        "document_processing_service": document_processing_service,
        "climate_risk_query_service": climate_risk_query_service,
        "pattern_evolution_service": pattern_evolution_service,
        "pattern_aware_rag": pattern_aware_rag,
        "event_service": event_service
    }


def test_document_processing(services):
    """Test document processing flow."""
    document_processing_service = services["document_processing_service"]
    
    # Process the document
    document_path = os.path.join(os.path.dirname(__file__), "../../data/climate_risk/climate_risk_marthas_vineyard.txt")
    
    if not os.path.exists(document_path):
        logger.error(f"Document not found at {document_path}")
        return False
    
    logger.info(f"Processing document: {document_path}")
    result = document_processing_service.process_document(document_path=document_path)
    
    # Verify that the document was processed successfully
    if result["status"] != "success":
        logger.error(f"Document processing failed: {result}")
        return False
    
    logger.info(f"Document processed successfully with {len(result['patterns'])} patterns")
    return True


def test_query_processing(services):
    """Test query processing flow."""
    climate_risk_query_service = services["climate_risk_query_service"]
    
    # Process a query
    query = "What are the projected impacts of climate change on Martha's Vineyard in terms of extreme precipitation and drought?"
    
    logger.info(f"Processing query: {query}")
    result = climate_risk_query_service.query(query)
    
    # Verify that the query was processed successfully
    if "response" not in result:
        logger.error(f"Query processing failed: {result}")
        return False
    
    logger.info(f"Query processed successfully")
    logger.info(f"Response: {result.get('response', '')[:200]}...")
    return True


def test_bidirectional_flow():
    """Test the complete bidirectional flow."""
    # Set up services
    logger.info("Setting up services...")
    services = setup_services()
    
    try:
        # Test document processing
        logger.info("Testing document processing...")
        if not test_document_processing(services):
            logger.error("Document processing test failed")
            return False
        
        # Wait for events to propagate
        logger.info("Waiting for events to propagate...")
        time.sleep(5)
        
        # Test query processing
        logger.info("Testing query processing...")
        if not test_query_processing(services):
            logger.error("Query processing test failed")
            return False
        
        # Wait for events to propagate
        logger.info("Waiting for events to propagate...")
        time.sleep(5)
        
        # Verify pattern evolution
        logger.info("Verifying pattern evolution...")
        patterns = services["pattern_evolution_service"].get_patterns()
        logger.info(f"Found {len(patterns)} patterns")
        
        # Success
        logger.info("Bidirectional flow test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error in bidirectional flow test: {e}")
        return False
    finally:
        # Clean up
        logger.info("Cleaning up...")
        services["document_processing_service"].shutdown()
        services["pattern_aware_rag"].shutdown()


if __name__ == "__main__":
    success = test_bidirectional_flow()
    if success:
        logger.info("All tests passed!")
    else:
        logger.error("Tests failed!")
