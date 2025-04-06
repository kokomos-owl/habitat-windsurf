"""
Test Queries as Actants in Habitat Evolution

This test demonstrates how queries act as actants in the Habitat Evolution system,
actively participating in and influencing pattern evolution.
"""

import os
import sys
import logging
import unittest
import asyncio
import time
from typing import Dict, List, Any
from datetime import datetime
import uuid
import json
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

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
load_dotenv(os.path.join(os.path.dirname(__file__), "../../pattern_aware_rag/test_env.example"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QueriesAsActantsTest(unittest.TestCase):
    """Test case for demonstrating queries as actants in Habitat Evolution."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        # Initialize services
        cls.event_service = EventService()
        cls.arangodb_connection = ArangoDBConnection(
            uri=os.getenv("ARANGODB_URI", "bolt://localhost:8529"),
            username=os.getenv("ARANGODB_USER", "root"),
            password=os.getenv("ARANGODB_PASSWORD", "habitat")
        )
        
        # Initialize pattern evolution service
        cls.pattern_evolution_service = PatternEvolutionService(
            db_connection=cls.arangodb_connection,
            event_service=cls.event_service
        )
        
        # Initialize field services
        cls.field_state_service = FieldStateService()
        cls.gradient_service = GradientService()
        cls.flow_dynamics_service = FlowDynamicsService()
        
        # Initialize metrics services
        cls.metrics_service = MetricsService()
        cls.quality_metrics_service = QualityMetricsService()
        
        # Initialize coherence analyzer
        cls.coherence_analyzer = CoherenceAnalyzer()
        
        # Initialize emergence flow
        cls.emergence_flow = PatternEmergenceFlow()
        
        # Initialize graph service
        cls.graph_service = GraphService()
        
        # Settings for PatternAwareRAG
        cls.settings = type('Settings', (), {
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
        cls.pattern_aware_rag = PatternAwareRAG(
            pattern_evolution_service=cls.pattern_evolution_service,
            field_state_service=cls.field_state_service,
            gradient_service=cls.gradient_service,
            flow_dynamics_service=cls.flow_dynamics_service,
            metrics_service=cls.metrics_service,
            quality_metrics_service=cls.quality_metrics_service,
            event_service=cls.event_service,
            coherence_analyzer=cls.coherence_analyzer,
            emergence_flow=cls.emergence_flow,
            settings=cls.settings,
            graph_service=cls.graph_service,
            claude_api_key=os.getenv("CLAUDE_API_KEY")
        )
        
        # Initialize document processing service
        cls.document_processing_service = DocumentProcessingService(
            pattern_evolution_service=cls.pattern_evolution_service,
            arangodb_connection=cls.arangodb_connection,
            claude_api_key=os.getenv("CLAUDE_API_KEY"),
            pattern_aware_rag_service=cls.pattern_aware_rag,
            event_service=cls.event_service
        )
        
        # Initialize climate risk query service
        cls.climate_risk_query_service = ClimateRiskQueryService(
            pattern_aware_rag_service=cls.pattern_aware_rag,
            pattern_evolution_service=cls.pattern_evolution_service,
            event_service=cls.event_service
        )
        
        # Process a document to initialize the pattern space
        document_path = os.path.join(os.path.dirname(__file__), "../../../data/climate_risk/climate_risk_marthas_vineyard.txt")
        if os.path.exists(document_path):
            cls.document_processing_service.process_document(document_path=document_path)
            time.sleep(5)  # Allow time for processing
        else:
            logger.warning(f"Test document not found at {document_path}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        cls.document_processing_service.shutdown()
        cls.pattern_aware_rag.shutdown()
    
    def test_query_pattern_extraction(self):
        """Test pattern extraction from queries."""
        # Create a query
        query = "What are the projected impacts of sea level rise on Martha's Vineyard coastal infrastructure by 2050?"
        
        # Process the query
        result = self.climate_risk_query_service.query(query)
        
        # Verify that patterns were extracted from the query
        self.assertIn("patterns", result)
        self.assertTrue(len(result["patterns"]) > 0)
        
        # Log the extracted patterns
        logger.info(f"Extracted {len(result['patterns'])} patterns from query")
        for pattern in result["patterns"]:
            logger.info(f"Pattern: {pattern.get('base_concept', 'unknown')} - Confidence: {pattern.get('confidence', 0)}")
    
    def test_query_influence_on_patterns(self):
        """Test how queries influence pattern evolution."""
        # Get initial patterns
        initial_patterns = self.pattern_evolution_service.get_patterns()
        initial_count = len(initial_patterns)
        logger.info(f"Initial pattern count: {initial_count}")
        
        # Process a sequence of related queries to influence pattern evolution
        queries = [
            "What are the projected sea level rise impacts on Martha's Vineyard by 2050?",
            "How will coastal flooding affect Martha's Vineyard infrastructure?",
            "What adaptation strategies are recommended for Martha's Vineyard to address sea level rise?",
            "How do sea level rise projections for Martha's Vineyard compare to other coastal areas?",
            "What are the economic impacts of sea level rise on Martha's Vineyard property values?"
        ]
        
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}: {query}")
            result = self.climate_risk_query_service.query(query)
            time.sleep(2)  # Allow time for pattern evolution
        
        # Get updated patterns
        updated_patterns = self.pattern_evolution_service.get_patterns()
        updated_count = len(updated_patterns)
        logger.info(f"Updated pattern count: {updated_count}")
        
        # Verify pattern evolution
        self.assertGreaterEqual(updated_count, initial_count)
        
        # Get pattern quality transitions
        transitions = self.pattern_evolution_service.get_pattern_quality_transitions()
        logger.info(f"Found {len(transitions)} pattern quality transitions")
        
        # Verify that some patterns have evolved in quality
        if transitions:
            for transition in transitions[:5]:  # Log first 5 transitions
                logger.info(f"Pattern {transition.get('pattern_id')} transitioned from {transition.get('from_state')} to {transition.get('to_state')}")
    
    def test_pattern_coherence_through_queries(self):
        """Test how pattern coherence evolves through query interactions."""
        # Process a query to establish baseline coherence
        query1 = "What are the drought risks for Martha's Vineyard agriculture?"
        result1 = self.climate_risk_query_service.query(query1)
        
        # Extract coherence metrics
        coherence1 = result1.get("coherence", {})
        flow_state1 = coherence1.get("flow_state", "unknown")
        emergence_potential1 = coherence1.get("emergence_potential", 0)
        
        logger.info(f"Initial query coherence - Flow state: {flow_state1}, Emergence potential: {emergence_potential1}")
        
        # Process a related follow-up query
        query2 = "How will drought conditions affect water resources for Martha's Vineyard residents?"
        result2 = self.climate_risk_query_service.query(query2)
        
        # Extract updated coherence metrics
        coherence2 = result2.get("coherence", {})
        flow_state2 = coherence2.get("flow_state", "unknown")
        emergence_potential2 = coherence2.get("emergence_potential", 0)
        
        logger.info(f"Follow-up query coherence - Flow state: {flow_state2}, Emergence potential: {emergence_potential2}")
        
        # Process a contradictory query
        query3 = "Is there evidence that Martha's Vineyard will not face significant drought risks?"
        result3 = self.climate_risk_query_service.query(query3)
        
        # Extract final coherence metrics
        coherence3 = result3.get("coherence", {})
        flow_state3 = coherence3.get("flow_state", "unknown")
        emergence_potential3 = coherence3.get("emergence_potential", 0)
        
        logger.info(f"Contradictory query coherence - Flow state: {flow_state3}, Emergence potential: {emergence_potential3}")
        
        # Verify coherence evolution
        logger.info("Coherence evolution through query sequence:")
        logger.info(f"Query 1: {flow_state1} - {emergence_potential1}")
        logger.info(f"Query 2: {flow_state2} - {emergence_potential2}")
        logger.info(f"Query 3: {flow_state3} - {emergence_potential3}")
    
    def test_bidirectional_flow_with_queries(self):
        """Test the complete bidirectional flow with queries as actants."""
        # Process initial document
        document_path = os.path.join(os.path.dirname(__file__), "../../../data/climate_risk/climate_risk_marthas_vineyard.txt")
        if not os.path.exists(document_path):
            self.skipTest(f"Test document not found at {document_path}")
        
        doc_result = self.document_processing_service.process_document(document_path=document_path)
        time.sleep(3)  # Allow time for processing
        
        # Get initial pattern state
        initial_patterns = self.pattern_evolution_service.get_patterns()
        initial_count = len(initial_patterns)
        logger.info(f"Initial pattern count after document processing: {initial_count}")
        
        # Process a sequence of queries that build on each other
        query_sequence = [
            "What are the key climate risks for Martha's Vineyard?",
            "How will sea level rise specifically impact Martha's Vineyard?",
            "What adaptation strategies are recommended for sea level rise on Martha's Vineyard?",
            "Which areas of Martha's Vineyard are most vulnerable to sea level rise?",
            "How do sea level rise projections for Martha's Vineyard compare to global averages?"
        ]
        
        # Process each query and track pattern evolution
        for i, query in enumerate(query_sequence):
            logger.info(f"\nProcessing query {i+1}/{len(query_sequence)}: {query}")
            result = self.climate_risk_query_service.query(query)
            
            # Log response
            response = result.get("response", "")
            logger.info(f"Response: {response[:100]}...")
            
            # Log patterns used
            patterns_used = result.get("patterns", [])
            logger.info(f"Used {len(patterns_used)} patterns in response")
            
            # Get current pattern count
            current_patterns = self.pattern_evolution_service.get_patterns()
            logger.info(f"Current pattern count: {len(current_patterns)}")
            
            # Allow time for pattern evolution
            time.sleep(2)
        
        # Get final pattern state
        final_patterns = self.pattern_evolution_service.get_patterns()
        final_count = len(final_patterns)
        logger.info(f"Final pattern count: {final_count}")
        
        # Verify pattern evolution through query sequence
        self.assertGreaterEqual(final_count, initial_count)
        
        # Get pattern usage statistics
        pattern_usage = self.pattern_evolution_service.get_pattern_usage()
        logger.info(f"Found usage data for {len(pattern_usage)} patterns")
        
        # Verify pattern feedback
        pattern_feedback = self.pattern_evolution_service.get_pattern_feedback()
        logger.info(f"Found feedback data for {len(pattern_feedback)} patterns")
        
        # Log the bidirectional flow results
        logger.info("\nBidirectional Flow Results:")
        logger.info(f"Document processing extracted {len(doc_result.get('patterns', []))} initial patterns")
        logger.info(f"Query sequence led to {final_count - initial_count} new patterns")
        logger.info(f"Pattern usage recorded for {len(pattern_usage)} patterns")
        logger.info(f"Pattern feedback recorded for {len(pattern_feedback)} patterns")


if __name__ == "__main__":
    unittest.main()
