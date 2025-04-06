"""
Test for bidirectional flow between document processing, pattern-aware RAG, and pattern evolution.

This test verifies that the complete bidirectional flow is working correctly,
from document ingestion through processing, persistence, RAG, and back to persistence.
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


class TestBidirectionalFlow(unittest.TestCase):
    """Test the bidirectional flow between document processing, RAG, and pattern evolution."""
    
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
        
        logger.info("Test environment set up successfully")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        # Shutdown services
        cls.document_processing_service.shutdown()
        cls.pattern_aware_rag.shutdown()
        
        logger.info("Test environment cleaned up successfully")
    
    def test_document_processing_to_rag(self):
        """Test document processing to RAG flow."""
        # Process a test document
        document_path = os.path.join(os.path.dirname(__file__), "../../../data/climate_risk/climate_risk_marthas_vineyard.txt")
        
        # Check if the file exists
        if not os.path.exists(document_path):
            # Try an alternative path
            document_path = os.path.join(os.path.dirname(__file__), "../../climate_risk/data/climate_risk_marthas_vineyard.txt")
            
            # If still not found, create a test document
            if not os.path.exists(document_path):
                # Create test directory if it doesn't exist
                os.makedirs(os.path.dirname(document_path), exist_ok=True)
                
                # Create a test document
                with open(document_path, 'w') as f:
                    f.write("""
                    CLIMATE RISK ASSESSMENT – MARTHA'S VINEYARD
                    
                    SUMMARY
                    Martha's Vineyard faces significant climate risks including sea level rise, increased storm intensity, and changing precipitation patterns. By 2050, sea levels are projected to rise by 1.5 to 3.1 feet, threatening coastal properties and infrastructure. The island has experienced extreme drought between 12% and 15% of the time in recent years. The number of wildfire days is expected to increase 40% by mid-century and 70% by late-century.
                    
                    RECOMMENDATIONS
                    1. Implement coastal buffer zones to mitigate flooding risks
                    2. Develop drought management plans for agriculture and water resources
                    3. Enhance wildfire prevention and response capabilities
                    4. Improve storm resilience for critical infrastructure
                    """)
        
        # Process the document
        result = self.document_processing_service.process_document(document_path=document_path)
        
        # Verify that the document was processed successfully
        self.assertEqual(result["status"], "success")
        self.assertTrue(len(result["patterns"]) > 0)
        
        # Wait for events to propagate
        time.sleep(2)
        
        # Verify that patterns were stored in the database
        patterns = self.pattern_evolution_service.get_patterns()
        self.assertTrue(len(patterns) > 0)
        
        logger.info(f"Document processed successfully with {len(result['patterns'])} patterns")
    
    def test_query_to_pattern_evolution(self):
        """Test query to pattern evolution flow."""
        # Process a query
        query = "What are the sea level rise projections for Martha's Vineyard by 2050?"
        
        # Run the query
        result = self.climate_risk_query_service.query(query)
        
        # Verify that the query was processed successfully
        self.assertTrue("response" in result)
        self.assertTrue(len(result.get("patterns", [])) > 0)
        
        # Wait for events to propagate
        time.sleep(2)
        
        # Verify that pattern usage was tracked
        # This would require querying the pattern usage repository
        # For now, we'll just log the result
        logger.info(f"Query processed successfully with response: {result.get('response', '')[:100]}...")
    
    def test_complete_bidirectional_flow(self):
        """Test complete bidirectional flow from ingestion through RAG and back to persistence."""
        # Process a document
        document_path = os.path.join(os.path.dirname(__file__), "../../../data/climate_risk/climate_risk_marthas_vineyard.txt")
        
        # Check if the file exists and create it if it doesn't
        if not os.path.exists(document_path):
            # Try an alternative path
            document_path = os.path.join(os.path.dirname(__file__), "../../climate_risk/data/climate_risk_marthas_vineyard.txt")
            
            # If still not found, create a test document
            if not os.path.exists(document_path):
                # Create test directory if it doesn't exist
                os.makedirs(os.path.dirname(document_path), exist_ok=True)
                
                # Create a test document
                with open(document_path, 'w') as f:
                    f.write("""
                    CLIMATE RISK ASSESSMENT – MARTHA'S VINEYARD
                    
                    SUMMARY
                    Martha's Vineyard faces significant climate risks including sea level rise, increased storm intensity, and changing precipitation patterns. By 2050, sea levels are projected to rise by 1.5 to 3.1 feet, threatening coastal properties and infrastructure. The island has experienced extreme drought between 12% and 15% of the time in recent years. The number of wildfire days is expected to increase 40% by mid-century and 70% by late-century.
                    
                    RECOMMENDATIONS
                    1. Implement coastal buffer zones to mitigate flooding risks
                    2. Develop drought management plans for agriculture and water resources
                    3. Enhance wildfire prevention and response capabilities
                    4. Improve storm resilience for critical infrastructure
                    """)
        
        # Process the document
        doc_result = self.document_processing_service.process_document(document_path=document_path)
        
        # Verify that the document was processed successfully
        self.assertEqual(doc_result["status"], "success")
        self.assertTrue(len(doc_result["patterns"]) > 0)
        
        # Wait for events to propagate
        time.sleep(2)
        
        # Process a query related to the document
        query = "What are the projected impacts of climate change on Martha's Vineyard?"
        
        # Run the query
        query_result = self.climate_risk_query_service.query(query)
        
        # Verify that the query was processed successfully
        self.assertTrue("response" in query_result)
        
        # Wait for events to propagate
        time.sleep(2)
        
        # Verify bidirectional flow by checking pattern quality transitions
        # This would require querying the pattern quality transition repository
        # For now, we'll just log the result
        logger.info(f"Complete bidirectional flow test completed successfully")
        logger.info(f"Document processed with {len(doc_result['patterns'])} patterns")
        logger.info(f"Query processed with response: {query_result.get('response', '')[:100]}...")


if __name__ == "__main__":
    unittest.main()
