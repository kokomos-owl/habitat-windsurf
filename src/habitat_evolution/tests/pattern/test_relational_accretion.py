"""
Test Relational Accretion for Queries as Actants

This test demonstrates the relational accretion approach to queries as actants,
where queries gradually accrete significance through interactions rather than
having patterns projected onto them.
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
from src.habitat_evolution.infrastructure.services.pattern_evolution_service import PatternEvolutionService
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.infrastructure.services.event_service import EventService
from src.habitat_evolution.pattern_aware_rag.accretive_pattern_rag import AccretivePatternRAG
from src.habitat_evolution.core.services.field.field_state_service import ConcreteFieldStateService
from src.habitat_evolution.core.storage.field_repository import ArangoFieldRepository
from src.habitat_evolution.core.services.field.gradient_service import GradientService
from src.habitat_evolution.core.services.field.flow_dynamics_service import FlowDynamicsService
from src.habitat_evolution.adaptive_core.services.metrics_service import MetricsService
from src.habitat_evolution.adaptive_core.services.quality_metrics_service import QualityMetricsService
from src.habitat_evolution.pattern_aware_rag.interfaces.pattern_emergence import PatternEmergenceFlow
from src.habitat_evolution.pattern_aware_rag.services.graph_service import GraphService
from src.habitat_evolution.pattern_aware_rag.core.coherence_analyzer import CoherenceAnalyzer
from src.habitat_evolution.pattern_aware_rag.services.significance_accretion_service import SignificanceAccretionService

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "../../pattern_aware_rag/test_env.example"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RelationalAccretionTest(unittest.TestCase):
    """Test case for demonstrating relational accretion for queries as actants."""
    
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
        
        # Initialize field repository
        cls.field_repository = ArangoFieldRepository(cls.arangodb_connection)
        
        # Initialize field services
        cls.field_state_service = ConcreteFieldStateService(
            field_repository=cls.field_repository,
            event_bus=cls.event_service
        )
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
        
        # Settings for AccretivePatternRAG
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
        
        # Initialize AccretivePatternRAG
        cls.accretive_rag = AccretivePatternRAG(
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
            db_connection=cls.arangodb_connection,
            claude_api_key=os.getenv("CLAUDE_API_KEY")
        )
        
        # Initialize significance accretion service
        cls.significance_service = SignificanceAccretionService(
            db_connection=cls.arangodb_connection,
            event_service=cls.event_service
        )
        
        # Process a document to initialize the pattern space
        document_path = os.path.join(os.path.dirname(__file__), "../../../data/climate_risk/climate_risk_marthas_vineyard.txt")
        if os.path.exists(document_path):
            with open(document_path, 'r') as f:
                document_content = f.read()
                
            document = {
                "id": f"doc-{uuid.uuid4()}",
                "content": document_content,
                "metadata": {
                    "source": "test",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            cls.accretive_rag.process_document(document)
            time.sleep(2)  # Allow time for processing
        else:
            logger.warning(f"Test document not found at {document_path}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        cls.accretive_rag.shutdown()
    
    def test_query_baseline_enhancement(self):
        """Test baseline enhancement of queries."""
        # Create a query
        query = "What are the projected impacts of sea level rise on Martha's Vineyard coastal infrastructure by 2050?"
        
        # Process the query
        result = self.accretive_rag.query(query)
        
        # Verify that the query was processed
        self.assertIn("query_id", result)
        self.assertIn("response", result)
        
        # Verify significance metrics
        self.assertIn("significance_level", result)
        self.assertIn("semantic_stability", result)
        self.assertIn("relational_density", result)
        
        # Log the results
        logger.info(f"Query processed with significance level: {result['significance_level']}")
        logger.info(f"Semantic stability: {result['semantic_stability']}")
        logger.info(f"Relational density: {result['relational_density']}")
        logger.info(f"Response: {result['response'][:100]}...")
    
    def test_significance_accretion_over_time(self):
        """Test how significance accretes over time with repeated queries."""
        # Create a sequence of related queries
        queries = [
            "What are the projected sea level rise impacts on Martha's Vineyard by 2050?",
            "How will sea level rise affect Martha's Vineyard infrastructure?",
            "What adaptation strategies are recommended for Martha's Vineyard to address sea level rise?"
        ]
        
        # Track significance levels
        significance_levels = []
        semantic_stability_levels = []
        relational_density_levels = []
        
        # Process each query in sequence
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}: {query}")
            result = self.accretive_rag.query(query)
            
            # Store significance metrics
            significance_levels.append(result["significance_level"])
            semantic_stability_levels.append(result["semantic_stability"])
            relational_density_levels.append(result["relational_density"])
            
            # Allow time for processing
            time.sleep(2)
        
        # Verify that significance accretes over time
        logger.info("Significance accretion over time:")
        for i, level in enumerate(significance_levels):
            logger.info(f"Query {i+1}: Significance={level:.2f}, Stability={semantic_stability_levels[i]:.2f}, Density={relational_density_levels[i]:.2f}")
        
        # Verify increasing trend
        self.assertGreaterEqual(significance_levels[-1], significance_levels[0])
        self.assertGreaterEqual(semantic_stability_levels[-1], semantic_stability_levels[0])
        self.assertGreaterEqual(relational_density_levels[-1], relational_density_levels[0])
    
    def test_pattern_emergence_from_significance(self):
        """Test how patterns emerge from query significance."""
        # Create a sequence of queries to build significance
        queries = [
            "What are the projected sea level rise impacts on Martha's Vineyard by 2050?",
            "How will sea level rise affect Martha's Vineyard infrastructure?",
            "What adaptation strategies are recommended for Martha's Vineyard to address sea level rise?",
            "Which areas of Martha's Vineyard are most vulnerable to sea level rise?",
            "How can Martha's Vineyard prepare its infrastructure for sea level rise?"
        ]
        
        # Get initial pattern count
        initial_patterns = self.pattern_evolution_service.get_patterns()
        initial_count = len(initial_patterns)
        logger.info(f"Initial pattern count: {initial_count}")
        
        # Process each query in sequence
        query_results = []
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}: {query}")
            result = self.accretive_rag.query(query)
            query_results.append(result)
            
            # Allow time for processing
            time.sleep(2)
        
        # Get final pattern count
        final_patterns = self.pattern_evolution_service.get_patterns()
        final_count = len(final_patterns)
        logger.info(f"Final pattern count: {final_count}")
        
        # Verify that patterns emerged from significance
        self.assertGreater(final_count, initial_count)
        
        # Log the pattern emergence
        logger.info(f"Pattern emergence: {final_count - initial_count} new patterns created")
        
        # Verify pattern IDs in results
        pattern_ids = [result.get("pattern_id") for result in query_results if result.get("pattern_id")]
        logger.info(f"Pattern IDs from queries: {pattern_ids}")
    
    def test_query_relationship_formation(self):
        """Test how relationships form between queries through shared patterns."""
        # Create two related queries
        query1 = "What are the projected sea level rise impacts on Martha's Vineyard by 2050?"
        query2 = "How will coastal flooding affect Martha's Vineyard infrastructure due to sea level rise?"
        
        # Process the first query
        result1 = self.accretive_rag.query(query1)
        query_id1 = result1["query_id"]
        
        # Allow time for processing
        time.sleep(2)
        
        # Process the second query
        result2 = self.accretive_rag.query(query2)
        query_id2 = result2["query_id"]
        
        # Allow time for processing
        time.sleep(2)
        
        # Get significance for both queries
        significance1 = self.significance_service.get_query_significance(query_id1)
        significance2 = self.significance_service.get_query_significance(query_id2)
        
        # Get related queries for the first query
        related_queries = self.significance_service.get_related_queries(query_id1, threshold=0.1)
        
        # Verify relationship formation
        logger.info(f"Related queries for '{query1}':")
        for related in related_queries:
            logger.info(f"  - '{related['query_text']}' (similarity: {related['similarity']:.2f})")
        
        # Verify that the second query is related to the first
        self.assertTrue(any(related["query_id"] == query_id2 for related in related_queries))
    
    def test_complete_accretion_flow(self):
        """Test the complete accretion flow from document processing to query evolution."""
        # Process a document
        document_path = os.path.join(os.path.dirname(__file__), "../../../data/climate_risk/climate_risk_marthas_vineyard.txt")
        if not os.path.exists(document_path):
            self.skipTest(f"Test document not found at {document_path}")
        
        with open(document_path, 'r') as f:
            document_content = f.read()
            
        document = {
            "id": f"doc-{uuid.uuid4()}",
            "content": document_content,
            "metadata": {
                "source": "test",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        doc_result = self.accretive_rag.process_document(document)
        time.sleep(2)  # Allow time for processing
        
        # Create a sequence of queries that build on each other
        query_sequence = [
            "What are the key climate risks for Martha's Vineyard?",
            "How will sea level rise specifically impact Martha's Vineyard?",
            "What adaptation strategies are recommended for sea level rise on Martha's Vineyard?"
        ]
        
        # Process each query and track significance evolution
        significance_evolution = []
        
        for i, query in enumerate(query_sequence):
            logger.info(f"\nProcessing query {i+1}/{len(query_sequence)}: {query}")
            result = self.accretive_rag.query(query)
            
            # Log response
            response = result.get("response", "")
            logger.info(f"Response: {response[:100]}...")
            
            # Log significance metrics
            significance = result.get("significance_level", 0)
            stability = result.get("semantic_stability", 0)
            density = result.get("relational_density", 0)
            
            significance_evolution.append({
                "query": query,
                "significance": significance,
                "stability": stability,
                "density": density
            })
            
            logger.info(f"Significance: {significance:.2f}, Stability: {stability:.2f}, Density: {density:.2f}")
            
            # Allow time for processing
            time.sleep(2)
        
        # Log the significance evolution
        logger.info("\nSignificance Evolution:")
        for i, data in enumerate(significance_evolution):
            logger.info(f"Query {i+1}: Significance={data['significance']:.2f}, Stability={data['stability']:.2f}, Density={data['density']:.2f}")
        
        # Verify increasing trend in significance
        self.assertGreaterEqual(significance_evolution[-1]["significance"], significance_evolution[0]["significance"])
        
        # Get pattern counts
        patterns = self.pattern_evolution_service.get_patterns()
        logger.info(f"Total patterns: {len(patterns)}")
        
        # Log the complete flow results
        logger.info("\nComplete Accretion Flow Results:")
        logger.info(f"Document processing completed with status: {doc_result.get('status')}")
        logger.info(f"Query sequence processed with final significance: {significance_evolution[-1]['significance']:.2f}")
        logger.info(f"Total patterns in system: {len(patterns)}")


if __name__ == "__main__":
    unittest.main()
