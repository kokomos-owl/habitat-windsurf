"""
test_habitat_evolution.py

Integration tests for knowledge graph evolution focusing on:
1. Document ingestion and graph creation
2. RAG enhancement with context
3. Pattern observation and evidence tracking
4. Graph evolution

Environment Setup:
- Requires .env file with: NEO4J_URI, MONGODB_URI, ANTHROPIC_API_KEY
- Requires running instances of Neo4j and MongoDB
- Requires network access for LLM service
"""

import os
import pytest
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from uuid import uuid4
import asyncio
import logging
from contextlib import asynccontextmanager

# Core imports
from core.bidirectional_processor import BidirectionalProcessor
from core.db_interface import DBInterface
from core.llm_interface import LLMInterface
from core.observable_interface import ObservableInterface
from core.rag_controller import RAGController

# Pattern and evolution imports
from adaptive_core.pattern_core import PatternCore
from adaptive_core.knowledge_coherence import KnowledgeCoherence
from meaning_evolution.structure_meaning_evolution import StructureMeaningEvolution

from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for validation
MIN_NODES_EXPECTED = 2
MIN_RELATIONSHIPS_EXPECTED = 1
REQUIRED_ENV_VARS = ['NEO4J_URI', 'MONGODB_URI', 'ANTHROPIC_API_KEY']

def verify_environment():
    """Verify all required environment variables are set."""
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}\n"
            f"Please ensure all required variables are set in .env file"
        )

# Mock classes for testing
class MockTimestampService:
    """Provides timestamps for testing."""
    def get_timestamp(self) -> str:
        return datetime.utcnow().isoformat()

class MockEventManager:
    """Handles event emission for testing."""
    async def emit(self, event_type: str, data: Dict[str, Any]):
        logger.info(f"Mock event emitted: {event_type}")
        pass

class MockVersionService:
    """Provides version information for testing."""
    def get_version(self) -> str:
        return "0.1.0-test"

class MockEvidenceManager:
    """Manages evidence storage for testing."""
    async def store_evidence(self, evidence: Dict[str, Any]) -> str:
        evidence_id = str(uuid4())
        logger.info(f"Stored evidence with ID: {evidence_id}")
        return evidence_id

class MockStateCoordinator:
    """Coordinates state updates for testing."""
    async def update_state(self, state: Dict[str, Any]):
        logger.info("State updated in mock coordinator")
        pass

class TestHabitatEvolution:
    """Test suite for Habitat knowledge evolution workflow."""
    
    @pytest.fixture(scope="class")
    async def setup_environment(self):
        """Setup and cleanup test environment."""
        # Verify environment before running tests
        verify_environment()
        
        from scripts.volume_management import ensure_volume_paths, cleanup_test_data
        
        try:
            # Clean previous test data
            await cleanup_test_data()
            # Ensure required paths exist
            ensure_volume_paths()
            
            yield
            
        finally:
            # Cleanup after tests
            await cleanup_test_data()
            logger.info("Test environment cleaned up")
    
    @pytest.fixture
    async def setup_components(self, setup_environment):
        """Initialize and configure test components."""
        settings = Settings()
        
        try:
            # Initialize core components
            db = DBInterface(settings)
            # Verify database connections
            await self._verify_db_connection(db)
            
            llm = LLMInterface(settings)
            # Verify LLM service
            await self._verify_llm_service(llm)
            
            observable = ObservableInterface(settings)
            
            # Initialize RAG controller
            rag_controller = RAGController(settings, llm)
            
            # Initialize pattern components with mocks
            pattern_core = PatternCore(
                timestamp_service=MockTimestampService(),
                event_manager=MockEventManager(),
                version_service=MockVersionService()
            )
            
            # Create components dictionary
            components = {
                'db': db,
                'llm': llm,
                'observable': observable,
                'rag_controller': rag_controller,
                'pattern_core': pattern_core,
                'coherence': KnowledgeCoherence(
                    pattern_core=pattern_core,
                    timestamp_service=MockTimestampService(),
                    event_manager=MockEventManager(),
                    version_service=MockVersionService()
                ),
                'evolution': StructureMeaningEvolution(),
                'processor': BidirectionalProcessor(
                    pattern_core=pattern_core,
                    evidence_manager=MockEvidenceManager(),
                    state_coordinator=MockStateCoordinator(),
                    timestamp_service=MockTimestampService(),
                    event_manager=MockEventManager()
                )
            }
            
            yield components
            
        except Exception as e:
            logger.error(f"Failed to setup components: {str(e)}")
            raise
        
        finally:
            # Cleanup
            for component in components.values():
                if hasattr(component, 'cleanup'):
                    await component.cleanup()
            logger.info("Components cleaned up")

    async def _verify_db_connection(self, db: DBInterface) -> None:
        """Verify database connections are working."""
        try:
            # Check MongoDB
            await db.mongo_client.admin.command('ping')
            
            # Check Neo4j
            await db.neo4j_driver.verify_connectivity()
            
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {str(e)}")

    async def _verify_llm_service(self, llm: LLMInterface) -> None:
        """Verify LLM service is accessible."""
        try:
            # Test simple completion
            result = await llm.process_text("Test connection.")
            if not result or "response" not in result:
                raise ConnectionError("LLM service test failed")
            
        except Exception as e:
            raise ConnectionError(f"LLM service verification failed: {str(e)}")

    async def _verify_graph_minimum(self, state: Dict[str, Any]) -> None:
        """Verify graph meets minimum requirements."""
        if state["node_count"] < MIN_NODES_EXPECTED:
            raise ValueError(
                f"Graph has fewer than {MIN_NODES_EXPECTED} nodes: {state['node_count']}"
            )
        if state["relationship_count"] < MIN_RELATIONSHIPS_EXPECTED:
            raise ValueError(
                f"Graph has fewer than {MIN_RELATIONSHIPS_EXPECTED} relationships: " \
                f"{state['relationship_count']}"
            )

    def _create_test_document(self, doc_id: str) -> Dict[str, Any]:
        """Create test document with verifiable structure."""
        timestamp = datetime.utcnow().isoformat()
        
        return {
            "doc_id": doc_id,
            "content": {
                "text": "Test document for climate impact analysis",
                "metadata": {
                    "source": "climate_assessment",
                    "timestamp": timestamp
                }
            },
            "structural_elements": {
                "nodes": [
                    {
                        "id": f"concept_1_{doc_id}",
                        "type": "concept",
                        "properties": {
                            "name": "climate_risk",
                            "confidence": 0.8
                        }
                    },
                    {
                        "id": f"concept_2_{doc_id}",
                        "type": "concept",
                        "properties": {
                            "name": "impact_assessment",
                            "confidence": 0.7
                        }
                    }
                ],
                "relationships": [
                    {
                        "source": f"concept_1_{doc_id}",
                        "target": f"concept_2_{doc_id}",
                        "type": "requires",
                        "properties": {
                            "confidence": 0.75,
                            "timestamp": timestamp
                        }
                    }
                ]
            },
            "semantic_elements": {
                "concepts": [
                    {
                        "id": f"concept_1_{doc_id}",
                        "properties": {
                            "domain": "climate",
                            "type": "risk",
                            "confidence": 0.8
                        }
                    },
                    {
                        "id": f"concept_2_{doc_id}",
                        "properties": {
                            "domain": "climate",
                            "type": "method",
                            "confidence": 0.7
                        }
                    }
                ],
                "contexts": [
                    {
                        "type": "domain",
                        "value": "climate",
                        "confidence": 0.9
                    }
                ]
            },
            "metadata": {
                "domain": "climate",
                "timestamp": timestamp,
                "confidence": 0.8,
                "test_id": doc_id
            }
        }

    async def test_knowledge_evolution_core(self, setup_components):
        """
        Test core knowledge evolution functionality with verification of:
        1. Document ingestion and graph creation
        2. RAG enhancement with context
        3. Pattern observation and evidence tracking
        4. Graph evolution
        """
        components = setup_components
        results = {}
        
        try:
            # 1. Document Ingestion and Graph Creation
            logger.info("Testing document ingestion and graph creation...")
            
            # Get initial graph state
            initial_state = await components['db'].get_graph_state()
            logger.info(f"Initial graph state: {initial_state}")
            
            # Create and process test document
            doc_id = str(uuid4())
            document = self._create_test_document(doc_id)
            
            # Process document
            logger.info(f"Processing document {doc_id}...")
            doc_result = await components['processor'].process_document_update(
                doc_id=doc_id,
                update=document
            )
            
            # Get post-ingestion state and verify
            post_ingest_state = await components['db'].get_graph_state()
            await self._verify_graph_minimum(post_ingest_state)
            
            nodes_created = post_ingest_state["node_count"] - initial_state["node_count"]
            relationships_created = post_ingest_state["relationship_count"] - initial_state["relationship_count"]
            
            assert nodes_created >= MIN_NODES_EXPECTED, f"Insufficient nodes created: {nodes_created}"
            assert relationships_created >= MIN_RELATIONSHIPS_EXPECTED, \
                f"Insufficient relationships created: {relationships_created}"
            
            results["ingestion"] = {
                "nodes_created": nodes_created,
                "relationships_created": relationships_created,
                "doc_id": doc_id
            }
            logger.info(f"Document ingestion results: {results['ingestion']}")
            
            # 2. RAG Enhancement
            logger.info("Testing RAG enhancement...")
            
            # Process through RAG
            rag_result = await components['rag_controller'].process_document(
                document={
                    "doc_id": doc_id,
                    "content": document["content"],
                    "metadata": document["metadata"]
                },
                domain_context={"domain": "climate"}
            )
            
            # Verify RAG enhancement produced meaningful changes
            assert len(rag_result["graph_context"]["nodes"]) > len(document["structural_elements"]["nodes"]), \
                "RAG failed to enrich nodes"
            assert len(rag_result["graph_context"]["relationships"]) > \
                   len(document["structural_elements"]["relationships"]), \
                "RAG failed to enrich relationships"
            
            results["rag"] = {
                "original_nodes": len(document["structural_elements"]["nodes"]),
                "enriched_nodes": len(rag_result["graph_context"]["nodes"]),
                "original_relationships": len(document["structural_elements"]["relationships"]),
                "enriched_relationships": len(rag_result["graph_context"]["relationships"])
            }
            logger.info(f"RAG enhancement results: {results['rag']}")
            
            # 3. Pattern Observation and Evidence
            logger.info("Testing pattern observation and evidence tracking...")
            
            # Observe patterns
            pattern = await components['pattern_core'].observe_evolution(
                structural_change=doc_result["graph_updates"]["structural"],
                semantic_change=doc_result["graph_updates"]["semantic"],
                evidence=rag_result["evidence"]
            )
            
            # Verify pattern and evidence creation
            assert pattern["pattern_id"], "No pattern ID generated"
            assert pattern["evidence"]["source"] == "climate_assessment", "Invalid evidence source"
            
            # Verify evidence chains
            evidence_chains = await components['db'].get_evidence_chains(limit=1)
            assert len(evidence_chains) > 0, "No evidence chains created"
            assert evidence_chains[0]["pattern_id"] == pattern["pattern_id"], \
                "Evidence chain not linked to pattern"
            
            results["patterns"] = {
                "pattern_id": pattern["pattern_id"],
                "evidence_chains": len(evidence_chains),
                "confidence": pattern["evidence"].get("confidence", 0)
            }
            logger.info(f"Pattern observation results: {results['patterns']}")
            
            # 4. Graph Evolution
            logger.info("Testing graph evolution...")
            
            # Get pre-evolution state
            pre_evolution_state = await components['db'].get_graph_state()
            
            evolution_context = {
                "structural": rag_result["rag_output"]["graph_updates"]["structural"],
                "semantic": rag_result["rag_output"]["graph_updates"]["semantic"],
                "evidence": pattern
            }
            
            # Process evolution
            evolution_result = await components['evolution'].evolve(
                data=evolution_context,
                context=evolution_context
            )
            
            # Apply evolution updates
            await components['db'].update_graph(evolution_result.evolved_data)
            
            # Get and verify post-evolution state
            post_evolution_state = await components['db'].get_graph_state()
            await self._verify_graph_minimum(post_evolution_state)
            
            # Extract and verify semantic changes
            initial_meanings = set(n["properties"].get("meaning", "") 
                                for n in pre_evolution_state["nodes"])
            evolved_meanings = set(n["properties"].get("meaning", "") 
                                for n in post_evolution_state["nodes"])
            new_meanings = len(evolved_meanings - initial_meanings)
            
            assert new_meanings > 0, "No semantic evolution occurred"
            
            results["evolution"] = {
                "nodes_added": post_evolution_state["node_count"] - pre_evolution_state["node_count"],
                "relationships_added": post_evolution_state["relationship_count"] - 
                                     pre_evolution_state["relationship_count"],
                "new_meanings": new_meanings,
                "success": evolution_result.success
            }
            logger.info(f"Evolution results: {results['evolution']}")
            
            # Final verification
            await self._verify_graph_minimum(post_evolution_state)
            
            logger.info("Knowledge evolution test completed successfully")
            return {
                "success": True,
                "stages": results,
                "final_state": {
                    "nodes": post_evolution_state["node_count"],
                    "relationships": post_evolution_state["relationship_count"]
                }
            }
            
        except Exception as e:
            logger.error(f"Knowledge evolution test failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "stage": "knowledge_evolution",
                "partial_results": results
            }

    async def test_document_ingestion(self, setup_components):
        """Verify document ingestion and graph creation in isolation."""
        components = setup_components
        
        try:
            # Get initial state
            initial_state = await components['db'].get_graph_state()
            logger.info(f"Initial graph state: {initial_state}")
            
            # Process document
            doc_id = str(uuid4())
            document = self._create_test_document(doc_id)
            
            logger.info(f"Processing test document {doc_id}")
            doc_result = await components['processor'].process_document_update(
                doc_id=doc_id,
                update=document
            )
            
            # Get and verify final state
            final_state = await components['db'].get_graph_state()
            await self._verify_graph_minimum(final_state)
            
            nodes_created = final_state["node_count"] - initial_state["node_count"]
            relationships_created = final_state["relationship_count"] - initial_state["relationship_count"]
            
            assert nodes_created >= MIN_NODES_EXPECTED, \
                f"Document ingestion created insufficient nodes: {nodes_created}"
            assert relationships_created >= MIN_RELATIONSHIPS_EXPECTED, \
                f"Document ingestion created insufficient relationships: {relationships_created}"
            
            return {
                "success": True,
                "doc_id": doc_id,
                "nodes_created": nodes_created,
                "relationships_created": relationships_created
            }
            
        except Exception as e:
            logger.error(f"Document ingestion test failed: {str(e)}")
            raise

    async def test_rag_enhancement(self, setup_components):
        """Verify RAG enhancement functionality in isolation."""
        components = setup_components
        
        try:
            # Create and process test document
            doc_id = str(uuid4())
            document = self._create_test_document(doc_id)
            
            logger.info(f"Testing RAG enhancement for document {doc_id}")
            
            # Get initial context size
            initial_nodes = len(document["structural_elements"]["nodes"])
            initial_relationships = len(document["structural_elements"]["relationships"])
            
            # Process through RAG
            rag_result = await components['rag_controller'].process_document(
                document={
                    "doc_id": doc_id,
                    "content": document["content"],
                    "metadata": document["metadata"]
                },
                domain_context={"domain": "climate"}
            )
            
            # Verify enrichment
            enriched_nodes = len(rag_result["graph_context"]["nodes"])
            enriched_relationships = len(rag_result["graph_context"]["relationships"])
            
            assert enriched_nodes > initial_nodes, \
                f"RAG failed to enrich nodes: {initial_nodes} -> {enriched_nodes}"
            assert enriched_relationships > initial_relationships, \
                f"RAG failed to enrich relationships: {initial_relationships} -> {enriched_relationships}"
            
            return {
                "success": True,
                "doc_id": doc_id,
                "node_enrichment": enriched_nodes - initial_nodes,
                "relationship_enrichment": enriched_relationships - initial_relationships
            }
            
        except Exception as e:
            logger.error(f"RAG enhancement test failed: {str(e)}")
            raise

    async def test_pattern_observation(self, setup_components):
        """Verify pattern observation and evidence tracking in isolation."""
        components = setup_components
        
        try:
            # Create and process initial document
            doc_id = str(uuid4())
            document = self._create_test_document(doc_id)
            
            logger.info(f"Testing pattern observation for document {doc_id}")
            
            # Get document result
            doc_result = await components['processor'].process_document_update(
                doc_id=doc_id,
                update=document
            )
            
            # Get RAG context
            rag_result = await components['rag_controller'].process_document(
                document={
                    "doc_id": doc_id,
                    "content": document["content"],
                    "metadata": document["metadata"]
                },
                domain_context={"domain": "climate"}
            )
            
            # Observe patterns
            pattern = await components['pattern_core'].observe_evolution(
                structural_change=doc_result["graph_updates"]["structural"],
                semantic_change=doc_result["graph_updates"]["semantic"],
                evidence=rag_result["evidence"]
            )
            
            # Verify pattern
            assert pattern["pattern_id"], "Pattern missing ID"
            assert pattern["evidence"], "Pattern missing evidence"
            assert pattern["evidence"]["source"] == "climate_assessment", "Invalid evidence source"
            
            # Verify evidence chains
            evidence_chains = await components['db'].get_evidence_chains(limit=1)
            assert len(evidence_chains) > 0, "No evidence chains created"
            assert evidence_chains[0]["pattern_id"] == pattern["pattern_id"], \
                "Evidence chain not linked to pattern"
            
            return {
                "success": True,
                "pattern_id": pattern["pattern_id"],
                "evidence_chains": len(evidence_chains)
            }
            
        except Exception as e:
            logger.error(f"Pattern observation test failed: {str(e)}")
            raise

    async def test_graph_evolution(self, setup_components):
        """Verify graph evolution functionality in isolation."""
        components = setup_components
        
        try:
            # Create and process test document
            doc_id = str(uuid4())
            document = self._create_test_document(doc_id)
            
            logger.info(f"Testing graph evolution for document {doc_id}")
            
            # Get document result
            doc_result = await components['processor'].process_document_update(
                doc_id=doc_id,
                update=document
            )
            
            # Get RAG context
            rag_result = await components['rag_controller'].process_document(
                document={
                    "doc_id": doc_id,
                    "content": document["content"],
                    "metadata": document["metadata"]
                },
                domain_context={"domain": "climate"}
            )
            
            # Get pattern
            pattern = await components['pattern_core'].observe_evolution(
                structural_change=doc_result["graph_updates"]["structural"],
                semantic_change=doc_result["graph_updates"]["semantic"],
                evidence=rag_result["evidence"]
            )
            
            # Get pre-evolution state
            pre_evolution_state = await components['db'].get_graph_state()
            
            # Process evolution
            evolution_context = {
                "structural": rag_result["rag_output"]["graph_updates"]["structural"],
                "semantic": rag_result["rag_output"]["graph_updates"]["semantic"],
                "evidence": pattern
            }
            
            evolution_result = await components['evolution'].evolve(
                data=evolution_context,
                context=evolution_context
            )
            
            assert evolution_result.success, "Evolution processing failed"
            
            # Apply evolution updates
            await components['db'].update_graph(evolution_result.evolved_data)
            
            # Get and verify post-evolution state
            post_evolution_state = await components['db'].get_graph_state()
            await self._verify_graph_minimum(post_evolution_state)
            
            # Verify changes
            nodes_added = post_evolution_state["node_count"] - pre_evolution_state["node_count"]
            relationships_added = post_evolution_state["relationship_count"] - \
                                pre_evolution_state["relationship_count"]
            
            assert nodes_added > 0, "No nodes added during evolution"
            assert relationships_added > 0, "No relationships added during evolution"
            
            return {
                "success": True,
                "nodes_added": nodes_added,
                "relationships_added": relationships_added,
                "evolution_id": evolution_result.evolution_id
            }
            
        except Exception as e:
            logger.error(f"Graph evolution test failed: {str(e)}")
            raise

    async def test_component_health(self, setup_components):
        """Verify health of core components."""
        components = setup_components
        health_status = {}
        
        try:
            # Check DB connection
            await self._verify_db_connection(components['db'])
            health_status['db'] = "healthy"
            
            # Check LLM service
            await self._verify_llm_service(components['llm'])
            health_status['llm'] = "healthy"
            
            # Check RAG metrics
            rag_metrics = components['rag_controller'].get_metrics()
            assert "queries" in rag_metrics, "RAG metrics unavailable"
            health_status['rag'] = "healthy"
            
            # Verify graph minimum state
            graph_state = await components['db'].get_graph_state()
            await self._verify_graph_minimum(graph_state)
            health_status['graph'] = "healthy"
            
            return {
                "success": True,
                "health_status": health_status,
                "graph_state": graph_state,
                "rag_metrics": rag_metrics
            }
            
        except Exception as e:
            logger.error(f"Component health check failed: {str(e)}")
            health_status['error'] = str(e)
            return {
                "success": False,
                "health_status": health_status
            }

if __name__ == "__main__":
    verify_environment()
    pytest.main(["-v", __file__])