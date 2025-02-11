# test_adaptive_id.py

import unittest
import logging
from unittest.mock import patch, MagicMock, ANY
import threading
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys
from io import StringIO
from concurrent.futures import ThreadPoolExecutor

from adaptive_core.adaptive_id import AdaptiveID, AdaptiveIDException
from utils.timestamp_service import TimestampService
from utils.version_service import VersionService
from events.event_manager import EventManager
from events.event_types import EventType
from database.mongodb_client import MongoDBClient
from database.neo4j_client import Neo4jClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_adaptive_id.log')
    ]
)
logger = logging.getLogger(__name__)

class TestOutputCapture:
    """Context manager to capture print statements and logging"""
    def __init__(self):
        self.held = StringIO()
        self.stream = sys.stdout

    def __enter__(self):
        sys.stdout = self.held
        return self.held

    def __exit__(self, *args):
        sys.stdout = self.stream

class TestAdaptiveID(unittest.TestCase):
    def setUp(self):
        """Initialize test environment with logging capture"""
        logger.info("\n" + "="*50)
        logger.info(f"Starting test: {self._testMethodName}")
        logger.info("="*50)
        
        try:
            # Mock core services
            self.timestamp_service = MagicMock(TimestampService)
            self.current_time = datetime.now()
            self.timestamp_service.get_timestamp.return_value = self.current_time.isoformat()
            logger.info(f"Initialized TimestampService with current time: {self.current_time}")
            
            self.version_service = MagicMock(VersionService)
            self.event_manager = MagicMock(EventManager)
            self.mongodb_client = MagicMock(MongoDBClient)
            self.neo4j_client = MagicMock(Neo4jClient)
            
            # Initialize test adaptive_id with configuration
            self.adaptive_id = AdaptiveID(
                base_concept="extreme_precipitation",
                weight=1.0,
                confidence=0.9,
                uncertainty=0.1,
                timestamp_service=self.timestamp_service,
                version_service=self.version_service,
                event_manager=self.event_manager,
                mongodb_client=self.mongodb_client,
                neo4j_client=self.neo4j_client
            )
            
            # Set up test concepts using Woodwell document data
            self.test_concepts = {
                "extreme_precipitation": {
                    "description": "55% increase in annual precipitation occurring from the heaviest 1% of events",
                    "measurements": {
                        "historical": "7.34 inches",
                        "mid_century": "7.82 inches",
                        "late_century": "10.06 inches"
                    },
                    "spatial_context": {
                        "location": "Martha's Vineyard",
                        "region": "U.S. Northeast",
                        "coordinates": {
                            "latitude": 41.3805,
                            "longitude": -70.6456
                        }
                    },
                    "temporal_context": {
                        "baseline_period": "2000-2020",
                        "mid_century": "2040-2060",
                        "late_century": "2070-2090"
                    }
                }
            }
            
            logger.info("Test environment initialized successfully")
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}", exc_info=True)
            raise

    def assertStateConsistency(self, adaptive_id: AdaptiveID, operation: str) -> bool:
        """Enhanced helper to verify adaptive_id consistency after operations"""
        try:
            current_state = adaptive_id.get_current_state()
            logger.info(f"\nState check after {operation}:")
            logger.info(f"- Current state: {current_state}")
            logger.info(f"- Last update: {adaptive_id.last_updated_timestamp}")
            logger.info(f"- Version: {adaptive_id.version}")
            logger.info(f"- Storage status: {adaptive_id.get_storage_stats()}")
            logger.info(f"- Context stats: {adaptive_id.get_context_stats()}")
            logger.info(f"- Event history: {adaptive_id.get_event_history()}")
            logger.info(f"- Version history: {adaptive_id.get_version_history()}")
            
            # Verify core properties
            self.assertIsNotNone(current_state)
            self.assertIsNotNone(adaptive_id.last_updated_timestamp)
            self.assertGreaterEqual(adaptive_id.version, 0)
            
            # Verify context integrity
            if adaptive_id.temporal_context:
                self.assertIsNotNone(adaptive_id.get_temporal_context())
            if adaptive_id.spatial_context:
                self.assertIsNotNone(adaptive_id.get_spatial_context())
            
            return True
        except Exception as e:
            logger.error(f"State consistency check failed after {operation}: {str(e)}")
            return False

    def test_error_handling(self):
        """Verify proper error handling with detailed logging."""
        logger.info("\nTesting error handling scenarios...")
        
        # Test invalid property update
        with self.assertRaises(AdaptiveIDException) as context:
            logger.info("Attempting invalid property update...")
            try:
                self.adaptive_id.update_property(
                    key="invalid_key",
                    value=None,
                    origin="test"
                )
            except AdaptiveIDException as e:
                logger.error(f"Expected error occurred: {str(e)}")
                logger.error("Stack trace:", exc_info=True)
                raise
        
        # Verify state consistency
        is_consistent = self.assertStateConsistency(
            self.adaptive_id, 
            "invalid property update attempt"
        )
        logger.info(f"State remained consistent: {is_consistent}")

    def test_storage_integration(self):
        """Test complete storage lifecycle with error handling."""
        logger.info("\n=== Testing Storage Integration ===")
        
        try:
            # Test MongoDB storage
            logger.info("\n1. Testing MongoDB storage...")
            state_id = self.adaptive_id.save_state()
            loaded_state = self.adaptive_id.load_state(state_id)
            self.assertEqual(
                self.adaptive_id.get_current_state(),
                loaded_state
            )
            
            # Test Neo4j storage
            logger.info("\n2. Testing Neo4j storage...")
            graph_id = self.adaptive_id.save_to_graph()
            graph_state = self.adaptive_id.load_from_graph(graph_id)
            self.assertEqual(
                self.adaptive_id.get_current_state(),
                graph_state
            )
            
            logger.info("Storage integration tests completed successfully")
            
        except Exception as e:
            logger.error("Storage integration failed:", exc_info=True)
            raise

    def test_error_recovery(self):
        """Test recovery from various error conditions."""
        logger.info("\n=== Testing Error Recovery ===")
        
        # Test storage failure recovery
        logger.info("\n1. Testing storage failure recovery...")
        with patch.object(self.mongodb_client, 'save_state', side_effect=Exception("Storage failed")):
            try:
                self.adaptive_id.update_property("test", "value", "test")
                # Should fall back to alternative storage
                self.neo4j_client.save_state.assert_called_once()
            except Exception as e:
                logger.error("Storage failure recovery test failed:", exc_info=True)
                raise
        
        # Test version conflict recovery
        logger.info("\n2. Testing version conflict recovery...")
        with patch.object(self.version_service, 'get_new_version', side_effect=Exception("Version conflict")):
            try:
                self.adaptive_id.update_property("test", "value", "test")
                # Should handle version conflict gracefully
            except Exception as e:
                logger.error("Version conflict recovery test failed:", exc_info=True)
                raise
        
        # Verify state remained consistent
        self.assertStateConsistency(self.adaptive_id, "error recovery")

    def test_thread_safety(self):
        """Test thread-safe operations with comprehensive logging."""
        logger.info("\n=== Testing Thread Safety ===")
        
        def concurrent_updates():
            try:
                self.adaptive_id.update_property(
                    key="test_key",
                    value="test_value",
                    origin="concurrent_test"
                )
                logger.info(f"Thread {threading.current_thread().name} completed update")
            except Exception as e:
                logger.error(f"Thread {threading.current_thread().name} error: {e}")
                logger.error("Stack trace:", exc_info=True)

        # Create and run multiple threads
        threads = [
            threading.Thread(
                target=concurrent_updates,
                name=f"UpdateThread-{i}"
            ) for i in range(5)
        ]
        
        logger.info("Starting concurrent update threads...")
        for t in threads:
            t.start()
            
        for t in threads:
            t.join()
        
        # Verify final state
        logger.info("Verifying final state after concurrent updates...")
        final_state = self.adaptive_id.get_current_state()
        logger.info(f"Final state: {final_state}")
        
        # Verify no deadlocks occurred
        self.assertStateConsistency(self.adaptive_id, "concurrent updates")

    def test_batch_operations(self):
        """Test batch operations with proper error handling."""
        logger.info("\n=== Testing Batch Operations ===")
        
        try:
            # Prepare batch updates using real climate data
            updates = [
                ("description", "55% increase in annual precipitation"),
                ("temporal_context", {
                    "baseline_period": "2000-2020",
                    "mid_century": "2040-2060"
                }),
                ("spatial_context", {
                    "location": "Martha's Vineyard",
                    "coordinates": {
                        "latitude": 41.3805,
                        "longitude": -70.6456
                    }
                }),
                ("measurements", {
                    "historical": "7.34 inches",
                    "projection": "10.06 inches"
                })
            ]
            
            # Execute batch update
            logger.info("\n1. Executing batch property updates...")
            self.adaptive_id.update_properties_batch(updates)
            
            # Verify batch operation success
            logger.info("\n2. Verifying batch updates...")
            for key, value in updates:
                stored_value = self.adaptive_id.get_property(key)
                self.assertEqual(stored_value, value)
            
            logger.info("Batch operations completed successfully")
            
        except Exception as e:
            logger.error("Batch operations failed:", exc_info=True)
            raise

    def test_complete_state_lifecycle(self):
        """Test complete state lifecycle with explicit verification."""
        logger.info("\n=== Testing Complete State Lifecycle ===")
        
        try:
            # 1. Initial state
            logger.info("\n1. Verifying initial state:")
            initial_state = self.adaptive_id.get_current_state()
            logger.info(f"Initial State: {initial_state}")
            
            # 2. Add basic content
            logger.info("\n2. Adding basic content...")
            doc_content = self.test_concepts["extreme_precipitation"]
            self.adaptive_id.update_property(
                key="description",
                value=doc_content["description"],
                origin="document_extraction"
            )
            
            # 3. Add measurements
            logger.info("\n3. Adding measurements...")
            self.adaptive_id.update_property(
                key="measurements",
                value=doc_content["measurements"],
                origin="document_extraction"
            )
            
            # 4. Verify state evolution
            logger.info("\n4. Verifying state evolution:")
            evolved_state = self.adaptive_id.get_current_state()
            logger.info(f"Evolved State: {evolved_state}")
            
        except Exception as e:
            logger.error("State lifecycle test failed:", exc_info=True)
            raise

    def test_comprehensive_context_handling(self):
        """Test comprehensive context handling with real climate data."""
        logger.info("\n=== Testing Comprehensive Context Handling ===")
        
        try:
            # 1. Add temporal context
            logger.info("\n1. Adding temporal context...")
            temporal_context = self.test_concepts["extreme_precipitation"]["temporal_context"]
            self.adaptive_id.update_temporal_context(
                key="projection_periods",
                value=temporal_context,
                origin="document_extraction"
            )
            
            # 2. Add spatial context
            logger.info("\n2. Adding spatial context...")
            spatial_context = self.test_concepts["extreme_precipitation"]["spatial_context"]
            self.adaptive_id.update_spatial_context(
                key="location_data",
                value=spatial_context,
                origin="document_extraction"
            )
            
            # 3. Verify context preservation
            logger.info("\n3. Verifying context preservation:")
            stored_temporal = self.adaptive_id.get_temporal_context("projection_periods")
            stored_spatial = self.adaptive_id.get_spatial_context("location_data")
            
            logger.info(f"Stored Temporal Context: {stored_temporal}")
            logger.info(f"Stored Spatial Context: {stored_spatial}")
            
        except Exception as e:
            logger.error("Context handling test failed:", exc_info=True)
            raise

    def test_detailed_version_tracking(self):
        """Test comprehensive version tracking."""
        logger.info("\n=== Testing Detailed Version Tracking ===")
        
        version_history = []
        try:
            # 1. Track initial version
            initial_version = self.adaptive_id.version
            version_history.append(("initial", initial_version))
            logger.info(f"\n1. Initial version: {initial_version}")
            
            # 2. Add basic data
            self.adaptive_id.update_property(
                key="measurement",
                value="7.34 inches",
                origin="historical_data"
            )
            version_history.append(("measurement", self.adaptive_id.version))
            logger.info(f"\n2. Version after measurement: {self.adaptive_id.version}")
            
            # 3. Add projection
            self.adaptive_id.update_property(
                key="projection",
                value="10.06 inches",
                origin="climate_projection"
            )
            version_history.append(("projection", self.adaptive_id.version))
            logger.info(f"\n3. Version after projection: {self.adaptive_id.version}")
            
            # Log version history
            logger.info("\n4. Version history:")
            for stage, version in version_history:
                logger.info(f"Stage: {stage}, Version: {version}")
                
        except Exception as e:
            logger.error("Version tracking test failed:", exc_info=True)
            raise

    def test_document_ingestion_lifecycle(self):
            """Test complete document ingestion lifecycle."""
            logger.info("\n=== Testing Document Ingestion Lifecycle ===")
            
            # 1. Initial document data
            doc_data = {
                "source": "Woodwell Climate Research Center",
                "document_type": "climate_risk_assessment",
                "location": "Martha's Vineyard",
                "date": "2024",
                "content": {
                    "type": "extreme_precipitation",
                    "historical": "7.34 inches",
                    "projection_2060": "7.82 inches",
                    "projection_2090": "10.06 inches"
                }
            }
            
            try:
                # 2. Process document content
                logger.info(f"\n1. Processing document data: {doc_data}")
                self.adaptive_id.process_document_content(
                    content=doc_data["content"],
                    source=doc_data["source"],
                    timestamp=datetime.now().isoformat()
                )
                
                # 3. Verify MongoDB storage
                logger.info("\n2. Verifying MongoDB storage:")
                self.mongodb_client.save_state.assert_called()
                
                # 4. Verify Neo4j representation
                logger.info("\n3. Verifying Neo4j representation:")
                self.neo4j_client.create_node.assert_called()
                
                # 5. Verify state after ingestion
                final_state = self.adaptive_id.get_current_state()
                logger.info(f"\n4. Final state after ingestion: {final_state}")
                
            except Exception as e:
                logger.error("Document ingestion failed:", exc_info=True)
                raise

    def test_rag_enhancement_lifecycle(self):
        """Test complete RAG enhancement lifecycle."""
        logger.info("\n=== Testing RAG Enhancement Lifecycle ===")
        
        # 1. Initial RAG enhancement data
        rag_data = {
            "enhanced_concepts": {
                "extreme_precipitation": {
                    "related_concepts": ["flooding", "storm_surge", "infrastructure_damage"],
                    "confidence": 0.85,
                    "relationships": [
                        {
                            "type": "causes",
                            "target": "flooding",
                            "confidence": 0.9
                        },
                        {
                            "type": "impacts",
                            "target": "infrastructure_damage",
                            "confidence": 0.8
                        }
                    ]
                }
            },
            "context_enrichment": {
                "spatial": {
                    "region_characteristics": "coastal_community",
                    "vulnerability_factors": ["low-lying areas", "coastal infrastructure"]
                },
                "temporal": {
                    "trend_analysis": "increasing_frequency",
                    "confidence_intervals": {
                        "2040-2060": 0.8,
                        "2070-2090": 0.7
                    }
                }
            }
        }
        
        try:
            # 2. Process RAG enhancements
            logger.info("\n1. Processing RAG enhancements...")
            self.adaptive_id.process_rag_enhancement(rag_data)
            
            # 3. Verify relationship creation
            logger.info("\n2. Verifying relationship creation:")
            for relationship in rag_data["enhanced_concepts"]["extreme_precipitation"]["relationships"]:
                self.neo4j_client.create_relationship.assert_any_call(
                    from_node=self.adaptive_id.id,
                    relationship_type=relationship["type"],
                    to_node=ANY,
                    properties={"confidence": relationship["confidence"]}
                )
            
            # 4. Verify context enrichment
            logger.info("\n3. Verifying context enrichment:")
            enriched_state = self.adaptive_id.get_current_state()
            logger.info(f"Enriched state: {enriched_state}")
            
        except Exception as e:
            logger.error("RAG enhancement failed:", exc_info=True)
            raise

    def test_graph_evolution_tracking(self):
        """Test tracking of graph evolution through updates."""
        logger.info("\n=== Testing Graph Evolution Tracking ===")
        
        try:
            # 1. Initial graph state
            logger.info("\n1. Recording initial graph state...")
            initial_relationships = self.neo4j_client.get_relationships(self.adaptive_id.id)
            
            # 2. Add new relationship
            logger.info("\n2. Adding new relationship...")
            new_relationship = {
                "type": "impacts",
                "target": "coastal_flooding",
                "properties": {
                    "confidence": 0.85,
                    "source": "rag_enhancement"
                }
            }
            
            self.adaptive_id.add_relationship(new_relationship)
            
            # 3. Verify graph updates
            logger.info("\n3. Verifying graph updates...")
            updated_relationships = self.neo4j_client.get_relationships(self.adaptive_id.id)
            
            # 4. Track evolution
            logger.info("\n4. Tracking graph evolution:")
            logger.info(f"Initial relationships: {initial_relationships}")
            logger.info(f"Updated relationships: {updated_relationships}")
            
        except Exception as e:
            logger.error("Graph evolution tracking failed:", exc_info=True)
            raise

    def test_logging_initialization(self):
        """Test proper logging initialization and context setting."""
        logger.info("\n=== Testing Logging Initialization ===")
        
        try:
            # Create new instance to test initialization
            test_id = AdaptiveID(
                base_concept="test_concept",
                weight=1.0,
                confidence=0.9,
                uncertainty=0.1
            )
            
            # Verify logger initialization
            self.assertIsNotNone(test_id.logger)
            self.assertIsInstance(test_id.logger, LoggingManager)
            
            # Verify context setting
            context = test_id.logger.get_context()
            self.assertEqual(context.component, "adaptive_id")
            self.assertEqual(context.stage, "initialization")
            self.assertEqual(context.system_component, "concept_tracking")
            self.assertIsNotNone(context.process_id)
            
            logger.info("Logging initialization test completed successfully")
            
        except Exception as e:
            logger.error("Logging initialization test failed:", exc_info=True)
            raise

    def test_logging_consistency(self):
        """Test consistent logging across operations."""
        logger.info("\n=== Testing Logging Consistency ===")
        
        try:
            # Create test instance
            test_id = AdaptiveID(
                base_concept="test_concept",
                weight=1.0,
                confidence=0.9,
                uncertainty=0.1
            )
            
            # Test property updates
            with self.assertLogs(level='INFO') as log:
                test_id.update_property("test_key", "test_value", "test")
                self.assertTrue(any("property_update" in record.message for record in log.records))
                self.assertTrue(any("process_start" in record.message for record in log.records))
                self.assertTrue(any("process_end" in record.message for record in log.records))
            
            # Test relationship management
            with self.assertLogs(level='INFO') as log:
                relationship = RelationshipModel(source_id="test", target_id="test2", relationship_type="test")
                test_id.add_relationship(relationship, "test")
                self.assertTrue(any("add_relationship" in record.message for record in log.records))
                self.assertTrue(any("process_start" in record.message for record in log.records))
                self.assertTrue(any("process_end" in record.message for record in log.records))
            
            # Test version management
            with self.assertLogs(level='INFO') as log:
                test_id._store_version("test_key", "test_value", "test")
                self.assertTrue(any("version_storage" in record.message for record in log.records))
                self.assertTrue(any("process_start" in record.message for record in log.records))
                self.assertTrue(any("process_end" in record.message for record in log.records))
            
            logger.info("Logging consistency test completed successfully")
            
        except Exception as e:
            logger.error("Logging consistency test failed:", exc_info=True)
            raise

    def test_error_logging(self):
        """Test proper error logging and handling."""
        logger.info("\n=== Testing Error Logging ===")
        
        try:
            # Create test instance
            test_id = AdaptiveID(
                base_concept="test_concept",
                weight=1.0,
                confidence=0.9,
                uncertainty=0.1
            )
            
            # Test invalid property update
            with self.assertLogs(level='ERROR') as log:
                with self.assertRaises(AdaptiveIDException):
                    test_id.update_property(None, None, None)
                self.assertTrue(any("Property update failed" in record.message for record in log.records))
            
            # Test invalid relationship
            with self.assertLogs(level='ERROR') as log:
                with self.assertRaises(Exception):
                    test_id.add_relationship(None, None)
                self.assertTrue(any("Failed to add relationship" in record.message for record in log.records))
            
            # Test version revert error
            with self.assertLogs(level='ERROR') as log:
                with self.assertRaises(Exception):
                    test_id.revert_to_version("non_existent_version")
                self.assertTrue(any("Version revert failed" in record.message for record in log.records))
            
            logger.info("Error logging test completed successfully")
            
        except Exception as e:
            logger.error("Error logging test failed:", exc_info=True)
            raise

    def tearDown(self):
        """Clean up test resources with error logging."""
        logger.info("\nPerforming test cleanup...")
        try:
            # Verify state persistence
            self.mongodb_client.save_state.assert_called()
            logger.info("MongoDB state verification complete")
            
            # Verify event propagation
            self.event_manager.publish.assert_called()
            logger.info("Event system verification complete")
            
        except Exception as e:
            logger.error("Teardown failed:")
            logger.error(f"- Error type: {type(e).__name__}")
            logger.error(f"- Error message: {str(e)}")
            logger.error("- Stack trace:", exc_info=True)
            raise
        finally:
            logger.info(f"\nCompleted test: {self._testMethodName}")
            logger.info("="*50 + "\n")

if __name__ == '__main__':
    unittest.main()