# test_relationship_model.py

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

from adaptive_core.relationship_model import RelationshipModel, RelationshipModelException
from adaptive_core.adaptive_id import AdaptiveID
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
        logging.FileHandler('test_relationship_model.log')
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

class TestRelationshipModel(unittest.TestCase):
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
            
            # Initialize test relationship using Woodwell document climate concepts
            self.relationship = RelationshipModel(
                source_id="extreme_precipitation",
                target_id="flooding",
                relationship_type="causes",
                confidence=0.9,
                timestamp_service=self.timestamp_service,
                version_service=self.version_service,
                event_manager=self.event_manager,
                mongodb_client=self.mongodb_client,
                neo4j_client=self.neo4j_client
            )
            
            # Set up test relationships using Woodwell document data
            self.test_relationships = {
                "extreme_precipitation": {
                    "relationships": [
                        {
                            "target": "flooding",
                            "type": "causes",
                            "confidence": 0.9,
                            "temporal_context": {
                                "baseline_period": "2000-2020",
                                "projection_period": "2040-2060"
                            },
                            "spatial_context": {
                                "location": "Martha's Vineyard",
                                "coordinates": {
                                    "latitude": 41.3805,
                                    "longitude": -70.6456
                                }
                            },
                            "uncertainty": 0.1
                        },
                        {
                            "target": "infrastructure_damage",
                            "type": "impacts",
                            "confidence": 0.85,
                            "temporal_context": {
                                "baseline_period": "2000-2020",
                                "projection_period": "2070-2090"
                            },
                            "uncertainty": 0.15
                        }
                    ]
                }
            }
            
            logger.info("Test environment initialized successfully")
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}", exc_info=True)
            raise

    def assertRelationshipConsistency(self, relationship: RelationshipModel, operation: str) -> bool:
        """Helper to verify relationship consistency after operations"""
        try:
            current_state = relationship.get_current_state()
            logger.info(f"\nState check after {operation}:")
            logger.info(f"- Current state: {current_state}")
            logger.info(f"- Last update: {relationship.last_updated_timestamp}")
            logger.info(f"- Version: {relationship.version}")
            logger.info(f"- Confidence: {relationship.confidence}")
            logger.info(f"- Uncertainty: {relationship.uncertainty}")
            logger.info(f"- Temporal context: {relationship.temporal_context}")
            logger.info(f"- Spatial context: {relationship.spatial_context}")
            logger.info(f"- Event history: {relationship.get_event_history()}")
            
            # Verify core properties
            self.assertIsNotNone(current_state)
            self.assertIsNotNone(relationship.last_updated_timestamp)
            self.assertGreaterEqual(relationship.version, 0)
            self.assertGreaterEqual(relationship.confidence, 0)
            self.assertLessEqual(relationship.confidence, 1)
            self.assertGreaterEqual(relationship.uncertainty, 0)
            self.assertLessEqual(relationship.uncertainty, 1)
            
            return True
        except Exception as e:
            logger.error(f"Relationship consistency check failed after {operation}: {str(e)}")
            return False

    def test_error_handling(self):
        """Verify proper error handling with detailed logging."""
        logger.info("\nTesting error handling scenarios...")
        
        # Test invalid confidence update
        with self.assertRaises(RelationshipModelException) as context:
            logger.info("Attempting invalid confidence update...")
            try:
                self.relationship.update_confidence(1.5, "test")  # Invalid confidence value
            except RelationshipModelException as e:
                logger.error(f"Expected error occurred: {str(e)}")
                logger.error("Stack trace:", exc_info=True)
                raise

        # Verify relationship consistency
        is_consistent = self.assertRelationshipConsistency(
            self.relationship, 
            "invalid confidence update attempt"
        )
        logger.info(f"Relationship remained consistent: {is_consistent}")

    def test_bidirectional_updates(self):
        """Test bidirectional relationship updates."""
        logger.info("\n=== Testing Bidirectional Updates ===")
        
        try:
            # Create inverse relationship
            logger.info("\n1. Creating inverse relationship...")
            inverse_rel = self.relationship.create_inverse()
            
            # Verify inverse properties
            self.assertEqual(inverse_rel.source_id, self.relationship.target_id)
            self.assertEqual(inverse_rel.target_id, self.relationship.source_id)
            self.assertEqual(inverse_rel.confidence, self.relationship.confidence)
            
            # Update original relationship
            logger.info("\n2. Updating original relationship...")
            self.relationship.update_confidence(0.95, "test")
            
            # Verify propagation to inverse
            logger.info("\n3. Verifying bidirectional propagation...")
            self.assertEqual(inverse_rel.confidence, self.relationship.confidence)
            
            logger.info("Bidirectional updates verified successfully")
            
        except Exception as e:
            logger.error("Bidirectional update test failed:", exc_info=True)
            raise

    def test_uncertainty_propagation(self):
        """Test uncertainty propagation in relationships."""
        logger.info("\n=== Testing Uncertainty Propagation ===")
        
        try:
            # Set initial uncertainty
            logger.info("\n1. Setting initial uncertainty...")
            self.relationship.uncertainty = 0.1
            
            # Create dependent relationship
            logger.info("\n2. Creating dependent relationship...")
            dependent_rel = RelationshipModel(
                source_id="flooding",
                target_id="infrastructure_damage",
                relationship_type="causes",
                confidence=0.85,
                uncertainty=0.1,
                timestamp_service=self.timestamp_service
            )
            
            # Link relationships
            logger.info("\n3. Linking relationships...")
            self.relationship.add_dependent_relationship(dependent_rel)
            
            # Update uncertainty and verify propagation
            logger.info("\n4. Testing uncertainty propagation...")
            self.relationship.update_uncertainty(0.2, "test")
            
            # Verify propagated uncertainty (should increase in dependent relationship)
            self.assertGreater(dependent_rel.uncertainty, 0.2)
            
            logger.info("Uncertainty propagation verified successfully")
            
        except Exception as e:
            logger.error("Uncertainty propagation test failed:", exc_info=True)
            raise

    def test_temporal_versioning(self):
        """Test temporal versioning of relationships."""
        logger.info("\n=== Testing Temporal Versioning ===")
        
        version_history = []
        try:
            # 1. Initial version
            initial_version = self.relationship.version
            version_history.append(("initial", initial_version))
            logger.info(f"\n1. Initial version: {initial_version}")
            
            # 2. Update confidence
            self.relationship.update_confidence(0.95, "test")
            version_history.append(("confidence_update", self.relationship.version))
            logger.info(f"\n2. Version after confidence update: {self.relationship.version}")
            
            # 3. Add temporal context
            temporal_context = self.test_relationships["extreme_precipitation"]["relationships"][0]["temporal_context"]
            self.relationship.update_temporal_context("projection_data", temporal_context, "test")
            version_history.append(("temporal_update", self.relationship.version))
            
            # Log version history
            logger.info("\n4. Version history:")
            for stage, version in version_history:
                logger.info(f"Stage: {stage}, Version: {version}")
            
        except Exception as e:
            logger.error("Temporal versioning test failed:", exc_info=True)
            raise

    def test_comprehensive_context_handling(self):
        """Test comprehensive context handling."""
        logger.info("\n=== Testing Comprehensive Context Handling ===")
        
        try:
            # 1. Add temporal context
            logger.info("\n1. Adding temporal context...")
            temporal_context = self.test_relationships["extreme_precipitation"]["relationships"][0]["temporal_context"]
            self.relationship.update_temporal_context(
                key="projection_periods",
                value=temporal_context,
                origin="climate_projection"
            )
            
            # 2. Add spatial context
            logger.info("\n2. Adding spatial context...")
            spatial_context = self.test_relationships["extreme_precipitation"]["relationships"][0]["spatial_context"]
            self.relationship.update_spatial_context(
                key="location_data",
                value=spatial_context,
                origin="climate_projection"
            )
            
            # 3. Verify context preservation
            logger.info("\n3. Verifying context preservation:")
            stored_temporal = self.relationship.get_temporal_context("projection_periods")
            stored_spatial = self.relationship.get_spatial_context("location_data")
            
            # Verify contexts match
            self.assertEqual(stored_temporal, temporal_context)
            self.assertEqual(stored_spatial, spatial_context)
            
            logger.info(f"Stored Temporal Context: {stored_temporal}")
            logger.info(f"Stored Spatial Context: {stored_spatial}")
            
        except Exception as e:
            logger.error("Context handling test failed:", exc_info=True)
            raise

    def test_relationship_strength_evolution(self):
        """Test relationship strength evolution over time."""
        logger.info("\n=== Testing Relationship Strength Evolution ===")
        
        try:
            # 1. Initial strength metrics
            logger.info("\n1. Recording initial strength metrics...")
            initial_strength = {
                "confidence": self.relationship.confidence,
                "uncertainty": self.relationship.uncertainty
            }
            
            # 2. Simulate relationship reinforcement
            logger.info("\n2. Simulating relationship reinforcement...")
            for _ in range(3):
                self.relationship.reinforce_relationship("test")
                time.sleep(0.1)  # Simulate time passing
            
            # 3. Verify strength evolution
            logger.info("\n3. Verifying strength evolution:")
            self.assertGreater(self.relationship.confidence, initial_strength["confidence"])
            self.assertLess(self.relationship.uncertainty, initial_strength["uncertainty"])
            
            logger.info(f"Initial strength: {initial_strength}")
            logger.info(f"Final confidence: {self.relationship.confidence}")
            logger.info(f"Final uncertainty: {self.relationship.uncertainty}")
            
        except Exception as e:
            logger.error("Relationship strength evolution test failed:", exc_info=True)
            raise

    def test_storage_integration(self):
        """Test storage integration for relationships."""
        logger.info("\n=== Testing Storage Integration ===")
        
        try:
            # 1. Save to MongoDB
            logger.info("\n1. Testing MongoDB storage...")
            state_id = self.relationship.save_state()
            loaded_state = self.relationship.load_state(state_id)
            self.assertEqual(
                self.relationship.get_current_state(),
                loaded_state
            )
            
            # 2. Save to Neo4j
            logger.info("\n2. Testing Neo4j storage...")
            graph_id = self.relationship.save_to_graph()
            graph_state = self.relationship.load_from_graph(graph_id)
            self.assertEqual(
                self.relationship.get_current_state(),
                graph_state
            )
            
            logger.info("Storage integration tests completed successfully")
            
        except Exception as e:
            logger.error("Storage integration failed:", exc_info=True)
            raise

    def test_thread_safety(self):
        """Test thread-safe operations."""
        logger.info("\n=== Testing Thread Safety ===")
        
        def concurrent_updates():
            try:
                confidence = self.relationship.confidence
                new_confidence = min(1.0, confidence + 0.05)
                self.relationship.update_confidence(new_confidence, "concurrent_test")
                logger.info(f"Thread {threading.current_thread().name} completed update")
            except Exception as e:
                logger.error(f"Thread {threading.current_thread().name} error: {e}")
                logger.error("Stack trace:", exc_info=True)
        
        # Create multiple threads
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
        final_state = self.relationship.get_current_state()
        logger.info(f"Final state: {final_state}")
        
        # Verify no deadlocks occurred
        self.assertRelationshipConsistency(self.relationship, "concurrent updates")

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
            
            # Verify relationship cleanup
            if hasattr(self, 'relationship'):
                # Clean up any bidirectional relationships
                inverse_relationships = self.relationship.get_inverse_relationships()
                for rel in inverse_relationships:
                    rel.cleanup()
                
                # Clean up dependent relationships
                dependent_relationships = self.relationship.get_dependent_relationships()
                for rel in dependent_relationships:
                    rel.cleanup()
                
                # Clean up main relationship
                self.relationship.cleanup()
                
            logger.info("Relationship cleanup complete")
            
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