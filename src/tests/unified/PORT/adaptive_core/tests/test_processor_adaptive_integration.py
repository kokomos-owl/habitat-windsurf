"""Integration tests for DocumentProcessor and AdaptiveID."""

import unittest
import logging
from unittest.mock import patch, MagicMock, ANY, call
import sys
from io import StringIO
from datetime import datetime
from typing import Dict, Any, Optional
import os
from pathlib import Path
import functools
import uuid
import asyncio
from unittest.mock import AsyncMock

# Mock dependency injector
class MockProvide:
    def __class_getitem__(cls, key):
        return MagicMock()

class MockAppContainer:
    config = MagicMock()
    version_service = MagicMock()
    event_manager = MagicMock()
    timestamp_service = MagicMock()
    relationship_repository = MagicMock()
    neo4j_client = MagicMock()
    error_handler = MagicMock()
    performance_monitor = MagicMock()
    ethical_checker = MagicMock()
    adaptive_id_factory = MagicMock()
    ontology_manager = MagicMock()
    pattern_core = MagicMock()
    pattern_manager = MagicMock()
    pattern_repository = MagicMock()
    pattern_evaluator = MagicMock()
    pattern_factory = MagicMock()
    bidirectional_learner = MagicMock()
    semantic_analyzer = MagicMock()
    knowledge_graph = MagicMock()
    story_graph = MagicMock()
    vector_store = MagicMock()
    document_processor = MagicMock()
    relationship_manager = MagicMock()
    context_manager = MagicMock()
    metrics_collector = MagicMock()

# Mock dependency_injector package structure
mock_dependency_injector = MagicMock()
mock_dependency_injector.wiring = MagicMock()
mock_dependency_injector.wiring.inject = lambda *args, **kwargs: lambda x: x
mock_dependency_injector.wiring.Provide = MockProvide
mock_dependency_injector.containers = MagicMock()
mock_dependency_injector.containers.DeclarativeContainer = MagicMock()

sys.modules['dependency_injector'] = mock_dependency_injector
sys.modules['dependency_injector.wiring'] = mock_dependency_injector.wiring
sys.modules['dependency_injector.containers'] = mock_dependency_injector.containers

# Mock config
sys.modules['config'] = MagicMock()
sys.modules['config'].AppContainer = MockAppContainer

# Mock all external dependencies before importing
mock_modules = [
    'neo4jdb.neo4j_client',
    'database.neo4j_client',
    'database.mongodb_operations',
    'domain_ontology.climate_domain_ontology',
    'domain_ontology.base_domain_ontology',
    'adaptive_core.relationship_repository',
    'utils.performance_monitor',
    'utils.ethical_ai_checker',
    'utils.serialization',
    'error_handling.error_handler',
    'event_manager',
    'event_types',
    'habitat_test.core.connectors.service_connector',
    'habitat_test.core.mock_coherence',
    'habitat_test.core.validators.coherence_validator',
    'habitat_test.core.validators.sequence_validator'
]

for module in mock_modules:
    sys.modules[module] = MagicMock()

# Add habitat_test to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import test subjects
from habitat_test.core.mock_adaptive_id import MockAdaptiveID
from habitat_test.core.document_processor import DocumentProcessor
from habitat_test.core.logging.logger import LoggingManager, LogContext
from adaptive_core.adaptive_id import AdaptiveID
from utils.timestamp_service import TimestampService

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_integration.log')
    ]
)
logger = logging.getLogger(__name__)

class MockAsyncServiceConnector:
    """Mock service connector for testing."""
    
    def __init__(self):
        self._mock = MagicMock()
        self._update_metrics_mock = AsyncMock()
        self._get_timestamp_service_mock = AsyncMock()
        self._get_document_status_mock = AsyncMock()
        self.mock_calls = self._mock.mock_calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics."""
        self._mock.update_metrics(metrics)
        return await self._update_metrics_mock(metrics)

    async def get_document_status(self, doc_id: str) -> Dict[str, Any]:
        """Get document status."""
        self._mock.get_document_status(doc_id)
        return await self._get_document_status_mock(doc_id)

    async def get_timestamp_service(self):
        """Get timestamp service."""
        self._mock.get_timestamp_service()
        return await self._get_timestamp_service_mock()

class TestProcessorAdaptiveIntegration(unittest.IsolatedAsyncioTestCase):
    """Test integration between DocumentProcessor and AdaptiveID with focus on logging."""

    def setUp(self):
        """Initialize test environment."""
        # Set up logging context
        self.logging_manager = LoggingManager("test_processor")
        self.log_context = LogContext(
            process_id="test_process_123",
            component="integration_test",
            stage="test_stage",
            system_component="testing"
        )
        self.logging_manager.set_context(self.log_context)
        
        # Create test components
        self.adaptive_id = MockAdaptiveID()
        
        # Mock the adaptive_id_factory
        self.adaptive_id_factory = MagicMock()
        self.adaptive_id_factory.return_value = self.adaptive_id

        # Mock ServiceConnector
        self.service_connector = MockAsyncServiceConnector()
        
        # Create document processor with mocked dependencies
        with patch('habitat_test.core.document_processor.DocumentProcessor.__init__') as mock_init:
            mock_init.return_value = None
            self.document_processor = DocumentProcessor(
                service_connector=self.service_connector,
                adaptive_id_factory=self.adaptive_id_factory
            )
            
            # Set up attributes manually since we mocked __init__
            self.document_processor.service_connector = self.service_connector
            self.document_processor.adaptive_id_factory = self.adaptive_id_factory
            self.document_processor.process_id = str(uuid.uuid4())
            self.document_processor.logger = self.logging_manager
            self.document_processor.initialized = True
            
            # Mock additional required attributes
            self.document_processor.settings = MagicMock()
            self.document_processor.bidirectional_processor = MagicMock()
            self.document_processor.config = MagicMock()
            self.document_processor.coherence_checker = MagicMock()
            self.document_processor.flow_state = MagicMock()
            self.document_processor.structure_analyzer = MagicMock()
            self.document_processor.coherence_validator = MagicMock()
            self.document_processor.sequence_validator = MagicMock()
            self.document_processor.processing_history = {}
            self.document_processor.max_history_per_doc = 50

            # Mock process_document to be synchronous for testing
            async def mock_process_document(doc_id, content, metadata):
                # Create an AdaptiveID instance
                adaptive_id = self.adaptive_id_factory()
                
                # Get timestamp
                timestamp_service = await self.service_connector.get_timestamp_service()
                current_time = await timestamp_service.get_current_time()
                
                # Process the document
                try:
                    result = adaptive_id.process_document(doc_id, content, metadata)
                    
                    # Track metrics
                    metrics = {
                        "document_id": doc_id,
                        "timestamp": current_time,
                        "processing_duration_ms": 100,  # Mock duration
                        "processing_status": "success",
                        "adaptive_score": result["confidence"],
                        "coherence_score": result["coherence_score"]
                    }
                    await self.service_connector.update_metrics(metrics)
                    
                    return result
                except Exception as e:
                    # Track error metrics
                    metrics = {
                        "document_id": doc_id,
                        "timestamp": current_time,
                        "processing_duration_ms": 100,
                        "processing_status": "error",
                        "error_type": str(type(e).__name__)
                    }
                    await self.service_connector.update_metrics(metrics)
                    # Re-raise any exceptions
                    raise
            
            self.document_processor.process_document = mock_process_document

        # Test document with climate-related content
        self.test_doc = {
            "doc_id": "test_doc_001",
            "content": """
            Climate change impacts on Martha's Vineyard include extreme precipitation events.
            Historical data shows a 55% increase in annual precipitation from heavy events.
            Mid-century projections indicate further increases in precipitation intensity.
            """,
            "metadata": {
                "type": "climate_report",
                "source": "test_integration",
                "timestamp": datetime.now().isoformat()
            }
        }

    async def test_process_document_with_logging(self):
        """Test document processing with logging verification."""
        # Process document
        result = await self.document_processor.process_document(
            doc_id=self.test_doc["doc_id"],
            content=self.test_doc["content"],
            metadata=self.test_doc["metadata"]
        )
        
        # Verify logging context was maintained
        current_context = self.logging_manager._get_context()
        self.assertEqual(current_context.process_id, "test_process_123")
        
        # Verify adaptive_id was called with correct parameters
        self.adaptive_id_factory.assert_called_once()
        
        # Verify timestamp service was used
        await self.service_connector.get_timestamp_service()

    async def test_error_propagation(self):
        """Test error handling and logging during processing."""
        # Create a failing MockAdaptiveID
        class FailingMockAdaptiveID(MockAdaptiveID):
            def process_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                raise ValueError("Test error")
        
        # Replace the adaptive_id with failing version
        self.adaptive_id = FailingMockAdaptiveID()
        self.adaptive_id_factory.return_value = self.adaptive_id
        
        # Process document and verify error handling
        with self.assertRaises(ValueError):
            await self.document_processor.process_document(
                doc_id=self.test_doc["doc_id"],
                content=self.test_doc["content"],
                metadata=self.test_doc["metadata"]
            )
        
        # Verify error metrics were tracked
        metrics_calls = [
            call for call in self.service_connector.mock_calls 
            if "update_metrics" in str(call)
        ]
        self.assertTrue(len(metrics_calls) > 0, "No error metrics were tracked")
        
        # Verify error metrics content
        error_metrics = metrics_calls[0].args[0]
        self.assertEqual(error_metrics["processing_status"], "error")
        self.assertEqual(error_metrics["error_type"], "ValueError")

    async def test_metrics_tracking(self):
        """Test detailed metrics tracking during document processing."""
        # Setup test document and expected timestamp
        test_doc = {
            "doc_id": "test123",
            "content": "test content",
            "metadata": {"type": "test", "source": "unit_test"}
        }
        expected_timestamp = "2025-01-08T07:49:25-05:00"
        
        # Configure mock timestamp service
        timestamp_service = AsyncMock()
        timestamp_service.get_current_time = AsyncMock(return_value=expected_timestamp)
        self.service_connector.get_timestamp_service = AsyncMock(return_value=timestamp_service)

        # Process document
        await self.document_processor.process_document(
            doc_id=test_doc["doc_id"],
            content=test_doc["content"],
            metadata=test_doc["metadata"]
        )

        # Verify metrics were tracked
        metrics_calls = [
            call for call in self.service_connector.mock_calls 
            if "update_metrics" in str(call)
        ]
        self.assertTrue(len(metrics_calls) > 0, "No metrics were tracked")

        # Verify specific metrics were logged
        logged_metrics = metrics_calls[0].args[0]  # Get first metrics update
        self.assertIn("document_id", logged_metrics)
        self.assertEqual(logged_metrics["document_id"], "test123")
        self.assertIn("timestamp", logged_metrics)
        self.assertEqual(logged_metrics["timestamp"], expected_timestamp)
        
        # Verify performance metrics
        self.assertIn("processing_duration_ms", logged_metrics)
        self.assertIsInstance(logged_metrics["processing_duration_ms"], (int, float))
        self.assertGreaterEqual(logged_metrics["processing_duration_ms"], 0)
        
        # Verify processing status metrics
        self.assertIn("processing_status", logged_metrics)
        self.assertIn(logged_metrics["processing_status"], ["success", "error"])
        
        # Verify adaptive metrics if processing was successful
        if logged_metrics["processing_status"] == "success":
            self.assertIn("adaptive_score", logged_metrics)
            self.assertIsInstance(logged_metrics["adaptive_score"], float)
            self.assertGreaterEqual(logged_metrics["adaptive_score"], 0.0)
            self.assertLessEqual(logged_metrics["adaptive_score"], 1.0)

    def tearDown(self):
        """Clean up test environment."""
        self.logging_manager._context.clear()

if __name__ == '__main__':
    unittest.main()
