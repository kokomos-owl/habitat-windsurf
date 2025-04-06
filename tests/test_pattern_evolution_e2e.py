"""
End-to-end test for pattern evolution with AdaptiveID integration.

This test processes a real document through the Habitat pipeline,
allowing the system to extract patterns, track their usage and evolution,
and demonstrate the AdaptiveID integration in a real-world scenario.
"""

import os
import pytest
from typing import Dict, List, Any, Optional

from src.habitat_evolution.infrastructure.services.pattern_evolution_service import PatternEvolutionService
from src.habitat_evolution.climate_risk.document_processing_service import DocumentProcessingService
from src.habitat_evolution.infrastructure.services.event_service import EventService
from src.habitat_evolution.infrastructure.services.bidirectional_flow_service import BidirectionalFlowService
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.infrastructure.di.container import DIContainer


@pytest.fixture
def di_container():
    """Create and configure the DI container for testing."""
    container = DIContainer()
    
    # Import necessary interfaces
    from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
    from src.habitat_evolution.infrastructure.interfaces.services.bidirectional_flow_interface import BidirectionalFlowInterface
    from src.habitat_evolution.infrastructure.interfaces.services.pattern_evolution_service_interface import PatternEvolutionServiceInterface
    from src.habitat_evolution.infrastructure.interfaces.persistence.arangodb_connection_interface import ArangoDBConnectionInterface
    from src.habitat_evolution.infrastructure.interfaces.services.pattern_aware_rag_interface import PatternAwareRAGInterface
    from src.habitat_evolution.infrastructure.interfaces.services.vector_tonic_service_interface import VectorTonicServiceInterface
    
    # Create a properly configured ArangoDBConnection with Docker credentials
    arangodb_connection = ArangoDBConnection(
        host="localhost",
        port=8529,
        username="root",
        password="habitat",  # From docker-compose.yml
        database_name="habitat_evolution"
    )
    
    # Register services with their interfaces
    container.register(EventServiceInterface, EventService)
    container.register(BidirectionalFlowInterface, BidirectionalFlowService)
    container.register(PatternEvolutionServiceInterface, PatternEvolutionService)
    container.register(ArangoDBConnectionInterface, lambda: arangodb_connection)
    
    # Import and register additional services
    from src.habitat_evolution.infrastructure.services.pattern_aware_rag_service import PatternAwareRAGService
    from src.habitat_evolution.infrastructure.services.vector_tonic_service import VectorTonicService
    
    # Create mock implementations for testing
    class MockPatternAwareRAGService(PatternAwareRAGService):
        def __init__(self, event_service: EventServiceInterface = None, vector_tonic_service: VectorTonicServiceInterface = None):
            self.event_service = event_service
            self.vector_tonic_service = vector_tonic_service
    
    class MockVectorTonicService(VectorTonicServiceInterface):
        def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
            pass
            
        def register_vector_space(self, name: str, dimensions: int, metadata: Optional[Dict[str, Any]] = None) -> str:
            return "mock_vector_space_id"
            
        def store_vector(self, vector_space_id: str, vector: List[float], entity_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
            return "mock_vector_id"
            
        def find_similar_vectors(self, vector_space_id: str, query_vector: List[float], limit: int = 10, threshold: float = 0.0) -> List[Dict[str, Any]]:
            return [{"id": "mock_vector_id", "similarity": 0.95, "metadata": {}}]
            
        def detect_tonic_patterns(self, vector_space_id: str, vectors: List[List[float]], threshold: float = 0.7) -> List[Dict[str, Any]]:
            return [{"pattern_id": "mock_pattern_id", "vectors": []}]
            
        def validate_harmonic_coherence(self, pattern_id: str, vectors: List[List[float]]) -> float:
            return 0.9
            
        def calculate_vector_gradient(self, vector_space_id: str, vector1: List[float], vector2: List[float]) -> Dict[str, Any]:
            return {"gradient": 0.1, "direction": [0.01, 0.01, 0.01]}
            
        def get_pattern_vectors(self, pattern_id: str) -> List[List[float]]:
            return [[0.1, 0.2, 0.3]]
            
        def get_pattern_centroid(self, pattern_id: str) -> List[float]:
            return [0.1, 0.2, 0.3]
            
        def update_pattern_with_vector(self, pattern_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> bool:
            return True
            
        def get_vector(self, text: str) -> List[float]:
            return [0.1, 0.2, 0.3]  # Mock vector
            
        def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
            return 0.95  # Mock high similarity
    
    # Register mock services
    container.register(VectorTonicServiceInterface, MockVectorTonicService)
    container.register(PatternAwareRAGInterface, MockPatternAwareRAGService)
    
    # Register DocumentProcessingService
    container.register(DocumentProcessingService, DocumentProcessingService)
    
    # Also register concrete implementations for direct resolution
    container.register(EventService, EventService)
    container.register(BidirectionalFlowService, BidirectionalFlowService)
    container.register(PatternEvolutionService, PatternEvolutionService)
    container.register(ArangoDBConnection, lambda: arangodb_connection)
    
    return container


@pytest.fixture
def event_service(di_container):
    """Get the event service from the DI container."""
    return di_container.resolve(EventService)


@pytest.fixture
def bidirectional_flow_service(di_container):
    """Get the bidirectional flow service from the DI container."""
    return di_container.resolve(BidirectionalFlowService)


@pytest.fixture
def arangodb_connection(di_container):
    """Get the ArangoDB connection from the DI container."""
    return di_container.resolve(ArangoDBConnection)


@pytest.fixture
def pattern_evolution_service(di_container):
    """Get the pattern evolution service from the DI container."""
    return di_container.resolve(PatternEvolutionService)


@pytest.fixture
def document_processing_service(di_container):
    """Get the document processing service from the DI container."""
    return di_container.resolve(DocumentProcessingService)


@pytest.fixture
def test_document_path():
    """Path to the test document."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                        "data", "climate_risk", "climate_risk_marthas_vineyard.txt")


@pytest.fixture
def test_document_content(test_document_path):
    """Content of the test document."""
    with open(test_document_path, "r") as f:
        return f.read()


class TestPatternEvolutionE2E:
    """End-to-end tests for pattern evolution with AdaptiveID integration."""
    
    def test_document_processing_and_pattern_evolution(self, pattern_evolution_service, 
                                                     document_processing_service,
                                                     test_document_path, test_document_content):
        """
        Test the full pattern lifecycle from document processing through evolution.
        
        This test:
        1. Processes a real climate risk document
        2. Extracts patterns from the document
        3. Tracks pattern usage and feedback
        4. Observes quality state transitions
        5. Verifies AdaptiveID integration for versioning and relationships
        """
        # Step 1: Initialize services
        pattern_evolution_service.initialize()
        # DocumentProcessingService doesn't need initialization
        
        # Step 2: Process the document to extract patterns
        document_id = os.path.basename(test_document_path)
        processing_result = document_processing_service.process_document(
            document_id=document_id,
            content=test_document_content,
            metadata={
                "source": "climate_risk",
                "location": "Martha's Vineyard",
                "type": "assessment"
            }
        )
        
        # Step 3: Verify patterns were extracted
        assert processing_result["status"] == "success"
        assert "patterns" in processing_result
        assert len(processing_result["patterns"]) > 0
        
        # Get the extracted patterns
        extracted_patterns = processing_result["patterns"]
        
        # Step 4: Simulate pattern usage
        for pattern in extracted_patterns[:2]:  # Use the first two patterns
            pattern_id = pattern["id"]
            
            # Track usage multiple times with different contexts
            for i in range(3):
                pattern_evolution_service.track_pattern_usage(
                    pattern_id=pattern_id,
                    context={
                        "query": f"Climate risk query {i}",
                        "user_id": f"test_user_{i % 2}",
                        "document_id": document_id
                    }
                )
        
        # Step 5: Simulate pattern feedback
        for pattern in extracted_patterns[:2]:  # Use the first two patterns
            pattern_id = pattern["id"]
            
            # Track feedback
            pattern_evolution_service.track_pattern_feedback(
                pattern_id=pattern_id,
                feedback={
                    "rating": 4,
                    "comment": "Useful climate risk information",
                    "user_id": "test_user_0"
                }
            )
        
        # Step 6: Get pattern evolution history for the first pattern
        first_pattern_id = extracted_patterns[0]["id"]
        evolution_history = pattern_evolution_service.get_pattern_evolution(first_pattern_id)
        
        # Step 7: Verify AdaptiveID integration
        assert evolution_history["status"] == "success"
        assert "adaptive_id" in evolution_history
        assert "timeline" in evolution_history
        assert len(evolution_history["timeline"]) > 0
        
        # Verify that versions were created
        versions = [event for event in evolution_history["timeline"] if event["type"] == "version"]
        assert len(versions) > 0
        
        # Step 8: Identify emerging patterns
        emerging_patterns = pattern_evolution_service.identify_emerging_patterns()
        
        # Step 9: Clean up
        pattern_evolution_service.shutdown()
        document_processing_service.shutdown()
