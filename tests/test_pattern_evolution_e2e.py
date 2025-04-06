"""
End-to-end test for pattern evolution with AdaptiveID integration.

This test processes a real document through the Habitat pipeline,
allowing the system to extract patterns, track their usage and evolution,
and demonstrate the AdaptiveID integration in a real-world scenario.
"""

import os
import pytest
from typing import Dict, List, Any

from src.habitat_evolution.infrastructure.services.pattern_evolution_service import PatternEvolutionService
from src.habitat_evolution.infrastructure.services.document_processing_service import DocumentProcessingService
from src.habitat_evolution.infrastructure.services.event_service import EventService
from src.habitat_evolution.infrastructure.services.bidirectional_flow_service import BidirectionalFlowService
from src.habitat_evolution.infrastructure.persistence.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.infrastructure.di.container import DIContainer


@pytest.fixture
def di_container():
    """Create and configure the DI container for testing."""
    container = DIContainer()
    container.initialize()
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
        document_processing_service.initialize()
        
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
