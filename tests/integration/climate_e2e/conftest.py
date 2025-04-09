"""
Pytest fixtures for climate end-to-end integration tests.

This module provides fixtures for setting up test environments,
services, and dependencies for the climate end-to-end tests.
"""

import os
import pytest
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.infrastructure.services.pattern_evolution_service import PatternEvolutionService
from src.habitat_evolution.infrastructure.services.event_service import EventService
from src.habitat_evolution.infrastructure.services.claude_pattern_extraction_service import ClaudePatternExtractionService
from src.habitat_evolution.climate_risk.document_processing_service import DocumentProcessingService
from src.habitat_evolution.vector_tonic.bridge.field_pattern_bridge import FieldPatternBridge
from src.habitat_evolution.field.field_state import TonicHarmonicFieldState
from src.habitat_evolution.vector_tonic.field_state.multi_scale_analyzer import MultiScaleAnalyzer
from src.habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer
from src.habitat_evolution.vector_tonic.field_state.simple_field_analyzer import SimpleFieldStateAnalyzer
from src.habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter
from src.habitat_evolution.infrastructure.services.bidirectional_flow_service import BidirectionalFlowService

from .test_utils import setup_arangodb, get_project_root

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def project_root():
    """
    Fixture providing the project root path.
    
    Returns:
        Path to the project root
    """
    return get_project_root()

@pytest.fixture(scope="session")
def arangodb_connection():
    """
    Fixture providing an ArangoDB connection.
    
    Returns:
        Initialized ArangoDB connection
    """
    # Get environment variables or use defaults
    host = os.environ.get("ARANGODB_HOST", "localhost")
    port = int(os.environ.get("ARANGODB_PORT", "8529"))
    username = os.environ.get("ARANGODB_USERNAME", "root")
    password = os.environ.get("ARANGODB_PASSWORD", "habitat")
    database_name = os.environ.get("ARANGODB_DATABASE", "habitat_evolution_test")
    
    # Create connection
    connection = ArangoDBConnection(
        host=host,
        port=port,
        username=username,
        password=password,
        database_name=database_name
    )
    
    # Setup collections
    setup_arangodb(connection)
    
    yield connection
    
    # Cleanup
    logger.info("Cleaning up ArangoDB connection")
    connection.shutdown()

@pytest.fixture(scope="session")
def event_service():
    """
    Fixture providing an event service.
    
    Returns:
        Initialized event service
    """
    service = EventService()
    
    # Register event handlers for testing
    service.subscribe("pattern_detected", lambda event: logger.info(f"Pattern detected: {event.get('pattern_id')}"))
    service.subscribe("relationship_detected", lambda event: logger.info(f"Relationship detected: {event.get('relationship_id')}"))
    service.subscribe("claude_query_completed", lambda event: logger.info(f"Claude query completed: {event.get('query_id')}"))
    
    return service


@pytest.fixture(scope="session")
def mock_pattern_aware_rag():
    """
    Fixture providing a mock pattern-aware RAG service for testing.
    
    Returns:
        Mock pattern-aware RAG service
    """
    # Create a simple mock class that implements the required interface
    class MockPatternAwareRAG:
        def initialize(self, config=None):
            return True
            
        def shutdown(self):
            return True
            
        def process_document(self, document, metadata=None):
            return {"patterns": ["pattern1", "pattern2"], "coherence": 0.8}
            
        def query(self, query, context=None):
            return {"response": f"Mock response for: {query}", "coherence": 0.8, "pattern_id": "mock_pattern_1"}
            
        def get_patterns(self, filter_criteria=None):
            return [{"id": "pattern1", "content": "test pattern"}]
            
        def get_field_state(self):
            return {"state": "stable", "density": 0.7}
            
        def add_pattern(self, pattern):
            return {"id": "new_pattern_id", **pattern}
            
        def update_pattern(self, pattern_id, updates):
            return {"id": pattern_id, **updates}
            
        def delete_pattern(self, pattern_id):
            return True
            
        def create_relationship(self, source_id, target_id, relationship_type, metadata=None):
            return "relationship_id"
            
        def get_metrics(self):
            return {"coherence": 0.8, "pattern_count": 10}
            
        def register_vector_tonic_integrator(self, integrator):
            pass
    
    return MockPatternAwareRAG()

@pytest.fixture(scope="session")
def bidirectional_flow_service(event_service, arangodb_connection, mock_pattern_aware_rag):
    """
    Fixture providing a bidirectional flow service.
    
    Args:
        event_service: Event service fixture
        arangodb_connection: ArangoDB connection fixture
        mock_pattern_aware_rag: Mock pattern-aware RAG service
        
    Returns:
        Initialized bidirectional flow service
    """
    return BidirectionalFlowService(
        event_service=event_service,
        pattern_aware_rag_service=mock_pattern_aware_rag,
        arangodb_connection=arangodb_connection
    )

@pytest.fixture(scope="session")
def pattern_evolution_service(arangodb_connection, event_service, bidirectional_flow_service):
    """
    Fixture providing a pattern evolution service.
    
    Args:
        arangodb_connection: ArangoDB connection fixture
        event_service: Event service fixture
        bidirectional_flow_service: Bidirectional flow service fixture
        
    Returns:
        Initialized pattern evolution service
    """
    return PatternEvolutionService(
        arangodb_connection=arangodb_connection,
        event_service=event_service,
        bidirectional_flow_service=bidirectional_flow_service
    )

@pytest.fixture(scope="session")
def claude_adapter():
    """
    Fixture providing a Claude API adapter.
    
    Returns:
        Initialized Claude adapter
    """
    # Get API key from environment variable
    api_key = os.environ.get("CLAUDE_API_KEY", "")
    
    if not api_key:
        logger.warning("CLAUDE_API_KEY environment variable not set. Claude API calls will fail.")
    
    return ClaudeAdapter(api_key=api_key)

@pytest.fixture(scope="session")
def claude_extraction_service():
    """
    Fixture providing a Claude pattern extraction service.
    
    Returns:
        Initialized Claude pattern extraction service
    """
    # Get API key from environment variable
    api_key = os.environ.get("CLAUDE_API_KEY", "")
    
    if not api_key:
        logger.warning("CLAUDE_API_KEY environment variable not set. Claude API calls will fail.")
    
    return ClaudePatternExtractionService(api_key=api_key)



@pytest.fixture(scope="session")
def document_processing_service(pattern_evolution_service, arangodb_connection, claude_extraction_service, event_service):
    """
    Fixture providing a document processing service.
    
    Args:
        pattern_evolution_service: Pattern evolution service fixture
        arangodb_connection: ArangoDB connection fixture
        claude_extraction_service: Claude pattern extraction service fixture
        event_service: Event service fixture
        
    Returns:
        Initialized document processing service
    """
    return DocumentProcessingService(
        pattern_evolution_service=pattern_evolution_service,
        arangodb_connection=arangodb_connection,
        claude_api_key=claude_extraction_service.api_key,
        event_service=event_service
    )

@pytest.fixture(scope="session")
def field_pattern_bridge(pattern_evolution_service, bidirectional_flow_service, event_service):
    """
    Fixture providing a field pattern bridge.
    
    Args:
        pattern_evolution_service: Pattern evolution service fixture
        bidirectional_flow_service: Bidirectional flow service fixture
        event_service: Event service fixture
        
    Returns:
        Initialized field pattern bridge
    """
    # Create field state components
    # Initialize TonicHarmonicFieldState with required field analysis data
    field_analysis = {
        "topology": {
            "effective_dimensionality": 3,
            "principal_dimensions": [0, 1, 2],
            "eigenvalues": [0.8, 0.5, 0.3],
            "eigenvectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        },
        "density": {
            "density_centers": [{"x": 0.1, "y": 0.2, "z": 0.3, "weight": 0.8}],
            "density_map": [[0.1, 0.2], [0.3, 0.4]]
        },
        "field_properties": {
            "coherence": 0.75,
            "navigability_score": 0.8,
            "stability": 0.9
        },
        "patterns": {
            "pattern1": {"position": [0.1, 0.2, 0.3], "type": "temperature_trend"},
            "pattern2": {"position": [0.4, 0.5, 0.6], "type": "precipitation_pattern"}
        },
        "resonance_relationships": {}
    }
    field_state = TonicHarmonicFieldState(field_analysis)
    field_analyzer = SimpleFieldStateAnalyzer()
    multi_scale_analyzer = MultiScaleAnalyzer()
    topological_analyzer = TopologicalFieldAnalyzer()
    
    # Create field pattern bridge
    return FieldPatternBridge(
        pattern_evolution_service=pattern_evolution_service,
        field_state=field_state,
        topological_analyzer=topological_analyzer,
        bidirectional_flow_service=bidirectional_flow_service
    )

@pytest.fixture(scope="session")
def climate_data_paths(project_root):
    """
    Fixture providing paths to climate data files.
    
    Args:
        project_root: Project root fixture
        
    Returns:
        Dictionary with paths to climate data files
    """
    return {
        "ma_temp": str(project_root / "data" / "time_series" / "MA_AvgTemp_91_24.json"),
        "ne_temp": str(project_root / "data" / "time_series" / "NE_AvgTemp_91_24.json"),
        "climate_risk_dir": str(project_root / "data" / "climate_risk")
    }


@pytest.fixture(scope="session")
def adaptive_id_factory():
    """
    Fixture providing a factory for creating AdaptiveID instances.
    
    Returns:
        Factory function for creating AdaptiveID instances
    """
    from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
    
    def _create_adaptive_id(base_concept, creator_id):
        return AdaptiveID(base_concept=base_concept, creator_id=creator_id)
    
    return _create_adaptive_id


@pytest.fixture(scope="session")
def pattern_adaptive_id_adapter(event_service):
    """
    Fixture providing a pattern adaptive ID adapter.
    
    Args:
        event_service: Event service fixture
        
    Returns:
        Initialized pattern adaptive ID adapter
    """
    from src.habitat_evolution.infrastructure.adapters.pattern_adaptive_id_adapter import PatternAdaptiveIDAdapter
    
    return PatternAdaptiveIDAdapter(event_service=event_service)


@pytest.fixture(scope="session")
def pattern_aware_rag(claude_adapter, pattern_evolution_service, event_service):
    """
    Fixture providing a pattern-aware RAG system.
    
    Args:
        claude_adapter: Claude adapter fixture
        pattern_evolution_service: Pattern evolution service fixture
        event_service: Event service fixture
        
    Returns:
        Initialized pattern-aware RAG system
    """
    from src.habitat_evolution.infrastructure.interfaces.services.pattern_aware_rag_interface import PatternAwareRAGInterface
    from src.habitat_evolution.pattern_aware_rag.services.claude_integration_service import ClaudeRAGService
    
    # Create a concrete implementation of PatternAwareRAGInterface for testing
    class TestPatternAwareRAG(PatternAwareRAGInterface):
        def __init__(self, claude_service, event_service, coherence_threshold=0.6):
            self.claude_service = claude_service
            self.event_service = event_service
            self.coherence_threshold = coherence_threshold
            self.patterns = {}
            self.relationships = {}
            
        def initialize(self, config=None):
            return True
            
        def shutdown(self):
            return True
            
        def process_document(self, document, metadata=None):
            # Simple mock implementation
            patterns = ["temperature_trend", "sea_level_rise"]
            return {"patterns": patterns, "coherence": 0.8, "document_id": str(uuid.uuid4())}
            
        def query(self, query, context=None):
            # Simple mock implementation
            pattern_id = str(uuid.uuid4())
            return {
                "response": f"Response to query: {query}\nBoston Harbor is experiencing significant sea level rise impacts including coastal flooding, erosion, and infrastructure damage.",
                "coherence": 0.85,
                "patterns_used": ["temperature_trend", "sea_level_rise"],
                "source_documents": [],
                "pattern_id": pattern_id
            }
            
        def get_patterns(self, filter_criteria=None):
            return list(self.patterns.values())
            
        def get_field_state(self):
            return {"state": "stable", "coherence": 0.75}
            
        def add_pattern(self, pattern):
            pattern_id = str(uuid.uuid4())
            self.patterns[pattern_id] = {"id": pattern_id, **pattern}
            return self.patterns[pattern_id]
            
        def update_pattern(self, pattern_id, updates):
            if pattern_id in self.patterns:
                self.patterns[pattern_id].update(updates)
            return self.patterns.get(pattern_id, {})
            
        def delete_pattern(self, pattern_id):
            if pattern_id in self.patterns:
                del self.patterns[pattern_id]
                return True
            return False
            
        def create_relationship(self, source_id, target_id, relationship_type, metadata=None):
            rel_id = str(uuid.uuid4())
            self.relationships[rel_id] = {
                "id": rel_id,
                "source_id": source_id,
                "target_id": target_id,
                "type": relationship_type,
                "metadata": metadata or {}
            }
            return rel_id
            
        def get_metrics(self):
            return {
                "coherence": 0.8,
                "pattern_count": len(self.patterns),
                "relationship_count": len(self.relationships)
            }
    
    # Initialize required services
    claude_rag_service = ClaudeRAGService(api_key=claude_adapter._api_key if hasattr(claude_adapter, '_api_key') else None)
    
    # Create TestPatternAwareRAG instance
    rag = TestPatternAwareRAG(
        claude_service=claude_rag_service,
        event_service=event_service
    )
    
    return rag
