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
from src.habitat_evolution.vector_tonic.field_state.tonic_harmonic_field_state import TonicHarmonicFieldState
from src.habitat_evolution.vector_tonic.field_state.multi_scale_analyzer import MultiScaleAnalyzer
from src.habitat_evolution.vector_tonic.field_state.topological_field_analyzer import TopologicalFieldAnalyzer
from src.habitat_evolution.vector_tonic.field_state.simple_field_analyzer import SimpleFieldAnalyzer
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
    password = os.environ.get("ARANGODB_PASSWORD", "")
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
def pattern_evolution_service(arangodb_connection, event_service):
    """
    Fixture providing a pattern evolution service.
    
    Args:
        arangodb_connection: ArangoDB connection fixture
        event_service: Event service fixture
        
    Returns:
        Initialized pattern evolution service
    """
    return PatternEvolutionService(
        db_connection=arangodb_connection,
        event_service=event_service
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
def bidirectional_flow_service(event_service):
    """
    Fixture providing a bidirectional flow service.
    
    Args:
        event_service: Event service fixture
        
    Returns:
        Initialized bidirectional flow service
    """
    return BidirectionalFlowService(event_service=event_service)

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
        claude_api_key=claude_extraction_service._api_key,
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
    field_state = TonicHarmonicFieldState()
    field_analyzer = SimpleFieldAnalyzer()
    multi_scale_analyzer = MultiScaleAnalyzer()
    topological_analyzer = TopologicalFieldAnalyzer()
    
    # Create field pattern bridge
    return FieldPatternBridge(
        pattern_evolution_service=pattern_evolution_service,
        field_state=field_state,
        field_analyzer=field_analyzer,
        multi_scale_analyzer=multi_scale_analyzer,
        topological_analyzer=topological_analyzer,
        bidirectional_flow_service=bidirectional_flow_service,
        event_service=event_service
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
