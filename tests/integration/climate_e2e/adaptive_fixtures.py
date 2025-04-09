"""
Fixtures for AdaptiveID and PatternAwareRAG integration in end-to-end tests.

This module provides fixtures for setting up AdaptiveID and PatternAwareRAG
components for integration testing, ensuring proper dependency management
and component interactions.
"""

import os
import pytest
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from src.habitat_evolution.adaptive_core.id.base_adaptive_id import BaseAdaptiveID, LoggingManager
from src.habitat_evolution.pattern_aware_rag.pattern_aware_rag import PatternAwareRAG, RAGPatternContext
from src.habitat_evolution.pattern_aware_rag.services.graph_service import GraphService
from src.habitat_evolution.pattern_aware_rag.services.claude_integration_service import ClaudeRAGService
from src.habitat_evolution.core.pattern import FieldDrivenPatternManager, PatternQualityAnalyzer
from src.habitat_evolution.core.services.field.interfaces import FieldStateService, GradientService, FlowDynamicsService
from src.habitat_evolution.adaptive_core.services.interfaces import (
    PatternEvolutionService,
    MetricsService,
    QualityMetricsService,
    EventManagementService
)
from src.habitat_evolution.infrastructure.services.event_service import EventService
from src.habitat_evolution.infrastructure.adapters.pattern_adaptive_id_adapter import PatternAdaptiveIDAdapter
from src.habitat_evolution.infrastructure.services.claude_pattern_extraction_service import ClaudePatternExtractionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def adaptive_id_factory():
    """
    Fixture providing a factory for creating AdaptiveID instances.
    
    Returns:
        Function to create AdaptiveID instances
    """
    def _create_adaptive_id(base_concept: str, creator_id: str = "test_user"):
        """
        Create an AdaptiveID instance.
        
        Args:
            base_concept: The base concept this ID represents
            creator_id: ID of the creator (default: "test_user")
            
        Returns:
            AdaptiveID instance
        """
        return AdaptiveID(
            base_concept=base_concept,
            creator_id=creator_id,
            weight=1.0,
            confidence=0.8,
            uncertainty=0.2
        )
    
    return _create_adaptive_id

@pytest.fixture(scope="session")
def pattern_adaptive_id_adapter(event_service):
    """
    Fixture providing a PatternAdaptiveIDAdapter.
    
    Args:
        event_service: Event service fixture
        
    Returns:
        Initialized PatternAdaptiveIDAdapter
    """
    return PatternAdaptiveIDAdapter(event_service=event_service)

@pytest.fixture(scope="session")
def field_state_service():
    """
    Fixture providing a mock FieldStateService.
    
    Returns:
        Mock FieldStateService
    """
    class MockFieldStateService(FieldStateService):
        """Mock implementation of FieldStateService."""
        
        def __init__(self):
            self.field_states = {}
            
        async def create_field_state(self, field_id):
            """Create a new field state."""
            field_state = {
                "id": field_id,
                "position": [0.5, 0.5, 0.5],
                "created_at": datetime.now().isoformat(),
                "patterns": []
            }
            self.field_states[field_id] = field_state
            return field_state
            
        async def get_field_state(self, field_id):
            """Get field state by ID."""
            return self.field_states.get(field_id, {})
            
        async def calculate_local_density(self, field_id, position):
            """Calculate local density."""
            return 0.5
            
        def add_pattern(self, pattern):
            """Add a pattern to the field state."""
            pattern_id = pattern.get("id")
            if pattern_id:
                for field_id, field_state in self.field_states.items():
                    if "patterns" not in field_state:
                        field_state["patterns"] = []
                    field_state["patterns"].append(pattern_id)
    
    return MockFieldStateService()

@pytest.fixture(scope="session")
def gradient_service():
    """
    Fixture providing a mock GradientService.
    
    Returns:
        Mock GradientService
    """
    class MockGradientService(GradientService):
        """Mock implementation of GradientService."""
        
        async def calculate_gradient(self, field_id, position):
            """Calculate gradient at position."""
            return [0.1, 0.1, 0.1]
            
        async def calculate_gradient_flow(self, field_id, start_position, end_position):
            """Calculate gradient flow between positions."""
            return 0.5
    
    return MockGradientService()

@pytest.fixture(scope="session")
def flow_dynamics_service():
    """
    Fixture providing a mock FlowDynamicsService.
    
    Returns:
        Mock FlowDynamicsService
    """
    class MockFlowDynamicsService(FlowDynamicsService):
        """Mock implementation of FlowDynamicsService."""
        
        async def calculate_back_pressure(self, field_id):
            """Calculate back pressure."""
            return 0.3
            
        async def calculate_flow_stability(self, field_id):
            """Calculate flow stability."""
            return 0.7
    
    return MockFlowDynamicsService()

@pytest.fixture(scope="session")
def metrics_service():
    """
    Fixture providing a mock MetricsService.
    
    Returns:
        Mock MetricsService
    """
    class MockMetricsService(MetricsService):
        """Mock implementation of MetricsService."""
        
        async def calculate_global_density(self):
            """Calculate global density."""
            return 0.5
            
        async def record_pattern_metrics(self, pattern_id, metrics):
            """Record pattern metrics."""
            pass
    
    return MockMetricsService()

@pytest.fixture(scope="session")
def quality_metrics_service():
    """
    Fixture providing a mock QualityMetricsService.
    
    Returns:
        Mock QualityMetricsService
    """
    class MockQualityMetricsService(QualityMetricsService):
        """Mock implementation of QualityMetricsService."""
        
        async def calculate_coherence(self, entity_id):
            """Calculate coherence."""
            return 0.7
            
        async def calculate_quality_metrics(self, pattern_id):
            """Calculate quality metrics."""
            return {
                "coherence": 0.7,
                "relevance": 0.8,
                "stability": 0.6
            }
    
    return MockQualityMetricsService()

@pytest.fixture(scope="session")
def coherence_analyzer():
    """
    Fixture providing a mock coherence analyzer.
    
    Returns:
        Mock coherence analyzer
    """
    class MockCoherenceAnalyzer:
        """Mock implementation of coherence analyzer."""
        
        async def analyze_coherence(self, patterns, context):
            """Analyze coherence."""
            return {
                "coherence": 0.7,
                "patterns": patterns,
                "context": context
            }
    
    return MockCoherenceAnalyzer()

@pytest.fixture(scope="session")
def emergence_flow():
    """
    Fixture providing a mock emergence flow.
    
    Returns:
        Mock emergence flow
    """
    class MockEmergenceFlow:
        """Mock implementation of emergence flow."""
        
        def __init__(self):
            self.context = type('obj', (object,), {
                'state_space': type('obj', (object,), {
                    'coherence': 0.7,
                    'density': 0.5,
                    'stability': 0.6
                })
            })
            
        def get_flow_state(self):
            """Get flow state."""
            return {
                "coherence": 0.7,
                "density": 0.5,
                "stability": 0.6
            }
            
        async def observe_emergence(self, patterns, state):
            """Observe emergence."""
            pass
    
    return MockEmergenceFlow()

@pytest.fixture(scope="session")
def graph_service():
    """
    Fixture providing a GraphService.
    
    Returns:
        Initialized GraphService
    """
    return GraphService()

@pytest.fixture(scope="session")
def pattern_aware_rag(
    pattern_evolution_service,
    field_state_service,
    gradient_service,
    flow_dynamics_service,
    metrics_service,
    quality_metrics_service,
    event_service,
    coherence_analyzer,
    emergence_flow,
    graph_service
):
    """
    Fixture providing a PatternAwareRAG instance.
    
    Args:
        pattern_evolution_service: Pattern evolution service fixture
        field_state_service: Field state service fixture
        gradient_service: Gradient service fixture
        flow_dynamics_service: Flow dynamics service fixture
        metrics_service: Metrics service fixture
        quality_metrics_service: Quality metrics service fixture
        event_service: Event service fixture
        coherence_analyzer: Coherence analyzer fixture
        emergence_flow: Emergence flow fixture
        graph_service: Graph service fixture
        
    Returns:
        Initialized PatternAwareRAG instance
    """
    # Get Claude API key from environment
    claude_api_key = os.environ.get("CLAUDE_API_KEY", "")
    
    # Create settings object
    settings = type('obj', (object,), {
        'VECTOR_STORE_DIR': '/tmp/vector_store',
        'THRESHOLDS': {
            'density': 0.3,
            'coherence': 0.5,
            'back_pressure': 0.7,
            'cross_paths': 2
        }
    })
    
    # Create PatternAwareRAG instance with mock config
    rag = PatternAwareRAG(
        pattern_evolution_service=pattern_evolution_service,
        field_state_service=field_state_service,
        gradient_service=gradient_service,
        flow_dynamics_service=flow_dynamics_service,
        metrics_service=metrics_service,
        quality_metrics_service=quality_metrics_service,
        event_service=event_service,
        coherence_analyzer=coherence_analyzer,
        emergence_flow=emergence_flow,
        settings=settings,
        graph_service=graph_service,
        claude_api_key=claude_api_key
    )
    
    # Add config attribute for window state determination
    rag.config = {
        "thresholds": {
            "density": 0.3,
            "coherence": 0.5,
            "back_pressure": 0.7,
            "cross_paths": 2
        }
    }
    
    yield rag
    
    # Cleanup
    rag.shutdown()
