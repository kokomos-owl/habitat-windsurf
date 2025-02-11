"""Test bidirectional evolution in coherence flow."""

import pytest
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass

from habitat_test.core.anthropic_coherence import AnthropicCoherenceAnalyzer, CoherenceInsight
from habitat_test.core.coherence_flow import CoherenceFlow, FlowDynamics
from habitat_test.core.graph_schema import NodeLabel, RelationType

@dataclass
class MockPatternContext:
    """Mock pattern context for testing."""
    query_patterns: List[str]
    retrieval_patterns: List[str]
    augmentation_patterns: List[str]
    coherence_level: float
    temporal_context: Dict[str, Any]

@pytest.fixture
def mock_anthropic():
    """Create mock Anthropic client."""
    class MockResponse:
        def __init__(self, content):
            self.content = content

    class MockMessages:
        def __init__(self):
            self.response_count = 0
            
        async def create(self, *args, **kwargs):
            self.response_count += 1
            if self.response_count == 1:
                return MockResponse({
                    'active_patterns': ['rainfall', 'drought', 'temporal'],
                    'confidence': 0.85,
                    'emergence_potential': 0.7,
                    'pattern_changes': {
                        'rainfall': 0.3,
                        'drought': 0.4,
                        'temporal': 0.2
                    }
                })
            else:
                return MockResponse({
                    'active_patterns': ['rainfall', 'drought', 'temporal', 'evolution', 'adaptation'],
                    'confidence': 0.9,
                    'emergence_potential': 0.85,
                    'pattern_changes': {
                        'rainfall': 0.4,
                        'drought': 0.5,
                        'temporal': 0.3,
                        'evolution': 0.6,
                        'adaptation': 0.7
                    }
                })

    class MockAnthropic:
        def __init__(self):
            self.messages = MockMessages()
            
    return MockAnthropic()

@pytest.fixture
def mock_coherence_flow():
    """Create mock coherence flow."""
    flow = CoherenceFlow()
    flow._flow_state._dynamics.energy = 0.8
    flow._flow_state._dynamics.velocity = 0.6
    flow._flow_state._dynamics.direction = 0.7
    flow._flow_state._dynamics.propensity = 0.8
    return flow

@pytest.fixture
def mock_graph_schema():
    """Create a mock graph schema for testing."""
    class MockGraphSchema:
        def create_graph_state_seed(self, nodes, relationships, projection=None):
            return {
                'nodes': nodes,
                'relationships': relationships,
                'projection': projection
            }
    return MockGraphSchema()

@pytest.fixture
def pattern_context():
    """Create a mock pattern context for testing."""
    return MockPatternContext(
        query_patterns=['measurement', 'impact'],
        retrieval_patterns=['relationship'],
        augmentation_patterns=['adaptation'],
        coherence_level=0.8,
        temporal_context={'timeframe': '2020-2100'}
    )

@pytest.mark.asyncio
async def test_bidirectional_evolution(
    mock_anthropic,
    mock_coherence_flow,
    mock_graph_schema,
    pattern_context
):
    """Test bidirectional evolution between document and graph patterns."""
    
    # Create coherence analyzer with mocks
    coherence_analyzer = AnthropicCoherenceAnalyzer(
        anthropic_client=mock_anthropic,
        coherence_flow=mock_coherence_flow,
        pattern_manager=None  # Not needed for this test
    )
    
    # Initial document content from climate risk assessment
    content = """
    As a result of climate change, the probability of the historical 100-year rainfall event is 
    expected to increase slightly by mid-century and be about five times more likely by late-century.
    The likelihood of extreme drought events will also increase, from an 8.5% annual likelihood in 
    today's climate to 13% and 26% by mid- and late-century, respectively.
    """
    
    # First pass: Document -> Graph Evolution
    insight1 = await coherence_analyzer.analyze_coherence(pattern_context, content)
    
    # Verify emergence conditions
    assert insight1.flow_state.energy > 0.6  # High energy from new patterns
    assert insight1.flow_state.velocity > 0  # Forward movement
    assert len(insight1.patterns) >= 2  # Multiple patterns detected
    
    # Create graph patterns from insight
    nodes = [
        {'type': NodeLabel.HAZARD.value, 'name': 'rainfall_event', 'properties': {'frequency': '100-year'}},
        {'type': NodeLabel.HAZARD.value, 'name': 'drought_event', 'properties': {'likelihood': '8.5%'}},
        {'type': NodeLabel.TIME_FRAME.value, 'name': 'mid_century', 'properties': {'period': '2050-2060'}},
        {'type': NodeLabel.TIME_FRAME.value, 'name': 'late_century', 'properties': {'period': '2080-2100'}}
    ]
    
    relationships = [
        {
            'source': 'rainfall_event',
            'target': 'mid_century',
            'type': RelationType.EVOLVES_TO.value,
            'properties': {'probability': 'slight_increase'}
        },
        {
            'source': 'drought_event',
            'target': 'mid_century',
            'type': RelationType.EVOLVES_TO.value,
            'properties': {'probability': '13%'}
        }
    ]
    
    # Create graph state that can evolve
    graph_state = mock_graph_schema.create_graph_state_seed(
        nodes=nodes,
        relationships=relationships,
        projection='likely'
    )
    
    # Second pass: Graph -> Document Evolution
    pattern_context.augmentation_patterns.extend(['evolution', 'projection'])
    pattern_context.temporal_context['graph_state'] = graph_state
    
    # Update flow state to maintain high velocity and energy
    mock_coherence_flow._flow_state._dynamics.velocity = 0.8
    mock_coherence_flow._flow_state._dynamics.energy = 0.9
    mock_coherence_flow._flow_state._dynamics.direction = 0.7
    mock_coherence_flow._flow_state.momentum = 0.8
    
    insight2 = await coherence_analyzer.analyze_coherence(pattern_context, content)
    
    # Verify bidirectional evolution
    assert insight2.emergence_potential > insight1.emergence_potential
    assert insight2.flow_state.direction > 0  # Convergent flow
    assert insight2.flow_state.propensity > 0.7  # High propensity for pattern formation
    
    # Verify new relationship patterns emerged
    new_patterns = set(insight2.patterns) - set(insight1.patterns)
    assert len(new_patterns) > 0  # New patterns emerged from graph interaction
    
    # Verify temporal evolution
    temporal_patterns = [p for p in insight2.patterns if 'temporal' in p]
    assert len(temporal_patterns) > 0  # Temporal patterns detected
    
    # Verify adaptation potential
    assert any('adaptation' in p for p in insight2.patterns)  # Adaptation patterns present
