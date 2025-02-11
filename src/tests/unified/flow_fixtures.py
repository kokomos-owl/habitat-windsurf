"""
Unified test fixtures for flow-based testing.
Combines patterns from both habitat-windsurf and habitat_poc repos.
"""

import pytest
from dataclasses import dataclass
from typing import Dict, List, Any, Set
from datetime import datetime

@dataclass
class UnifiedFlowState:
    """Combined flow state from both repos."""
    energy: float = 0.0
    velocity: float = 0.0
    direction: float = 0.0
    propensity: float = 0.0
    coherence: float = 0.0
    patterns: Set[str] = None
    
    def __post_init__(self):
        self.patterns = set() if self.patterns is None else self.patterns

@dataclass
class UnifiedPatternContext:
    """Combined pattern context from both repos."""
    flow_state: UnifiedFlowState
    query_patterns: List[str]
    retrieval_patterns: List[str]
    augmentation_patterns: List[str]
    coherence_level: float
    temporal_context: Dict[str, Any]

@dataclass
class UnifiedLearningWindow:
    """Combined learning window from both repos."""
    window_data: Dict[str, Any]
    flow_metrics: Dict[str, float]
    density_field: Dict[str, float]
    cross_paths: List[str]

@pytest.fixture
def unified_flow_state():
    """Provide unified flow state fixture."""
    return UnifiedFlowState(
        energy=0.8,        # From habitat_poc energy threshold
        velocity=0.6,      # From habitat_poc velocity measure
        direction=0.7,     # From habitat_poc direction metric
        propensity=0.8,    # From habitat_poc propensity threshold
        coherence=0.85,    # From habitat_poc coherence threshold
        patterns={"climate_risk", "adaptation", "temporal"}
    )

@pytest.fixture
def unified_pattern_context(unified_flow_state):
    """Provide unified pattern context fixture."""
    return UnifiedPatternContext(
        flow_state=unified_flow_state,
        query_patterns=["rainfall", "drought", "temporal"],
        retrieval_patterns=["climate", "adaptation"],
        augmentation_patterns=["evolution"],
        coherence_level=0.85,
        temporal_context={"window": "2020-2050"}
    )

@pytest.fixture
def unified_learning_window(unified_flow_state):
    """Provide unified learning window fixture."""
    return UnifiedLearningWindow(
        window_data={
            "score": 0.9117,
            "potential": 1.0000,
            "channels": {
                "structural": {"strength": 0.8869},
                "semantic": {"strength": 1.0000}
            }
        },
        flow_metrics={
            "density": 0.85,
            "coherence": unified_flow_state.coherence
        },
        density_field={
            "local": 0.82,
            "global": 0.75
        },
        cross_paths=["climate->adaptation", "temporal->projection"]
    )

@pytest.fixture
def mock_pattern_observer():
    """Provide mock pattern observer."""
    class MockPatternObserver:
        def __init__(self):
            self.observations = []
            
        def observe_patterns(self, flow_state: UnifiedFlowState) -> Dict[str, Any]:
            self.observations.append(flow_state)
            return {
                "timestamp": datetime.now().isoformat(),
                "patterns": list(flow_state.patterns),
                "metrics": {
                    "energy": flow_state.energy,
                    "coherence": flow_state.coherence
                }
            }
    return MockPatternObserver()

@pytest.fixture
def mock_density_analyzer():
    """Provide mock density analyzer."""
    class MockDensityAnalyzer:
        def __init__(self):
            self.analyses = []
            
        def analyze_density(self, window: UnifiedLearningWindow) -> Dict[str, float]:
            self.analyses.append(window)
            return {
                "local_density": window.density_field["local"],
                "global_density": window.density_field["global"],
                "cross_path_count": len(window.cross_paths)
            }
    return MockDensityAnalyzer()
