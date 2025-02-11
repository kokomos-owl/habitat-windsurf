"""
Unified test configuration for Habitat Flow testing environment.
Combines infrastructure from habitat-windsurf and test patterns from habitat_test.
"""

import os
import pytest
import pytest_asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Test environment configuration
os.environ["TEST_ENVIRONMENT"] = "unified"
TEST_DATA_DIR = Path(__file__).parent / "test_data"

@dataclass
class FlowTestConfig:
    """Configuration for flow-based tests."""
    energy_threshold: float = 0.6
    velocity_threshold: float = 0.0
    direction_threshold: float = 0.0
    propensity_threshold: float = 0.7
    coherence_threshold: float = 0.75
    evolution_threshold: float = 0.8

@pytest.fixture
def flow_config():
    """Provide flow test configuration."""
    return FlowTestConfig()

@dataclass
class TestDocument:
    """Test document with flow metrics."""
    content: str
    metadata: Dict[str, Any]
    flow_state: Dict[str, float]
    structure_data: Optional[Dict[str, Any]] = None
    meaning_data: Optional[Dict[str, Any]] = None

@pytest.fixture
def test_document():
    """Provide test document with initial flow state."""
    return TestDocument(
        content="Test document content for flow analysis",
        metadata={"source": "test", "type": "flow_test"},
        flow_state={
            "energy": 0.8,
            "velocity": 0.6,
            "direction": 0.7,
            "propensity": 0.8
        }
    )

@pytest.fixture
def mock_flow_observer():
    """Provide mock flow observer."""
    class MockFlowObserver:
        def __init__(self):
            self.observations = []
            
        def observe_flow(self, flow_state: Dict[str, float]) -> Dict[str, Any]:
            self.observations.append(flow_state)
            return {
                "timestamp": datetime.now().isoformat(),
                "metrics": flow_state,
                "patterns": ["pattern1", "pattern2"]
            }
    return MockFlowObserver()

@pytest.fixture
def mock_coherence_checker():
    """Provide mock coherence checker."""
    class MockCoherenceChecker:
        def __init__(self):
            self.checks = []
            
        def check_coherence(self, document: TestDocument) -> Dict[str, float]:
            self.checks.append(document)
            return {
                "structure_coherence": 0.85,
                "meaning_coherence": 0.82,
                "flow_coherence": 0.88
            }
    return MockCoherenceChecker()

# Import and configure mocks from both repos
import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from tests.mocks.mock_db import MockDB
from tests.mocks.mock_websocket import MockWebSocket

@pytest.fixture
def mock_db():
    """Provide unified mock database."""
    return MockDB()

@pytest.fixture
def mock_websocket():
    """Provide unified mock websocket."""
    return MockWebSocket()

# Test data setup
def pytest_configure(config):
    """Set up test data directory."""
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
