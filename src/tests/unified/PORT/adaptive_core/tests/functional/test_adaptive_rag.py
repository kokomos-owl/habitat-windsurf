"""
Test suite for adaptive RAG enhancement in Habitat POC.
Tests the system's ability to adapt and enhance RAG operations based on context and feedback.
"""

import os
import sys
import pytest
import pytest_asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Set test environment
os.environ["TEST_ENVIRONMENT"] = "test"

from habitat_test.config.mock_settings import MockSettings
from habitat_test.core.mock_rag import MockRAGController
from habitat_test.core.mock_bidirectional_processor import MockBidirectionalProcessor
from habitat_test.core.mock_adaptive_id import MockAdaptiveID
from habitat_test.core.mock_coherence import CoherenceChecker, CoherenceThresholds
from habitat_test.core.mock_observable import MockObservable

# Configure logging
logging.basicConfig(level=logging.INFO)

@pytest.fixture
def settings():
    """Test settings fixture."""
    return MockSettings()

@pytest.fixture
def mock_adaptive_id():
    """Mock adaptive ID fixture."""
    return MockAdaptiveID({
        "name": "test_concept",
        "type": "climate_concept",
        "attributes": {
            "domain": "climate_science",
            "confidence": 0.85
        }
    })

@pytest.fixture
def mock_bidirectional_processor():
    """Mock bidirectional processor fixture."""
    return MockBidirectionalProcessor()

@pytest.fixture
def mock_observable():
    """Mock Observable interface fixture."""
    return MockObservable("test-notebook")

@pytest.fixture
def coherence_checker():
    """Coherence checker fixture."""
    return CoherenceChecker(CoherenceThresholds(
        min_semantic_coherence=0.7,
        min_structural_coherence=0.6,
        min_overall_coherence=0.65
    ))

@pytest.fixture
def rag_controller(settings, mock_bidirectional_processor, coherence_checker, mock_observable):
    """RAG controller fixture."""
    return MockRAGController(
        settings=settings,
        bidirectional_processor=mock_bidirectional_processor,
        coherence_checker=coherence_checker,
        observable=mock_observable
    )

@pytest.mark.asyncio
async def test_adaptive_rag_enhancement(
    rag_controller,
    mock_adaptive_id,
    mock_bidirectional_processor,
    coherence_checker
):
    """Test adaptive RAG enhancement with mock components."""
    # Test data
    test_document = {
        "id": "doc1",
        "content": "Climate change affects global temperatures.",
        "metadata": {
            "source": "test_source",
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    # Process document with adaptive context
    result = await rag_controller.process_document(test_document, mock_adaptive_id)
    
    # Verify structure processing
    assert "structure" in result
    assert result["structure"]["structure_score"] >= 0.7
    
    # Verify meaning processing
    assert "meaning" in result
    assert result["meaning"]["meaning_score"] >= 0.8
    
    # Verify coherence
    coherence_history = coherence_checker.get_coherence_history()
    assert len(coherence_history) > 0
    assert all(check["semantic"] >= 0.7 for check in coherence_history)
    assert all(check["structural"] >= 0.6 for check in coherence_history)
    
    # Verify bidirectional processing
    processing_history = mock_bidirectional_processor.get_processing_history()
    assert len(processing_history) > 0
    assert any(proc["type"] == "structure" for proc in processing_history)
    assert any(proc["type"] == "meaning" for proc in processing_history)
    
    # Verify adaptive context updates
    adaptive_state = mock_adaptive_id.get_current_state()
    assert adaptive_state["version"] > 1  # Should have been updated
    assert "confidence" in adaptive_state["data"]["attributes"]
