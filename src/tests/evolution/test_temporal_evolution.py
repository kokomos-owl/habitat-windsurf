"""Natural evolution tests for temporal patterns."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.core.evolution.temporal_core import TemporalCore
from src.core.evolution.pattern_core import PatternEvidence, TemporalContext
from src.core.types import UncertaintyMetrics

@pytest.fixture
def temporal_core():
    """Create TemporalCore with natural dependencies."""
    timestamp_service = Mock()
    timestamp_service.get_timestamp.return_value = datetime.now()
    
    event_manager = Mock()
    version_service = Mock()
    
    return TemporalCore(
        timestamp_service=timestamp_service,
        event_manager=event_manager,
        version_service=version_service
    )

@pytest.fixture
def pattern_evidence():
    """Create natural pattern evidence."""
    return PatternEvidence(
        evidence_id="test_pattern",
        timestamp=datetime.now().isoformat(),
        pattern_type="test",
        source_data={"key": "value"},
        temporal_context=TemporalContext(
            start_time=datetime.now(),
            confidence=0.8
        ),
        uncertainty_metrics=UncertaintyMetrics(
            temporal_stability=0.85
        )
    )

@pytest.mark.asyncio
async def test_natural_evolution(temporal_core, pattern_evidence):
    """Test natural temporal evolution."""
    # First observation
    result1 = await temporal_core.observe_temporal_evolution(pattern_evidence)
    
    assert result1["pattern_id"] == pattern_evidence.evidence_id
    assert "stability" in result1
    assert "confidence" in result1
    
    # Allow natural evolution
    pattern_evidence.source_data["key"] = "evolved"
    result2 = await temporal_core.observe_temporal_evolution(pattern_evidence)
    
    # Verify natural progression
    assert result2["stability"] <= result1["stability"]
    assert len(temporal_core.temporal_evidence[pattern_evidence.evidence_id]) == 2

@pytest.mark.asyncio
async def test_stability_emergence(temporal_core, pattern_evidence):
    """Test natural stability emergence."""
    # Create evolution sequence
    results = []
    for i in range(5):
        pattern_evidence.source_data["value"] = i
        result = await temporal_core.observe_temporal_evolution(pattern_evidence)
        results.append(result)
        
    # Verify natural stability patterns
    stabilities = [r["stability"] for r in results]
    assert all(0 <= s <= 1 for s in stabilities)
    
    # Natural variance should emerge
    stability_changes = [
        abs(stabilities[i] - stabilities[i-1])
        for i in range(1, len(stabilities))
    ]
    assert any(change > 0 for change in stability_changes)

@pytest.mark.asyncio
async def test_confidence_adaptation(temporal_core, pattern_evidence):
    """Test natural confidence adaptation."""
    # Evolve with varying confidence
    confidences = [0.9, 0.7, 0.8, 0.85]
    results = []
    
    for conf in confidences:
        pattern_evidence.temporal_context.confidence = conf
        result = await temporal_core.observe_temporal_evolution(pattern_evidence)
        results.append(result)
    
    # Verify natural confidence adaptation
    assert all(0 <= r["confidence"] <= 1 for r in results)
    
    # Should show some natural variation
    confidence_changes = [
        abs(results[i]["confidence"] - results[i-1]["confidence"])
        for i in range(1, len(results))
    ]
    assert any(change > 0 for change in confidence_changes)
