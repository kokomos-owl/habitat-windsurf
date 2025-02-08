"""Tests for natural pattern emergence tracking."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.core.analysis.pattern_emergence import (
    PatternEmergenceTracker,
    EmergentPattern
)
from src.core.analysis.structure_analysis import (
    StructuralElement,
    StructureContext
)
from src.core.types import (
    DensityMetrics,
    PatternEvolutionMetrics,
    PatternEvidence,
    TemporalContext
)

@pytest.fixture
def pattern_tracker():
    """Create PatternEmergenceTracker with natural dependencies."""
    timestamp_service = Mock()
    timestamp_service.get_timestamp.return_value = datetime.now()
    
    temporal_core = Mock()
    
    return PatternEmergenceTracker(
        timestamp_service=timestamp_service,
        temporal_core=temporal_core
    )

@pytest.fixture
def analysis_context():
    """Create analysis context."""
    return StructureContext(
        start_time=datetime.now(),
        content_type="text",
        analysis_depth=2
    )

@pytest.fixture
def pattern_evidence():
    """Create pattern evidence elements."""
    base_time = datetime.now()
    
    def create_evidence(idx, time_offset):
        return PatternEvidence(
            evidence_id=f"evidence{idx}",
            pattern_type="climate_risk",
            evolution_metrics=PatternEvolutionMetrics(
                coherence_level=0.7 + (idx * 0.1),
                stability=0.6 + (idx * 0.1),
                transition_rate=0.2
            ),
            density_metrics=DensityMetrics(
                local_density=0.8,
                global_density=0.6
            ),
            temporal_context=TemporalContext(
                start_time=base_time + timedelta(minutes=time_offset),
                end_time=base_time + timedelta(minutes=time_offset+60)
            )
        )
    
    return [create_evidence(i, i*60) for i in range(3)]

@pytest.fixture
def similar_elements():
    """Create naturally similar structural elements."""
    base_time = datetime.now()
    base_content = "This is a section about machine learning"
    
    def create_element(idx, time_offset):
        return StructuralElement(
            element_id=f"elem{idx}",
            element_type="section",
            content=f"{base_content} {['focusing on neural networks', 'discussing deep learning', 'exploring reinforcement learning'][idx-1]}",
            metadata={"depth": 2},
            density=0.7,
            emergence_time=base_time + timedelta(minutes=time_offset)
        )
    
    return [
        create_element(1, 0),    # First observation
        create_element(2, 30),   # After 30 minutes
        create_element(3, 60)    # After 1 hour
    ]

@pytest.fixture
def diverse_elements():
    """Create naturally diverse structural elements."""
    now = datetime.now()
    
    return [
        StructuralElement(
            element_id="elem4",
            element_type="list",
            content="- First point\n- Second point",
            metadata={"depth": 1},
            density=0.5,
            emergence_time=now
        ),
        StructuralElement(
            element_id="elem5",
            element_type="paragraph",
            content="A completely different topic about databases",
            metadata={"depth": 0},
            density=0.3,
            emergence_time=now
        ),
        StructuralElement(
            element_id="elem6",
            element_type="code",
            content="def example(): pass",
            metadata={"depth": 1},
            density=0.6,
            emergence_time=now
        )
    ]

@pytest.mark.asyncio
async def test_natural_pattern_discovery(
    pattern_tracker,
    analysis_context,
    similar_elements
):
    """Test natural emergence of patterns."""
    result = await pattern_tracker.observe_elements(similar_elements, analysis_context)
    
    # Verify pattern emergence
    assert "new_patterns" in result
    assert len(result["new_patterns"]) > 0
    
    # Check pattern properties
    pattern = result["new_patterns"][0]
    assert pattern.pattern_type.startswith("section")
    assert pattern.confidence > pattern_tracker.confidence_threshold
    assert len(pattern.elements) > 1

@pytest.mark.asyncio
async def test_pattern_evolution(
    pattern_tracker,
    analysis_context,
    similar_elements,
    diverse_elements
):
    """Test natural pattern evolution."""
    # First observation
    await pattern_tracker.observe_elements(similar_elements[:2], analysis_context)
    
    # Second observation with new element
    result = await pattern_tracker.observe_elements(
        [similar_elements[2]] + diverse_elements,
        analysis_context
    )
    
    # Verify pattern evolution
    assert "evolved_patterns" in result
    assert len(result["evolved_patterns"]) > 0
    
    # Check evolution properties
    evolved = result["evolved_patterns"][0]
    assert len(evolved.elements) > 2
    assert evolved.stability > 0.0

@pytest.mark.asyncio
async def test_pattern_stability(pattern_tracker, analysis_context, similar_elements):
    """Test natural stability emergence."""
    base_time = datetime.now()
    pattern_tracker.timestamp_service.get_timestamp.return_value = base_time
    
    # First observation
    await pattern_tracker.observe_elements(
        similar_elements[:2],
        analysis_context
    )
    
    # Second observation after some time
    pattern_tracker.timestamp_service.get_timestamp.return_value = base_time + timedelta(minutes=30)
    await pattern_tracker.observe_elements(
        similar_elements[1:],
        analysis_context
    )
    
    # Third observation after more time
    pattern_tracker.timestamp_service.get_timestamp.return_value = base_time + timedelta(hours=1)
    await pattern_tracker.observe_elements(
        similar_elements,
        analysis_context
    )
    
    # Get stable patterns
    stable_patterns = [
        p for p in pattern_tracker.emergent_patterns.values()
        if p.stability >= pattern_tracker.stability_threshold
    ]
    
    assert len(stable_patterns) > 0
    assert all(p.age > 0 for p in stable_patterns)
    
    # Verify increasing stability
    pattern = stable_patterns[0]
    assert pattern.stability > 0.5  # Should be quite stable after consistent observations

@pytest.mark.asyncio
async def test_diverse_pattern_handling(
    pattern_tracker,
    analysis_context,
    diverse_elements
):
    """Test handling of naturally diverse elements."""
    result = await pattern_tracker.observe_elements(diverse_elements, analysis_context)
    
    # Should find fewer patterns in diverse elements
    assert len(result["new_patterns"]) < len(diverse_elements)
    
    # Patterns should have lower confidence
    for pattern in result["new_patterns"]:
        assert pattern.confidence < 0.9  # Not too confident about diverse elements

@pytest.mark.asyncio
async def test_pattern_coherence(
    pattern_tracker,
    similar_elements
):
    """Test natural coherence calculation."""
    # Calculate coherence for similar elements
    coherence = pattern_tracker._calculate_group_coherence(similar_elements)
    assert coherence > 0.7  # Should be highly coherent
    
    # Calculate coherence for mixed elements
    mixed_elements = similar_elements[:1] + [
        StructuralElement(
            element_id="elem7",
            element_type="paragraph",
            content="Completely different content",
            metadata={"depth": 0},
            density=0.3,
            emergence_time=datetime.now()
        )
    ]
    mixed_coherence = pattern_tracker._calculate_group_coherence(mixed_elements)
    assert mixed_coherence < coherence  # Should be less coherent
