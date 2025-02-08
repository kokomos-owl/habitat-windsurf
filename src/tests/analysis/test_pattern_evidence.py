"""Tests for pattern evidence handling in pattern emergence."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.core.analysis.pattern_emergence import (
    PatternEmergenceTracker,
    EmergentPattern
)
from src.core.types import (
    DensityMetrics,
    PatternEvolutionMetrics,
    PatternEvidence,
    TemporalContext,
    LearningWindow
)

@pytest.fixture
def pattern_tracker():
    """Create PatternEmergenceTracker with dependencies."""
    timestamp_service = Mock()
    timestamp_service.get_timestamp.return_value = datetime.now()
    
    temporal_core = Mock()
    
    return PatternEmergenceTracker(
        timestamp_service=timestamp_service,
        temporal_core=temporal_core
    )

@pytest.fixture
def pattern_evidence():
    """Create pattern evidence elements."""
    base_time = datetime.now()
    
    def create_evidence(idx, time_offset):
        return PatternEvidence(
            evidence_id=f"evidence{idx}",
            pattern_type="climate_risk",
            timestamp=base_time + timedelta(minutes=time_offset),
            source_data={"test_data": f"data{idx}"},
            evolution_metrics=PatternEvolutionMetrics(
                coherence_level=0.7 + (idx * 0.1),
                stability=0.6 + (idx * 0.1),
                emergence_rate=0.2
            ),
            density_metrics=DensityMetrics(
                local_density=0.8,
                global_density=0.6
            ),
            temporal_context=TemporalContext(
                start_time=base_time + timedelta(minutes=time_offset),
                learning_window=LearningWindow(
                    window_id=f"window{idx}",
                    start_time=base_time + timedelta(minutes=time_offset),
                    patterns=[],
                    density_metrics=DensityMetrics(
                        local_density=0.8,
                        global_density=0.6
                    ),
                    coherence_level=0.7,
                    viscosity_gradient=0.2
                )
            )
        )
    
    return [create_evidence(i, i*60) for i in range(3)]

@pytest.mark.asyncio
async def test_pattern_evidence_handling(
    pattern_tracker,
    pattern_evidence
):
    """Test handling of PatternEvidence elements."""
    # Test pattern discovery
    timestamp = datetime.now()
    patterns = await pattern_tracker._discover_patterns(pattern_evidence, timestamp)
    
    assert len(patterns) > 0, "Should discover patterns from evidence"
    pattern = patterns[0]
    
    # Test pattern type
    assert pattern.pattern_type == "climate_risk", "Pattern should match evidence type"
    
    # Test evolution metrics in metadata
    assert "coherence" in pattern.metadata, "Pattern should have coherence metadata"
    assert "stability" in pattern.metadata, "Pattern should have stability metadata"
    assert pattern.metadata["coherence"] > 0.7, "Pattern should have high coherence"
    assert pattern.metadata["stability"] > 0.6, "Pattern should have good stability"
    
    # Test element fit calculation
    new_evidence = PatternEvidence(
        evidence_id="test_evidence",
        pattern_type="climate_risk",
        timestamp=timestamp,
        source_data={"test_data": "new_data"},
        evolution_metrics=PatternEvolutionMetrics(
            coherence_level=0.8,
            stability=0.7,
            emergence_rate=0.2
        ),
        density_metrics=DensityMetrics(
            local_density=0.8,
            global_density=0.6
        ),
        temporal_context=TemporalContext(
            start_time=timestamp,
            learning_window=LearningWindow(
                window_id="test_window",
                start_time=timestamp,
                patterns=[],
                density_metrics=DensityMetrics(
                    local_density=0.8,
                    global_density=0.6
                ),
                coherence_level=0.7,
                viscosity_gradient=0.2
            )
        )
    )
    
    fit_score = pattern_tracker._calculate_element_fit(new_evidence, pattern)
    assert fit_score > 0.7, "Similar evidence should have high fit score"
    
    # Test pattern stability calculation
    stability = pattern_tracker._calculate_pattern_stability(pattern)
    assert stability > 0.6, "Pattern should maintain good stability"

@pytest.mark.asyncio
async def test_mixed_element_handling(
    pattern_tracker,
    pattern_evidence
):
    """Test handling of mixed element types."""
    # Create a mixed list of elements
    timestamp = datetime.now()
    mixed_elements = pattern_evidence[:1] + [
        PatternEvidence(
            evidence_id="different_evidence",
            pattern_type="flood_risk",  # Different type
            timestamp=timestamp,
            source_data={"test_data": "different_data"},
            evolution_metrics=PatternEvolutionMetrics(
                coherence_level=0.4,  # Lower coherence
                stability=0.3,  # Lower stability
                emergence_rate=0.5
            ),
            density_metrics=DensityMetrics(
                local_density=0.4,
                global_density=0.3
            ),
            temporal_context=TemporalContext(
                start_time=timestamp,
                learning_window=LearningWindow(
                    window_id="different_window",
                    start_time=timestamp,
                    patterns=[],
                    density_metrics=DensityMetrics(
                        local_density=0.4,
                        global_density=0.3
                    ),
                    coherence_level=0.4,
                    viscosity_gradient=0.5
                )
            )
        )
    ]
    
    # Test pattern discovery with mixed elements
    patterns = await pattern_tracker._discover_patterns(mixed_elements, timestamp)
    
    # Should find fewer patterns due to type differences
    assert len(patterns) <= 1, "Should not group different pattern types"
    
    if patterns:
        pattern = patterns[0]
        # Test coherence calculation
        coherence = pattern_tracker._calculate_group_coherence(pattern.elements)
        assert coherence < 0.7, "Mixed elements should have lower coherence"
        
        # Test stability calculation
        stability = pattern_tracker._calculate_pattern_stability(pattern)
        assert stability < 0.7, "Mixed elements should have lower stability"
