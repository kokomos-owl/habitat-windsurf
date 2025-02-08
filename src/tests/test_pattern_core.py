import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock, patch

from core.evolution.pattern_core import PatternCore
from core.types import (
    PatternEvidence,
    TemporalContext,
    UncertaintyMetrics,
    DensityMetrics,
    LearningWindow,
    PatternEvolutionMetrics
)

@pytest.fixture
def pattern_core():
    """Create a PatternCore instance with mocked dependencies."""
    timestamp_service = Mock()
    timestamp_service.get_timestamp.return_value = datetime.now()
    
    event_manager = Mock()
    version_service = Mock()
    version_service.get_version.return_value = "1.0.0"
    
    return PatternCore(
        timestamp_service=timestamp_service,
        event_manager=event_manager,
        version_service=version_service
    )

@pytest.fixture
def sample_pattern():
    """Create a sample pattern observation."""
    return {
        "id": str(uuid4()),
        "type": "test_pattern",
        "data": {"key": "value"},
        "metadata": {"source": "test"}
    }

@pytest.fixture
def learning_window():
    """Create a sample learning window."""
    return LearningWindow(
        window_id=str(uuid4()),
        start_time=datetime.now(),
        patterns=[],
        density_metrics=DensityMetrics(),
        coherence_level=0.75,
        viscosity_gradient=0.5
    )

class TestPatternCore:
    def test_create_learning_window(self, pattern_core):
        """Test learning window creation."""
        patterns = [str(uuid4()) for _ in range(3)]
        window_id = pattern_core.create_learning_window(patterns)
        
        assert window_id in pattern_core.active_windows
        window = pattern_core.active_windows[window_id]
        assert window.patterns == patterns
        assert isinstance(window.density_metrics, DensityMetrics)
        assert 0 <= window.coherence_level <= 1
        assert 0 <= window.viscosity_gradient <= 1

    def test_observe_pattern(self, pattern_core, sample_pattern):
        """Test pattern observation and tracking."""
        result = pattern_core.observe_pattern(
            sample_pattern["id"],
            sample_pattern
        )
        
        assert result["pattern_id"] == sample_pattern["id"]
        assert "evolution_metrics" in result
        assert "interface_metrics" in result
        assert "window_metrics" in result
        
        # Verify evidence chain
        assert sample_pattern["id"] in pattern_core.evidence_chains
        evidence = pattern_core.evidence_chains[sample_pattern["id"]][-1]
        assert evidence.pattern_type == sample_pattern["type"]
        assert evidence.source_data == sample_pattern

    def test_pattern_evolution_tracking(self, pattern_core, sample_pattern):
        """Test pattern evolution metrics calculation."""
        # Create initial observation
        window_id = pattern_core.create_learning_window([sample_pattern["id"]])
        result1 = pattern_core.observe_pattern(
            sample_pattern["id"],
            sample_pattern,
            window_id
        )
        
        # Create second observation
        sample_pattern["data"]["key"] = "updated"
        result2 = pattern_core.observe_pattern(
            sample_pattern["id"],
            sample_pattern,
            window_id
        )
        
        # Verify evolution tracking
        assert len(pattern_core.evidence_chains[sample_pattern["id"]]) == 2
        evolution = result2["evolution_metrics"]
        assert "gradient" in evolution
        assert "stability" in evolution
        assert "emergence_rate" in evolution

    def test_interface_strength(self, pattern_core, sample_pattern):
        """Test interface strength calculation."""
        window_id = pattern_core.create_learning_window([sample_pattern["id"]])
        pattern_core.observe_pattern(sample_pattern["id"], sample_pattern, window_id)
        
        metrics = pattern_core.assess_interface_strength(
            sample_pattern["id"],
            window_id
        )
        
        assert "recognition" in metrics
        assert "confidence" in metrics
        assert "stability" in metrics
        assert "alignment" in metrics
        assert "overall_strength" in metrics
        assert all(0 <= v <= 1 for v in metrics.values())

    def test_batch_processing(self, pattern_core):
        """Test batch pattern processing."""
        patterns = [
            {
                "id": str(uuid4()),
                "type": "test_pattern",
                "data": {"value": i}
            }
            for i in range(3)
        ]
        
        result = pattern_core.process_pattern_batch(patterns)
        
        assert result["processed_count"] == len(patterns)
        assert len(result["pattern_results"]) == len(patterns)
        assert "window_metrics" in result
        
        # Verify all patterns were processed
        for pattern in patterns:
            assert pattern["id"] in result["pattern_results"]

    def test_window_cleanup(self, pattern_core, sample_pattern):
        """Test inactive window cleanup."""
        # Create old window
        window_id = pattern_core.create_learning_window([sample_pattern["id"]])
        window = pattern_core.active_windows[window_id]
        window.start_time = datetime.now() - timedelta(minutes=61)
        
        # Create recent window
        recent_id = pattern_core.create_learning_window([str(uuid4())])
        
        removed = pattern_core.cleanup_inactive_windows(max_age_minutes=60)
        
        assert window_id in removed
        assert recent_id not in removed
        assert window_id not in pattern_core.active_windows
        assert recent_id in pattern_core.active_windows

    def test_pattern_history_pruning(self, pattern_core, sample_pattern):
        """Test pattern history pruning."""
        # Create multiple observations
        for _ in range(10):
            pattern_core.observe_pattern(sample_pattern["id"], sample_pattern)
        
        initial_size = len(pattern_core.pattern_history)
        removed = pattern_core.prune_pattern_history(max_entries=5)
        
        assert removed == initial_size - 5
        assert len(pattern_core.pattern_history) == 5
        
    def test_window_metrics_calculation(self, pattern_core, sample_pattern):
        """Test window metrics calculation."""
        window_id = pattern_core.create_learning_window([sample_pattern["id"]])
        pattern_core.observe_pattern(sample_pattern["id"], sample_pattern, window_id)
        
        metrics = pattern_core.calculate_window_metrics(window_id)
        
        assert "global_density" in metrics
        assert "local_density" in metrics
        assert "cross_domain_strength" in metrics
        assert "interface_recognition" in metrics
        assert "viscosity" in metrics
        assert "coherence_level" in metrics
        assert "viscosity_gradient" in metrics
