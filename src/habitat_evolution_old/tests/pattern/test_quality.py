"""Tests for pattern quality analysis."""

import pytest
from typing import Dict, Any, List
from datetime import datetime

from ...pattern.quality import (
    PatternQualityAnalyzer,
    SignalMetrics,
    FlowMetrics,
    PatternState
)

def create_test_pattern(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Create a test pattern with given metrics."""
    return {
        "type": "test",
        "metrics": metrics,
        "content": {"value": 42}
    }

def create_test_history(base_metrics: Dict[str, float],
                       variations: List[float]) -> List[Dict[str, Any]]:
    """Create test history with metric variations."""
    history = []
    for var in variations:
        metrics = {k: v * (1 + var) for k, v in base_metrics.items()}
        history.append(create_test_pattern(metrics))
    return history

class TestPatternQualityAnalyzer:
    """Tests for PatternQualityAnalyzer."""
    
    def test_analyze_strong_signal(self):
        """Test analysis of strong, clear pattern."""
        analyzer = PatternQualityAnalyzer()
        
        # Create pattern with strong metrics
        pattern = create_test_pattern({
            "coherence": 0.9,
            "stability": 0.9,
            "emergence_rate": 0.8,
            "energy_state": 0.8,
            "adaptation_rate": 0.2,
            "cross_pattern_flow": 0.7
        })
        
        # Create stable history
        history = create_test_history(pattern["metrics"], [
            -0.05, -0.02, 0.0, 0.02, 0.05
        ])
        
        # Analyze signal
        signal_metrics = analyzer.analyze_signal(pattern, history)
        
        assert signal_metrics.strength > 0.8  # Strong signal
        assert signal_metrics.noise_ratio < 0.2  # Low noise
        assert signal_metrics.persistence > 0.8  # High persistence
        assert signal_metrics.reproducibility > 0.8  # High reproducibility
    
    def test_analyze_noisy_signal(self):
        """Test analysis of noisy pattern."""
        analyzer = PatternQualityAnalyzer()
        
        # Create pattern with moderate metrics
        pattern = create_test_pattern({
            "coherence": 0.5,
            "stability": 0.4,
            "emergence_rate": 0.6,
            "energy_state": 0.5,
            "adaptation_rate": 0.7,
            "cross_pattern_flow": 0.4
        })
        
        # Create volatile history
        history = create_test_history(pattern["metrics"], [
            -0.4, 0.3, -0.2, 0.4, -0.3
        ])
        
        # Analyze signal
        signal_metrics = analyzer.analyze_signal(pattern, history)
        
        assert signal_metrics.strength < 0.6  # Moderate signal
        assert signal_metrics.noise_ratio > 0.6  # High noise
        assert signal_metrics.persistence < 0.4  # Low persistence
        assert signal_metrics.reproducibility < 0.5  # Low reproducibility
    
    def test_analyze_flow_dynamics(self):
        """Test analysis of pattern flow."""
        analyzer = PatternQualityAnalyzer()
        
        # Create pattern with flow metrics
        pattern = create_test_pattern({
            "cross_pattern_flow": 0.8,
            "adaptation_rate": 0.3,
            "energy_state": 0.6
        })
        
        # Create related patterns with varying energy
        related_patterns = [
            create_test_pattern({"energy_state": 0.4}),
            create_test_pattern({"energy_state": 0.8}),
            create_test_pattern({"energy_state": 0.7})
        ]
        
        # Analyze flow
        flow_metrics = analyzer.analyze_flow(pattern, related_patterns)
        
        assert flow_metrics.viscosity < 0.3  # Low resistance (high flow)
        assert 0.3 < flow_metrics.back_pressure < 0.7  # Moderate pressure
        assert flow_metrics.volume == 0.3  # 3 related patterns
        assert flow_metrics.current < 0  # Negative flow (adaptation < 0.5)
    
    def test_state_transitions(self):
        """Test pattern state determination."""
        analyzer = PatternQualityAnalyzer()
        
        # Test emerging pattern
        emerging_signal = SignalMetrics(
            strength=0.6,
            noise_ratio=0.3,
            persistence=0.2,
            reproducibility=0.4
        )
        emerging_flow = FlowMetrics(
            viscosity=0.3,
            back_pressure=0.2,
            volume=0.1,
            current=0.8
        )
        
        state = analyzer.determine_state(emerging_signal, emerging_flow)
        assert state == PatternState.EMERGING
        
        # Test stable pattern
        stable_signal = SignalMetrics(
            strength=0.8,
            noise_ratio=0.1,
            persistence=0.9,
            reproducibility=0.8
        )
        stable_flow = FlowMetrics(
            viscosity=0.2,
            back_pressure=0.3,
            volume=0.6,
            current=0.1
        )
        
        state = analyzer.determine_state(stable_signal, stable_flow)
        assert state == PatternState.STABLE
        
        # Test declining pattern
        declining_signal = SignalMetrics(
            strength=0.3,
            noise_ratio=0.6,
            persistence=0.4,
            reproducibility=0.3
        )
        declining_flow = FlowMetrics(
            viscosity=0.7,
            back_pressure=0.8,
            volume=0.2,
            current=-0.6
        )
        
        state = analyzer.determine_state(declining_signal, declining_flow)
        assert state == PatternState.DECLINING
    
    def test_dynamic_thresholds(self):
        """Test dynamic threshold adjustment."""
        analyzer = PatternQualityAnalyzer()
        
        # Initial thresholds
        initial_signal = analyzer._dynamic_signal_threshold
        initial_noise = analyzer._dynamic_noise_threshold
        
        # Process series of patterns
        for _ in range(10):
            pattern = create_test_pattern({
                "coherence": 0.7,
                "stability": 0.7,
                "emergence_rate": 0.7,
                "energy_state": 0.7
            })
            history = create_test_history(pattern["metrics"], [-0.1, 0, 0.1])
            analyzer.analyze_signal(pattern, history)
        
        # Thresholds should adapt
        assert analyzer._dynamic_signal_threshold != initial_signal
        assert analyzer._dynamic_noise_threshold != initial_noise
        
        # Should adapt towards pattern characteristics
        assert 0.5 < analyzer._dynamic_signal_threshold < 0.7
    
    def test_noise_filtering(self):
        """Test filtering of noise patterns."""
        analyzer = PatternQualityAnalyzer()
        
        # Create noisy pattern
        pattern = create_test_pattern({
            "coherence": 0.2,
            "stability": 0.1,
            "emergence_rate": 0.3,
            "energy_state": 0.2
        })
        
        # Create volatile history
        history = create_test_history(pattern["metrics"], [
            -0.8, 0.7, -0.6, 0.9, -0.7
        ])
        
        signal_metrics = analyzer.analyze_signal(pattern, history)
        flow_metrics = analyzer.analyze_flow(pattern, [])
        
        state = analyzer.determine_state(signal_metrics, flow_metrics)
        assert state == PatternState.NOISE
