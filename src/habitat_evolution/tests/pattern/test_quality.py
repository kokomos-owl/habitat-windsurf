"""Tests for pattern quality analysis."""

import pytest
import math
from typing import Dict, Any, List
from datetime import datetime

from habitat_evolution.core.pattern.quality import (
    PatternQualityAnalyzer,
    SignalMetrics,
    FlowMetrics,
    PatternState
)

def create_test_pattern(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Create a test pattern with given metrics."""
    # Default metrics with field context
    default_metrics = {
        "coherence": 0.5,
        "stability": 0.5,
        "emergence_rate": 0.5,
        "energy_state": 0.5,
        "adaptation_rate": 0.5,
        "cross_pattern_flow": 0.5
    }
    
    # Default field context
    default_context = {
        "field_gradients": {
            "coherence": 0.5,
            "energy": 0.5,
            "density": 1.0,
            "turbulence": 0.1
        },
        "initial_strength": 0.5,
        "phase": 0.0,
        "wavelength": 2 * math.pi
    }
    
    # Update metrics with provided values
    default_metrics.update(metrics)
    
    return {
        "type": "test",
        "metrics": default_metrics,
        "context": default_context,
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
    
    @pytest.mark.asyncio
    async def test_analyze_flow_dynamics(self):
        """Observe pattern flow dynamics and their co-process relationships.
        
        This test explores how patterns naturally establish flow relationships
        through:
        1. Cross-pattern energy gradients
        2. Field-driven regulation
        3. Emergent viscosity states
        4. Natural flow networks
        """
        analyzer = PatternQualityAnalyzer()
        
        # Create a network of related patterns with varying energy states
        patterns = [
            create_test_pattern({
                "cross_pattern_flow": 0.8,  # High flow potential
                "adaptation_rate": 0.3,     # Moderate adaptation
                "energy_state": 0.6,        # Medium-high energy
                "coherence": 0.7           # Strong coherence
            }),
            create_test_pattern({
                "energy_state": 0.4,        # Lower energy
                "cross_pattern_flow": 0.6,  # Moderate flow
                "coherence": 0.5           # Moderate coherence
            }),
            create_test_pattern({
                "energy_state": 0.8,        # Higher energy
                "cross_pattern_flow": 0.7,  # High flow
                "coherence": 0.8           # High coherence
            }),
            create_test_pattern({
                "energy_state": 0.7,        # Medium-high energy
                "cross_pattern_flow": 0.5,  # Moderate flow
                "coherence": 0.6           # Moderate coherence
            })
        ]
        
        # Observe initial flow conditions
        flow_metrics = analyzer.analyze_flow(patterns[0], patterns[1:])
        
        print("\n=== Flow Network Analysis ===")
        print(f"Primary Pattern:")
        print(f"  Flow Potential: {patterns[0]['metrics']['cross_pattern_flow']}")
        print(f"  Adaptation Rate: {patterns[0]['metrics']['adaptation_rate']}")
        print(f"  Energy State: {patterns[0]['metrics']['energy_state']}")
        
        print("\nRelated Pattern Energy States:")
        for i, p in enumerate(patterns[1:], 1):
            print(f"  Pattern {i}: {p['metrics']['energy_state']}")
        
        print("\nEmergent Flow Metrics:")
        print(f"  Viscosity: {flow_metrics.viscosity}")
        print(f"  Back Pressure: {flow_metrics.back_pressure}")
        print(f"  Volume: {flow_metrics.volume}")
        print(f"  Current: {flow_metrics.current}")
        
        # Instead of asserting specific values, observe relationships
        insights = {
            "viscosity_state": "high" if flow_metrics.viscosity > 0.7 else "medium" if flow_metrics.viscosity > 0.3 else "low",
            "pressure_balance": flow_metrics.back_pressure / patterns[0]['metrics']['energy_state'],
            "flow_efficiency": abs(flow_metrics.current) / flow_metrics.viscosity if flow_metrics.viscosity > 0 else float('inf'),
            "volume_utilization": flow_metrics.volume / len(patterns)
        }
        
        print("\nNetwork Insights:")
        print(f"  Viscosity State: {insights['viscosity_state']}")
        print(f"  Pressure Balance: {insights['pressure_balance']:.2f}")
        print(f"  Flow Efficiency: {insights['flow_efficiency']:.2f}")
        print(f"  Volume Utilization: {insights['volume_utilization']:.2f}")
        
        # Document emergent conditions that could enable co-processes
        if insights['viscosity_state'] == "high":
            print("\nPotential Co-Processes:")
            print("1. High viscosity suggests strong pattern coupling")
            print("2. Consider energy gradient redistribution")
            print("3. Look for natural flow channels in the network")
            print("4. Observe temporal evolution of viscosity state")
        
        # Record the conditions for future analysis
        return {
            "flow_metrics": flow_metrics,
            "insights": insights,
            "pattern_network": patterns
        }
    
    @pytest.mark.asyncio
    async def test_observe_flow_dynamics(self):
        """Observe pattern flow dynamics and their co-process relationships.
        
        This test explores how patterns naturally establish flow relationships
        through:
        1. Cross-pattern energy gradients
        2. Field-driven regulation
        3. Emergent viscosity states
        4. Natural flow networks
        """
        analyzer = PatternQualityAnalyzer()
        
        # Create a network of related patterns with varying energy states
        patterns = [
            create_test_pattern({
                "cross_pattern_flow": 0.8,  # High flow potential
                "adaptation_rate": 0.3,     # Moderate adaptation
                "energy_state": 0.6         # Medium-high energy
            }),
            create_test_pattern({"energy_state": 0.4}),  # Lower energy
            create_test_pattern({"energy_state": 0.8}),  # Higher energy
            create_test_pattern({"energy_state": 0.7})   # Medium-high energy
        ]
        
        # Observe initial flow conditions
        flow_metrics = analyzer.analyze_flow(patterns[0], patterns[1:])
        
        print("\n=== Flow Network Analysis ===")
        print(f"Primary Pattern:")
        print(f"  Flow Potential: {patterns[0]['metrics']['cross_pattern_flow']}")
        print(f"  Adaptation Rate: {patterns[0]['metrics']['adaptation_rate']}")
        print(f"  Energy State: {patterns[0]['metrics']['energy_state']}")
        
        print("\nRelated Pattern Energy States:")
        for i, p in enumerate(patterns[1:], 1):
            print(f"  Pattern {i}: {p['metrics']['energy_state']}")
        
        print("\nEmergent Flow Metrics:")
        print(f"  Viscosity: {flow_metrics.viscosity}")
        print(f"  Back Pressure: {flow_metrics.back_pressure}")
        print(f"  Volume: {flow_metrics.volume}")
        print(f"  Current: {flow_metrics.current}")
        
        # Instead of asserting specific values, observe relationships
        insights = {
            "viscosity_state": "high" if flow_metrics.viscosity > 0.7 else "medium" if flow_metrics.viscosity > 0.3 else "low",
            "pressure_balance": flow_metrics.back_pressure / patterns[0]['metrics']['energy_state'],
            "flow_efficiency": abs(flow_metrics.current) / flow_metrics.viscosity if flow_metrics.viscosity > 0 else float('inf'),
            "volume_utilization": flow_metrics.volume / len(patterns)
        }
        
        print("\nNetwork Insights:")
        print(f"  Viscosity State: {insights['viscosity_state']}")
        print(f"  Pressure Balance: {insights['pressure_balance']:.2f}")
        print(f"  Flow Efficiency: {insights['flow_efficiency']:.2f}")
        print(f"  Volume Utilization: {insights['volume_utilization']:.2f}")
        
        # Document emergent conditions that could enable co-processes
        if insights['viscosity_state'] == "high":
            print("\nPotential Co-Processes:")
            print("1. High viscosity suggests strong pattern coupling")
            print("2. Consider energy gradient redistribution")
            print("3. Look for natural flow channels in the network")
            print("4. Observe temporal evolution of viscosity state")
        
        # Record the conditions for future analysis
        return {
            "flow_metrics": flow_metrics,
            "insights": insights,
            "pattern_network": patterns
        }

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
