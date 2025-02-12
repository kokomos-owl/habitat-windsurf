"""Tests for the gradient-based pattern regulation system."""

import math
import pytest
import asyncio
from typing import Dict, Any, List

from habitat_evolution.core.pattern.quality import PatternQualityAnalyzer, FlowMetrics
from habitat_evolution.core.pattern.evolution import PatternEvolutionManager

class TestGradientRegulation:
    """Test suite for gradient-based pattern regulation."""
    
    @pytest.fixture
    def quality_analyzer(self):
        """Create a PatternQualityAnalyzer instance."""
        return PatternQualityAnalyzer(
            signal_threshold=0.3,
            noise_threshold=0.7,
            persistence_window=10
        )
    
    @pytest.fixture
    def evolution_manager(self):
        """Create a PatternEvolutionManager instance."""
        return PatternEvolutionManager()

    def create_test_pattern(self, coherence: float = 0.5,
                         energy: float = 0.5,
                         turbulence: float = 0.0) -> Dict[str, Any]:
        """Create a test pattern with specified characteristics."""
        return {
            "id": "test_pattern",
            "metrics": {
                "coherence": coherence,
                "energy_state": energy,
                "stability": 0.5,
                "emergence_rate": 0.5,
                "cross_pattern_flow": 0.5
            },
            "context": {
                "field_gradients": {
                    "turbulence": turbulence,
                    "coherence": coherence,
                    "energy": energy,
                    "density": 1.0
                }
            },
            "history": []
        }

    def test_turbulence_impact_on_viscosity(self, quality_analyzer):
        """Test how turbulence affects pattern viscosity."""
        # Test incoherent pattern
        low_coherence = self.create_test_pattern(
            coherence=0.2, energy=0.5, turbulence=0.5
        )
        flow_metrics = quality_analyzer.analyze_flow(low_coherence, [])
        assert flow_metrics.viscosity > 0.8, "High turbulence should increase viscosity"

        # Test coherent pattern
        high_coherence = self.create_test_pattern(
            coherence=0.8, energy=0.5, turbulence=0.5
        )
        flow_metrics = quality_analyzer.analyze_flow(high_coherence, [])
        assert flow_metrics.viscosity < 0.5, "Coherent patterns should resist turbulence"

    def test_density_impact_on_volume(self, quality_analyzer):
        """Test how density affects pattern volume."""
        # Create patterns with different densities
        base_pattern = self.create_test_pattern(coherence=0.7, energy=0.7)
        base_pattern["context"]["field_gradients"]["density"] = 0.5
        low_density = quality_analyzer.analyze_flow(base_pattern, [])

        base_pattern["context"]["field_gradients"]["density"] = 2.0
        high_density = quality_analyzer.analyze_flow(base_pattern, [])

        assert high_density.volume > low_density.volume
        assert high_density.back_pressure > low_density.back_pressure

    def test_gradient_based_flow(self, quality_analyzer):
        """Test how gradients affect pattern flow."""
        # Create pattern with strong gradient
        strong_gradient = self.create_test_pattern(coherence=0.8)
        strong_gradient["context"]["field_gradients"]["coherence"] = 0.4
        strong_gradient["context"]["field_gradients"]["energy"] = 0.3
        
        flow_strong = quality_analyzer.analyze_flow(strong_gradient, [])
        
        # Create pattern with weak gradient
        weak_gradient = self.create_test_pattern(coherence=0.8)
        weak_gradient["context"]["field_gradients"]["coherence"] = 0.7
        weak_gradient["context"]["field_gradients"]["energy"] = 0.7
        
        flow_weak = quality_analyzer.analyze_flow(weak_gradient, [])
        
        assert abs(flow_strong.current) > abs(flow_weak.current)

    def test_incoherent_pattern_dissipation(self, quality_analyzer):
        """Test that incoherent patterns dissipate properly."""
        # Create incoherent pattern with high turbulence
        pattern = self.create_test_pattern(
            coherence=0.2,
            energy=0.3,
            turbulence=0.8
        )
        
        flow_metrics = quality_analyzer.analyze_flow(pattern, [])
        
        # Verify dissipation properties
        assert flow_metrics.current < -1.5, "Should have strong negative flow"
        assert flow_metrics.viscosity > 0.9, "Should have high viscosity"
        assert flow_metrics.volume < 0.3, "Should have reduced volume"

    def test_coherent_pattern_stability(self, quality_analyzer):
        """Test that coherent patterns remain stable."""
        # Create coherent pattern with moderate turbulence
        pattern = self.create_test_pattern(
            coherence=0.9,
            energy=0.8,
            turbulence=0.4
        )
        
        flow_metrics = quality_analyzer.analyze_flow(pattern, [])
        
        # Verify stability properties
        assert flow_metrics.viscosity < 0.4, "Should maintain low viscosity"
        assert flow_metrics.volume > 0.6, "Should maintain significant volume"
        assert abs(flow_metrics.current) < 1.0, "Should have moderate flow"

    def test_adaptive_regulation(self, quality_analyzer):
        """Test the adaptive regulation mechanism."""
        patterns = []
        flows = []
        
        # Create pattern with increasing turbulence
        for turbulence in [0.0, 0.2, 0.4, 0.6, 0.8]:
            pattern = self.create_test_pattern(
                coherence=0.5,
                energy=0.5,
                turbulence=turbulence
            )
            patterns.append(pattern)
            flows.append(quality_analyzer.analyze_flow(pattern, []))
        
        # Verify adaptive response
        viscosities = [f.viscosity for f in flows]
        volumes = [f.volume for f in flows]
        currents = [abs(f.current) for f in flows]
        
        # Check for monotonic changes
        assert all(v1 <= v2 for v1, v2 in zip(viscosities, viscosities[1:]))
        assert all(v1 >= v2 for v1, v2 in zip(volumes, volumes[1:]))
        assert all(c1 >= c2 for c1, c2 in zip(currents, currents[1:]))
