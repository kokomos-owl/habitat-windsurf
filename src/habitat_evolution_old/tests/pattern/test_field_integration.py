"""
Integration tests for field-driven pattern evolution.
"""
import pytest
from typing import Dict, Any, List

from habitat_evolution.core.flow.gradient.controller import GradientFlowController, FieldGradients
from habitat_evolution.core.pattern.evolution import FieldDrivenPatternManager


class TestFieldIntegration:
    """Tests field-driven pattern integration."""
    
    @pytest.fixture
    def gradient_controller(self):
        return GradientFlowController()
    
    @pytest.fixture
    def pattern_manager(self):
        return FieldDrivenPatternManager()
    
    def create_test_field(self) -> Dict[str, Any]:
        """Creates a test field with known gradients."""
        return {
            'gradients': FieldGradients(
                coherence=0.7,
                energy=0.6,
                density=0.5,
                turbulence=0.2
            ),
            'patterns': []
        }
    
    def create_test_pattern(self, coherence: float = 0.8,
                          energy: float = 0.7) -> Dict[str, Any]:
        """Creates a test pattern with specified metrics."""
        return {
            'id': 'test_pattern',
            'coherence': coherence,
            'energy': energy,
            'state': 'ACTIVE',
            'relationships': []
        }
    
    async def test_pattern_field_coupling(self, pattern_manager):
        """Tests pattern-field coupling mechanics."""
        # Create test field and pattern
        field = self.create_test_field()
        pattern = self.create_test_pattern(coherence=0.4)
        
        # Evolve pattern in field
        evolved_pattern = await pattern_manager.evolve_patterns(
            field_state=field,
            patterns=[pattern]
        )
        
        # Verify pattern adapts to field
        assert len(evolved_pattern) == 1
        assert evolved_pattern[0]['coherence'] > pattern['coherence']
        assert evolved_pattern[0]['state'] == 'ACTIVE'
    
    async def test_gradient_flow_effects(self, gradient_controller):
        """Tests gradient effects on pattern flow."""
        # Create strong gradient field
        strong_gradients = FieldGradients(
            coherence=0.9,
            energy=0.8,
            density=0.6,
            turbulence=0.1
        )
        
        # Create test pattern
        pattern = self.create_test_pattern(coherence=0.5, energy=0.5)
        
        # Calculate flow metrics
        flow = gradient_controller.calculate_flow(
            gradients=strong_gradients,
            pattern=pattern,
            related_patterns=[]
        )
        
        # Verify flow responds to gradients
        assert flow.current > 0  # Flow towards higher coherence
        assert flow.viscosity > 0  # Non-zero resistance
        assert 0.2 <= flow.volume <= 1.0  # Volume in valid range
    
    async def test_turbulence_handling(self, gradient_controller):
        """Tests pattern behavior in turbulent fields."""
        # Create turbulent field
        turbulent_gradients = FieldGradients(
            coherence=0.6,
            energy=0.6,
            density=0.5,
            turbulence=0.8  # High turbulence
        )
        
        # Test coherent pattern
        coherent_pattern = self.create_test_pattern(coherence=0.8)
        coherent_flow = gradient_controller.calculate_flow(
            gradients=turbulent_gradients,
            pattern=coherent_pattern,
            related_patterns=[]
        )
        
        # Test incoherent pattern
        incoherent_pattern = self.create_test_pattern(coherence=0.2)
        incoherent_flow = gradient_controller.calculate_flow(
            gradients=turbulent_gradients,
            pattern=incoherent_pattern,
            related_patterns=[]
        )
        
        # Verify turbulence effects
        assert coherent_flow.viscosity > incoherent_flow.viscosity
        assert coherent_flow.volume > incoherent_flow.volume
        assert abs(coherent_flow.current) < abs(incoherent_flow.current)
