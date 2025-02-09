"""Tests for topology-based visualization components."""

import pytest
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from src.core.metrics.flow_metrics import MetricFlowManager, VectorFieldState

@dataclass
class VizTestCase:
    """Test case for visualization verification."""
    name: str
    field_params: Dict[str, float]
    expected_visuals: Dict[str, any]
    collapse_expected: bool

class TopologyTestHarness:
    """Test harness for topology visualization."""
    
    def __init__(self):
        self.flow_manager = MetricFlowManager()
        
    def generate_vector_field(self, 
                            center: Tuple[float, float],
                            strength: float,
                            field_type: str) -> List[VectorFieldState]:
        """Generate a predictable vector field."""
        states = []
        t = np.linspace(0, 2*np.pi, 20)
        
        if field_type == 'spiral':
            # Spiral source/sink
            for i in t:
                r = 0.1 * i
                x = center[0] + r * np.cos(i)
                y = center[1] + r * np.sin(i)
                
                states.append(VectorFieldState(
                    magnitude=strength * r,
                    direction=i,
                    divergence=strength if field_type == 'source' else -strength,
                    curl=0.3,
                    critical_points=[{
                        'type': 'source' if field_type == 'source' else 'attractor',
                        'position': center,
                        'strength': strength
                    }]
                ))
        
        elif field_type == 'saddle':
            # Saddle point
            for i in t:
                x = center[0] + np.cos(i)
                y = center[1] + np.sin(i)
                
                states.append(VectorFieldState(
                    magnitude=strength * np.sqrt(x*x + y*y),
                    direction=np.arctan2(y, x),
                    divergence=0,
                    curl=0,
                    critical_points=[{
                        'type': 'saddle',
                        'position': center,
                        'strength': strength
                    }]
                ))
                
        return states

    def create_test_cases(self) -> List[VizTestCase]:
        """Create standard test cases."""
        return [
            VizTestCase(
                name="stable_attractor",
                field_params={
                    'center': (0.5, 0.5),
                    'strength': 0.3,
                    'type': 'attractor'
                },
                expected_visuals={
                    'glyph': '◉',
                    'size_range': (8, 12),
                    'color': 'green',
                    'pulse': False
                },
                collapse_expected=False
            ),
            VizTestCase(
                name="strong_source",
                field_params={
                    'center': (0.5, 0.5),
                    'strength': 0.9,
                    'type': 'source'
                },
                expected_visuals={
                    'glyph': '◎',
                    'size_range': (16, 25),
                    'color': 'red',
                    'pulse': True
                },
                collapse_expected=True
            ),
            VizTestCase(
                name="saddle_point",
                field_params={
                    'center': (0.5, 0.5),
                    'strength': 0.5,
                    'type': 'saddle'
                },
                expected_visuals={
                    'glyph': '⊗',
                    'size_range': (12, 16),
                    'color': 'yellow',
                    'pulse': False
                },
                collapse_expected=False
            )
        ]

def test_critical_point_visualization(viz_test_harness):
    """Test visualization of critical points."""
    test_cases = viz_test_harness.create_test_cases()
    
    for case in test_cases:
        field_states = viz_test_harness.generate_vector_field(
            center=case.field_params['center'],
            strength=case.field_params['strength'],
            field_type=case.field_params['type']
        )
        
        # Verify visual properties
        critical_point = field_states[-1].critical_points[0]
        assert critical_point['type'] == case.field_params['type']
        
        # Size within expected range
        size = critical_point['strength'] * 20  # Assuming BASE_SIZE = 20
        assert case.expected_visuals['size_range'][0] <= size <= case.expected_visuals['size_range'][1]
        
        # Verify collapse warning
        if case.collapse_expected:
            assert field_states[-1].divergence > viz_test_harness.flow_manager.collapse_threshold
        else:
            assert field_states[-1].divergence <= viz_test_harness.flow_manager.collapse_threshold

def test_vector_field_animation(viz_test_harness):
    """Test vector field animation properties."""
    test_cases = viz_test_harness.create_test_cases()
    
    for case in test_cases:
        field_states = viz_test_harness.generate_vector_field(
            center=case.field_params['center'],
            strength=case.field_params['strength'],
            field_type=case.field_params['type']
        )
        
        # Verify animation properties
        final_state = field_states[-1]
        
        # Check arrow density
        arrow_density = final_state.magnitude * 100  # Assuming BASE_DENSITY = 100
        assert 20 <= arrow_density <= 200  # Reasonable range for visualization
        
        # Check rotation
        if case.field_params['type'] == 'spiral':
            assert abs(final_state.curl) > 0
        elif case.field_params['type'] == 'saddle':
            assert abs(final_state.curl) < 0.1

def test_collapse_warning_system(viz_test_harness):
    """Test collapse warning visualization."""
    test_cases = viz_test_harness.create_test_cases()
    
    for case in test_cases:
        field_states = viz_test_harness.generate_vector_field(
            center=case.field_params['center'],
            strength=case.field_params['strength'],
            field_type=case.field_params['type']
        )
        
        final_state = field_states[-1]
        
        if case.collapse_expected:
            # Verify warning properties
            assert final_state.divergence > viz_test_harness.flow_manager.collapse_threshold
            assert case.expected_visuals['pulse']
            
            # Recovery chance should be lower for stronger sources
            recovery_chance = 1.0 - (final_state.magnitude / 2)
            assert recovery_chance < 0.7 if case.field_params['strength'] > 0.7 else True
        else:
            assert final_state.divergence <= viz_test_harness.flow_manager.collapse_threshold
            assert not case.expected_visuals['pulse']
