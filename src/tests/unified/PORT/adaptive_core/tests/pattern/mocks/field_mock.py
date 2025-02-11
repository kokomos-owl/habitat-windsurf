"""Mock framework for field-driven pattern testing."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
from unittest.mock import MagicMock

@dataclass
class MockFieldState:
    """Mock field state for testing."""
    coherence_map: np.ndarray
    energy_map: np.ndarray
    density_map: np.ndarray
    turbulence_map: np.ndarray
    size: tuple = (100, 100)

    @classmethod
    def create_uniform(cls, coherence: float = 0.5, energy: float = 0.5,
                      density: float = 0.5, turbulence: float = 0.2,
                      size: tuple = (100, 100)) -> 'MockFieldState':
        """Create uniform field state."""
        return cls(
            coherence_map=np.full(size, coherence),
            energy_map=np.full(size, energy),
            density_map=np.full(size, density),
            turbulence_map=np.full(size, turbulence),
            size=size
        )
    
    @classmethod
    def create_gradient(cls, direction: str = 'horizontal',
                       size: tuple = (100, 100)) -> 'MockFieldState':
        """Create gradient field state."""
        x = np.linspace(0, 1, size[1])
        y = np.linspace(0, 1, size[0])
        if direction == 'horizontal':
            gradient = np.tile(x, (size[0], 1))
        else:  # vertical
            gradient = np.tile(y, (size[1], 1)).T
            
        return cls(
            coherence_map=gradient.copy(),
            energy_map=gradient[::-1, :].copy(),  # inverse gradient
            density_map=np.full(size, 0.5),
            turbulence_map=np.random.uniform(0, 0.3, size),
            size=size
        )
    
    @classmethod
    def create_turbulent(cls, turbulence_scale: float = 1.0,
                        size: tuple = (100, 100)) -> 'MockFieldState':
        """Create turbulent field state."""
        return cls(
            coherence_map=np.random.uniform(0, 1, size),
            energy_map=np.random.uniform(0, 1, size),
            density_map=np.random.uniform(0, 1, size),
            turbulence_map=np.random.uniform(0, turbulence_scale, size),
            size=size
        )

class MockPatternGenerator:
    """Generates mock patterns for testing."""
    
    @staticmethod
    def create_pattern(coherence: float = 0.8, energy: float = 0.7,
                      position: tuple = (50, 50)) -> Dict[str, Any]:
        """Create a single mock pattern."""
        return {
            'id': f'pattern_{np.random.randint(10000)}',
            'coherence': coherence,
            'energy': energy,
            'position': position,
            'state': 'ACTIVE',
            'relationships': []
        }
    
    @classmethod
    def create_pattern_cluster(cls, num_patterns: int = 5,
                             radius: float = 10.0,
                             center: tuple = (50, 50),
                             base_coherence: float = 0.8) -> List[Dict[str, Any]]:
        """Create a cluster of related patterns."""
        patterns = []
        for _ in range(num_patterns):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius)
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            coherence = base_coherence + np.random.uniform(-0.1, 0.1)
            patterns.append(cls.create_pattern(coherence=coherence,
                                            position=(int(x), int(y))))
        return patterns

class MockGradientFlowController:
    """Mock gradient flow controller for testing."""
    
    def __init__(self, field_state: Optional[MockFieldState] = None):
        self.field_state = field_state or MockFieldState.create_uniform()
        self.mock = MagicMock()
    
    def calculate_flow(self, pattern: Dict[str, Any],
                      related_patterns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate mock flow metrics."""
        x, y = pattern['position']
        coherence = self.field_state.coherence_map[y, x]
        energy = self.field_state.energy_map[y, x]
        turbulence = self.field_state.turbulence_map[y, x]
        
        # Record the call for verification
        self.mock.calculate_flow(pattern=pattern,
                               related_patterns=related_patterns)
        
        return {
            'viscosity': 0.3 + (turbulence * 0.5),
            'back_pressure': 0.2 * len(related_patterns),
            'volume': min(1.0, coherence * 0.6 + energy * 0.4),
            'current': (coherence - pattern['coherence']) * 2.0
        }

def create_stress_test_scenario(size: tuple = (1000, 1000),
                              num_patterns: int = 1000,
                              turbulence_scale: float = 2.0) -> tuple:
    """Create a large-scale stress test scenario."""
    field_state = MockFieldState.create_turbulent(
        turbulence_scale=turbulence_scale,
        size=size
    )
    
    patterns = []
    for _ in range(num_patterns):
        x = np.random.randint(0, size[1])
        y = np.random.randint(0, size[0])
        coherence = np.random.uniform(0, 1)
        patterns.append(MockPatternGenerator.create_pattern(
            coherence=coherence,
            position=(x, y)
        ))
    
    return field_state, patterns
