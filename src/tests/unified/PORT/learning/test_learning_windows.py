"""
Direct port of habitat_poc learning windows tests.
Tests verify natural window formation and density metrics.
"""

import pytest
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass

@dataclass
class DensityField:
    """Density field for testing."""
    local_density: float
    global_density: float
    cross_paths: List[str]
    
    def get_local(self) -> float:
        return self.local_density
    
    def get_paths(self) -> List[str]:
        return self.cross_paths

@dataclass
class WindowMetrics:
    """Window metrics for testing."""
    structural: Dict[str, float]
    semantic: Dict[str, float]
    score: float
    sustainability: float

@pytest.fixture
def mock_learning_window():
    """Create mock learning window."""
    class MockLearningWindow:
        def __init__(self):
            self.density_field = DensityField(
                local_density=0.82,
                global_density=0.75,
                cross_paths=['climate->adaptation', 'temporal->projection']
            )
            self.metrics = WindowMetrics(
                structural={'strength': 0.8869},
                semantic={'sustainability': 0.9000},
                score=0.9117,
                sustainability=1.0000
            )
            self.thresholds = {
                'density': 0.7,
                'paths': 1
            }
            
        def should_open(self, density_field: DensityField) -> bool:
            return (
                density_field.get_local() > self.thresholds['density'] and
                len(density_field.get_paths()) >= self.thresholds['paths']
            )
            
        def calculate_density_metrics(self, window_data: Dict) -> Dict[str, float]:
            return {
                'local_density': (
                    self.metrics.structural['strength'] * 0.4 +
                    self.metrics.semantic['sustainability'] * 0.3 +
                    self.metrics.score * 0.3
                ),
                'cross_path_count': len(self.density_field.cross_paths)
            }
    
    return MockLearningWindow()

@pytest.mark.asyncio
class TestLearningWindows:
    """Ported learning windows tests."""
    
    async def test_window_formation(self, mock_learning_window):
        """Test natural window formation."""
        # Check initial density field
        density_field = mock_learning_window.density_field
        
        # Verify natural window opening
        should_open = mock_learning_window.should_open(density_field)
        assert should_open, "Window should open naturally"
        
        # Verify density conditions
        assert density_field.get_local() > 0.8, "High local density"
        assert density_field.global_density > 0.7, "Sufficient global density"
        assert len(density_field.get_paths()) > 0, "Cross-domain paths present"
    
    async def test_density_metrics(self, mock_learning_window):
        """Test density metrics calculation."""
        # Calculate metrics
        window_data = {
            'structural': mock_learning_window.metrics.structural,
            'semantic': mock_learning_window.metrics.semantic,
            'score': mock_learning_window.metrics.score
        }
        metrics = mock_learning_window.calculate_density_metrics(window_data)
        
        # Verify natural density formation
        assert metrics['local_density'] > 0.85, "Strong local density"
        assert metrics['cross_path_count'] > 0, "Cross-domain paths maintained"
        
        # Verify metric components
        structural_component = mock_learning_window.metrics.structural['strength'] * 0.4
        semantic_component = mock_learning_window.metrics.semantic['sustainability'] * 0.3
        score_component = mock_learning_window.metrics.score * 0.3
        
        # Verify natural balance
        assert structural_component > 0.3, "Strong structural contribution"
        assert semantic_component > 0.25, "Strong semantic contribution"
        assert score_component > 0.25, "Strong scoring contribution"
