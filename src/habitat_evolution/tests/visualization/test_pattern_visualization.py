"""Tests for the test-focused visualization toolset."""

import pytest
import numpy as np
from habitat_evolution.visualization.test_visualization import (
    TestVisualizationConfig,
    TestPatternVisualizer
)

@pytest.fixture
def mock_field():
    """Create a mock field with known patterns."""
    size = 20
    field = np.zeros((size, size))
    
    # Add some Gaussian patterns
    for center, strength in [((5, 5), 1.0), ((15, 15), 0.8)]:
        y, x = np.ogrid[-center[0]:size-center[0], -center[1]:size-center[1]]
        r2 = x*x + y*y
        field += strength * np.exp(-r2 / (2.0 * 3.0**2))
    
    return field

@pytest.fixture
def mock_patterns():
    """Create mock pattern data."""
    return [
        {
            "position": np.array([5, 5]),
            "metrics": {
                "energy_state": 0.9,
                "coherence": 0.8,
                "flow_magnitude": 0.7
            }
        },
        {
            "position": np.array([15, 15]),
            "metrics": {
                "energy_state": 0.7,
                "coherence": 0.6,
                "flow_magnitude": 0.5
            }
        }
    ]

def test_visualization_config():
    """Test visualization configuration initialization."""
    config = TestVisualizationConfig()
    
    assert config.hazard_types == ['precipitation', 'drought', 'wildfire']
    assert config.hazard_thresholds['precipitation'] == 7.34
    assert config.hazard_thresholds['drought'] == 0.26
    assert config.hazard_thresholds['wildfire'] == 0.94

def test_pattern_visualizer_initialization():
    """Test pattern visualizer initialization."""
    visualizer = TestPatternVisualizer()
    assert len(visualizer.test_results) == 0
    assert visualizer.config is not None

def test_capture_test_state(mock_field, mock_patterns):
    """Test capturing test state."""
    visualizer = TestPatternVisualizer()
    metrics = {'coherence': 0.8, 'energy': 0.7}
    
    visualizer.capture_test_state('test1', mock_field, mock_patterns, metrics)
    
    assert len(visualizer.test_results) == 1
    result = visualizer.test_results[0]
    assert result['test_name'] == 'test1'
    assert np.array_equal(result['field_state'], mock_field)
    assert result['metrics'] == metrics

def test_climate_pattern_visualization(mock_field, mock_patterns):
    """Test climate pattern visualization generation."""
    visualizer = TestPatternVisualizer()
    
    # Test precipitation visualization
    fig, metrics = visualizer.visualize_climate_patterns(
        mock_field, mock_patterns, 'precipitation'
    )
    
    assert fig is not None
    assert 'coherence' in metrics
    assert 'energy' in metrics
    assert 'above_threshold' in metrics
    assert 'max_intensity' in metrics
    
    # Clean up
    plt.close(fig)

def test_invalid_hazard_type(mock_field, mock_patterns):
    """Test handling of invalid hazard type."""
    visualizer = TestPatternVisualizer()
    
    with pytest.raises(ValueError, match="Unsupported hazard type"):
        visualizer.visualize_climate_patterns(
            mock_field, mock_patterns, 'invalid_hazard'
        )

@pytest.mark.asyncio
async def test_pattern_evolution_visualization(mock_field, mock_patterns):
    """Test visualization of pattern evolution over time."""
    visualizer = TestPatternVisualizer()
    
    # Capture multiple states
    for i in range(3):
        # Modify field and patterns slightly for evolution
        evolved_field = mock_field * (1.0 - 0.1 * i)
        evolved_patterns = [
            {
                **pattern,
                'metrics': {
                    **pattern['metrics'],
                    'energy_state': pattern['metrics']['energy_state'] * (1.0 - 0.1 * i)
                }
            }
            for pattern in mock_patterns
        ]
        
        metrics = {
            'coherence': 0.8 - 0.1 * i,
            'energy': 0.7 - 0.1 * i
        }
        
        visualizer.capture_test_state(
            'evolution_test',
            evolved_field,
            evolved_patterns,
            metrics
        )
    
    # Generate visualization for precipitation
    fig, metrics = visualizer.visualize_climate_patterns(
        evolved_field, evolved_patterns, 'precipitation'
    )
    
    assert fig is not None
    assert len(visualizer.test_results) == 3
    
    # Clean up
    plt.close(fig)

def test_hazard_metrics_calculation(mock_field, mock_patterns):
    """Test calculation of hazard-specific metrics."""
    visualizer = TestPatternVisualizer()
    threshold = visualizer.config.hazard_thresholds['precipitation']
    
    metrics = visualizer._calculate_hazard_metrics(
        mock_field, mock_patterns, threshold
    )
    
    assert 'coherence' in metrics
    assert 'energy' in metrics
    assert 'above_threshold' in metrics
    assert 'max_intensity' in metrics
    
    assert 0 <= metrics['coherence'] <= 1
    assert 0 <= metrics['energy'] <= 1
    assert 0 <= metrics['above_threshold'] <= 1
    assert metrics['max_intensity'] >= 0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
