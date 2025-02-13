"""Tests for the test-focused visualization toolset."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from habitat_evolution.visualization.test_visualization import (
    TestVisualizationConfig,
    TestPatternVisualizer
)

@pytest.fixture
def mock_field():
    """Create a mock field factory that generates fields for different time periods.
    
    Returns:
        callable: A function that takes a time_period argument and returns a field array
    """
    def field_generator(time_period='current'):
        """Generate a field for a specific time period.
        
        Args:
            time_period (str): One of 'current', 'mid_century', or 'late_century'
        """
        config = TestVisualizationConfig()
        size = 20
        field = np.zeros((size, size))
        
        # Create SW-NE gradient for drought (as per Martha's Vineyard data)
        x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
        sw_ne_gradient = (x + y) / 2.0  # SW (0,0) to NE (1,1) gradient
        
        # Get risk factors for the specified time period
        drought_risk = config.risk_factors['drought'][time_period]
        precip_risk = config.risk_factors['precipitation'][time_period]
        wildfire_risk = config.risk_factors['wildfire'][time_period]
        
        # Add climate-specific patterns
        # 1. Drought pattern with SW-NE gradient and temporal evolution
        field += drought_risk * sw_ne_gradient
        
        # 2. Precipitation patterns with temporal scaling
        base_precip = 0.734  # historical 100-year event baseline
        for center, strength in [((5, 5), base_precip), ((15, 15), base_precip)]:
            y, x = np.ogrid[-center[0]:size-center[0], -center[1]:size-center[1]]
            r2 = x*x + y*y
            field += (strength * precip_risk) * np.exp(-r2 / (2.0 * 3.0**2))
        
        # 3. Wildfire danger zones with temporal evolution
        field *= wildfire_risk
        
        return field
    
    return field_generator

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
    
    # Generate field for current time period
    field = mock_field('current')
    
    visualizer.capture_test_state('test1', field, mock_patterns, metrics)
    
    assert len(visualizer.test_results) == 1
    result = visualizer.test_results[0]
    assert result['test_name'] == 'test1'
    assert np.array_equal(result['field_state'], field)
    assert result['metrics'] == metrics

def test_climate_pattern_visualization(mock_field, mock_patterns):
    """Test climate pattern visualization generation."""
    visualizer = TestPatternVisualizer()
    
    # Create output directory if it doesn't exist
    import os
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Test visualization for each hazard type
    for hazard_type in ['precipitation', 'drought', 'wildfire']:
        # Generate field for current time period
        field = mock_field('current')
        
        fig, metrics = visualizer.visualize_climate_patterns(
            field, mock_patterns, hazard_type
        )
        
        assert fig is not None
        assert 'coherence' in metrics
        assert 'energy' in metrics
        assert 'above_threshold' in metrics
        assert 'max_intensity' in metrics
        
        # Save the figure
        output_path = os.path.join(output_dir, f'climate_pattern_{hazard_type}.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved visualization to: {output_path}')
        
        # Clean up
        plt.close(fig)

def test_invalid_hazard_type(mock_field, mock_patterns):
    """Test handling of invalid hazard type."""
    visualizer = TestPatternVisualizer()
    
    # Generate field for current time period
    field = mock_field('current')
    
    with pytest.raises(ValueError, match="Unsupported hazard type"):
        visualizer.visualize_climate_patterns(
            field, mock_patterns, 'invalid_hazard'
        )

def test_hazard_metrics_calculation(mock_field, mock_patterns):
    """Test calculation of hazard-specific metrics."""
    visualizer = TestPatternVisualizer()
    
    # Generate field for current time period
    field = mock_field('current')
    threshold = visualizer.config.hazard_thresholds['precipitation']
    
    metrics = visualizer._calculate_hazard_metrics(
        field, mock_patterns, threshold
    )
    
    assert 'coherence' in metrics
    assert 'energy' in metrics
    assert 'above_threshold' in metrics
    assert 'max_intensity' in metrics
    
    assert 0 <= metrics['coherence'] <= 1
    assert metrics['energy'] >= 0
    assert 0 <= metrics['above_threshold'] <= 1

@pytest.mark.asyncio
async def test_pattern_evolution_visualization(mock_patterns, request):
    """Test visualization of pattern evolution across time periods."""
    visualizer = TestPatternVisualizer()
    config = TestVisualizationConfig()
    
    # Create output directory if it doesn't exist
    import os
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get mock_field fixture
    mock_field = request.getfixturevalue('mock_field')
    
    # Generate fields for different time periods
    time_periods = ['current', 'mid_century', 'late_century']
    
    # Visualize each time period
    for i, period in enumerate(time_periods):
        # Generate field for this time period
        field = mock_field(period)
        
        # Create time-evolved patterns with hazard-specific scaling
        evolved_patterns = [
            {
                **pattern,
                'hazard_type': hazard_type,  # Assign a hazard type to each pattern
                'strength': pattern['metrics']['energy_state'] * config.risk_factors[hazard_type][period],
                'metrics': {
                    **pattern['metrics'],
                    'energy_state': pattern['metrics']['energy_state'] * config.risk_factors[hazard_type][period]
                }
            }
            for pattern, hazard_type in zip(mock_patterns, config.hazard_types)
        ]
        
        # Create visualization
        fig = visualizer.visualize_field_state(field, evolved_patterns)
        
        # Add time period annotation
        year = config.time_periods[period]
        plt.text(0.02, 0.98, 
                f'Time Period: {period}\nYear: {year}\n\nRisk Factors:\n' +
                '\n'.join(f'{h}: {config.risk_factors[h][period]:.2f}x' 
                          for h in config.hazard_types),
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        # Save evolution state
        output_path = f'{output_dir}/pattern_evolution_state_{i}.png'
        fig.savefig(output_path)
        print(f'Saved evolution state to: {output_path}')
        plt.close(fig)
        
        # Capture test state
        metrics = {
            'coherence': np.mean([p['metrics']['coherence'] for p in evolved_patterns]),
            'energy': np.mean([p['metrics']['energy_state'] for p in evolved_patterns]),
            'flow': np.mean([p['metrics']['flow_magnitude'] for p in evolved_patterns])
        }
        
        visualizer.capture_test_state(
            f'evolution_test_{period}',
            field,
            evolved_patterns,
            metrics
        )
    
    assert len(visualizer.test_results) == 3

@pytest.mark.asyncio
async def test_pattern_graph_visualization(mock_patterns, request):
    """Test transformation of field patterns into graph representation."""
    visualizer = TestPatternVisualizer()
    config = TestVisualizationConfig()
    
    # Get mock_field fixture
    mock_field = request.getfixturevalue('mock_field')
    field = mock_field('current')
    
    # Create graph nodes from patterns
    pattern_nodes = [
        {
            'id': f'pattern_{i}',
            'type': 'pattern',
            'position': pattern['position'].tolist(),
            'metrics': pattern['metrics'],
            'embedded_data': {
                'field_state': field[tuple(pattern['position'])].item(),
                'hazard_type': config.hazard_types[i % len(config.hazard_types)]
            }
        }
        for i, pattern in enumerate(mock_patterns)
    ]
    
    # Create edges based on pattern relationships
    pattern_edges = []
    for i, node1 in enumerate(pattern_nodes):
        for j, node2 in enumerate(pattern_nodes[i+1:], i+1):
            # Calculate spatial distance
            pos1 = np.array(node1['position'])
            pos2 = np.array(node2['position'])
            distance = np.linalg.norm(pos2 - pos1)
            
            # Calculate coherence relationship
            coherence_diff = abs(node1['metrics']['coherence'] - node2['metrics']['coherence'])
            
            # Add edge if patterns are related
            if distance < 15 and coherence_diff < 0.3:  # Threshold values
                pattern_edges.append({
                    'source': node1['id'],
                    'target': node2['id'],
                    'type': 'pattern_interaction',
                    'metrics': {
                        'spatial_distance': distance,
                        'coherence_similarity': 1 - coherence_diff,
                        'combined_strength': (node1['metrics']['energy_state'] + 
                                            node2['metrics']['energy_state']) / 2
                    }
                })
    
    # Create output directory if it doesn't exist
    import os
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize the graph representation
    fig = visualizer.visualize_pattern_graph(
        pattern_nodes,
        pattern_edges,
        field
    )
    
    # Add visualization metadata
    plt.title('Pattern Interaction Graph')
    plt.text(0.02, 0.98,
             'Node color: pattern coherence\n' +
             'Node size: energy state\n' +
             'Edge width: interaction strength\n' +
             f'Total patterns: {len(pattern_nodes)}\n' +
             f'Total interactions: {len(pattern_edges)}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the graph visualization
    output_path = os.path.join(output_dir, 'pattern_graph.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved pattern graph to: {output_path}')
    plt.close(fig)
    
    # Verify graph structure
    assert len(pattern_nodes) == len(mock_patterns)
    assert all('id' in node and 'embedded_data' in node for node in pattern_nodes)
    assert all('source' in edge and 'metrics' in edge for edge in pattern_edges)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
