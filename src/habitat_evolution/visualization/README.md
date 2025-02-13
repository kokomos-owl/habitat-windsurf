# Habitat Evolution Visualization Package

## Overview
This package provides visualization tools for the habitat evolution system, with a focus on climate pattern observation and analysis.

## Components

### 1. Test Visualization (`test_visualization.py`)
- Test-focused visualization toolset
- Neo4j export capabilities
- Climate hazard-specific visualizations
- Pattern evolution tracking

### 2. Field Visualization
- Pattern field state visualization
- Flow field visualization
- Coherence landscape plotting
- Hazard metrics visualization

## Usage

### Climate Pattern Visualization
```python
from habitat_evolution.visualization import TestVisualizationConfig, TestPatternVisualizer

# Initialize visualizer
config = TestVisualizationConfig()
visualizer = TestPatternVisualizer(config)

# Visualize climate patterns
fig, metrics = visualizer.visualize_climate_patterns(
    field,
    patterns,
    'precipitation'  # or 'drought', 'wildfire'
)

# Export results to Neo4j
visualizer.export_to_neo4j('test_name')
```

## Integration Points

### 1. Neo4j Integration
- Test results storage
- Pattern evolution tracking
- Relationship visualization
- Temporal analysis

### 2. Climate Risk Analysis
- Hazard zone visualization
- Risk intensity mapping
- Cross-hazard relationships
- Adaptation opportunity identification

## Testing
Tests are located in `tests/visualization/` and include:
- Pattern visualization tests
- Climate hazard visualization tests
- Neo4j export tests
- Evolution tracking tests
