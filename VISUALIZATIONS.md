# Flow Pattern Visualization Guide

## Vector Space Visualization Principles

Habitat's visualizations now leverage the dimensionalized vector space architecture through:

1. Vector Field Visualization
   - Streamplot visualization of pattern flow
   - Critical point identification
   - Vector field topology mapping
   - Pattern trajectory tracking

2. Multi-dimensional Display
   - Interactive 3D pattern space
   - Coherence matrix heatmaps
   - Emergence potential gradients
   - Pattern velocity vectors

3. Network Dynamics
   - Force-directed layouts
   - Flow-based edge bundling
   - Natural pattern clustering
   - Emergence visualization

4. Interactive Elements
   - Real-time flow control
   - Pressure point adjustment
   - Flow path exploration
   - Pattern evolution tracking

**Document Date**: 2025-02-08T11:07:18-05:00

## Overview

The habitat-windsurf visualization system provides interactive, real-time visualization of flow patterns and their evolution. This guide explains how to work with and extend the visualization components.

## Core Components

### 1. Flow Visualizer

The `FlowVisualizer` class (`src/visualization/core/flow_visualizer.py`) is the main visualization engine:

```python
from src.visualization.core.flow_visualizer import FlowVisualizer

visualizer = FlowVisualizer()
result = await visualizer.visualize_flow(flow_data)
```

#### Visualization Output Format

```python
{
    "nodes": [
        {
            "id": "pattern_1",
            "label": "Pattern 1",
            "strength": 0.85,
            "coherence": 0.92,
            "stage": "stable"
        }
    ],
    "edges": [
        {
            "source": "pattern_1",
            "target": "pattern_2",
            "weight": 0.75,
            "type": "evolution"
        }
    ],
    "metrics": {
        "flow_velocity": 0.23,
        "pattern_density": 0.85,
        "emergence_rate": 0.45
    },
    "visualization": {
        # Plotly figure dict
        "data": [...],
        "layout": {...}
    }
}
```

### 2. WebSocket Integration

Real-time updates are handled through WebSocket connections (`src/visualization/static/js/websocket.js`):

```javascript
const ws = new PatternWebSocket('ws://localhost:8000/ws');

ws.onPatternUpdate = (data) => {
    // Update visualization with new pattern data
    updateVisualization(data);
};
```

## Visualization Features

### 1. Interactive Graph

- **Node Colors**: Based on coherence values (0.0-1.0)
  - Red: Low coherence (0.0-0.3)
  - Yellow: Medium coherence (0.3-0.7)
  - Green: High coherence (0.7-1.0)

- **Node Size**: Based on pattern strength
  - Minimum size: 10px
  - Maximum size: 30px

- **Edge Thickness**: Based on relationship weight
  - Thin: Weak relationship (0.0-0.3)
  - Medium: Moderate relationship (0.3-0.7)
  - Thick: Strong relationship (0.7-1.0)

### 2. Evolution Stages

Patterns progress through stages with enhanced metrics:
- "emerging": Early formation (DensityMetrics tracking)
- "evolving": Active development (Interface recognition)
- "stabilizing": Nearing stability (Cross-domain strength)
- "stable": Consistent pattern (Global density)

### 3. Metrics Visualization

- **Flow Velocity**: Rate of pattern change
- **Pattern Density**: Concentration of related patterns
- **Structure-Meaning**: Relationship strength

## Integration Examples

### 1. Basic Integration

```javascript
import { FlowGraph } from './flow_graph.js';

const graph = new FlowGraph('#visualization-container');

// Initial render
graph.render(initialData);

// Handle updates
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    graph.update(data);
};
```

### 2. Custom Styling

```javascript
const customConfig = {
    node: {
        colorScale: 'Viridis',
        minSize: 8,
        maxSize: 25
    },
    edge: {
        colorScale: 'Greys',
        minWidth: 1,
        maxWidth: 5
    }
};

graph.setConfig(customConfig);
```

### 3. Event Handling

```javascript
graph.on('nodeClick', (node) => {
    console.log('Selected pattern:', node.id);
    showPatternDetails(node);
});

graph.on('edgeClick', (edge) => {
    console.log('Selected relationship:', edge);
    showRelationshipDetails(edge);
});
```

## Performance Considerations

1. **Large Datasets**
   - Use `maxNodes` config to limit visible nodes
   - Enable `dynamicLoading` for large graphs
   - Set `updateThrottle` for smooth animations

2. **Real-time Updates**
   - Batch updates using `batchSize` config
   - Use `updateInterval` to control refresh rate
   - Enable `smoothTransitions` for better UX

3. **Memory Management**
   - Call `dispose()` when removing visualization
   - Use `clearCache()` periodically
   - Enable `autoCleanup` for long-running sessions

## Best Practices

1. **Initialization**
   - Always specify container dimensions
   - Set reasonable defaults for styling
   - Initialize WebSocket connection early

2. **Updates**
   - Batch multiple updates when possible
   - Use requestAnimationFrame for smooth transitions
   - Implement error handling for data updates

3. **Event Handling**
   - Debounce user interaction events
   - Implement loading states for async operations
   - Provide feedback for user interactions

## Common Patterns

### 1. Progressive Loading
```javascript
graph.enableProgressiveLoading({
    initialBatch: 50,
    batchSize: 20,
    interval: 100
});
```

### 2. Filtered Views
```javascript
graph.setFilter({
    minCoherence: 0.3,
    minStrength: 0.5,
    stage: ['stable', 'stabilizing']
});
```

### 3. Custom Layouts
```javascript
graph.setLayout({
    type: 'force',
    iterations: 300,
    linkDistance: 100,
    charge: -30
});
```

## Troubleshooting

1. **Visualization Not Updating**
   - Check WebSocket connection status
   - Verify data format matches schema
   - Check browser console for errors

2. **Performance Issues**
   - Reduce number of visible nodes
   - Increase update throttle interval
   - Disable animations for large graphs

3. **Layout Problems**
   - Adjust force layout parameters
   - Check for disconnected components
   - Verify edge weights are normalized

## API Reference

See [API Documentation](src/visualization/api/README.md) for detailed endpoint specifications and WebSocket protocol.
