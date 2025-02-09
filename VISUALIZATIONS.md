# Flow Pattern Visualization Guide

## Current Implementation State

Habitat's visualization system now implements town-based visualization with metric overlays and controlled pattern evolution:

### Town Visualization
1. **Boundary Display**
   - GeoJSON-based polygon overlays
   - Dynamic boundary generation
   - Interactive town selection

2. **Metric Visualization**
   - Color blending based on metrics:
     * Red: 40% coherence + 20% emergence rate
     * Green: 40% cross pattern flow + 20% social support
     * Blue: 40% emergence rate + 20% coherence
   - Interactive popups with detailed metrics
   - Heatmap layer for aggregate view

### Pattern Evolution

1. Pattern Evolution Server
   - WebSocket-based real-time updates
   - Controlled step-wise evolution
   - Pattern-specific characteristics
   - Gradient-based state changes

2. Pattern Types and Metrics
   - Precipitation patterns (high stability)
   - Drought patterns (high coherence)
   - Wildfire patterns (high energy)
   - Cross-pattern relationships

3. Evolution Controls
   - Initial state loading
   - Step-by-step evolution
   - Temporal state jumps (t=0, t=20, t=50, t=100)
   - Pattern reset capability

4. Status Tracking
   - Connection status
   - Data loading status
   - Evolution step status
   - Error reporting

**Document Date**: 2025-02-09T11:36:15-05:00

## Pattern Evolution Implementation

### 1. Data Structure
```json
{
    "patterns": [
        {
            "risk_type": "precipitation",
            "initial_metrics": {
                "stability": 0.85,
                "coherence": 0.75,
                "energy_state": 0.65
            },
            "nodes": [...],
            "links": [...]
        }
    ]
}
```

### 2. Evolution Characteristics

#### Precipitation Patterns
- High base stability (0.85)
- Moderate coherence (0.75)
- Moderate energy (0.65)
- Low noise scale (0.02)

#### Drought Patterns
- Moderate stability (0.65)
- High coherence (0.85)
- Low energy (0.45)
- Medium noise scale (0.03)

#### Wildfire Patterns
- Moderate stability (0.75)
- Moderate coherence (0.65)
- High energy (0.75)
- High noise scale (0.04)

### 3. State Transitions
- Controlled drift toward base values
- Time-scaled random fluctuations
- Pattern-specific evolution rates
- Bounded metric ranges

## Topology-Based Visual Language

### 1. Critical Point Visualization

```typescript
interface CriticalPoint {
    type: 'attractor' | 'source' | 'saddle'
    position: [number, number]
    strength: number
    field_state: VectorFieldState
}

class TopologyVisualizer {
    renderCriticalPoint(point: CriticalPoint) {
        const glyphMap = {
            attractor: '◉',  // Stable point
            source: '◎',     // Unstable point
            saddle: '⊗'      // Saddle point
        }
        
        return {
            glyph: glyphMap[point.type],
            size: BASE_SIZE * point.strength,
            color: this.getStabilityColor(point.field_state),
            pulseRate: point.type === 'source' ? 
                      PULSE_RATE * point.strength : 0
        }
    }
}
```

### 2. Vector Field Indicators

```typescript
interface FieldIndicator {
    magnitude: number
    direction: number
    divergence: number
    curl: number
}

class VectorFieldDisplay {
    updateFieldVisualization(field: FieldIndicator) {
        // Arrow density shows magnitude
        const arrowDensity = BASE_DENSITY * field.magnitude
        
        // Arrow rotation shows curl
        const rotation = field.curl * MAX_ROTATION
        
        // Color intensity shows divergence
        const intensity = this.mapDivergence(field.divergence)
        
        // Line thickness shows field strength
        const thickness = BASE_THICKNESS * 
            (1 + Math.abs(field.divergence))
            
        return {
            arrows: this.generateArrowField(arrowDensity),
            rotation: rotation,
            color: this.getFieldColor(intensity),
            lineWidth: thickness
        }
    }
}
```

### 3. Pattern State Indicators

```typescript
interface PatternState {
    stability: number
    coherence: number
    emergence_rate: number
    energy_state: number
}

class PatternStateVisualizer {
    updatePatternDisplay(state: PatternState) {
        // Node size shows stability
        const size = MIN_SIZE + 
            (MAX_SIZE - MIN_SIZE) * state.stability
        
        // Color saturation shows coherence
        const color = this.getCoherenceColor(state.coherence)
        
        // Glow effect shows emergence
        const glow = state.emergence_rate > 0.7 ? {
            color: EMERGENCE_COLOR,
            intensity: state.emergence_rate
        } : null
        
        // Pulse rate shows energy state
        const pulse = state.energy_state * MAX_PULSE_RATE
        
        return {
            nodeSize: size,
            nodeColor: color,
            glowEffect: glow,
            pulseRate: pulse
        }
    }
}
```

### 4. Collapse Warning System

```typescript
interface CollapseWarning {
    severity: number
    type: 'topology_based'
    recovery_chance: number
    field_state: VectorFieldState
}

class CollapseVisualizer {
    showCollapseWarning(warning: CollapseWarning) {
        // Warning intensity based on severity
        const intensity = warning.severity
        
        // Color based on recovery chance
        const color = this.getRecoveryColor(warning.recovery_chance)
        
        // Animation speed based on urgency
        const speed = BASE_SPEED * (1 + intensity)
        
        // Visual elements
        return {
            border: {
                style: 'dashed',
                width: 2 + (3 * intensity),
                color: color
            },
            warning: {
                icon: '⚠️',
                size: 16 + (8 * intensity),
                flash: intensity > 0.7
            },
            tooltip: {
                title: 'Pattern Collapse Warning',
                details: this.getWarningDetails(warning)
            }
        }
    }
}
```

### Benefits of Visual Language

1. **Immediate Pattern State Recognition**
   - Critical points show stability centers
   - Vector field shows flow dynamics
   - Color coding indicates health
   - Animations show energy levels

2. **Early Warning System**
   - Visual alerts for potential collapse
   - Clear recovery indicators
   - Progression tracking
   - Intervention points

3. **System-Wide Coherence**
   - Consistent visual language
   - Clear state transitions
   - Intuitive monitoring
   - Pattern relationship visibility

### Next Steps

1. Enhanced Document Processing
   - Implement streaming document ingestion
   - Add real-time pattern extraction
   - Enable incremental updates

2. Vector Space Integration
   - Add dynamic dimension weighting
   - Implement topology-based pattern matching
   - Enable real-time field analysis

3. Visualization Enhancements
   - Add interactive dimension controls
   - Implement pattern trajectory prediction
   - Create energy flow animations

4. System Integration
   - Implement WebSocket-based updates
   - Add real-time metric streaming
   - Enable collaborative visualization

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
