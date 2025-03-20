# Pattern Topology Framework

## Overview

The Pattern Topology Framework is designed to identify, document, and persist topological features of the semantic landscape in the pattern co-evolution system. This framework enables the system to not only exhibit topological features but to actively document, identify, and navigate them as physical properties within a semantic landscape.

## Core Concepts

### Frequency Domains

Frequency domains represent regions in the semantic landscape characterized by similar frequency signatures. Patterns within the same frequency domain tend to evolve at similar rates and exhibit similar harmonic properties.

**Properties:**
- **Dominant Frequency**: The primary frequency at which patterns in this domain evolve
- **Bandwidth**: The range of frequencies contained within the domain
- **Phase Coherence**: The degree to which patterns in this domain are phase-aligned
- **Center Coordinates**: The central position of the domain in the semantic space
- **Radius**: The approximate size/reach of the domain
- **Pattern IDs**: The patterns contained within this domain

### Boundaries

Boundaries represent transitions between frequency domains, where the harmonic properties of patterns change significantly.

**Properties:**
- **Sharpness**: How abrupt the transition is between domains
- **Permeability**: How easily patterns can cross between domains
- **Stability**: How consistent the boundary remains over time
- **Dimensionality**: The complexity of the boundary (e.g., 1D line, 2D surface)
- **Coordinates**: Points defining the boundary's location

### Resonance Points

Resonance points are locations in the semantic landscape where multiple patterns interact harmoniously, creating stable attractors.

**Properties:**
- **Strength**: The intensity of the resonance
- **Stability**: How resistant the resonance point is to perturbations
- **Attractor Radius**: How far the resonance point's influence extends
- **Contributing Patterns**: Which patterns contribute to the resonance and their relative contributions

### Field Metrics

Field metrics provide overall measurements of the semantic landscape's properties.

**Properties:**
- **Coherence**: The overall harmony of the field
- **Energy Density**: Distribution of energy across regions
- **Adaptation Rate**: How quickly the field adapts to changes
- **Homeostasis Index**: The field's ability to maintain stability
- **Entropy**: The level of disorder in the field

## Architecture

The Topology Framework consists of three main components:

1. **Models (`models.py`)**: Core data structures for representing topological features
2. **Detector (`detector.py`)**: Analyzes pattern histories to detect topological features
3. **Manager (`manager.py`)**: Manages detection, persistence, and retrieval of topology constructs

### Integration with Neo4j

The framework integrates with Neo4j for persisting topology states and querying historical data. This enables:

- Long-term storage of topology evolution
- Querying for historical patterns and trends
- Analysis of topology changes over time

### API Layer

The framework exposes its functionality through a RESTful API built with FastAPI, providing endpoints for:

- Analyzing patterns to generate topology states
- Retrieving current and historical topology states
- Comparing topology states to identify changes
- Accessing specific topological features (domains, boundaries, resonance points)
- Exporting and importing topology states

## Usage Examples

### Analyzing Patterns

```python
from habitat_evolution.pattern_aware_rag.topology.manager import TopologyManager
from habitat_evolution.pattern_aware_rag.learning.pattern_id import PatternID
from habitat_evolution.pattern_aware_rag.learning.learning_control import LearningWindow

# Create topology manager
manager = TopologyManager(persistence_mode=True)

# Get patterns and learning windows
patterns = [pattern_a, pattern_b, pattern_c]
windows = [window_a, window_b]

# Define time period for analysis
time_period = {
    "start": datetime.now() - timedelta(hours=1),
    "end": datetime.now()
}

# Analyze patterns to generate topology state
state = manager.analyze_patterns(patterns, windows, time_period)

# Access topology features
for domain_id, domain in state.frequency_domains.items():
    print(f"Domain {domain_id}: {domain.dominant_frequency} Hz")

for boundary_id, boundary in state.boundaries.items():
    print(f"Boundary {boundary_id}: Sharpness {boundary.sharpness}")

for point_id, point in state.resonance_points.items():
    print(f"Resonance Point {point_id}: Strength {point.strength}")

print(f"Field Coherence: {state.field_metrics.coherence}")
```

### Serializing and Deserializing Topology States

```python
# Serialize current state to JSON
json_str = manager.serialize_current_state()

# Save to file
with open("topology_state.json", "w") as f:
    f.write(json_str)

# Load from file
with open("topology_state.json", "r") as f:
    json_str = f.read()

# Create new manager and load state
new_manager = TopologyManager(persistence_mode=False)
loaded_state = new_manager.load_from_serialized(json_str)
```

### Comparing Topology States

```python
# Get difference between current state and a previous state
diff = manager.get_topology_diff("previous_state_id")

# Examine changes
print(f"Added domains: {diff['added_domains']}")
print(f"Removed domains: {diff['removed_domains']}")
print(f"Modified boundaries: {diff['modified_boundaries']}")
print(f"Field metrics changes: {diff['field_metrics_changes']}")
```

### Using the API

```python
import requests
import json
from datetime import datetime, timedelta

# Base URL for API
base_url = "http://localhost:8000"

# Analyze patterns
response = requests.post(
    f"{base_url}/topology/analyze",
    json={
        "pattern_ids": ["pattern-a", "pattern-b", "pattern-c"],
        "window_ids": ["window-a", "window-b"],
        "time_range": {
            "start": (datetime.now() - timedelta(hours=1)).isoformat(),
            "end": datetime.now().isoformat()
        }
    }
)
state = response.json()
state_id = state["id"]

# Get frequency domains
domains = requests.get(f"{base_url}/topology/frequency-domains?state_id={state_id}").json()

# Get boundaries
boundaries = requests.get(f"{base_url}/topology/boundaries?state_id={state_id}").json()

# Get field metrics
metrics = requests.get(f"{base_url}/topology/field-metrics?state_id={state_id}").json()

# Export state
export = requests.post(f"{base_url}/topology/export?state_id={state_id}").json()

# Import state
requests.post(f"{base_url}/topology/import", json=export)
```

## Integration with Pattern Co-Evolution

The Topology Framework integrates with the pattern co-evolution system through:

1. **Learning Windows**: Analyzing window frequencies and stability thresholds to identify frequency domains
2. **Pattern Evolution History**: Using pattern evolution events to detect harmonic properties
3. **Tonic-Harmonic Values**: Leveraging tonic and stability values to calculate field metrics

This integration creates a feedback loop where:
- Pattern evolution generates topological features
- Topological features influence future pattern evolution
- The system becomes increasingly self-aware of its own structure

## Testing

The framework includes comprehensive tests:

1. **Model Tests**: Verify data structures, serialization, and diff calculation
2. **Detector Tests**: Validate frequency domain detection, boundary detection, and field analysis
3. **Manager Tests**: Test persistence, state management, and integration with Neo4j
4. **Integration Tests**: Verify interaction with the pattern co-evolution system

## Future Directions

1. **Visualization**: Develop tools to visualize the semantic landscape and its topological features
2. **Predictive Analysis**: Use topology history to predict future pattern evolution
3. **Adaptive Navigation**: Enable intelligent navigation of the semantic landscape based on topological features
4. **Emergent Properties**: Identify higher-order emergent properties arising from topology evolution

## Conclusion

The Pattern Topology Framework provides a powerful mechanism for understanding and navigating the semantic landscape of the pattern co-evolution system. By making topological features explicit and navigable, it enables the system to become increasingly self-aware and self-organizing.
