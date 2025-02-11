# Pattern Evolution Testing Framework

This directory contains the testing framework for pattern evolution in the Habitat system. The tests validate field behavior, pattern propagation, and coherence detection using principles from physics, dynamics, and information theory.

## Core Components

### 1. Field Configuration
```python
@dataclass
class FieldConfig:
    field_size: int = 10
    propagation_speed: float = 1.0
    decay_rate: float = 0.1
    boundary_condition: str = 'periodic'  # or 'reflective', 'absorbing'
```

The field configuration manages:
- Field dimensions and properties
- Wave propagation characteristics
- Boundary conditions
- Energy conservation parameters

### 2. Pattern Metrics
```python
@dataclass
class PatternMetrics:
    coherence: float = 0.0
    energy_state: float = 0.0
    emergence_rate: float = 0.0
```

Tracks key pattern properties:
- Coherence levels
- Energy states
- Emergence dynamics

### 3. Visualization Tools (`test_field_visualization.py`)
- Field state plots
- Flow field visualization
- Coherence landscapes
- Evolution metrics tracking

## Test Categories

### 1. Basic Field Tests (`minimal_test.py`)
- Field creation and initialization
- Pattern storage and retrieval
- Basic field-pattern interaction
- Metric preservation

### 2. Field Dynamics (`test_field_basics.py`)
- Pattern propagation
- Wave behavior
- Energy conservation
- Information flow

### 3. Edge Cases
- Singularities (threshold: 0.9)
- Chaos onset (threshold: 0.7)
- Bifurcation points (threshold: 0.5)
- Phase transitions (temperature: 1.0)

## Running Tests

### Basic Test
```bash
cd /path/to/pattern/tests
PYTHONPATH=/path/to/habitat-windsurf/src pytest minimal_test.py -v
```

### Visualization Test
```bash
pytest test_field_visualization.py -v
```

### All Tests
```bash
pytest . -v
```

## Key Learnings

1. Testing Infrastructure:
   - Modular test structure
   - Isolated test environments
   - Fixture-based configuration

2. Pattern Properties:
   - Measurable metrics
   - State persistence
   - Position accuracy

3. Field Behavior:
   - Disturbance propagation
   - Energy conservation
   - Boundary effects

4. Visualization Benefits:
   - Debug complex behaviors
   - Track evolution
   - Validate interactions

## Next Steps

1. Pattern Propagation:
   - Wave equation validation
   - Group velocity testing
   - Phase relationships

2. Interaction Testing:
   - Pattern collisions
   - Coherence interference
   - Energy exchange

3. Edge Case Validation:
   - Singularity handling
   - Chaos emergence
   - Phase transitions

4. Performance Testing:
   - Large field behavior
   - Many-pattern scenarios
   - Long-term evolution

## Contributing

When adding new tests:
1. Start with minimal test cases
2. Add appropriate visualizations
3. Document edge cases
4. Include performance considerations

## References

- Pattern Navigation Fields (../../../docs/theory/pattern_navigation_fields.md)
- Field Visualization Guide (test_field_visualization.py)
- Basic Field Tests (test_field_basics.py)
