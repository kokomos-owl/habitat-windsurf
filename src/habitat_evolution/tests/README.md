# Tests Package

Comprehensive test suite validating pattern evolution, coherence detection, and field dynamics.

## Test Structure

### Pattern Tests (`pattern/`)

#### Core Pattern Tests
1. **Evolution** (`test_evolution.py`)
   - Pattern lifecycle management
   - State transitions
   - Relationship tracking

2. **Quality** (`test_quality.py`)
   - Signal strength analysis
   - Coherence measurement
   - Flow metrics validation

3. **Pattern Dynamics** (`test_pattern_dynamics.py`)
   - Pattern behavior over time
   - Interaction effects
   - State evolution

#### Field Tests
1. **Field Basics** (`test_field_basics.py`)
   - Field initialization
   - Gradient calculations
   - Basic field operations

2. **Field Integration** (`test_field_integration.py`)
   - Field-pattern interactions
   - Multi-field dynamics
   - Complex field behaviors

#### Gradient Regulation (`test_gradient_regulation.py`)
Validates core pattern regulation functionality:
- Turbulence impact on viscosity
- Density effects on volume
- Gradient-based flow
- Pattern dissipation
- Coherence stability
- Adaptive regulation

### Service Tests (`services/`)

1. **Event Bus** (`test_event_bus.py`)
   - Event distribution
   - Handler registration
   - Event filtering

2. **Time Provider** (`test_time_provider.py`)
   - Time synchronization
   - Temporal operations
   - Time-based triggers

### Storage Tests (`storage/`)

1. **Memory Storage** (`test_memory_storage.py`)
   - Pattern persistence
   - Relationship storage
   - State management

## Running Tests

### Full Suite
```bash
PYTHONPATH=/path/to/src python -m pytest src/habitat_evolution/tests -v
```

### Pattern Tests
```bash
# Run all pattern tests
pytest src/habitat_evolution/tests/pattern -v

# Run specific pattern test
pytest src/habitat_evolution/tests/pattern/test_evolution.py -v
```

### Service Tests
```bash
# Run all service tests
pytest src/habitat_evolution/tests/services -v

# Run specific service test
pytest src/habitat_evolution/tests/services/test_event_bus.py -v
```

### Storage Tests
```bash
# Run storage tests
pytest src/habitat_evolution/tests/storage -v
```

## Test Design Principles

1. **Scientific Validation**
   - Wave mechanics verification
   - Field theory compliance
   - Information theory metrics
   - Flow dynamics accuracy

2. **Pattern Quality**
   - Signal strength validation
   - Noise ratio assessment
   - Persistence verification
   - Coherence measurement

3. **System Dynamics**
   - Gradient propagation
   - Turbulence modeling
   - Density regulation
   - Energy conservation

## Adding New Tests

When adding new tests:
1. Follow the existing pattern validation structure
2. Include both positive and negative test cases
3. Validate across multiple scientific domains
4. Ensure proper setup and teardown
5. Document test purpose and validation criteria
