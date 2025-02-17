# Pattern-Aware RAG Sequential Tests

This directory contains tests for the sequential foundation of the Pattern-Aware RAG system. The tests ensure proper functionality of pattern extraction, state management, prompt formation, and learning window control.

## Test Coverage

### 1. Learning Window Control (`test_learning_window_control.py`)
- **Window Lifecycle Management**
  - State transitions: CLOSED → OPENING → OPEN → SATURATED
  - Semantic validation during initialization
  - Structural validation during operations
  - Window saturation handling

- **Stability and Back Pressure**
  - Delay calculation based on stability trends
  - Threshold-based pressure control
  - Proportional delay increases/decreases
  - Fine-grained stability gradients

- **Event Coordination**
  - Atomic event processing
  - State consistency maintenance
  - Change count tracking
  - Event queue management

- **State Management**
  - Two-level validation pattern
  - Atomic state updates with rollback
  - Event processing order
  - Saturation state maintenance

### 2. Pattern Processing (`test_pattern_processing.py`)
- **Initial State Loading**: Validates state loading and initialization
- **State Persistence**: Tests state persistence through transformations
- **Invalid State Handling**: Verifies error handling for invalid states
- **Prompt Formation**: Tests dynamic prompt construction and validation
  - Basic variable substitution
  - Nested variable substitution
  - Template validation
  - Error handling for invalid inputs
  - Context integration
- **Consensus Mechanism**: Tests state agreement processes
- **Pattern Extraction**: Validates pattern extraction from documents
- **Adaptive ID Assignment**: Tests ID assignment to patterns
- **Graph Ready State**: Ensures patterns reach graph-ready state
- **Sequential Dependency**: Verifies correct operation sequence
- **Provenance Tracking**: Tests provenance establishment and tracking

## Validation Architecture

The Pattern-Aware RAG system implements a two-level validation pattern across all components:

1. **Semantic Validation** (During Initialization)
   - Validates individual data elements
   - Runs immediately during object creation
   - Catches invalid data early
   - Examples:
     * Relations must have valid types and weights
     * Pattern confidence must exceed threshold
     * Window parameters must be valid

2. **Structural Validation** (During Operations)
   - Validates component relationships
   - Runs when needed during operations
   - Ensures state completeness
   - Examples:
     * Relations must reference valid nodes
     * Window state must be consistent
     * Event processing must maintain order

This pattern is applied consistently across:
- Learning Window Control
- Pattern Processing
- Event Coordination
- State Management

### State Management

1. **Atomic Operations**
   - State updates are atomic with rollback
   - Window transitions are protected
   - Event processing maintains consistency
   - Change counts are accurately tracked

2. **Back Pressure Control**
   - Delay calculation based on stability
   - Threshold-based pressure adjustment
   - Proportional response to trends
   - Fine-grained stability handling

3. **Error Handling**
   - Invalid states raise appropriate errors
   - State rollback on operation failure
   - Clear error messages for validation
   - Protection against race conditions

## Running Tests

```bash
# Run all sequential tests
pytest test_pattern_processing.py

# Run specific test class
pytest test_pattern_processing.py::TestSequentialFoundation

# Run specific test
pytest test_pattern_processing.py::TestSequentialFoundation::test_prompt_formation
```

## Related Documentation
- [TESTING.md](../../../../../TESTING.md): Main testing documentation
- [STATE.md](../../../../../STATE.md): System state documentation
- [Pattern Processor](../../../habitat_evolution/pattern_aware_rag/core/pattern_processor.py): Implementation details
- [Learning Control](../../../habitat_evolution/pattern_aware_rag/learning/learning_control.py): Learning window implementation

## Key Insights

1. **Natural System Modeling**
   - Back pressure mimics natural stress responses
   - Stability trends influence delay calculations
   - System develops "memory" of stress patterns
   - Response strengthens in areas of repeated stress

2. **State Consistency**
   - Two-level validation ensures data integrity
   - Atomic operations maintain consistency
   - Clear separation of concerns in validation
   - Proper handling of concurrent operations

3. **Testing Strategy**
   - Comprehensive lifecycle testing
   - Fine-grained stability validation
   - Concurrent operation verification
   - Edge case coverage for state transitions
