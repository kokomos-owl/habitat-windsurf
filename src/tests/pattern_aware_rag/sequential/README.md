# Pattern-Aware RAG Sequential Tests

This directory contains tests for the sequential foundation of the Pattern-Aware RAG system. The tests ensure proper functionality of pattern extraction, state management, and prompt formation.

## Test Coverage

### 1. Pattern Processing (`test_pattern_processing.py`)
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

## Prompt Formation System

The prompt formation system follows a two-level validation pattern:

1. **Template Validation**
   - Validates template key existence
   - Ensures required template variables are present
   - Handles both basic and nested templates

2. **Variable Validation**
   - Checks for missing or empty required fields
   - Validates variable types and content
   - Ensures state completeness

### Error Handling
- Invalid template keys raise ValueError
- Missing required variables trigger appropriate errors
- Empty or invalid state components are caught early

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
