# Testing Guide for Pattern-Aware RAG

## Environment Setup

1. **Create Test Environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   ```bash
   # Copy example configuration
   cp test_env.example .env.test
   
   # Edit .env.test with your values
   # REQUIRED variables:
   # - CLAUDE_API_KEY
   # - NEO4J_PASSWORD
   ```

3. **Setup Test Infrastructure**
   ```bash
   # Create necessary directories
   mkdir -p .habitat/test_data
   mkdir -p .habitat/test_cache
   mkdir -p .habitat/embeddings
   ```

## Running Tests

### 1. Sequential Foundation Tests
These must run first and pass before concurrent tests:

```bash
pytest tests/test_pattern_processing.py -v
```

Verifies:
- Pattern extraction
- ID assignment
- Graph-ready state
- Provenance tracking

### 2. Coherence Interface Tests
Test core coherence mechanisms:

```bash
pytest tests/test_coherence.py -v
```

Verifies:
- State alignment
- Back pressure controls
- Learning windows

### 3. Integration Tests
Full system integration tests:

```bash
pytest tests/test_integration.py -v
```

Verifies:
- Complete state cycle
- Claude interaction
- Graph state management

### 4. All Tests
Run complete test suite:

```bash
pytest -v
```

## Test Categories

1. **Functional Tests**
   - Pattern processing
   - State transitions
   - Coherence mechanisms

2. **Integration Tests**
   - Full state cycle
   - External system interaction
   - Event coordination

3. **System Tests**
   - Learning windows
   - Back pressure
   - Stability control

## Common Issues

1. **Environment Variables**
   - Ensure all required variables are set
   - Check variable format
   - Verify API keys are valid

2. **Infrastructure**
   - Neo4j must be running
   - Required directories must exist
   - Proper permissions set

3. **State Management**
   - Clear test data between runs
   - Reset learning windows
   - Clean cache directories

## Test Data Management

1. **Before Tests**
   ```bash
   # Clear test data
   rm -rf .habitat/test_data/*
   rm -rf .habitat/test_cache/*
   ```

2. **After Tests**
   ```bash
   # Preserve important test artifacts
   cp -r .habitat/test_data/important_test .habitat/preserved/
   ```

## Debugging Tests

1. **Enable Debug Logging**
   ```bash
   export TEST_LOG_LEVEL=DEBUG
   pytest -v --log-cli-level=DEBUG
   ```

2. **Inspect State**
   ```bash
   # View current test state
   python -m habitat_evolution.debug.view_state
   ```

3. **Check Learning Windows**
   ```bash
   # View learning window status
   python -m habitat_evolution.debug.view_windows
   ```

## Best Practices

1. **Sequential Testing**
   - Run foundation tests first
   - Verify pattern processing
   - Check state readiness

2. **Coherence Testing**
   - Monitor state alignment
   - Verify back pressure
   - Check learning windows

3. **Integration Testing**
   - Test complete cycles
   - Verify all components
   - Check system stability

Remember: Pattern-Aware RAG is a coherence interface, not a traditional RAG system. Tests must verify coherence and state agreement, not just query-response functionality.
