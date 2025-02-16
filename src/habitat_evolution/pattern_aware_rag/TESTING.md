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
pytest tests/test_coherence_interface.py -v
```

Verifies:
- State alignment
- Coherence scoring
- Pattern validation

### 3. Integration Tests
Run the full cycle integration tests:

```bash
pytest tests/pattern_aware_rag/integration/test_full_cycle.py -v
```

This test suite validates both sequential and concurrent operations:

#### Sequential Foundation Test
```python
async def test_full_state_cycle(self, pattern_processor, coherence_interface, 
                              state_stores, adaptive_bridge, sample_document):
    # 1. Sequential Foundation
    pattern = await pattern_processor.extract_pattern(sample_document)
    adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
    initial_state = await pattern_processor.prepare_graph_state(pattern, adaptive_id)
    
    # 2. Coherence Interface
    alignment = await coherence_interface.align_state(initial_state)
    assert alignment.coherence_score > 0.0
    
    # 3. State Storage
    neo4j_id = await state_stores["neo4j"].store_graph_state(initial_state)
    mongo_id = await state_stores["mongo"].store_state_history(initial_state)
    
    # 4. Evolution
    evolved_state = await adaptive_bridge.evolve_state(initial_state)
    
    # 5. Verify Evolution
    stored_state = await state_stores["neo4j"].get_graph_state(neo4j_id)
    history = await state_stores["mongo"].get_state_evolution(mongo_id)
```

#### Concurrent Operations Test
```python
async def test_concurrent_operations(self, pattern_processor, coherence_interface,
                                  state_stores, adaptive_bridge, sample_document):
    # 1. Establish Foundation
    pattern = await pattern_processor.extract_pattern(sample_document)
    adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
    initial_state = await pattern_processor.prepare_graph_state(pattern, adaptive_id)
    
    # 2. Verify Concurrent Operations
    tasks = [
        adaptive_bridge.enhance_pattern(initial_state),
        state_stores["neo4j"].store_graph_state(initial_state),
        state_stores["mongo"].store_state_history(initial_state)
    ]
    
    results = await asyncio.gather(*tasks)
```

### Test Infrastructure

#### Key Fixtures
```python
@pytest.fixture
def state_stores():
    return {
        "neo4j": Neo4jStateStore(),
        "mongo": MongoStateStore()
    }

@pytest.fixture
def adaptive_bridge():
    return AdaptiveStateBridge()

@pytest.fixture
def sample_document():
    return {
        "id": "test_doc_1",
        "content": "Test document content",
        "metadata": {
            "source": "test",
            "timestamp": datetime.now().isoformat()
        }
    }
```

## Test Success Status

✅ **All Integration Tests Passing**
- Sequential foundation validated
- Concurrent operations verified
- State management confirmed
- Database integration successful

### Key Achievements
1. **Sequential Processing**
   - Pattern extraction with provenance
   - Adaptive ID assignment
   - Graph state preparation
   - Coherence alignment

2. **Concurrent Operations**
   - Pattern enhancement
   - Parallel state storage
   - Event coordination

3. **State Management**
   - Version control
   - Evolution tracking
   - History preservation

The test suite demonstrates our system's ability to maintain strict sequential foundations while enabling efficient concurrent operations where appropriate.

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
