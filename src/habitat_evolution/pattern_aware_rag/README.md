# Pattern-Aware RAG

## Overview

Pattern-Aware RAG serves as a coherence interface in the Habitat system, managing the alignment between external and internal coherence states through controlled state agreement. It is not a traditional RAG system - instead, it focuses on maintaining and evolving pattern coherence through careful state management.

## Module Structure

The system is organized into focused modules, each with specific responsibilities:

```
pattern_aware_rag/
├── core/               # Core system components
│   └── __init__.py    # Exports core functionality
├── bridges/           # Integration bridges
│   ├── __init__.py
│   └── adaptive_state_bridge.py
├── learning/          # Learning and control systems
│   ├── __init__.py
│   └── learning_control.py
├── state/             # State management
│   ├── __init__.py
│   ├── state_handler.py
│   ├── state_evolution.py
│   ├── claude_state_handler.py
│   ├── graph_service.py
│   └── langchain_config.py
├── services/          # Service layer
│   └── __init__.py    # Exports service interfaces
└── __init__.py        # Main package interface
```

### Module Descriptions

#### Core (`core/`)
- Essential system components
- State management foundations
- Evolution tracking interfaces

#### Bridges (`bridges/`)
- Adaptive state management
- Pattern evolution tracking
- Version history maintenance

#### Learning (`learning/`)
- Learning window control
- Back pressure mechanisms
- Event coordination

#### State (`state/`)
- Graph state management
- Pattern state tracking
- Evolution history

## Testing Success

The Pattern-Aware RAG system has successfully passed comprehensive integration tests, validating both its sequential foundation and concurrent operation capabilities:

### 1. Sequential Foundation
```
Document → Pattern Extraction → Adaptive ID → Graph State → Evolution
```

Key validations:
- Pattern extraction with provenance tracking
- Adaptive ID assignment and verification
- Graph-ready state preparation
- Coherence alignment scoring
- State evolution tracking

### 2. Concurrent Operations
After establishing the sequential foundation, the system successfully demonstrated concurrent:
- Pattern enhancement
- State storage (Neo4j)
- Evolution history (MongoDB)
- Event coordination

### 3. Test Coverage
The integration test suite (`test_full_cycle.py`) provides comprehensive coverage:

```python
# Sequential Foundation Test
async def test_full_state_cycle(...):
    # 1. Sequential Foundation
    pattern = await pattern_processor.extract_pattern(sample_document)
    adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
    initial_state = await pattern_processor.prepare_graph_state(pattern, adaptive_id)
    
    # 2. Coherence Interface
    alignment = await coherence_interface.align_state(initial_state)
    
    # 3. State Storage & Evolution
    evolved_state = await adaptive_bridge.evolve_state(initial_state)

# Concurrent Operations Test
async def test_concurrent_operations(...):
    # Parallel execution of enhancement and storage
    tasks = [
        adaptive_bridge.enhance_pattern(initial_state),
        state_stores["neo4j"].store_graph_state(initial_state),
        state_stores["mongo"].store_state_history(initial_state)
    ]
```

All tests are passing, demonstrating the system's ability to maintain coherence while enabling efficient parallel processing where appropriate.
- State transitions
- Claude integration
- LangChain configuration

#### Services (`services/`)
- High-level interfaces
- Integration services
- Operation coordination

## Core Architecture

```
Graph State → Prompt Formation → Query → Claude → Response → Pattern-Aware RAG → Graph State
```

### Key Components

1. **Graph State Foundation**
   - Initial state loading
   - State coherence metrics
   - Relationship tracking
   - History maintenance

2. **Learning Windows**
   - State transitions (CLOSED → OPENING → OPEN)
   - Back pressure control
   - Agreement formation
   - Evolution guidance

3. **Pattern-Coherence Co-Evolution**
   - State alignment
   - Relationship development
   - Coherence maintenance
   - Evolution tracking

## Implementation

### Key Files

#### State Management
- `state/state_handler.py`: Core state management
- `state/state_evolution.py`: Evolution tracking
- `state/graph_service.py`: Graph operations

#### Integration
- `bridges/adaptive_state_bridge.py`: Adaptive ID integration
- `state/claude_state_handler.py`: Claude integration
- `state/langchain_config.py`: LangChain configuration

#### Learning Control
- `learning/learning_control.py`: Learning windows and back pressure

#### Documentation
- `BIDIRECTIONAL_FLOW.md`: Architecture documentation
- `TEST_PATTERN_AWARE_RAG_OVERVIEW.md`: Testing documentation

### Dependencies
- Neo4j for graph state
- Claude for language model
- Core pattern evolution services
- Event management system

## Testing

See `TEST_PATTERN_AWARE_RAG_OVERVIEW.md` for detailed testing information.

1. **Functional Testing**
   - Graph state foundation
   - Learning window mechanics
   - Pattern-coherence evolution

2. **Integration Testing**
   - Complete state cycle
   - Claude interaction
   - System stability

## Usage

The Pattern-Aware RAG system must be used with careful attention to:
1. Graph state as the foundation
2. Back pressure control for state changes
3. Coherence maintenance throughout operations
4. Proper state agreement formation

## Development Status

Current focus:
1. Functional test implementation
2. Integration infrastructure setup
3. Live testing preparation

See `STATE.md` in the root directory for current development status.
