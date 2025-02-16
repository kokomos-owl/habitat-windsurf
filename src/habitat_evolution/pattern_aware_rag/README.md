# Pattern-Aware RAG

## Overview

Pattern-Aware RAG serves as a coherence interface in the Habitat system, managing the alignment between external and internal coherence states through controlled state agreement. It is not a traditional RAG system - instead, it focuses on maintaining and evolving pattern coherence through careful state management.

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

### Core Files
- `pattern_aware_rag.py`: Main implementation
- `graph_service.py`: Graph state management
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
