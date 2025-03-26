# Habitat Evolution Framework: Development Handoff

## Current Development Status

As of March 25, 2025, we have successfully implemented and demonstrated the core components of the Habitat Evolution framework, with a particular focus on Query Actants and Meaning Bridges. The system demonstrates the ability to process queries across multiple modalities while maintaining semantic identity and forming meaningful relationships with other actants in the system.

## System Architecture Overview

The Habitat Evolution framework is built around several key components that work together to enable semantic emergence across domains:

1. **Query Actants**: First-class citizens in the semantic landscape that maintain identity across transformations
2. **Meaning Bridges**: Relationships that enable meaning to emerge across domains (renamed from Semantic Affordances)
3. **Pattern-Aware RAG**: Retrieval Augmented Generation system with pattern awareness
4. **Field Analysis**: Topological analysis of semantic fields
5. **Adaptive Identity**: System for maintaining semantic identity across transformations
6. **Actant Journeys**: Tracking of actants' paths through semantic domains

The system architecture is visualized in the [Habitat System Schematic](/docs/habitat_system_schematic.md), which shows the flow from ingestion to persistence.

## Key Components to Understand

### 1. Pattern-Aware RAG (`pattern_aware_rag.py`)

The Pattern-Aware RAG component is the central processing engine that integrates pattern awareness into the retrieval and generation process. It works with the Field State to maintain a topological understanding of the semantic landscape and detect emergent patterns.

Key aspects to understand:
- How it processes queries with pattern awareness
- How it integrates with field state
- How it detects and tracks meaning bridges
- How it manages learning windows for pattern emergence

### 2. Learning Control (`learning_control.py`)

The Learning Control component manages the learning process, determining when and how the system should adapt to new information. It works with eigenspace windows to detect potential pattern emergence.

Key aspects to understand:
- Window state management (CLOSED, OPENING, OPEN)
- Eigenspace window management
- Learning health integration
- Pattern coevolution persistence

### 3. Adaptive ID (`adaptive_id.py`)

The Adaptive ID system is responsible for maintaining semantic identity across transformations. It enables actants to evolve while preserving their core identity.

Key aspects to understand:
- Temporal context management
- State change notifications
- Relationship tracking
- Identity preservation across transformations

### 4. Field Components (`field/`)

The Field components provide topological analysis of semantic spaces, enabling the detection of emergent patterns and the tracking of semantic flows.

Key aspects to understand:
- Field state management
- Topological field analysis
- Field-RAG bridge
- Field-AdaptiveID bridge
- Harmonic field IO bridge

### 5. Topology Components

The Topology components analyze the topological properties of semantic fields, identifying effective dimensionality, principal dimensions, and flow dynamics.

Key aspects to understand:
- Effective dimensionality analysis
- Principal component analysis
- Flow dynamics analysis
- Topological transitions

## Current Implementation Status

### Completed Features

1. **Query Actant Implementation**:
   - Creation and processing of queries as first-class actants
   - Transformation of queries across modalities (text, image, audio)
   - Evolution of queries based on new information
   - Formation of meaning bridges between queries and other actants

2. **Meaning Bridge Detection**:
   - Detection of co-occurrence relationships
   - Tracking of domain crossing relationships
   - Analysis of transformation relationships

3. **Actant Journey Tracking**:
   - Tracking of actants' paths through semantic domains
   - Visualization of actant journeys
   - Narrative generation for actant journeys

4. **Field Analysis**:
   - Topological analysis of semantic fields
   - Detection of effective dimensionality
   - Analysis of flow dynamics

### In Progress Features

1. **Integration with Real AI Systems**:
   - Currently using mock handlers for different modalities
   - Planning to integrate with real AI systems for each modality

2. **Enhanced Visualization**:
   - Developing more sophisticated visualization tools for semantic networks
   - Implementing interactive exploration of actant journeys

3. **Performance Optimization**:
   - Optimizing database operations through the Harmonic IO Service
   - Improving pattern detection algorithms

## Key Files and Modules to Understand

### Core Implementation Files

1. `/src/habitat_evolution/pattern_aware_rag/pattern_aware_rag.py` - Central processing engine
2. `/src/habitat_evolution/pattern_aware_rag/learning/learning_control.py` - Learning window management
3. `/src/habitat_evolution/adaptive_core/id/adaptive_id.py` - Adaptive identity system
4. `/src/habitat_evolution/field/field_state.py` - Field state management
5. `/src/habitat_evolution/field/topological_field_analyzer.py` - Topological analysis
6. `/src/habitat_evolution/adaptive_core/query/query_actant.py` - Query actant implementation
7. `/src/habitat_evolution/adaptive_core/query/query_interaction.py` - Query processing
8. `/src/habitat_evolution/adaptive_core/transformation/meaning_bridges.py` - Meaning bridge implementation

### Test Modules

1. `/tests/adaptive_core/query/test_query_actant.py` - Tests for QueryActant
2. `/tests/adaptive_core/query/test_query_interaction.py` - Tests for query processing
3. `/tests/adaptive_core/transformation/test_meaning_bridges.py` - Tests for meaning bridges
4. `/tests/adaptive_core/identity/test_adaptive_id.py` - Tests for AdaptiveID
5. `/tests/adaptive_core/journey/test_actant_journey.py` - Tests for actant journeys
6. `/tests/adaptive_core/semantic/test_semantic_proposition.py` - Tests for semantic propositions
7. `/tests/adaptive_core/semantic/test_emergent_propensity.py` - Tests for emergent propensity
8. `/tests/adaptive_core/transformation/test_transformation_rules.py` - Tests for transformation rules
9. `/tests/integration/test_query_evolution.py` - Tests for query evolution
10. `/tests/integration/test_cross_modal_preservation.py` - Tests for cross-modal preservation

### Documentation Files

1. `/docs/query_actant_process.md` - Guide to query actants
2. `/docs/pattern_language_green_paper.md` - Theoretical foundation
3. `/docs/habitat_system_schematic.md` - System architecture visualization
4. `/docs/meaning_bridges.md` - Guide to meaning bridges
5. `/docs/adaptive_identity.md` - Guide to adaptive identity

## Demo Files

1. `/demos/query_actant_demo.py` - Demonstrates query actants
2. `/demos/output/query_actants/` - Contains output files from the demo

## Next Steps

1. **Integration with Real AI Systems**:
   - Implement real handlers for text, image, and audio modalities
   - Test cross-modal preservation with real AI systems

2. **Enhanced Visualization**:
   - Develop interactive visualization tools for semantic networks
   - Implement 3D visualization of field topology

3. **Performance Optimization**:
   - Optimize database operations
   - Implement parallel processing for pattern detection

4. **Extended Modalities**:
   - Add support for video and interactive modalities
   - Test cross-modal preservation across all modalities

## Theoretical Foundation

The Habitat Evolution framework is built on three core principles:

1. **Pattern Evolution**: Semantic patterns evolve as they move across domains, adapting to new contexts while maintaining core elements of their identity.
2. **Co-Evolution**: Patterns and domains co-evolve, with patterns shaping domains and domains shaping patterns in a continuous feedback loop.
3. **Observable Semantic Change**: By tracking how actants carry predicates across domain boundaries, we can observe semantic change as it happens.

These principles are detailed in the [Pattern Language Green Paper](/docs/pattern_language_green_paper.md).

## Conclusion

The Habitat Evolution framework represents a significant advancement in semantic processing, enabling the detection and tracking of emergent patterns across domains. By treating queries as first-class actants and implementing meaning bridges, the system creates a rich semantic landscape where meaning emerges from relationships rather than being statically defined.

This handoff document provides a starting point for understanding the system architecture and implementation. For more detailed information, refer to the documentation files and test modules listed above.
