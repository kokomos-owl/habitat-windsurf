Based on my analysis of the codebase, here's a list of the files that must be refactored to integrate scalar mathematics for a minimum viable test of the pattern_aware_rag interface:

Core Modules to Refactor
Pattern-Aware RAG Core
/src/habitat_evolution/pattern_aware_rag/pattern_aware_rag.py
Replace vector similarity operations with scalar field calculations
Update pattern processing to use field-state architecture
Modify coherence calculations to use scalar metrics
Field Services
/src/habitat_evolution/core/services/field/field_state_service.py
Update field state calculations to use scalar mathematics
Enhance stability calculations with tonic-harmonic patterns
/src/habitat_evolution/core/services/field/gradient_service.py
Replace vector-based gradient calculations with scalar field gradients
Implement energy flow calculations based on field coherence
/src/habitat_evolution/core/services/field/flow_dynamics_service.py
Update flow calculations to use scalar-based metrics
Implement natural boundary detection using tonic-harmonic patterns
Pattern Processing
/src/habitat_evolution/pattern_aware_rag/core/pattern_processor.py
Replace vector embeddings with natural frequency calculations
Update pattern extraction to use field-aware metrics
/src/habitat_evolution/pattern_aware_rag/core/coherence_interface.py
Replace vector similarity with field interaction metrics
Implement multi-level coherence calculations
State Management
/src/habitat_evolution/pattern_aware_rag/state/state_evolution.py
Update state evolution to use field-state transitions
Replace vector-based coherence with scalar field coherence
/src/habitat_evolution/pattern_aware_rag/state/state_handler.py
Update state validation to use field-state metrics
Implement tonic-harmonic pattern detection
Learning Window Control
/src/habitat_evolution/pattern_aware_rag/learning/learning_control.py
Update window state transitions to align with field boundaries
Enhance event coordination with field-aware triggers
/src/habitat_evolution/pattern_aware_rag/learning/field_neo4j_bridge.py
Update pattern alignment to use field-state metrics
Ensure field state integrity during persistence
Test Modules to Refactor
Integration Tests
/src/tests/pattern_aware_rag/integration/test_full_cycle.py
Update test cases to validate scalar-based calculations
Add tests for field-state transitions and boundary detection
Field State Tests
/src/habitat_evolution/tests/learning/test_field_neo4j_bridge.py
Update tests to validate field-state persistence
Add tests for tonic-harmonic pattern detection
Learning Window Tests
/src/tests/pattern_aware_rag/learning/test_learning_window_field_control.py
Update tests to validate field-aware window transitions
Add tests for natural boundary detection
Pattern Processing Tests
Create new test file: /src/tests/pattern_aware_rag/core/test_scalar_pattern_processor.py
Test natural frequency calculations
Validate multi-level coherence metrics
Test field-state transitions
Field Service Tests
Create new test file: /src/tests/core/services/field/test_scalar_field_services.py
Test scalar-based gradient calculations
Validate field state metrics
Test energy flow calculations
Minimum Viable Test Implementation
For a minimum viable test of the pattern_aware_rag interface with scalar mathematics, I recommend focusing on:

Core Components:
Implement scalar-based pattern positioning in pattern_processor.py
Update coherence calculations in coherence_interface.py
Modify field state transitions in state_evolution.py
Field Services:
Implement scalar-based gradient calculations in gradient_service.py
Update flow dynamics in flow_dynamics_service.py
Test Implementation:
Create a focused test in test_full_cycle.py that validates:
Pattern positioning using scalar calculations
Coherence measurements using field interactions
Natural boundary detection with tonic-harmonic patterns
Field state transitions aligned with natural boundaries
This approach will allow you to validate the core functionality of the field-state architecture while minimizing the initial refactoring effort. Once the minimum viable test passes, you can progressively refactor the remaining components to fully implement the field-state architecture.