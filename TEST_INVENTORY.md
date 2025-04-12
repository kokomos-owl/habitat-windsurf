# Habitat Evolution Test Inventory

This document provides a comprehensive inventory of all tests in the Habitat Evolution system. This inventory should be used during migration to ensure no tests are missed.

## Custom Test Scripts Created During Development

These scripts were created to address specific issues and should be migrated to the new repository:

1. **Fixed Test Utilities**:
   - `/tests/integration/climate_e2e/test_utils_fix.py`
   - Improved error handling and fixed document processing methods

2. **Run Fixed Test Script**:
   - `/run_fixed_test.py`
   - Script to run integrated tests with patched components

3. **Document Processing Service Fix Test**:
   - `/src/habitat_evolution/climate_risk/document_processing_service_fix.py`
   - Fixed version of the document processing service with improved error handling

4. **Event Service Test**:
   - `/test_event_service.py`
   - Tests for the EventService initialization

5. **Event Service Fix Test**:
   - `/test_event_service_fix.py`
   - Tests for the fixed EventService with patching mechanism

6. **Vector Tonic Initialization Test**:
   - `/test_vector_tonic_init.py`
   - Tests for vector-tonic component initialization with fallback implementations

7. **Climate E2E Fixed Test**:
   - `/tests/integration/climate_e2e/test_climate_e2e_fix.py`
   - Fixed version of the end-to-end climate test

8. **Image PKM Demo Test**:
   - `/demo_image_pkm.py`
   - Demonstrates image processing and pattern association capabilities

9. **Run Image Demo**:
   - `/run_image_demo.py`
   - User-friendly wrapper for the image PKM demo

## Existing Test Files in Repository

These are the existing test files in the repository that should be migrated:

### Adaptive Core Tests
- `/src/habitat_evolution/adaptive_core/emergence/test_event_integration.py`
- `/src/habitat_evolution/adaptive_core/emergence/test_event_integration_with_climate_data.py`
- `/src/habitat_evolution/adaptive_core/emergence/test_learning_window_integration.py`
- `/src/habitat_evolution/adaptive_core/emergence/test_meta_pattern_propensity.py`
- `/src/habitat_evolution/adaptive_core/emergence/test_tonic_harmonic_integration.py`
- `/src/habitat_evolution/adaptive_core/emergence/test_vector_tonic_window_integration.py`
- `/src/habitat_evolution/tests/adaptive_core/test_dimensional_context.py`
- `/src/habitat_evolution/tests/adaptive_core/test_resonance_cascade_tracker.py`
- `/src/habitat_evolution/tests/adaptive_core/test_resonance_visualization.py`
- `/src/habitat_evolution/tests/adaptive_core/test_tonic_harmonic_metrics.py`
- `/src/habitat_evolution/tests/adaptive_core/test_tonic_harmonic_resonance.py`
- `/src/habitat_evolution/tests/adaptive_core/test_wave_resonance_analyzer.py`

### Infrastructure Tests
- `/src/habitat_evolution/infrastructure/tests/test_bidirectional_flow.py`
- `/src/habitat_evolution/infrastructure/tests/test_di_system.py`
- `/src/habitat_evolution/infrastructure/tests/test_pattern_adaptive_id_adapter.py`
- `/src/habitat_evolution/infrastructure/tests/test_user_interaction.py`

### Pattern-Aware RAG Tests
- `/src/habitat_evolution/pattern_aware_rag/state/test_states.py`
- `/src/habitat_evolution/tests/pattern/test_pattern_aware_rag.py`

### API Tests
- `/src/habitat_evolution/test_claude_adapter.py`
- `/src/habitat_evolution/tests/api/test_claude_adapter.py`

### Field Tests
- `/src/habitat_evolution/tests/field/test_field_navigator.py`
- `/src/habitat_evolution/tests/field/test_resonance_pattern_detector.py`
- `/src/habitat_evolution/tests/field/test_topological_field_analyzer.py`
- `/src/habitat_evolution/tests/pattern/test_field_basics.py`
- `/src/habitat_evolution/tests/pattern/test_field_integration.py`
- `/src/habitat_evolution/tests/pattern/test_field_navigation.py`
- `/src/habitat_evolution/tests/pattern/test_field_visualization.py`

### Integration Tests
- `/src/habitat_evolution/tests/integration/test_bidirectional_flow.py`
- `/src/habitat_evolution/tests/integration/test_claude_api_integration.py`
- `/src/habitat_evolution/tests/integration/test_claude_service_integration.py`
- `/src/habitat_evolution/tests/integration/test_climate_risk_pattern_extraction.py`
- `/src/habitat_evolution/tests/integration/test_medical_case_evolution.py`
- `/src/habitat_evolution/tests/integration/test_pattern_aware_rag_integration.py`
- `/tests/integration/climate_e2e/test_climate_e2e.py`

### Pattern Tests
- `/src/habitat_evolution/tests/pattern/test_claude_state_handler.py`
- `/src/habitat_evolution/tests/pattern/test_climate_patterns.py`
- `/src/habitat_evolution/tests/pattern/test_enhanced_relational_accretion.py`
- `/src/habitat_evolution/tests/pattern/test_evolution.py`
- `/src/habitat_evolution/tests/pattern/test_gradient_regulation.py`
- `/src/habitat_evolution/tests/pattern/test_langchain_config.py`
- `/src/habitat_evolution/tests/pattern/test_pattern_dynamics.py`
- `/src/habitat_evolution/tests/pattern/test_quality.py`
- `/src/habitat_evolution/tests/pattern/test_queries_as_actants.py`
- `/src/habitat_evolution/tests/pattern/test_relational_accretion.py`
- `/src/habitat_evolution/tests/pattern/test_relational_accretion_demo.py`
- `/src/habitat_evolution/tests/pattern/test_relational_accretion_integration.py`
- `/src/habitat_evolution/tests/pattern/test_social_patterns.py`
- `/src/habitat_evolution/tests/pattern/test_state_handler.py`

### Other Tests
- `/src/habitat_evolution/tests/ghost_integration/test_toolbar.py`
- `/src/habitat_evolution/tests/learning/test_field_neo4j_bridge.py`
- `/scripts/test_predicate_relationships.py`

## Test Categories for Migration

When migrating tests to the new repository, consider organizing them into these categories:

1. **Unit Tests**: Tests for individual components
2. **Integration Tests**: Tests for component interactions
3. **End-to-End Tests**: Tests for complete workflows
4. **Performance Tests**: Tests for system performance
5. **Resilience Tests**: Tests for error handling and recovery

## Migration Checklist

- [ ] Copy all test files to the new repository
- [ ] Update import paths as needed
- [ ] Ensure test dependencies are installed
- [ ] Verify test configuration files are migrated
- [ ] Run tests in the new environment to verify functionality
- [ ] Update test documentation as needed

## Important Test Fixtures and Data

Ensure these test fixtures and data files are migrated:

- Test climate data files
- Mock Claude API responses
- Test document corpus
- ArangoDB test configuration
- Environment variable configurations for tests

## Test Dependencies

The following dependencies are required for running the tests:

- pytest
- pytest-asyncio
- mock
- pytest-mock
- pytest-cov (for coverage reporting)
- OpenCV and Pillow (for image processing tests)
