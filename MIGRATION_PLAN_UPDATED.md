# Habitat Evolution Migration Plan

## Executive Summary

This document outlines the plan for migrating the Habitat Evolution system from the `habitat_alpha` repository to the new `habitat_mvp` repository. The migration will focus on implementing a robust, test-driven approach that addresses the current system's limitations while preserving its core pattern evolution and co-evolution principles.

## Progress Update (April 12, 2025)

**MILESTONE ACHIEVED**: We have successfully implemented a robust system initialization framework with comprehensive dependency tracking and error handling. This provides a solid foundation for the migration process.

Key accomplishments:

1. Fixed critical initialization issues in PatternAwareRAGService
2. Resolved ArangoDB client compatibility issues
3. Implemented comprehensive dependency tracking
4. Created test infrastructure for system initialization
5. Achieved full system initialization with proper component ordering
6. Implemented detailed logging and error reporting

The system can now reliably initialize all components in the correct order, with proper dependency verification and explicit error reporting.

## 1. Repository Structure

Create the following directory structure in the new repository:

```
habitat_mvp/
├── docs/
│   ├── green_papers/
│   └── architecture/
├── src/
│   └── habitat_evolution/
│       ├── adaptive_core/
│       ├── climate_risk/
│       ├── infrastructure/
│       │   └── services/
│       ├── pattern_aware_rag/
│       └── pkm/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── scripts/
└── examples/
    └── image_pkm_demo/
```

## 2. Component Migration Priority

Migrate components in the following order to ensure dependencies are properly handled:

1. **Foundation Components**:
   - ArangoDB Connection
   - EventService
   - Component Initializer

2. **Service Components**:
   - Document Processing Service
   - Field-Pattern Bridge
   - Claude Adapter

3. **Vector-Tonic Components** (with critical fixes integrated):
   - SemanticCurrentObserver with FieldNavigator and ActantJourneyTracker
   - EventAwarePatternDetector with proper semantic observer initialization
   - VectorTonicIntegrator with complete dependency chain
   - TonicDetector with proper base detector configuration
   - HarmonicIOService with event bus integration

4. **Pattern-Aware Components**:
   - PatternAwareRAG
   - PKM Factory
   - BidirectionalFlowService

## 3. Critical Integration Fixes

### 3.1 EventService Initialization Fix

**IMPORTANT**: The EventService is a foundational component that must be properly initialized before other components. Multiple warnings about "EventService not initialized" indicate this critical issue must be addressed.

Key components of the fix:

1. **Global Singleton Pattern**:
   - Implement EventService as a true singleton with proper initialization checks
   - Ensure the EventService is initialized before any other components
   - Add a global access method (get_instance) that initializes if needed

2. **Automatic Initialization**:
   - Add automatic initialization when EventService methods are called
   - Implement lazy loading of event handlers
   - Ensure thread safety for concurrent initialization attempts

3. **Integration Strategy**:
   - Move the EventService initialization to the earliest stage of system startup
   - Add explicit initialization in component factory methods
   - Implement comprehensive logging for initialization status
   - Add health check methods to verify EventService status

### 3.2 Vector Tonic Integration Fix

**IMPORTANT**: The Vector Tonic integration fix must be incorporated directly into the core initialization logic, not as a separate patch. This is critical for processing both semantic and statistical data.

Key components of the fix:

1. **Complete Dependency Chain**:
   - SemanticCurrentObserver requires FieldNavigator and ActantJourneyTracker
   - EventAwarePatternDetector requires SemanticCurrentObserver
   - LearningWindowAwareDetector requires EventAwarePatternDetector
   - TonicHarmonicPatternDetector requires LearningWindowAwareDetector

2. **Simplified Component Implementations**:
   - Include SimpleFieldNavigator and SimpleActantJourneyTracker
   - Implement graceful fallback mechanisms for all Vector Tonic components
   - Ensure proper error handling throughout the initialization sequence

3. **Integration Strategy**:
   - Move the initialization logic from `vector_tonic_fix.py` to a core factory module
   - Update all dependent components to use this factory
   - Implement comprehensive error handling and logging
   - Ensure the system can operate with limited functionality when components are unavailable

### 3.2 PatternAwareRAG Service Fixes

The PatternAwareRAG service has several issues that need to be addressed during migration:

1. **Claude API Integration**:
   - Fix the `expected string or bytes-like object, got 'dict'` error in Claude API queries
   - Ensure proper type checking and conversion before API calls
   - Implement more robust error handling for API communication

2. **Relationship Enhancement**:
   - Address the `'float' object has no attribute 'get'` error in relationship enhancement
   - Implement proper type checking before attribute access
   - Add defensive programming to handle unexpected data types

3. **Graceful Degradation**:
   - Enhance fallback mechanisms when components are unavailable
   - Ensure meaningful responses even when some services fail
   - Improve error logging and reporting for troubleshooting

## 4. Test Migration

Follow the test inventory in `TEST_INVENTORY.md` to migrate all tests, organizing them into:

- `tests/unit/`: Tests for individual components
- `tests/integration/`: Tests for component interactions
- `tests/e2e/`: End-to-end tests for complete workflows

Ensure all fixed versions of tests are included, particularly:
- Fixed EventService tests
- Document processing service fix tests
- Climate E2E fixed tests with Vector Tonic integration
- Vector-tonic initialization tests with complete dependency chain

## 5. Documentation Migration

Migrate and update the following documentation:

- Green papers (especially the updated Habitat Evolution green paper)
- Architecture diagrams
- Component interaction documentation
- Dependency chain management documentation
- API documentation
- Vector Tonic initialization sequence documentation

## 6. Demo Migration

Migrate the Image PKM demo to `examples/image_pkm_demo/` directory:
- `demo_image_pkm.py`
- `run_image_demo.py`
- `README_image_demo.md`
- Supporting files

## 7. Configuration Updates

Update configuration files and paths:
- Update import paths in all files
- Create new configuration files for the MVP environment
- Update environment variable references
- Create a comprehensive `.gitignore` file based on the one from habitat_alpha

## 8. Dependency Management

Create proper dependency management files:
- `requirements.txt` for basic dependencies
- `requirements-dev.txt` for development dependencies
- `requirements-test.txt` for test dependencies

Include all necessary dependencies, particularly:
- pytest and related plugins
- ArangoDB driver
- OpenCV and Pillow for image processing
- FastAPI for future web UI development

## 9. Migration Verification

After migration, verify:
- All components initialize properly with the Vector Tonic integration fix
- Tests pass in the new environment, particularly the climate E2E test
- Documentation is accessible and up-to-date
- Demo applications run correctly

## 10. Next Steps After System Initialization

With the robust system initialization framework now in place, our next priorities are:

1. **Implement Comprehensive Error Recovery Strategies**:
   - Create standardized error recovery patterns for all components
   - Implement graceful degradation when components fail
   - Add self-healing mechanisms for common initialization failures
   - Develop a unified error reporting system across all components

2. **Enhance Dependency Verification**:
   - Expand dependency verification to include configuration validation
   - Add runtime dependency health checks
   - Implement dynamic dependency resolution for optional components
   - Create visualization tools for the dependency graph

3. **Extend Test Coverage**:
   - Create stress tests for the initialization system
   - Implement fault injection tests to verify error handling
   - Add performance benchmarks for initialization sequence
   - Develop integration tests for all component combinations

4. **Prepare Component Migration Templates**:
   - Create standardized migration templates for each component type
   - Develop migration verification checklists
   - Build automated migration validation tools
   - Document component-specific migration considerations

## 11. Future Development

After successful migration, proceed with:
- FastAPI-based web UI development
- Enhanced pattern persistence capabilities
- Expanded multimodal support
- Improved visualization tools

## Migration Command Reference

Here are some helpful commands for the migration process:

```bash
# Clone the new repository
git clone https://github.com/kokomos-owl/habitat_mvp.git

# Copy core components
cp -r habitat_alpha/src/habitat_evolution/infrastructure habitat_mvp/src/habitat_evolution/

# Copy tests
cp -r habitat_alpha/tests/integration habitat_mvp/tests/

# Update import paths (example)
find habitat_mvp -type f -name "*.py" -exec sed -i '' 's/from habitat_alpha/from habitat_mvp/g' {} \;
```

Remember to commit changes frequently during the migration process to maintain a clear history of the transition.
