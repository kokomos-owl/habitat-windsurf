# Habitat Evolution Migration Plan

This document outlines the plan for migrating the Habitat Evolution system from the `habitat_alpha` repository to the new `habitat_mvp` repository.

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

3. **Vector-Tonic Components**:
   - VectorTonicIntegrator
   - TonicDetector
   - HarmonicIOService
   - EventBus

4. **Pattern-Aware Components**:
   - PatternAwareRAG
   - PKM Factory
   - BidirectionalFlowService

## 3. Test Migration

Follow the test inventory in `TEST_INVENTORY.md` to migrate all tests, organizing them into:

- `tests/unit/`: Tests for individual components
- `tests/integration/`: Tests for component interactions
- `tests/e2e/`: End-to-end tests for complete workflows

Ensure all fixed versions of tests are included, particularly:
- Fixed EventService tests
- Document processing service fix tests
- Climate E2E fixed tests
- Vector-tonic initialization tests

## 4. Documentation Migration

Migrate and update the following documentation:

- Green papers (especially the updated Habitat Evolution green paper)
- Architecture diagrams
- Component interaction documentation
- Dependency chain management documentation
- API documentation

## 5. Demo Migration

Migrate the Image PKM demo to `examples/image_pkm_demo/` directory:
- `demo_image_pkm.py`
- `run_image_demo.py`
- `README_image_demo.md`
- Supporting files

## 6. Configuration Updates

Update configuration files and paths:
- Update import paths in all files
- Create new configuration files for the MVP environment
- Update environment variable references
- Create a comprehensive `.gitignore` file based on the one from habitat_alpha

## 7. Dependency Management

Create proper dependency management files:
- `requirements.txt` for basic dependencies
- `requirements-dev.txt` for development dependencies
- `requirements-test.txt` for test dependencies

Include all necessary dependencies, particularly:
- pytest and related plugins
- ArangoDB driver
- OpenCV and Pillow for image processing
- FastAPI for future web UI development

## 8. CI/CD Setup

Set up continuous integration and deployment:
- GitHub Actions workflows for testing
- Containerization configuration (Dockerfile, docker-compose.yml)
- Test coverage reporting

## 9. Migration Verification

After migration, verify:
- All components initialize properly
- Tests pass in the new environment
- Documentation is accessible and up-to-date
- Demo applications run correctly

## 10. Future Development

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
