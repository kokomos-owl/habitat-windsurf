# Habitat Windsurf UI Course Handoff

**Document Date**: 2025-02-05T07:47:06-05:00

## Project Overview
Habitat Windsurf UI Course is a proof-of-concept educational environment focused on teaching visualization component development using the Windsurf IDE.

## Required Reading
1. [POC Requirements](habitat_windsurf_ui_poc.md) - Core requirements and course structure
2. [Current State](STATE.md) - Project status and next steps

## Repository Structure
```
habitat-windsurf/
├── docs/
├── src/
├── tests/
└── notebooks/
```

## Development Setup

### Prerequisites
1. Python 3.11+
2. Windsurf IDE installed
3. Git
4. Access to habitat_poc repository (for component migration)

### Local Environment Setup
```bash
# Clone repository
git clone https://github.com/your-org/habitat-windsurf.git
cd habitat-windsurf

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run test suite
pytest tests/

# Start development server (when implemented)
python -m src.core.server
```

### Configuration
1. Copy `.env.example` to `.env`
2. Update environment variables as needed:
   ```env
   WINDSURF_HOME=/path/to/windsurf
   HABITAT_POC_PATH=/path/to/habitat_poc
   ```

### Verification Steps
1. Run `pytest tests/test_visualization.py` - should pass basic visualization tests
2. Open `notebooks/lesson_01/basics.ipynb` in Jupyter - should load without errors
3. Check GraphVisualizer initialization

## Current Progress
- Initial documentation and planning phase complete
- Repository structure defined
- Core requirements established

## Immediate Tasks
1. Repository Setup
   - Initialize project structure
   - Set up dependency management
   - Configure test environment

2. Component Migration
   - GraphVisualizer
   - LayoutEngine
   - Basic WebSocket support

3. Course Development
   - Create notebook templates
   - Develop first lesson content
   - Set up example code

## Points of Contact
- Project Lead: [TBD]
- Technical Lead: [TBD]
- Course Instructor: [TBD]

## Development Guidelines

### Code Organization
- Core visualization components in `src/core/visualization/`
- Course-specific modules in `src/lessons/`
- Tests mirror source structure in `tests/`
- Notebooks organized by lesson in `notebooks/`

### Testing Strategy
1. Unit Tests: `pytest tests/unit/`
   - Component initialization
   - Basic functionality
   - Error handling

2. Integration Tests: `pytest tests/integration/`
   - Component interactions
   - Data flow validation
   - WebSocket communication

3. Course Material Tests: `pytest tests/course/`
   - Notebook execution
   - Exercise validation

### Common Issues & Solutions
1. GraphVisualizer Import Errors
   - Check PYTHONPATH includes project root
   - Verify habitat_poc components accessible

2. Notebook Kernel Issues
   - Use `python -m ipykernel install --user --name=habitat-windsurf`
   - Select 'habitat-windsurf' kernel in Jupyter

### Development Workflow
1. Create feature branch from `main`
2. Implement changes with tests
3. Verify notebook compatibility
4. Submit PR with documentation updates

## Handoff Checklist
- [ ] Repository access granted
- [ ] Development environment documented
- [ ] All documentation reviewed
- [ ] Initial requirements understood
- [ ] Next steps clearly defined
- [ ] Contact information shared
