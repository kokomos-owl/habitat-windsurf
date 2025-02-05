# Habitat Windsurf UI Course Handoff

**Document Date**: 2025-02-05T08:14:17-05:00

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
- Core visualization components implemented
- Initial test suite passing
- First lesson notebook created
- Documentation structure established
- Agentic workshop builder created

## Component Status
1. Core Visualization
   - GraphVisualizer (implemented with plotly)
   - LayoutEngine (multiple layouts supported)
   - DI Container (basic setup complete)
   - WebSocket (pending implementation)
   - Agentic Builder (workshop automation)

2. Course Materials
   - Directory structure complete
   - Basic notebook implemented
   - Exercise templates ready
   - Advanced content pending

3. Testing
   - Basic test suite passing
   - Integration tests pending
   - Exercise validation pending

## Development Tasks

### Agentic Workshop
The repository now includes an agentic workshop builder that can:
- Automatically create the workshop environment
- Implement core visualization components
- Set up notebooks and tests
- Clean up the environment when needed

To use the builder:
```bash
# Build workshop
python scripts/agentic/workshop_builder.py

# Clean up
python scripts/agentic/workshop_builder.py clean
```

### Senior Developer Focus
1. Architecture & Integration
   - Design WebSocket protocol for real-time updates
   - Plan component event system
   - Define scaling strategy
   - Establish security guidelines

2. Course Architecture
   - Design advanced lesson structure
   - Plan integration tutorials
   - Create architecture documentation

3. Code Quality
   - Set up code review process
   - Implement CI/CD for notebooks
   - Define performance metrics

### Junior Developer Focus
1. Feature Implementation
   - Add visualization customization options
   - Implement user interaction features
   - Create basic exercise notebooks

2. Testing & Documentation
   - Write additional unit tests
   - Create usage examples
   - Document common issues

3. Learning Path
   - Study existing components
   - Practice with notebook examples
   - Start with small enhancements

## Points of Contact
- Project Lead: [TBD]
- Technical Lead: [TBD]
- Course Instructor: [TBD]

## Development Guidelines

### Repository Structure
```
habitat-windsurf/
├── src/
│   ├── core/
│   │   ├── container.py         # DI container
│   │   └── visualization/       # Core components
│   │       ├── graph.py         # GraphVisualizer
│   │       └── layout.py        # LayoutEngine
│   └── lessons/                 # Course modules
├── tests/
│   ├── conftest.py
│   └── test_visualization.py
├── docs/
│   └── lessons/
│       └── 01_windsurf_basics.md
└── notebooks/
    ├── templates/
    │   └── lesson_template.ipynb
    ├── lesson_01/
    └── lesson_02/
```

### Testing Strategy
1. Component Tests
   - Basic initialization tests implemented
   - More tests to be added as components are developed

2. Integration Tests (Planned)
   - Component interactions
   - WebSocket communication
   - Graph visualization flow

### Known Issues & Solutions
1. Component Placeholders
   - GraphVisualizer and LayoutEngine are stubs
   - Implementation coming from habitat_poc

2. Environment Setup
   - Use provided virtual environment
   - Run tests to verify setup

### Development Workflow
1. Check current component status in STATE.md
2. Refer to POC requirements for implementation details
3. Follow test-driven development approach
4. Update documentation as components are implemented

## Handoff Checklist
- [ ] Repository access granted
- [ ] Development environment documented
- [ ] All documentation reviewed
- [ ] Initial requirements understood
- [ ] Next steps clearly defined
- [ ] Contact information shared
