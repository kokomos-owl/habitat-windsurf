# Habitat Windsurf UI Course Handoff

**Document Date**: 2025-02-05T07:55:53-05:00

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
- Repository structure implemented
- Core component placeholders created
- Initial test framework established
- Documentation framework in place

## Component Status
1. Core Visualization
   - GraphVisualizer (placeholder ready for implementation)
   - LayoutEngine (placeholder ready for implementation)
   - DI Container (basic setup complete)

2. Course Materials
   - Directory structure complete
   - Notebook templates created
   - Initial lesson documentation started

3. Testing
   - Basic test structure in place
   - Initial component tests implemented

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
