# Habitat Windsurf UI Course Handoff

**Document Date**: 2025-02-07T10:30:30-05:00

## Project Overview

Habitat Windsurf UI Course is an interactive visualization workshop focused on real-time network visualization and status tracking. The project demonstrates modern web development practices with a focus on user experience and error handling.

## Component Architecture

### Visualization Layer
- `flow_visualizer.py`: Plotly-based flow pattern visualization
- Interactive graph visualization with coherence-based coloring
- Real-time pattern evolution tracking
- Structure-meaning relationship visualization
- Pattern density and temporal metrics

### Pattern Evolution System
- Dynamic pattern state management
- Coherence calculation and tracking
- Temporal relationship analysis
- Evolution stage determination
- Pattern density metrics

### Status System
- Real-time process tracking
- Glowing status indicators
- Step-by-step progress visualization
- Comprehensive error handling

### Backend Integration
- FastAPI REST endpoints
- WebSocket real-time updates
- MongoDB data persistence
- Robust error handling and status codes

## Developer Guides

### For Senior Developers
- Architecture focuses on modularity and maintainability
- Key extension points in network.js for additional visualization types
- Error handling system can be extended for new error types
- Performance considerations in WebSocket implementation
- Consider memory management for large datasets

### For Junior Developers
- Start with `main.js` to understand the application flow
- Status indicators in `network.js` show how to handle UI states
- Error handling patterns demonstrate good practices
- CSS shows modern styling techniques
- Follow existing patterns for new features

### For AI Agents
- Component boundaries are clearly defined
- Error states are explicitly handled
- Data structures are consistently formatted
- Status updates follow predictable patterns
- Documentation provides context for decisions

## Current Status

### Implemented Features
- [x] Flow pattern visualization with Plotly
- [x] Pattern evolution tracking
- [x] Coherence calculation system
- [x] Interactive graph visualization
- [x] Pattern density metrics
- [x] Structure-meaning analysis
- [x] Temporal evolution tracking
- [x] Real-time WebSocket updates

### Pending Tasks
- [ ] Performance optimization for large pattern sets
- [ ] Cross-browser testing of visualization
- [ ] Pattern evolution documentation updates
- [ ] Memory optimization for long-running evolution
- [ ] Load testing with concurrent pattern updates
- [ ] Cross-document pattern analysis
- [ ] Advanced temporal modeling
- [ ] Pattern state transition validation

## Next Development Phase: Climate Risk Pattern Evolution

### Overview
Implement dynamic pattern learning and coherence analysis for climate risk data, building on the initial implementation with `climate_risk_marthas_vineyard.txt` as source data.

### High-Level Steps

1. **Pattern Evolution System**
   - Semantic pattern extraction with emphasis detection
   - Temporal relationship tracking
   - Pattern evolution metrics
   - Coherence scoring system
   - Note: See `docs/PATTERN_EVOLUTION.md` for details

2. **Pattern Storage and Analysis**
   - Pattern-aware document schema in MongoDB
   - Evolution tracking endpoints in FastAPI
   - Coherence calculation service
   - Pattern relationship storage
   - Temporal context preservation

3. **Pattern Visualization**
   - Dynamic network graph for pattern relationships
   - Temporal evolution visualization
   - Coherence strength indicators
   - Pattern density visualization
   - Real-time evolution tracking

4. **Evolution Analysis Tools**
   - Pattern evolution tracking interface
   - Coherence analysis dashboard
   - Temporal relationship explorer
   - Pattern density analysis
   - Cross-document pattern viewer

5. **Testing and Validation**
   - Pattern extraction validation
   - Coherence calculation tests
   - Evolution tracking verification
   - Performance benchmarking
   - Cross-document pattern tests
   - Visualization rendering tests
   - Mock data validation

### Key Considerations
- Keep mock pipeline modular for easy replacement
- Document all assumptions about future NLP integration
- Maintain compatibility with habitat_evolution ID system
- Focus on visualization features that will remain relevant

### Internal Development (Not Part of Workshop)
A parallel development effort is underway to migrate and enhance core visualization components. This work is internal and separate from the workshop content.

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
2. Docker and Docker Compose
3. MongoDB
4. Neo4j (optional)
5. Windsurf IDE installed
6. Git
7. Access to habitat_poc repository (for component migration)

### Local Environment Setup
```bash
# Clone repository
git clone https://github.com/your-org/habitat-windsurf.git
cd habitat-windsurf

# Start required services
docker-compose up -d mongodb neo4j

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run test suite
python -m pytest

# Start development server
uvicorn src.visualization.api.app:app --reload
```

### Configuration
1. Copy `.env.example` to `.env`
2. Update environment variables as needed:
   ```env
   WINDSURF_HOME=/path/to/windsurf
   HABITAT_POC_PATH=/path/to/habitat_poc
   MONGO_USERNAME=your_username
   MONGO_PASSWORD=your_password
   MONGO_HOST=localhost
   MONGO_PORT=27017
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   NEO4J_HOST=localhost
   NEO4J_PORT=7687
   ```

### Verification Steps
1. Run `pytest tests/test_visualization.py` - should pass basic visualization tests
2. Open `notebooks/lesson_01/basics.ipynb` in Jupyter - should load without errors
3. Check GraphVisualizer initialization

## Current Progress

### Core Implementation
- GraphVisualizer with Plotly integration
- MongoDB integration with authentication
- Neo4j integration (optional)
- WebSocket for real-time updates
- FastAPI REST endpoints
- Comprehensive test suite
- First lesson notebook (balanced approach)
- Documentation structure (collaborative)
- Workshop builder (hybrid implementation)

### Learning Approach
- Progressive complexity model
- Balanced user-agent interactions
- Hands-on exercises with agent support
- Clear learning objectives for each module

## Component Status
1. Core Visualization
   - GraphVisualizer (implemented with Plotly)
   - MongoDB Client (authentication enabled)
   - Neo4j Client (optional integration)
   - WebSocket Manager (real-time updates)
   - FastAPI Router (REST endpoints)
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

### Workshop Technical Stack (Current Release)

1. **Backend** (`src/backend/`)
   - FastAPI for REST and WebSocket
   - SQLAlchemy for ORM
   - Redis for real-time state
   - GraphQL for complex queries

2. **Frontend** (`src/frontend/`)
   - React with TypeScript
   - Plotly.js for visualization
   - TailwindCSS for styling
   - Apollo Client for GraphQL

3. **Infrastructure** (`infra/`)
   - Docker for containerization
   - Redis for pub/sub
   - PostgreSQL for persistence
   - Nginx for routing

### Workshop Development Environment (Current Release)

1. **Prerequisites**
   ```bash
   # Required software:
   python >= 3.9
   node >= 16.0
   docker >= 20.0
   redis >= 6.0
   ```

2. **Setup Commands**
   ```bash
   # Backend setup
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # Frontend setup
   cd frontend
   npm install
   ```

3. **Development Flow**
   ```bash
   # Start services
   docker-compose up -d     # Start infrastructure
   make dev-backend        # Start FastAPI
   make dev-frontend       # Start React
   
   # Development tools
   make lint              # Run linters
   make test              # Run tests
   make docs              # Generate docs
   ```

### Workshop Structure
The repository implements a balanced user-agent approach:

#### User-Driven Components
- Manual implementation exercises
- Code exploration tasks
- Custom feature development
- Testing and validation

#### Agent-Assisted Features
- Environment setup automation
- Code scaffolding
- Test suite generation
- Documentation updates

#### Collaborative Elements
- Interactive notebooks
- Guided implementations
- Code reviews
- Problem-solving sessions

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
