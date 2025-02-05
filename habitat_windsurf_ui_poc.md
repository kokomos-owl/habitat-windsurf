# Habitat Windsurf UI Course - POC Requirements

## Overview
A proof-of-concept course environment for teaching Windsurf UI development, focusing on visualization components from habitat_poc. Initial scope covers basic Windsurf setup and guided introduction.

## Core Requirements

### 1. Environment Setup
- Python 3.11+
- Key Dependencies:
  ```
  fastapi>=0.109.0
  pydantic>=2.5.0
  dependency-injector>=4.41.0
  pytest>=7.4.3
  pytest-asyncio>=0.21.1
  networkx>=3.2.1
  plotly>=5.18.0
  ```

### 2. Repository Structure
```
habitat-windsurf/
├── README.md                # Course overview and quick start
├── pyproject.toml           # Project dependencies and config
├── src/
│   ├── core/               # Core visualization components
│   │   ├── __init__.py
│   │   ├── container.py    # DI container
│   │   └── visualization/  # Migrated from habitat_poc
│   │       ├── __init__.py
│   │       ├── graph.py
│   │       └── layout.py
│   └── lessons/            # Course-specific modules
│       ├── __init__.py
│       ├── lesson_01/      # Windsurf Basics
│       │   ├── __init__.py
│       │   ├── graph_examples.py
│       │   └── utils.py
│       └── lesson_02/      # Guided Setup
│           ├── __init__.py
│           ├── setup_helpers.py
│           └── viz_components.py
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── conftest.py
│   └── test_visualization.py
├── docs/                   # Documentation
│   ├── lessons/
│   │   ├── 01_windsurf_basics.md
│   │   └── 02_guided_setup.md
│   └── examples/           # Example code and outputs
└── notebooks/             # Interactive notebooks
    ├── lesson_01/
    │   ├── basics.ipynb
    │   └── exercises/
    │       └── graph_creation.ipynb
    └── lesson_02/
        ├── setup.ipynb
        └── exercises/
            └── component_integration.ipynb
```

### 3. Course Content and Notebook Structure

#### Jupyter Notebook Template Structure
```
notebooks/
├── templates/
│   ├── lesson_template.ipynb      # Base template for all lessons
│   ├── exercise_template.ipynb    # Template for interactive exercises
│   └── solution_template.ipynb    # Template for exercise solutions
├── lessons/
│   ├── 01_windsurf_basics/
│   │   ├── lesson.ipynb
│   │   ├── exercises/
│   │   │   ├── 01_navigation.ipynb
│   │   │   └── 02_ui_interaction.ipynb
│   │   └── solutions/
│   │       ├── 01_navigation_solution.ipynb
│   │       └── 02_ui_interaction_solution.ipynb
│   └── 02_guided_setup/
│       ├── lesson.ipynb
│       ├── exercises/
│       │   ├── 01_environment_setup.ipynb
│       │   └── 02_visualization_basics.ipynb
│       └── solutions/
│           ├── 01_environment_setup_solution.ipynb
│           └── 02_visualization_basics_solution.ipynb
└── resources/
    ├── images/
    └── data/
```

#### Lesson .01 - Getting Started with Visualization
1. Welcome to the Course
   - Repository overview and structure
   - Quick verification of Windsurf setup
   - Course objectives and flow
   - How to use this repository

2. Understanding Graph Visualization
   - Core visualization concepts
   - Graph data structures in habitat
   - Basic graph layouts and styling
   - Real-time updates and WebSocket basics

3. Your First Graph Component
   - Loading the example repository
   - Exploring the visualization codebase
   - Understanding component relationships
   - Running the example graph

4. Interactive Exercises
   - Graph Component Exploration
   - Basic Graph Customization
   - Real-time Update Implementation
   - Component Integration Practice

#### Lesson .02 - Guided Setup
1. Development Environment Setup
   - Python environment configuration
   - Required dependencies installation
   - Project structure creation
   - Virtual environment setup

2. Visualization Components
   - Understanding component architecture
   - Loading and initializing components
   - Basic graph visualization setup
   - WebSocket integration basics

3. Creating Your First Graph View
   - Graph data structure setup
   - Basic layout configuration
   - Node and edge styling
   - Interactive graph features

4. Interactive Exercises
   - Environment Setup Verification
   - Component Integration Practice
   - Basic Graph Creation Exercise
   - Real-time Update Implementation

### 4. Core Components to Migrate
- GraphVisualizer (simplified version)
- LayoutEngine (basic functionality)
- Basic WebSocket support
- Minimal API endpoints

### 5. Testing Requirements
- Basic unit tests for visualization
- Simple integration test
- Test fixtures for course exercises

### 6. Documentation
- README.md with setup instructions
- Basic API documentation
- Lesson guides in Markdown
- Exercise solutions

## Implementation Status and Next Steps

### Phase 1: Foundation (Completed)
- [x] Project structure created
- [x] Dependency injection configured
- [x] Core visualization components implemented
  - GraphVisualizer with plotly integration
  - LayoutEngine with multiple layouts
  - Basic test suite passing

### Phase 2: Course Development (In Progress)
- [x] Notebook templates created
- [x] First lesson notebook implemented
- [ ] Complete exercise notebooks
- [ ] Implement WebSocket integration
- [ ] Create advanced visualization examples

### Phase 3: Documentation & Testing (In Progress)
- [x] Basic tests implemented
- [x] Initial documentation created
- [ ] Complete instructor guides
- [ ] Add integration tests
- [ ] Create troubleshooting guides

### Senior Developer Priorities
1. Architecture
   - WebSocket integration design
   - Component event system
   - Performance optimization
   - Security considerations

2. Course Structure
   - Advanced lesson planning
   - Integration examples
   - Review process setup

### Junior Developer Priorities
1. Features
   - Visualization customization
   - User interaction improvements
   - Basic exercises

2. Documentation
   - API documentation
   - Usage examples
   - Setup guides

### Learning Approach
- **Self-Paced**: Course adapts to individual learning speeds
- **Interactive**: Combines hands-on coding with guided exploration
- **Flexible**: Supports both quick runs and deep dives
- **Instructor-Supported**: Provides guidance while maintaining independence

## Technical Constraints
- Keep dependencies minimal
- Focus on core visualization functionality
- Use simple file-based storage
- Limit WebSocket complexity

## Success Criteria
- [x] Participants can set up Windsurf environment
- [x] Basic visualization components work
- [x] Course modules are completable independently
- [x] Tests pass
- [x] Documentation is clear and sufficient

## Out of Scope
- Advanced visualization features
- Complex data persistence
- Production deployment
- Multiple user support
- Advanced error handling
- Performance optimization

## Notes
- Focus on instructor-led scenarios
- Keep setup process simple
- Prioritize working examples over completeness
- Document known limitations