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

### 2. Minimal Course Structure
```
habitat-windsurf/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── container.py      # Minimal DI container
│   │   └── visualization/    # Migrated from habitat_poc
│   │       ├── __init__.py
│   │       ├── graph.py
│   │       └── layout.py
│   └── course/
│       ├── __init__.py
│       ├── lessons/
│       │   ├── lesson_01_windsurf_basics.py
│       │   └── lesson_02_guided_setup.py
│       └── utils/
│           └── feedback.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_visualization.py
└── notebooks/
    ├── 01_windsurf_basics.ipynb
    └── 02_guided_setup.ipynb
```

### 3. Lesson Content and Notebook Structure

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

#### Lesson .01 - Windsurf Basics
1. Introduction to Windsurf IDE
   - Installation and system requirements
   - Authentication and account setup
   - Editor theme selection
   - Keybinding options (VS Code default vs Vim)

2. Core Windsurf Features
   - Editor layout and components
   - File navigation and management
   - Command palette usage
   - Settings configuration
   - Extension management

3. Basic UI Interactions
   - Opening and managing workspaces
   - File editing and saving
   - Terminal integration
   - Split views and panels
   - Basic shortcuts

4. Interactive Exercises
   - Workspace Navigation Challenge
   - UI Component Discovery Quest
   - Settings Configuration Practice
   - Extension Installation Workshop

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
   - Basic Graph Creation Workshop
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

## Development Phases

### Phase 1: Initial Setup (1-2 days)
- [ ] Create project structure
- [ ] Set up basic dependency injection
- [ ] Migrate core visualization components

### Phase 2: Course Content (2-3 days)
- [ ] Create Lesson .01 content
- [ ] Create Lesson .02 content
- [ ] Develop basic exercises

### Phase 3: Testing & Documentation (1-2 days)
- [ ] Implement core tests
- [ ] Write setup documentation
- [ ] Create exercise solutions

## Technical Constraints
- Keep dependencies minimal
- Focus on core visualization functionality
- Use simple file-based storage
- Limit WebSocket complexity

## Success Criteria
- [x] Students can set up Windsurf environment
- [x] Basic visualization components work
- [x] Lessons are completable independently
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