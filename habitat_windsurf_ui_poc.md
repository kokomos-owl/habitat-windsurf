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

### 3. Lesson Content

#### Lesson .01 - Windsurf Basics
- Introduction to Windsurf IDE
- Basic concepts and terminology
- Simple visualization example
- Interactive exercises:
  - Opening and navigating Windsurf
  - Basic UI component interaction

#### Lesson .02 - Guided Setup
- Setting up development environment
- Loading visualization components
- Creating first graph view
- Interactive exercises:
  - Environment configuration
  - Running basic visualization

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