# Habitat Windsurf UI Workshop

A hands-on workshop for learning visualization component development using the Windsurf IDE. This workshop implements a balanced approach between user-driven learning and agent-assisted development, ensuring optimal skill development and understanding.

## Features

### Real-time Network Visualization
- Interactive force-directed graph visualization
- Dark theme with grid background
- Smooth animations and transitions
- Drag and zoom capabilities
- Stage-based filtering

### Error Handling and Status
- Real-time status updates with glowing indicators
- Step-by-step process visualization
- Clear error messaging and state tracking
- WebSocket-based live updates

### Developer Experience
- Modular component architecture
- Clear separation of concerns
- Comprehensive error handling
- Real-time debugging capabilities
- Extensive documentation

## Innovation and Value Proposition

This approach is genuinely innovative and powerful for several key reasons:

### Learning Evolution 🌱
- Traditional workshops: Fixed pace, one-size-fits-all
- This approach: Adaptive learning that respects individual rhythms
- Users can deeply explore areas of interest while maintaining forward momentum

### Agent-Human Partnership 🤝
- Not just automation, but true collaboration
- Agents provide scaffolding without taking over
- Users maintain agency while having support exactly when needed
- Creates a "learning conversation" rather than just following instructions

### Habitat-Windsurf Synergy ⚡
- Habitat's strength: Knowledge evolution and coherence
- Windsurf's strength: Agent-assisted development
- Combined: A powerful platform for teaching complex technical concepts
- Each platform enhances the other's value proposition

### Workshop Builder Innovation 🏗️
- Goes beyond static content
- Environment adapts to user progress
- Validates learning in real-time
- Provides just-in-time resources
- Creates a living, breathing learning space

### User-Centric Design 👤
- Users control their learning journey
- Support available but not intrusive
- Progress at natural pace
- Build confidence through guided exploration
- Learn by doing, not just reading

### Scalability 📈
- Workshop content can evolve
- New components can be added
- Learning patterns can be analyzed
- Community can contribute
- Framework can expand to new domains

### Workshop Differentiation 🎯
- Unique blend of self-paced and guided learning
- Leverages cutting-edge AI capabilities
- Creates memorable learning experiences
- Builds user confidence and competence
- Sets new standard for technical education

## Evolution in Technical Education

This approach represents a significant evolution in technical education:

- It respects the learner's autonomy
- Provides intelligent support
- Creates a rich learning environment
- Scales efficiently
- Delivers measurable results

Most importantly, it creates a model that could revolutionize how we think about technical workshops - moving from static content delivery to dynamic, intelligent learning environments that grow with their users.

## 🚀 Quick Start

### Balanced Learning Approach
This workshop combines:
- User-driven exercises for hands-on learning
- Agent-assisted development for efficient implementation
- Collaborative problem-solving for optimal understanding

### Option 1: Workshop Builder
Start with our hybrid approach that balances automation and manual implementation:

```bash
# Clone repository
git clone https://github.com/your-org/habitat-windsurf.git
cd habitat-windsurf

# Build workshop environment
python scripts/agentic/workshop_builder.py

# Start with first lesson
jupyter notebook notebooks/jumpstart/habitat_windsurf_jumpstart.ipynb
```

### Option 2: Step-by-Step Setup
1. Clone repository (user action)
2. Install dependencies (agent-assisted)
3. Explore notebooks (guided learning)
4. Implement components (hands-on practice)

## 📚 Course Structure

### 1. Jumpstart (Getting Started)
- Workshop builder exploration
- Environment setup
- Basic concepts

### 2. Core Components
- GraphVisualizer with Plotly integration
- MongoDB for data persistence
- Neo4j for graph operations (optional)
- WebSocket for real-time updates
- FastAPI REST endpoints

### 3. Advanced Topics
- Custom graph layouts
- Real-time visualization updates
- MongoDB and Neo4j integration
- Performance optimization
- Containerized deployment

## 🛠 Development

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- MongoDB
- Neo4j (optional)
- FastAPI
- NetworkX
- Plotly

### Directory Structure
```
habitat-windsurf/
├── notebooks/           # Interactive lessons
│   ├── jumpstart/      # Getting started
│   ├── lesson_01/      # Basic visualization
│   └── lesson_02/      # Advanced features
├── scripts/
│   └── agentic/        # Automation tools
├── src/
│   ├── core/           # Core components
│   ├── visualization/  # Visualization service
│   │   ├── api/       # FastAPI endpoints
│   │   ├── core/      # Core visualization logic
│   │   └── websocket/ # Real-time updates
│   └── tests/         # Test suite
└── docker/            # Container configurations
```

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Submit pull request

## 📖 Documentation

This repository is documented through several key files:

### Core Documentation
- [State](STATE.md) - Current project state, development status, and next steps
- [Handoff](HANDOFF.md) - Development handoff notes, setup instructions, and component status
- [POC](habitat_windsurf_ui_poc.md) - Proof of concept details and requirements

### Additional Resources
- [API Documentation](src/visualization/api/README.md) - REST API endpoints and WebSocket interface
- [Component Guide](src/visualization/core/README.md) - Core visualization components
- [Testing Guide](src/tests/README.md) - Test suite organization and execution

### Important Notes
1. MongoDB document IDs (_id and doc_id) are currently removed from responses for workshop compatibility
2. These fields will be preserved in future versions for habitat_evolution integration
3. Neo4j integration is optional and can be disabled if not needed

## 🧪 Testing

```bash
# Start required services
docker-compose up -d mongodb neo4j

# Run test suite
python -m pytest

# Run specific test category
python -m pytest src/tests/unit/
python -m pytest src/tests/integration/
```

The test suite includes:
- Unit tests for core components
- Integration tests for API endpoints
- WebSocket connection tests
- Database client tests

## 🧹 Cleanup

To reset workshop environment:
```bash
# Stop and remove containers
docker-compose down

# Clean workshop environment
python scripts/agentic/workshop_builder.py clean
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.