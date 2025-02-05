# Habitat Windsurf UI Course

A hands-on workshop for learning visualization component development using the Windsurf IDE. This course implements a balanced approach between user-driven learning and agent-assisted development, ensuring optimal skill development and understanding.

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
- GraphVisualizer
- LayoutEngine
- WebSocket integration

### 3. Advanced Topics
- Custom layouts
- Real-time updates
- Performance optimization

## 🛠 Development

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- NetworkX
- Plotly

### Directory Structure
```
habitat-windsurf/
├── notebooks/           # Interactive lessons
│   ├── jumpstart/      # Getting started
│   └── lesson_01/      # Basic visualization
├── scripts/
│   └── agentic/        # Automation tools
├── src/
│   └── core/           # Core components
└── tests/              # Test suite
```

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Submit pull request

## 📖 Documentation

- [State](STATE.md) - Current project state
- [Handoff](HANDOFF.md) - Development handoff notes
- [POC](habitat_windsurf_ui_poc.md) - Proof of concept details

## 🧪 Testing

```bash
# Run test suite
python -m pytest tests/

# Run specific test
python -m pytest tests/test_graph_visualization.py
```

## 🧹 Cleanup

To reset workshop environment:
```bash
python scripts/agentic/workshop_builder.py clean
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.