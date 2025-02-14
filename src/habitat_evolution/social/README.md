# Social Pattern Evolution

This package implements social pattern detection, analysis, and evolution tracking across personal, community, and organizational scales.

## Directory Structure

```
social/
├── core/                   # Core implementation
│   ├── field.py           # Social field dynamics
│   ├── pattern.py         # Pattern detection/evolution
│   └── practice.py        # Practice formation/stability
│
├── data/                  # Data management
│   ├── crawlers/          # Source-specific crawlers
│   ├── processors/        # Data cleaning/preparation
│   └── storage/           # Data persistence
│
├── analysis/             # Analysis tools
│   ├── personal/         # Individual pattern analysis
│   ├── community/        # Group pattern analysis
│   └── organization/     # Institutional pattern analysis
│
├── visualization/        # Visualization tools
│   ├── field_viz/       # Field state visualization
│   ├── pattern_viz/     # Pattern evolution viz
│   └── network_viz/     # Neo4j graph visualization
│
└── tests/               # Test suite
    ├── integration/     # Integration tests
    ├── unit/           # Unit tests
    └── data/           # Test data
```

## Components

### Core Components

1. Field Dynamics (`core/field.py`)
   - Social field state management
   - Energy/flow calculations
   - Field interactions

2. Pattern Evolution (`core/pattern.py`)
   - Pattern detection
   - Evolution tracking
   - State transitions

3. Practice Formation (`core/practice.py`)
   - Practice emergence
   - Stability analysis
   - Pattern institutionalization

### Data Management

1. Crawlers (`data/crawlers/`)
   - Source-specific data collection
   - Rate limiting
   - Content extraction

2. Processors (`data/processors/`)
   - Data cleaning
   - Feature extraction
   - Pattern tagging

3. Storage (`data/storage/`)
   - Neo4j integration
   - Pattern persistence
   - History tracking

### Analysis Tools

1. Personal Analysis
   - Individual pattern detection
   - Practice formation
   - Adaptation tracking

2. Community Analysis
   - Group pattern emergence
   - Resource flow analysis
   - Network effects

3. Organization Analysis
   - Institutional dynamics
   - Practice diffusion
   - Culture evolution

## Usage

See individual component READMEs for detailed usage instructions.
