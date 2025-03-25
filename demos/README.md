# HarmonicIO Dataflow System Demos

This directory contains demonstration scripts for the HarmonicIO dataflow system, showcasing how data flows through the system without preset transformations, allowing patterns to emerge naturally.

## Overview

The HarmonicIO system is built on the principles of pattern evolution and co-evolution, enabling the detection and evolution of coherent patterns while observing semantic change across the system. These demos illustrate the modality-agnostic nature of the Habitat Pattern Language framework, where semantic patterns can be preserved across different representations.

## Demos

### 1. HarmonicIO Dataflow Demo

The `harmonic_io_demo.py` script demonstrates the core functionality of the HarmonicIO dataflow system:

- Processing climate risk data through the harmonic I/O service
- Tracking actant journeys across semantic domains
- Visualizing the dataflow as a network graph
- Creating an "Actant Transformation Story" that narrates how actants transform as they journey through different domains

To run this demo:

```bash
cd /path/to/habitat-windsurf
python demos/harmonic_io_demo.py
```

The demo will generate:

- A dataflow visualization image (`harmonic_io_dataflow.png`)
- An actant transformation story markdown file (`actant_transformation_story.md`)
- A log file (`harmonic_io_demo.log`)

### 2. Query as Actant Demo

The `query_as_actant_demo.py` script demonstrates how queries can function as first-class actants in the semantic ecosystem:

- Creating query actants with AdaptiveIDs
- Processing queries through different semantic domains
- Tracking how queries evolve and transform as they journey through the system
- Visualizing query journeys as a network graph
- Creating a narrative of query transformations

To run this demo:

```bash
cd /path/to/habitat-windsurf
python demos/query_as_actant_demo.py
```

The demo will generate:

- A query journey visualization image (`query_journey_visualization.png`)
- A query transformation narrative markdown file (`query_transformation_narrative.md`)
- A log file (`query_as_actant_demo.log`)

## Sample Data

The demos use sample climate risk data located in the `data/climate_risk` directory:

- `sea_level_rise.json`: Data about sea level rise projections
- `economic_impact.json`: Data about economic impacts of sea level rise
- `policy_response.json`: Data about policy responses to address sea level rise

## Requirements

The demos require the following dependencies:

- Python 3.8+
- matplotlib
- networkx
- The Habitat Evolution package (included in the repository)

## Key Concepts

### Actant Journeys

Actants are entities that carry predicates across domain boundaries, creating a form of narrative structure or "character building" as concepts transform. The `ActantJourney` class tracks how actants move through different semantic domains and how their relationships change over time.

### AdaptiveID

The `AdaptiveID` class represents adaptive concepts with versioning, relationships, and context tracking capabilities. It enables patterns to function as first-class entities that can influence the system's behavior through feedback loops.

### HarmonicIO Service

The `HarmonicIOService` harmonizes I/O operations with system rhythms, ensuring that database operations don't disrupt the natural evolution of eigenspaces and pattern detection. It implements a priority queue for operations, scheduling them according to harmonic timing.

### Pattern Propensities

Pattern propensities describe how patterns tend to evolve and transform. Key metrics include:

- **Coherence**: How well a pattern maintains its identity across contexts
- **Capaciousness**: A pattern's ability to absorb and integrate new information
- **Directionality**: A pattern's tendency to evolve in particular directions

### Modality-Agnostic Approach

The Habitat Pattern Language framework enables a modality-agnostic approach to knowledge representation, where semantic patterns can be preserved across different representations (text, image, audio, video, interactive systems). This allows for medium-independent knowledge representation and preservation of semantic identity across transformations.
