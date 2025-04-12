#!/usr/bin/env python3
"""
Script to clean up the Habitat POC repository by removing all unnecessary files.
This will keep only the files needed for the climate_e2e tests to run successfully.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Define paths
HABITAT_POC_PATH = "/Users/prphillips/Documents/GitHub/habitat_poc.01"

# Required files from our successful POC
REQUIRED_FILES = [
    # Test Files
    "tests/integration/climate_e2e/__init__.py",
    "tests/integration/climate_e2e/conftest.py",
    "tests/integration/climate_e2e/test_climate_e2e.py",
    "tests/integration/climate_e2e/test_utils.py",
    
    # Core Infrastructure
    "src/habitat_evolution/__init__.py",
    "src/habitat_evolution/infrastructure/__init__.py",
    "src/habitat_evolution/infrastructure/adapters/__init__.py",
    "src/habitat_evolution/infrastructure/adapters/claude_adapter.py",
    "src/habitat_evolution/infrastructure/adapters/claude_cache.py",
    "src/habitat_evolution/infrastructure/adapters/pattern_adapter.py",
    "src/habitat_evolution/infrastructure/adapters/pattern_bridge.py",
    "src/habitat_evolution/infrastructure/adapters/pattern_monkey_patch.py",
    "src/habitat_evolution/infrastructure/adapters/pattern_adaptive_id_adapter.py",
    "src/habitat_evolution/infrastructure/interfaces/__init__.py",
    "src/habitat_evolution/infrastructure/interfaces/persistence/__init__.py",
    "src/habitat_evolution/infrastructure/interfaces/persistence/arangodb_connection_interface.py",
    "src/habitat_evolution/infrastructure/interfaces/persistence/database_connection_interface.py",
    "src/habitat_evolution/infrastructure/interfaces/repositories/__init__.py",
    "src/habitat_evolution/infrastructure/interfaces/repositories/repository_interface.py",
    "src/habitat_evolution/infrastructure/interfaces/repositories/graph_repository_interface.py",
    "src/habitat_evolution/infrastructure/interfaces/service_interface.py",
    "src/habitat_evolution/infrastructure/interfaces/services/__init__.py",
    "src/habitat_evolution/infrastructure/interfaces/services/bidirectional_flow_interface.py",
    "src/habitat_evolution/infrastructure/interfaces/services/event_service_interface.py",
    "src/habitat_evolution/infrastructure/interfaces/services/pattern_aware_rag_interface.py",
    "src/habitat_evolution/infrastructure/interfaces/services/pattern_evolution_interface.py",
    "src/habitat_evolution/infrastructure/metrics/__init__.py",
    "src/habitat_evolution/infrastructure/metrics/claude_api_metrics.py",
    "src/habitat_evolution/infrastructure/persistence/__init__.py",
    "src/habitat_evolution/infrastructure/persistence/arangodb/__init__.py",
    "src/habitat_evolution/infrastructure/persistence/arangodb/arangodb_connection.py",
    "src/habitat_evolution/infrastructure/persistence/arangodb/arangodb_graph_repository.py",
    "src/habitat_evolution/infrastructure/persistence/arangodb/arangodb_pattern_repository.py",
    "src/habitat_evolution/infrastructure/persistence/arangodb/arangodb_repository.py",
    "src/habitat_evolution/infrastructure/services/__init__.py",
    "src/habitat_evolution/infrastructure/services/bidirectional_flow_service.py",
    "src/habitat_evolution/infrastructure/services/claude_pattern_extraction_service.py",
    "src/habitat_evolution/infrastructure/services/event_service.py",
    "src/habitat_evolution/infrastructure/services/pattern_evolution_service.py",
    
    # Adaptive Core
    "src/habitat_evolution/adaptive_core/__init__.py",
    "src/habitat_evolution/adaptive_core/emergence/__init__.py",
    "src/habitat_evolution/adaptive_core/emergence/climate_data_loader.py",
    "src/habitat_evolution/adaptive_core/emergence/emergent_pattern_detector.py",
    "src/habitat_evolution/adaptive_core/emergence/enhanced_semantic_observer.py",
    "src/habitat_evolution/adaptive_core/emergence/event_aware_detector.py",
    "src/habitat_evolution/adaptive_core/emergence/event_bus_integration.py",
    "src/habitat_evolution/adaptive_core/emergence/integration_service.py",
    "src/habitat_evolution/adaptive_core/emergence/interfaces/__init__.py",
    "src/habitat_evolution/adaptive_core/emergence/interfaces/field_observer.py",
    "src/habitat_evolution/adaptive_core/emergence/interfaces/field_state_repository.py",
    "src/habitat_evolution/adaptive_core/emergence/interfaces/learning_window_observer.py",
    "src/habitat_evolution/adaptive_core/emergence/interfaces/pattern_observer.py",
    "src/habitat_evolution/adaptive_core/emergence/interfaces/pattern_repository.py",
    "src/habitat_evolution/adaptive_core/emergence/interfaces/relationship_repository.py",
    "src/habitat_evolution/adaptive_core/emergence/interfaces/topology_repository.py",
    "src/habitat_evolution/adaptive_core/emergence/learning_window_integration.py",
    "src/habitat_evolution/adaptive_core/emergence/pattern_integration.py",
    "src/habitat_evolution/adaptive_core/emergence/persistence_integration.py",
    "src/habitat_evolution/adaptive_core/emergence/repository_factory.py",
    "src/habitat_evolution/adaptive_core/emergence/resonance_trail_observer.py",
    "src/habitat_evolution/adaptive_core/emergence/semantic_current_observer.py",
    "src/habitat_evolution/adaptive_core/emergence/tonic_harmonic_integration.py",
    "src/habitat_evolution/adaptive_core/emergence/vector_tonic_persistence_connector.py",
    "src/habitat_evolution/adaptive_core/emergence/vector_tonic_window_integration.py",
    "src/habitat_evolution/adaptive_core/id/__init__.py",
    "src/habitat_evolution/adaptive_core/id/adaptive_id.py",
    "src/habitat_evolution/adaptive_core/id/base_adaptive_id.py",
    "src/habitat_evolution/adaptive_core/io/__init__.py",
    "src/habitat_evolution/adaptive_core/io/harmonic_io_service.py",
    "src/habitat_evolution/adaptive_core/models/__init__.py",
    "src/habitat_evolution/adaptive_core/models/pattern.py",
    "src/habitat_evolution/adaptive_core/models/relationship.py",
    "src/habitat_evolution/adaptive_core/models/pattern_metadata.py",
    "src/habitat_evolution/adaptive_core/persistence/__init__.py",
    "src/habitat_evolution/adaptive_core/persistence/adapters/__init__.py",
    "src/habitat_evolution/adaptive_core/persistence/adapters/field_state_repository_adapter.py",
    "src/habitat_evolution/adaptive_core/persistence/adapters/pattern_repository_adapter.py",
    "src/habitat_evolution/adaptive_core/persistence/adapters/relationship_repository_adapter.py",
    "src/habitat_evolution/adaptive_core/persistence/adapters/topology_repository_adapter.py",
    "src/habitat_evolution/adaptive_core/persistence/arangodb/__init__.py",
    "src/habitat_evolution/adaptive_core/persistence/arangodb/base_repository.py",
    "src/habitat_evolution/adaptive_core/persistence/arangodb/connection.py",
    "src/habitat_evolution/adaptive_core/persistence/arangodb/graph_state_repository.py",
    "src/habitat_evolution/adaptive_core/persistence/factory.py",
    "src/habitat_evolution/adaptive_core/persistence/interfaces.py",
    "src/habitat_evolution/adaptive_core/persistence/interfaces/__init__.py",
    "src/habitat_evolution/adaptive_core/persistence/interfaces/field_state_repository.py",
    "src/habitat_evolution/adaptive_core/persistence/interfaces/graph_state_repository.py",
    "src/habitat_evolution/adaptive_core/persistence/interfaces/learning_window_repository.py",
    "src/habitat_evolution/adaptive_core/persistence/interfaces/metrics_repository.py",
    "src/habitat_evolution/adaptive_core/persistence/interfaces/pattern_repository.py",
    "src/habitat_evolution/adaptive_core/persistence/interfaces/relationship_repository.py",
    "src/habitat_evolution/adaptive_core/persistence/interfaces/topology_repository.py",
    "src/habitat_evolution/adaptive_core/resonance/__init__.py",
    "src/habitat_evolution/adaptive_core/resonance/tonic_harmonic_metrics.py",
    "src/habitat_evolution/adaptive_core/services/__init__.py",
    "src/habitat_evolution/adaptive_core/services/interfaces.py",
    "src/habitat_evolution/adaptive_core/transformation/__init__.py",
    "src/habitat_evolution/adaptive_core/transformation/actant_journey_tracker.py",
    
    # Climate Risk Processing
    "src/habitat_evolution/climate_risk/__init__.py",
    "src/habitat_evolution/climate_risk/document_processing_service.py",
    
    # Core Services
    "src/habitat_evolution/core/__init__.py",
    "src/habitat_evolution/core/container.py",
    "src/habitat_evolution/core/config/__init__.py",
    "src/habitat_evolution/core/config/field_config.py",
    "src/habitat_evolution/core/pattern/__init__.py",
    "src/habitat_evolution/core/pattern/evolution.py",
    "src/habitat_evolution/core/pattern/quality.py",
    "src/habitat_evolution/core/services/__init__.py",
    "src/habitat_evolution/core/services/event_bus.py",
    "src/habitat_evolution/core/services/time_provider.py",
    "src/habitat_evolution/core/storage/__init__.py",
    "src/habitat_evolution/core/storage/interfaces.py",
    "src/habitat_evolution/core/services/field/__init__.py",
    "src/habitat_evolution/core/services/field/interfaces.py",
    
    # Field Components
    "src/habitat_evolution/field/__init__.py",
    "src/habitat_evolution/field/field_state.py",
    "src/habitat_evolution/field/field_navigator.py",
    "src/habitat_evolution/field/harmonic_field_io_bridge.py",
    "src/habitat_evolution/field/topological_field_analyzer.py",
    
    # Pattern-Aware RAG Components
    "src/habitat_evolution/pattern_aware_rag/__init__.py",
    "src/habitat_evolution/pattern_aware_rag/core/__init__.py",
    "src/habitat_evolution/pattern_aware_rag/core/coherence_interface.py",
    "src/habitat_evolution/pattern_aware_rag/learning/__init__.py",
    "src/habitat_evolution/pattern_aware_rag/learning/learning_control.py",
    "src/habitat_evolution/pattern_aware_rag/learning/window_manager.py",
    "src/habitat_evolution/pattern_aware_rag/pattern_aware_rag.py",
    "src/habitat_evolution/pattern_aware_rag/persistence/__init__.py",
    "src/habitat_evolution/pattern_aware_rag/persistence/arangodb/__init__.py",
    "src/habitat_evolution/pattern_aware_rag/persistence/arangodb/boundary_repository.py",
    "src/habitat_evolution/pattern_aware_rag/persistence/arangodb/connection_manager.py",
    "src/habitat_evolution/pattern_aware_rag/persistence/arangodb/field_state_repository.py",
    "src/habitat_evolution/pattern_aware_rag/persistence/arangodb/pattern_repository.py",
    "src/habitat_evolution/pattern_aware_rag/persistence/arangodb/predicate_relationship_repository.py",
    "src/habitat_evolution/pattern_aware_rag/persistence/arangodb/topology_repository.py",
    "src/habitat_evolution/pattern_aware_rag/semantic/__init__.py",
    "src/habitat_evolution/pattern_aware_rag/semantic/pattern_semantic.py",
    "src/habitat_evolution/pattern_aware_rag/services/__init__.py",
    "src/habitat_evolution/pattern_aware_rag/services/claude_integration_service.py",
    "src/habitat_evolution/pattern_aware_rag/services/graph_service.py",
    "src/habitat_evolution/pattern_aware_rag/state/__init__.py",
    "src/habitat_evolution/pattern_aware_rag/state/test_states.py",
    "src/habitat_evolution/pattern_aware_rag/topology/__init__.py",
    "src/habitat_evolution/pattern_aware_rag/topology/detector.py",
    "src/habitat_evolution/pattern_aware_rag/topology/manager.py",
    "src/habitat_evolution/pattern_aware_rag/topology/models.py",
    "src/habitat_evolution/pattern_aware_rag/topology/semantic_topology_enhancer.py",
    
    # Vector Tonic Components
    "src/habitat_evolution/vector_tonic/__init__.py",
    "src/habitat_evolution/vector_tonic/bridge/__init__.py",
    "src/habitat_evolution/vector_tonic/bridge/field_pattern_bridge.py",
    "src/habitat_evolution/vector_tonic/field_state/__init__.py",
    "src/habitat_evolution/vector_tonic/field_state/multi_scale_analyzer.py",
    "src/habitat_evolution/vector_tonic/field_state/simple_field_analyzer.py",
    
    # Visualization Components
    "src/habitat_evolution/visualization/__init__.py",
    "src/habitat_evolution/visualization/eigenspace_visualizer.py",
    "src/habitat_evolution/visualization/field_topology_visualizer.py",
    "src/habitat_evolution/visualization/pattern_id.py",
    "src/habitat_evolution/visualization/run_visualizer.py",
    "src/habitat_evolution/visualization/semantic_validation.py",
    "src/habitat_evolution/visualization/topological_temporal_visualizer.py",
    
    # Data Files
    "data/time_series/MA_AvgTemp_91_24.json",
    "data/time_series/NE_AvgTemp_91_24.json",
    
    # Configuration Files
    "pytest.ini",
    "requirements.txt",
]

# Additional directories to keep
REQUIRED_DIRS = [
    "data/climate_risk",
]

def is_required_file(file_path):
    """Check if a file is in the required files list."""
    return file_path in REQUIRED_FILES

def is_in_required_dir(file_path):
    """Check if a file is in one of the required directories."""
    for required_dir in REQUIRED_DIRS:
        if file_path.startswith(required_dir):
            return True
    return False

def should_keep_file(file_path):
    """Determine if a file should be kept."""
    return is_required_file(file_path) or is_in_required_dir(file_path)

def clean_repository():
    """Remove all files that are not in the required list."""
    print(f"Cleaning repository at {HABITAT_POC_PATH}...")
    
    # Get all files in the repository
    all_files = []
    for root, dirs, files in os.walk(HABITAT_POC_PATH):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, HABITAT_POC_PATH)
            
            # Skip .git directory
            if '.git' in rel_path.split(os.sep):
                continue
                
            all_files.append(rel_path)
    
    # Remove files that are not required
    for file_path in all_files:
        if not should_keep_file(file_path):
            full_path = os.path.join(HABITAT_POC_PATH, file_path)
            try:
                os.remove(full_path)
                print(f"Removed {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    
    # Clean up empty directories
    for root, dirs, files in os.walk(HABITAT_POC_PATH, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path) and '.git' not in dir_path.split(os.sep):
                try:
                    os.rmdir(dir_path)
                    print(f"Removed empty directory {os.path.relpath(dir_path, HABITAT_POC_PATH)}")
                except Exception as e:
                    print(f"Error removing directory {dir_path}: {e}")

def test_repository():
    """Test that the climate_e2e test still runs after cleanup."""
    print("Testing climate_e2e test after cleanup...")
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/integration/climate_e2e/test_climate_e2e.py", "-v"],
        cwd=HABITAT_POC_PATH,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("Test successful! All tests passed after cleanup.")
        return True
    else:
        print("Test failed with the following output:")
        print(result.stdout)
        print(result.stderr)
        return False

def create_readme():
    """Create a README.md file for the POC repository."""
    readme_path = os.path.join(HABITAT_POC_PATH, "README.md")
    
    readme_content = """# Habitat Evolution POC

This repository contains a minimal, isolated Proof of Concept (POC) for the Habitat Evolution system, focusing on the climate_e2e test.

## Overview

Habitat Evolution is built on the principles of pattern evolution and co-evolution. This system is designed to detect and evolve coherent patterns, while enabling the observation of semantic change across the system.

## Key Capabilities

1. **Cross-Modal Pattern Integration**: Detects and analyzes relationships between semantic patterns (extracted from climate risk text documents) and statistical patterns (derived from temperature data).

2. **AdaptiveID Coherence Tracking**: Tracks pattern coherence over time, with versioning and contextual awareness that allows patterns to evolve while maintaining their identity.

3. **Pattern-Enhanced RAG**: Incorporates pattern information into retrieval-augmented generation responses, producing more comprehensive and contextually relevant information.

4. **Spatial-Temporal Context Integration**: Incorporates spatial context (regions) and temporal context (time ranges) into pattern representations.

5. **Field-Pattern Bridge**: Detects multiple types of relationships between patterns, including spatial proximity, temporal sequence, and regional association.

## Data

The system works with:

1. **Temperature Data**: Monthly average temperatures and anomalies from 1991-2024, structured with timestamps in YYYYMM format.

2. **Climate Risk Documents**: Regional assessments containing structured sections covering observations, projections, and impacts.

## Running the Tests

To run the end-to-end test:

```bash
python -m pytest tests/integration/climate_e2e/test_climate_e2e.py -v
```

## Dependencies

See `requirements.txt` for the list of dependencies.
"""
    
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    print(f"Created README.md at {readme_path}")

def main():
    """Main function."""
    if not os.path.exists(HABITAT_POC_PATH):
        print(f"Error: {HABITAT_POC_PATH} does not exist.")
        return
    
    # Clean the repository
    clean_repository()
    
    # Create README.md
    create_readme()
    
    # Test the repository
    test_repository()
    
    print("Cleanup complete!")

if __name__ == "__main__":
    main()
