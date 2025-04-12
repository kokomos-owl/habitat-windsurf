#!/usr/bin/env python3
"""
Script to create a clean Habitat POC with only the necessary files.

This script:
1. Clones the habitat_alpha repository to habitat_poc.01
2. Creates a temporary directory with only the required files
3. Tests that the climate_e2e test runs in the temporary directory
4. If successful, removes all unnecessary files from habitat_poc.01
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Define paths
HABITAT_ALPHA_PATH = "/Users/prphillips/Documents/GitHub/habitat_alpha"
HABITAT_POC_PATH = "/Users/prphillips/Documents/GitHub/habitat_poc.01"
TEMP_DIR = "/Users/prphillips/Documents/GitHub/habitat_poc_temp"

# Required files from dependency analysis
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
    "src/habitat_evolution/infrastructure/adapters/pattern_adaptive_id_adapter.py",
    "src/habitat_evolution/infrastructure/interfaces/__init__.py",
    "src/habitat_evolution/infrastructure/interfaces/persistence/__init__.py",
    "src/habitat_evolution/infrastructure/interfaces/persistence/arangodb_connection_interface.py",
    "src/habitat_evolution/infrastructure/interfaces/persistence/database_connection_interface.py",
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
    "src/habitat_evolution/infrastructure/services/__init__.py",
    "src/habitat_evolution/infrastructure/services/bidirectional_flow_service.py",
    "src/habitat_evolution/infrastructure/services/claude_pattern_extraction_service.py",
    "src/habitat_evolution/infrastructure/services/event_service.py",
    "src/habitat_evolution/infrastructure/services/pattern_evolution_service.py",
    
    # Adaptive Core
    "src/habitat_evolution/adaptive_core/__init__.py",
    "src/habitat_evolution/adaptive_core/emergence/__init__.py",
    "src/habitat_evolution/adaptive_core/emergence/climate_data_loader.py",
    "src/habitat_evolution/adaptive_core/emergence/event_aware_detector.py",
    "src/habitat_evolution/adaptive_core/emergence/event_bus_integration.py",
    "src/habitat_evolution/adaptive_core/emergence/interfaces/__init__.py",
    "src/habitat_evolution/adaptive_core/emergence/interfaces/field_observer.py",
    "src/habitat_evolution/adaptive_core/emergence/interfaces/field_state_repository.py",
    "src/habitat_evolution/adaptive_core/emergence/interfaces/learning_window_observer.py",
    "src/habitat_evolution/adaptive_core/emergence/interfaces/pattern_observer.py",
    "src/habitat_evolution/adaptive_core/emergence/interfaces/pattern_repository.py",
    "src/habitat_evolution/adaptive_core/emergence/interfaces/relationship_repository.py",
    "src/habitat_evolution/adaptive_core/emergence/interfaces/topology_repository.py",
    "src/habitat_evolution/adaptive_core/emergence/learning_window_integration.py",
    "src/habitat_evolution/adaptive_core/emergence/persistence_integration.py",
    "src/habitat_evolution/adaptive_core/emergence/repository_factory.py",
    "src/habitat_evolution/adaptive_core/emergence/tonic_harmonic_integration.py",
    "src/habitat_evolution/adaptive_core/emergence/vector_tonic_persistence_connector.py",
    "src/habitat_evolution/adaptive_core/emergence/vector_tonic_window_integration.py",
    "src/habitat_evolution/adaptive_core/id/__init__.py",
    "src/habitat_evolution/adaptive_core/id/adaptive_id.py",
    "src/habitat_evolution/adaptive_core/io/__init__.py",
    "src/habitat_evolution/adaptive_core/io/harmonic_io_service.py",
    "src/habitat_evolution/adaptive_core/models/__init__.py",
    "src/habitat_evolution/adaptive_core/models/pattern.py",
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
    "src/habitat_evolution/adaptive_core/persistence/interfaces/pattern_repository.py",
    "src/habitat_evolution/adaptive_core/persistence/interfaces/relationship_repository.py",
    "src/habitat_evolution/adaptive_core/persistence/interfaces/topology_repository.py",
    "src/habitat_evolution/adaptive_core/resonance/__init__.py",
    "src/habitat_evolution/adaptive_core/resonance/tonic_harmonic_metrics.py",
    "src/habitat_evolution/adaptive_core/services/__init__.py",
    "src/habitat_evolution/adaptive_core/services/interfaces.py",
    
    # Climate Risk Processing
    "src/habitat_evolution/climate_risk/__init__.py",
    "src/habitat_evolution/climate_risk/document_processing_service.py",
    
    # Core Services
    "src/habitat_evolution/core/__init__.py",
    "src/habitat_evolution/core/pattern/__init__.py",
    "src/habitat_evolution/core/pattern/pattern_model.py",
    "src/habitat_evolution/core/pattern/pattern_quality.py",
    "src/habitat_evolution/core/services/__init__.py",
    "src/habitat_evolution/core/services/event_bus.py",
    "src/habitat_evolution/core/services/field/__init__.py",
    "src/habitat_evolution/core/services/field/interfaces.py",
    
    # Field Components
    "src/habitat_evolution/field/__init__.py",
    "src/habitat_evolution/field/field_state.py",
    "src/habitat_evolution/field/harmonic_field_io_bridge.py",
    "src/habitat_evolution/field/topological_field_analyzer.py",
    
    # Pattern-Aware RAG
    "src/habitat_evolution/pattern_aware_rag/__init__.py",
    "src/habitat_evolution/pattern_aware_rag/core/__init__.py",
    "src/habitat_evolution/pattern_aware_rag/core/coherence_interface.py",
    "src/habitat_evolution/pattern_aware_rag/learning/__init__.py",
    "src/habitat_evolution/pattern_aware_rag/learning/learning_control.py",
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
    
    # Data Files
    "data/time_series/MA_AvgTemp_91_24.json",
    "data/time_series/NE_AvgTemp_91_24.json",
    
    # Configuration Files
    "pytest.ini",
    "requirements.txt",
]

def clone_repository():
    """Clone the repository to the POC directory."""
    if os.path.exists(HABITAT_POC_PATH):
        print(f"Directory {HABITAT_POC_PATH} already exists. Please remove it first.")
        return False
    
    print(f"Cloning repository to {HABITAT_POC_PATH}...")
    subprocess.run(["git", "clone", HABITAT_ALPHA_PATH, HABITAT_POC_PATH], check=True)
    return True

def create_temp_directory():
    """Create a temporary directory with only the required files."""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    
    os.makedirs(TEMP_DIR)
    print(f"Created temporary directory at {TEMP_DIR}")
    
    # Copy required files
    for file_path in REQUIRED_FILES:
        src_path = os.path.join(HABITAT_ALPHA_PATH, file_path)
        dst_path = os.path.join(TEMP_DIR, file_path)
        
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # Copy the file
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {file_path}")
        else:
            print(f"Warning: {file_path} does not exist")
            
            # Create empty __init__.py files for missing directories
            if file_path.endswith("__init__.py"):
                with open(dst_path, "w") as f:
                    f.write("# Auto-generated __init__.py\n")
                print(f"Created empty {file_path}")
    
    # Fix circular import in core/__init__.py
    core_init_path = os.path.join(TEMP_DIR, "src", "habitat_evolution", "core", "__init__.py")
    if os.path.exists(core_init_path):
        with open(core_init_path, "w") as f:
            f.write("# Fixed to avoid circular imports\n")
        print("Fixed circular import in core/__init__.py")
    
    # Copy climate_risk directory
    climate_risk_src = os.path.join(HABITAT_ALPHA_PATH, "data", "climate_risk")
    climate_risk_dst = os.path.join(TEMP_DIR, "data", "climate_risk")
    if os.path.exists(climate_risk_src):
        shutil.copytree(climate_risk_src, climate_risk_dst)
        print(f"Copied data/climate_risk directory")
    else:
        print(f"Warning: data/climate_risk directory does not exist")
    
    # Create time_series directory and sample data files if they don't exist
    time_series_dst = os.path.join(TEMP_DIR, "data", "time_series")
    os.makedirs(time_series_dst, exist_ok=True)
    
    # Create sample MA_AvgTemp_91_24.json if it doesn't exist
    ma_temp_path = os.path.join(time_series_dst, "MA_AvgTemp_91_24.json")
    if not os.path.exists(ma_temp_path):
        with open(ma_temp_path, "w") as f:
            f.write('{\n  "199101": {"avg_temp": 32.1, "anomaly": 1.2},\n  "199102": {"avg_temp": 33.5, "anomaly": 0.8}\n}')
        print("Created sample MA_AvgTemp_91_24.json")
    
    # Create sample NE_AvgTemp_91_24.json if it doesn't exist
    ne_temp_path = os.path.join(time_series_dst, "NE_AvgTemp_91_24.json")
    if not os.path.exists(ne_temp_path):
        with open(ne_temp_path, "w") as f:
            f.write('{\n  "199101": {"avg_temp": 30.5, "anomaly": 0.9},\n  "199102": {"avg_temp": 31.8, "anomaly": 0.6}\n}')
        print("Created sample NE_AvgTemp_91_24.json")
    
    return True

def test_temp_directory():
    """Test that the climate_e2e test runs in the temporary directory."""
    print("Testing climate_e2e test in temporary directory...")
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/integration/climate_e2e/test_climate_e2e.py", "-v"],
        cwd=TEMP_DIR,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("Test successful!")
        return True
    else:
        print("Test failed with the following output:")
        print(result.stdout)
        print(result.stderr)
        return False

def clean_poc_directory():
    """Remove unnecessary files from the POC directory."""
    print("Cleaning POC directory...")
    
    # Get all files in the POC directory
    all_files = []
    for root, dirs, files in os.walk(HABITAT_POC_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, HABITAT_POC_PATH)
            all_files.append(rel_path)
    
    # Convert required files to set for faster lookup
    required_files_set = set(REQUIRED_FILES)
    
    # Add climate_risk files to required files
    climate_risk_dir = os.path.join(HABITAT_POC_PATH, "data", "climate_risk")
    if os.path.exists(climate_risk_dir):
        for root, dirs, files in os.walk(climate_risk_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, HABITAT_POC_PATH)
                required_files_set.add(rel_path)
    
    # Remove files that are not required
    for file_path in all_files:
        if file_path not in required_files_set and not file_path.startswith(".git/"):
            full_path = os.path.join(HABITAT_POC_PATH, file_path)
            os.remove(full_path)
            print(f"Removed {file_path}")
    
    # Remove empty directories
    for root, dirs, files in os.walk(HABITAT_POC_PATH, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path) and ".git" not in dir_path:
                os.rmdir(dir_path)
                print(f"Removed empty directory {os.path.relpath(dir_path, HABITAT_POC_PATH)}")
    
    # Fix circular import in core/__init__.py in POC directory
    core_init_path = os.path.join(HABITAT_POC_PATH, "src", "habitat_evolution", "core", "__init__.py")
    if os.path.exists(core_init_path):
        with open(core_init_path, "w") as f:
            f.write("# Fixed to avoid circular imports\n")
        print("Fixed circular import in core/__init__.py in POC directory")
    
    # Create missing __init__.py files in POC directory
    for file_path in REQUIRED_FILES:
        if file_path.endswith("__init__.py"):
            dst_path = os.path.join(HABITAT_POC_PATH, file_path)
            if not os.path.exists(dst_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                with open(dst_path, "w") as f:
                    f.write("# Auto-generated __init__.py\n")
                print(f"Created empty {file_path} in POC directory")
    
    # Create time_series directory and sample data files in POC directory if they don't exist
    time_series_dst = os.path.join(HABITAT_POC_PATH, "data", "time_series")
    os.makedirs(time_series_dst, exist_ok=True)
    
    # Create sample MA_AvgTemp_91_24.json if it doesn't exist
    ma_temp_path = os.path.join(time_series_dst, "MA_AvgTemp_91_24.json")
    if not os.path.exists(ma_temp_path):
        with open(ma_temp_path, "w") as f:
            f.write('{\n  "199101": {"avg_temp": 32.1, "anomaly": 1.2},\n  "199102": {"avg_temp": 33.5, "anomaly": 0.8}\n}')
        print("Created sample MA_AvgTemp_91_24.json in POC directory")
    
    # Create sample NE_AvgTemp_91_24.json if it doesn't exist
    ne_temp_path = os.path.join(time_series_dst, "NE_AvgTemp_91_24.json")
    if not os.path.exists(ne_temp_path):
        with open(ne_temp_path, "w") as f:
            f.write('{\n  "199101": {"avg_temp": 30.5, "anomaly": 0.9},\n  "199102": {"avg_temp": 31.8, "anomaly": 0.6}\n}')
        print("Created sample NE_AvgTemp_91_24.json in POC directory")
    
    return True

def main():
    """Main function."""
    print("Creating Habitat POC...")
    
    # Clone repository
    if not clone_repository():
        return
    
    # Create temporary directory
    if not create_temp_directory():
        return
    
    # Test temporary directory
    if not test_temp_directory():
        print("Test failed. Aborting.")
        return
    
    # Clean POC directory
    if not clean_poc_directory():
        return
    
    print("Habitat POC created successfully!")
    print(f"POC directory: {HABITAT_POC_PATH}")
    print("You can now run the climate_e2e test in the POC directory:")
    print(f"cd {HABITAT_POC_PATH} && python -m pytest tests/integration/climate_e2e/test_climate_e2e.py -v")

if __name__ == "__main__":
    main()
