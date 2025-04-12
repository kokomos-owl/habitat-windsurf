#!/usr/bin/env python3
"""
Script to create a clean Habitat POC with only the necessary files.
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

# Required files from dependency analysis - same as before but adding visualization module
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
    
    # Fix circular imports in __init__.py files
    fix_init_files()
    
    # Copy climate_risk directory
    climate_risk_src = os.path.join(HABITAT_ALPHA_PATH, "data", "climate_risk")
    climate_risk_dst = os.path.join(TEMP_DIR, "data", "climate_risk")
    if os.path.exists(climate_risk_src):
        shutil.copytree(climate_risk_src, climate_risk_dst)
        print(f"Copied data/climate_risk directory")
    else:
        print(f"Warning: data/climate_risk directory does not exist")
    
    # Create time_series directory and sample data files
    create_sample_data_files(TEMP_DIR)
    
    return True

def fix_init_files():
    """Fix circular imports in __init__.py files."""
    # Fix habitat_evolution/__init__.py
    habitat_init_path = os.path.join(TEMP_DIR, "src", "habitat_evolution", "__init__.py")
    if os.path.exists(habitat_init_path):
        with open(habitat_init_path, "w") as f:
            f.write("# Fixed to avoid circular imports\n")
        print("Fixed circular import in habitat_evolution/__init__.py")
    
    # Fix core/__init__.py
    core_init_path = os.path.join(TEMP_DIR, "src", "habitat_evolution", "core", "__init__.py")
    if os.path.exists(core_init_path):
        with open(core_init_path, "w") as f:
            f.write("# Fixed to avoid circular imports\n")
        print("Fixed circular import in core/__init__.py")
    
    # Fix visualization/__init__.py
    vis_init_path = os.path.join(TEMP_DIR, "src", "habitat_evolution", "visualization", "__init__.py")
    if os.path.exists(vis_init_path):
        with open(vis_init_path, "w") as f:
            f.write("# Fixed to avoid circular imports\n")
        print("Fixed circular import in visualization/__init__.py")

def create_sample_data_files(base_dir):
    """Create sample climate data files."""
    time_series_dst = os.path.join(base_dir, "data", "time_series")
    os.makedirs(time_series_dst, exist_ok=True)
    
    # Create sample MA_AvgTemp_91_24.json
    ma_temp_path = os.path.join(time_series_dst, "MA_AvgTemp_91_24.json")
    if not os.path.exists(ma_temp_path):
        sample_data = {
            "199101": {"avg_temp": 32.1, "anomaly": 1.2},
            "199102": {"avg_temp": 33.5, "anomaly": 0.8},
            "199103": {"avg_temp": 42.3, "anomaly": 2.1},
            "199104": {"avg_temp": 51.7, "anomaly": 1.5},
            "199105": {"avg_temp": 62.4, "anomaly": 0.9},
            "199106": {"avg_temp": 72.1, "anomaly": 1.7},
            "202301": {"avg_temp": 35.2, "anomaly": 4.3},
            "202302": {"avg_temp": 36.8, "anomaly": 4.1},
            "202303": {"avg_temp": 45.6, "anomaly": 5.4},
            "202304": {"avg_temp": 54.9, "anomaly": 4.7}
        }
        with open(ma_temp_path, "w") as f:
            import json
            json.dump(sample_data, f, indent=2)
        print("Created sample MA_AvgTemp_91_24.json")
    
    # Create sample NE_AvgTemp_91_24.json
    ne_temp_path = os.path.join(time_series_dst, "NE_AvgTemp_91_24.json")
    if not os.path.exists(ne_temp_path):
        sample_data = {
            "199101": {"avg_temp": 30.5, "anomaly": 0.9},
            "199102": {"avg_temp": 31.8, "anomaly": 0.6},
            "199103": {"avg_temp": 40.7, "anomaly": 1.8},
            "199104": {"avg_temp": 49.2, "anomaly": 1.2},
            "199105": {"avg_temp": 60.1, "anomaly": 0.7},
            "199106": {"avg_temp": 70.3, "anomaly": 1.5},
            "202301": {"avg_temp": 33.7, "anomaly": 4.1},
            "202302": {"avg_temp": 35.2, "anomaly": 4.0},
            "202303": {"avg_temp": 43.9, "anomaly": 5.0},
            "202304": {"avg_temp": 52.4, "anomaly": 4.4}
        }
        with open(ne_temp_path, "w") as f:
            import json
            json.dump(sample_data, f, indent=2)
        print("Created sample NE_AvgTemp_91_24.json")

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
    
    # Fix circular imports in POC directory
    fix_init_files_poc()
    
    # Create sample data files in POC directory
    create_sample_data_files(HABITAT_POC_PATH)
    
    # Copy all required files to POC directory
    for file_path in REQUIRED_FILES:
        src_path = os.path.join(TEMP_DIR, file_path)
        dst_path = os.path.join(HABITAT_POC_PATH, file_path)
        
        if os.path.exists(src_path):
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # Copy the file
            shutil.copy2(src_path, dst_path)
            print(f"Copied {file_path} to POC directory")
    
    # Copy climate_risk directory to POC
    climate_risk_src = os.path.join(TEMP_DIR, "data", "climate_risk")
    climate_risk_dst = os.path.join(HABITAT_POC_PATH, "data", "climate_risk")
    if os.path.exists(climate_risk_src) and not os.path.exists(climate_risk_dst):
        shutil.copytree(climate_risk_src, climate_risk_dst)
        print(f"Copied data/climate_risk directory to POC")
    
    return True

def fix_init_files_poc():
    """Fix circular imports in __init__.py files in POC directory."""
    # Fix habitat_evolution/__init__.py
    habitat_init_path = os.path.join(HABITAT_POC_PATH, "src", "habitat_evolution", "__init__.py")
    if os.path.exists(habitat_init_path):
        with open(habitat_init_path, "w") as f:
            f.write("# Fixed to avoid circular imports\n")
        print("Fixed circular import in habitat_evolution/__init__.py in POC directory")
    
    # Fix core/__init__.py
    core_init_path = os.path.join(HABITAT_POC_PATH, "src", "habitat_evolution", "core", "__init__.py")
    if os.path.exists(core_init_path):
        with open(core_init_path, "w") as f:
            f.write("# Fixed to avoid circular imports\n")
        print("Fixed circular import in core/__init__.py in POC directory")
    
    # Fix visualization/__init__.py
    vis_init_path = os.path.join(HABITAT_POC_PATH, "src", "habitat_evolution", "visualization", "__init__.py")
    if os.path.exists(vis_init_path):
        with open(vis_init_path, "w") as f:
            f.write("# Fixed to avoid circular imports\n")
        print("Fixed circular import in visualization/__init__.py in POC directory")

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
        print("Test failed. Will try to fix POC directory anyway.")
    
    # Clean POC directory
    if not clean_poc_directory():
        return
    
    print("Habitat POC created successfully!")
    print(f"POC directory: {HABITAT_POC_PATH}")
    print("You can now run the climate_e2e test in the POC directory:")
    print(f"cd {HABITAT_POC_PATH} && python -m pytest tests/integration/climate_e2e/test_climate_e2e.py -v")

if __name__ == "__main__":
    main()
