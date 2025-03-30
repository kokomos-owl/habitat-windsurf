"""
Simple test to verify that the topology manager can be imported and instantiated.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from habitat_evolution.pattern_aware_rag.topology.manager import TopologyManager
    print("Successfully imported TopologyManager")
    
    # Try to instantiate the manager
    manager = TopologyManager()
    print("Successfully instantiated TopologyManager")
    
    # Print some basic info about the manager
    print(f"Detector: {manager.detector.__class__.__name__}")
    print(f"Persistence mode: {manager.persistence_mode}")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("\nAvailable modules in habitat_evolution:")
    try:
        import habitat_evolution
        print(dir(habitat_evolution))
    except ImportError:
        print("Could not import habitat_evolution")
