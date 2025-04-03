"""
Import fix for Habitat Evolution.

This module sets up the Python path to allow both import styles to work together:
1. Imports with the src. prefix (e.g., from src.habitat_evolution.module import X)
2. Imports without the src. prefix (e.g., from habitat_evolution.module import X)
"""

import sys
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent.parent.parent.parent

# Add the project root to sys.path for imports with src. prefix
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add the src directory to sys.path for imports without src. prefix
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Print debug information
if __name__ == "__main__":
    print(f"Project root: {project_root}")
    print(f"src directory: {src_dir}")
    print("Python path:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    # Test imports
    try:
        import habitat_evolution
        print("✓ import habitat_evolution succeeded")
    except ImportError as e:
        print(f"✗ import habitat_evolution failed: {e}")
    
    try:
        import src.habitat_evolution
        print("✓ import src.habitat_evolution succeeded")
    except ImportError as e:
        print(f"✗ import src.habitat_evolution failed: {e}")
