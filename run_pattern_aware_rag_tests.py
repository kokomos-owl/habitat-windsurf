#!/usr/bin/env python
"""
Run Pattern-Aware RAG integration tests with proper path setup.
This script ensures that all necessary modules are in the Python path.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Set up paths and run the tests."""
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    
    # Add src directory to Python path
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Print current Python path for debugging
    print("Python path:")
    for path in sys.path:
        print(f"  - {path}")
    
    # Run the tests with pytest
    test_path = os.path.join(
        project_root, 
        "src/habitat_evolution/tests/integration/test_pattern_aware_rag_integration.py"
    )
    
    print(f"\nRunning tests: {test_path}\n")
    
    # Use subprocess to run pytest with the current environment
    result = subprocess.run(
        ["python", "-m", "pytest", test_path, "-v"],
        env=os.environ.copy()
    )
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
