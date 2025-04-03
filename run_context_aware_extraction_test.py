#!/usr/bin/env python
"""
Wrapper script to run the context-aware extraction test.

This script sets up the Python path correctly and then runs the test.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Run the test module
if __name__ == "__main__":
    print("Running context-aware extraction test...")
    from src.habitat_evolution.adaptive_core.demos.context_aware_extraction_test import ContextAwareExtractionTest
    
    # Create and run the test
    test = ContextAwareExtractionTest()
    test.run()
    print("Test completed.")
