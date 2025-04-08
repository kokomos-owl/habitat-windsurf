#!/usr/bin/env python
"""
Vector-Tonic Climate Pattern Demo Runner

This script runs the climate pattern demo using the vector-tonic statistical
analysis framework. It ensures proper path resolution regardless of how
the script is executed.
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the demo
from src.habitat_evolution.vector_tonic.demo.climate_pattern_demo import main

if __name__ == "__main__":
    main()
