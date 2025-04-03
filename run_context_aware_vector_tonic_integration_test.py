#!/usr/bin/env python3
"""
Runner script for the Context-Aware NER with Vector-Tonic-Window Integration Test.

This script executes the integration test that combines context-aware NER evolution
with the vector-tonic-window system for enhanced topological and temporal analysis.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import the integration test
from habitat_evolution.adaptive_core.demos.context_aware_vector_tonic_integration_test import run_context_aware_vector_tonic_integration_test

if __name__ == "__main__":
    print("Starting Context-Aware NER with Vector-Tonic-Window Integration Test...")
    results = run_context_aware_vector_tonic_integration_test()
    print(f"Test completed successfully!")
    print(f"Processed {results['metrics']['documents_processed']} documents")
    print(f"Extracted {results['entity_count']} entities with {results['relationship_count']} relationships")
    print(f"Visualizations generated in the 'visualizations' directory")
