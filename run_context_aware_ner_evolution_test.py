#!/usr/bin/env python3
"""
Runner script for Context-Aware NER Evolution Test.

This script executes the context-aware NER evolution test to demonstrate
how domain-specific Named Entity Recognition (NER) evolves through document
ingestion, leveraging contextual reinforcement and quality transitions.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import the test module
from habitat_evolution.adaptive_core.demos.context_aware_ner_evolution_test import run_context_aware_ner_evolution_test

if __name__ == "__main__":
    print("Starting Context-Aware NER Evolution Test...")
    results = run_context_aware_ner_evolution_test()
    print("Test completed successfully!")
    print(f"Documents processed: {results['documents_processed']}")
    print(f"Total entities: {results['entities']['total']}")
    print(f"Total relationships: {results['relationships']['total']}")
    print(f"Domain relevance improvement: {results['domain_relevance']['initial']:.2f} → {results['domain_relevance']['final']:.2f}")
