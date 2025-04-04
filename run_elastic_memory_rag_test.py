#!/usr/bin/env python3
"""
Runner script for the Elastic Memory RAG Integration Test.

This script executes the integration test that combines predicate quality tracking,
semantic memory persistence, and quality-enhanced retrieval to create a complete
RAG↔Evolution↔Persistence loop for elastic semantic memory.
"""

import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_dir = str(project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import the integration test
from src.habitat_evolution.pattern_aware_rag.elastic_memory_rag_integration import run_elastic_memory_rag_integration_test

if __name__ == "__main__":
    print("Starting Elastic Memory RAG Integration Test...")
    results = run_elastic_memory_rag_integration_test()
    
    print("\nElastic Memory RAG Integration Test Summary:")
    print("-------------------------------------")
    print(f"Entity count: {results['entity_count']}")
    print(f"Relationship count: {results['relationship_count']}")
    print(f"Predicate count: {results['predicate_count']}")
    print("\nRetrieval Results:")
    print(f"  Pattern count: {results['retrieval_result']['pattern_count']}")
    print(f"  Quality distribution: {results['retrieval_result']['quality_distribution']}")
    print(f"  Confidence: {results['retrieval_result']['confidence']}")
    print("\nReinforcement Results:")
    print(f"  Reinforced entities: {len(results['reinforcement_result']['reinforced_entities'])}")
    print(f"  Reinforced predicates: {len(results['reinforcement_result']['reinforced_predicates'])}")
    print("\nPersistence Results:")
    print(f"  Save timestamp: {results['persistence_result']['save']['timestamp']}")
    print(f"  Load timestamp: {results['persistence_result']['load']['timestamp']}")
    print("-------------------------------------")
    print("Test completed successfully!")
