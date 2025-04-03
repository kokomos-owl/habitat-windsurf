"""
Quality State Machine Test for Habitat Evolution.

This script demonstrates how our system uses the uncertain, poor, good quality states
as a state machine and how we improve our domain NER through the ingestion process.
It processes multiple climate risk documents sequentially to show quality transitions.
"""

import logging
import sys
import json
import os
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List, Any, Optional

# Import the fix for handling both import styles
from .import_fix import *

from habitat_evolution.adaptive_core.emergence.context_aware_extraction.context_aware_extractor import ContextAwareExtractor
from habitat_evolution.pattern_aware_rag.context.quality_aware_context import QualityAwarePatternContext
from habitat_evolution.pattern_aware_rag.quality_rag.context_aware_rag import ContextAwareRAG
from habitat_evolution.pattern_aware_rag.quality_rag.quality_enhanced_retrieval import QualityEnhancedRetrieval
from habitat_evolution.adaptive_core.persistence.interfaces.repository_adapter import InMemoryPatternRepository

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path("src/habitat_evolution/adaptive_core/demos/analysis_results")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file and console
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"quality_state_machine_test_{timestamp}.log"
    results_file = log_dir / f"quality_state_machine_test_{timestamp}.json"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file, results_file

# Define domain-specific entity categories for comprehensive testing
ENTITY_CATEGORIES = {
    'CLIMATE_HAZARD': [
        'Sea level rise', 'Coastal erosion', 'Storm surge', 'Extreme precipitation',
        'Drought', 'Extreme heat', 'Wildfire', 'Flooding'
    ],
    'ECOSYSTEM': [
        'Salt marsh complexes', 'Barrier beaches', 'Coastal dunes', 'Freshwater wetlands',
        'Vernal pools', 'Upland forests', 'Grasslands', 'Estuaries'
    ],
    'INFRASTRUCTURE': [
        'Roads', 'Bridges', 'Culverts', 'Stormwater systems', 'Wastewater treatment',
        'Drinking water supply', 'Power grid', 'Telecommunications'
    ],
    'ADAPTATION_STRATEGY': [
        'Living shorelines', 'Managed retreat', 'Green infrastructure', 'Beach nourishment',
        'Floodplain restoration', 'Building elevation', 'Permeable pavement', 'Rain gardens'
    ],
    'ASSESSMENT_COMPONENT': [
        'Vulnerability assessment', 'Risk analysis', 'Adaptation planning', 'Resilience metrics',
        'Stakeholder engagement', 'Implementation timeline', 'Funding mechanisms', 'Monitoring protocols'
    ]
}

# Define relationship types for comprehensive testing
RELATIONSHIP_TYPES = {
    'STRUCTURAL': [
        'part_of', 'contains', 'component_of'
    ],
    'CAUSAL': [
        'causes', 'affects', 'damages', 'mitigates'
    ],
    'FUNCTIONAL': [
        'protects_against', 'analyzes', 'evaluates'
    ],
    'TEMPORAL': [
        'precedes', 'concurrent_with'
    ]
}

def load_document(file_path: str) -> str:
    """Load document from file."""
    with open(file_path, 'r') as f:
        return f.read()

def run_quality_state_machine_test():
    """Run the quality state machine test with multiple documents."""
    # Set up logging
    log_file, results_file = setup_logging()
    logger = logging.getLogger()
    
    logger.info("Running quality state machine test with multiple climate risk documents")
    
    # Initialize components
    pattern_repository = InMemoryPatternRepository()
    quality_retrieval = QualityEnhancedRetrieval(
        quality_weight=0.7,
        coherence_threshold=0.6
    )
    
    # Create context-aware RAG component
    context_aware_rag = ContextAwareRAG(
        pattern_repository=pattern_repository,
        window_sizes=[2, 3, 5],
        quality_threshold=0.7,
        quality_weight=0.7,
        coherence_threshold=0.6
    )
    
    # Define document paths
    data_dir = Path("/Users/prphillips/Documents/GitHub/habitat-windsurf/data/climate_risk")
    documents = [
        data_dir / "basic_test_doc_cape_code.txt",
        data_dir / "complex_test_doc_boston_harbor_islands.txt",
        data_dir / "climate_risk_marthas_vineyard.txt"
    ]
    
    # Initialize tracking for quality transitions
    quality_transitions = {}
    entity_states = {}
    
    # Process each document in sequence to show quality transitions
    for i, doc_path in enumerate(documents):
        doc_name = doc_path.name
        logger.info(f"\n\n===== Processing document {i+1}/{len(documents)}: {doc_name} =====")
        
        # Load document
        document_text = load_document(str(doc_path))
        
        # Process document
        logger.info(f"Extracting entities and relationships from {doc_name}")
        extraction_results = context_aware_rag.process_with_context_aware_patterns(
            query="climate risk assessment",
            document=document_text
        )
        
        # Record quality states after processing this document
        logger.info(f"\n----- Quality State Distribution After Processing {doc_name} -----")
        quality_states = extraction_results.get("quality_states", {})
        
        # Log quality state distribution
        logger.info(f"  Good entities: {len(quality_states.get('good', {}))} entities")
        logger.info(f"  Uncertain entities: {len(quality_states.get('uncertain', {}))} entities")
        logger.info(f"  Poor entities: {len(quality_states.get('poor', {}))} entities")
        
        # Track entity state changes
        for state, entities in quality_states.items():
            for entity, data in entities.items():
                if entity not in entity_states:
                    # First appearance of this entity
                    entity_states[entity] = {
                        "initial_state": state,
                        "current_state": state,
                        "state_history": [state],
                        "document_history": [doc_name],
                        "metrics_history": [data.get("metrics", {})],
                        "category": data.get("category", "UNKNOWN")
                    }
                else:
                    # Entity already seen, track state change
                    prev_state = entity_states[entity]["current_state"]
                    entity_states[entity]["current_state"] = state
                    entity_states[entity]["state_history"].append(state)
                    entity_states[entity]["document_history"].append(doc_name)
                    entity_states[entity]["metrics_history"].append(data.get("metrics", {}))
                    
                    # Record transition
                    transition_key = f"{prev_state}_to_{state}"
                    if transition_key not in quality_transitions:
                        quality_transitions[transition_key] = 0
                    quality_transitions[transition_key] += 1
        
        # Save patterns to repository to demonstrate learning
        for entity, data in quality_states.get("good", {}).items():
            pattern = {
                "id": f"pattern_{entity.lower().replace(' ', '_')}",
                "text": entity,
                "category": data.get("category", "UNKNOWN"),
                "quality_state": "good",
                "metrics": data.get("metrics", {}),
                "source_document": doc_name
            }
            pattern_repository.save(pattern)
            logger.info(f"Saved good pattern to repository: {entity}")
        
        # Also save promising uncertain entities
        for entity, data in quality_states.get("uncertain", {}).items():
            if data.get("metrics", {}).get("coherence", 0) > 0.4:
                pattern = {
                    "id": f"pattern_{entity.lower().replace(' ', '_')}_uncertain",
                    "text": entity,
                    "category": data.get("category", "UNKNOWN"),
                    "quality_state": "uncertain",
                    "metrics": data.get("metrics", {}),
                    "source_document": doc_name
                }
                pattern_repository.save(pattern)
                logger.info(f"Saved promising uncertain pattern to repository: {entity}")
    
    # Analyze quality transitions
    logger.info("\n\n===== Quality State Machine Analysis =====")
    logger.info(f"Total quality state transitions: {sum(quality_transitions.values())}")
    for transition, count in quality_transitions.items():
        logger.info(f"  {transition}: {count} transitions")
    
    # Analyze entities that improved in quality
    improved_entities = [
        entity for entity, data in entity_states.items()
        if data["initial_state"] in ["poor", "uncertain"] and data["current_state"] == "good"
    ]
    
    logger.info(f"\nEntities that improved to 'good' quality: {len(improved_entities)}")
    for entity in improved_entities:
        data = entity_states[entity]
        logger.info(f"  {entity} ({data['category']}): {' -> '.join(data['state_history'])}")
        logger.info(f"    Documents: {' -> '.join(data['document_history'])}")
        
        # Show metric improvements
        initial_metrics = data['metrics_history'][0]
        final_metrics = data['metrics_history'][-1]
        logger.info(f"    Initial metrics: coherence={initial_metrics.get('coherence', 0):.2f}, stability={initial_metrics.get('stability', 0):.2f}")
        logger.info(f"    Final metrics: coherence={final_metrics.get('coherence', 0):.2f}, stability={final_metrics.get('stability', 0):.2f}")
    
    # Save results to JSON for further analysis
    results = {
        "quality_transitions": quality_transitions,
        "entity_states": entity_states,
        "improved_entities": [entity_states[e] for e in improved_entities],
        "document_sequence": [d.name for d in documents],
        "summary": {
            "total_entities": len(entity_states),
            "final_good_entities": sum(1 for e in entity_states.values() if e["current_state"] == "good"),
            "final_uncertain_entities": sum(1 for e in entity_states.values() if e["current_state"] == "uncertain"),
            "final_poor_entities": sum(1 for e in entity_states.values() if e["current_state"] == "poor"),
            "improved_entities": len(improved_entities),
            "total_transitions": sum(quality_transitions.values())
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    logger.info("\nQuality state machine test completed successfully.")
    
    return True

if __name__ == "__main__":
    success = run_quality_state_machine_test()
    if success:
        logging.info("Quality state machine test completed successfully")
    else:
        logging.error("Test failed")
