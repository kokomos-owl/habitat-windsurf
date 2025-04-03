"""
Quality State Machine Demo for Habitat Evolution.

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
from habitat_evolution.adaptive_core.emergence.context_aware_extraction.quality_assessment import QualityAssessment
from habitat_evolution.adaptive_core.persistence.interfaces.repository_adapter import InMemoryPatternRepository

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path("src/habitat_evolution/adaptive_core/demos/analysis_results")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file and console
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"quality_state_machine_demo_{timestamp}.log"
    results_file = log_dir / f"quality_state_machine_demo_{timestamp}.json"
    
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

def run_quality_state_machine_demo():
    """Run the quality state machine demo with multiple documents."""
    # Set up logging
    log_file, results_file = setup_logging()
    logger = logging.getLogger()
    
    logger.info("Running quality state machine demo with multiple climate risk documents")
    
    # Initialize components
    pattern_repository = InMemoryPatternRepository()
    
    # Create context-aware extractor
    extractor = ContextAwareExtractor(
        window_sizes=[2, 3, 5],
        quality_threshold=0.7
    )
    
    # Define document paths
    data_dir = Path("/Users/prphillips/Documents/GitHub/habitat-windsurf/data/climate_risk")
    
    # Get all text files in the climate_risk directory
    documents = list(data_dir.glob("*.txt"))
    
    # Sort documents to ensure consistent processing order
    documents.sort()
    
    logger.info(f"Found {len(documents)} climate risk documents to process")
    for doc in documents:
        logger.info(f"  - {doc.name}")
    
    # Initialize tracking for quality transitions
    quality_transitions = {}
    entity_states = {}
    quality_assessment = QualityAssessment()
    
    # Process each document in sequence to show quality transitions
    for i, doc_path in enumerate(documents):
        doc_name = doc_path.name
        logger.info(f"\n\n===== Processing document {i+1}/{len(documents)}: {doc_name} =====")
        
        # Load document
        document_text = load_document(str(doc_path))
        
        # Process document with context-aware extractor
        logger.info(f"Extracting entities and relationships from {doc_name}")
        extraction_results = extractor.process_document(document_text)
        
        # Get quality states from the extractor
        quality_states = extractor.quality_assessor.get_quality_states()
        
        # Log quality state distribution
        logger.info(f"\n----- Quality State Distribution After Processing {doc_name} -----")
        logger.info(f"  Good entities: {len(quality_states.get('good', {}))} entities")
        logger.info(f"  Uncertain entities: {len(quality_states.get('uncertain', {}))} entities")
        logger.info(f"  Poor entities: {len(quality_states.get('poor', {}))} entities")
        
        # Track entity state changes
        for state, entities in quality_states.items():
            for entity, data in entities.items():
                # Skip very short entities (likely noise)
                if len(entity.strip()) < 3:
                    continue
                    
                # Skip entities that are just numbers or punctuation
                if entity.strip().isdigit() or all(c in '.,;:!?-()[]{}' for c in entity.strip()):
                    continue
                
                # Check for domain relevance (optional filtering)
                is_domain_relevant = False
                for category, terms in ENTITY_CATEGORIES.items():
                    if any(term.lower() in entity.lower() for term in terms):
                        is_domain_relevant = True
                        break
                
                if entity not in entity_states:
                    # First appearance of this entity
                    entity_states[entity] = {
                        "initial_state": state,
                        "current_state": state,
                        "state_history": [state],
                        "document_history": [doc_name],
                        "metrics_history": [data.get("metrics", {})],
                        "category": data.get("category", "UNKNOWN"),
                        "appearances": 1,
                        "is_domain_relevant": is_domain_relevant
                    }
                else:
                    # Entity already seen, update appearance count
                    entity_states[entity]["appearances"] += 1
                    
                    # Track state change if it occurred
                    prev_state = entity_states[entity]["current_state"]
                    if prev_state != state:
                        # State transition occurred
                        entity_states[entity]["current_state"] = state
                        entity_states[entity]["state_history"].append(state)
                        entity_states[entity]["document_history"].append(doc_name)
                        entity_states[entity]["metrics_history"].append(data.get("metrics", {}))
                        
                        # Record transition
                        transition_key = f"{prev_state}_to_{state}"
                        if transition_key not in quality_transitions:
                            quality_transitions[transition_key] = 0
                        quality_transitions[transition_key] += 1
                        
                        # Log significant transitions (especially improvements)
                        if state == "good" or (prev_state == "poor" and state == "uncertain"):
                            logger.info(f"  IMPROVEMENT: '{entity}' {prev_state} -> {state}")
                            logger.info(f"    Metrics: {data.get('metrics', {})}")
                        else:
                            logger.info(f"  Transition: '{entity}' {prev_state} -> {state}")
        
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
    
    # Calculate transition percentages
    total_transitions = sum(quality_transitions.values())
    if total_transitions > 0:
        logger.info("\nTransition percentages:")
        for transition, count in quality_transitions.items():
            percentage = (count / total_transitions) * 100
            logger.info(f"  {transition}: {percentage:.1f}%")
    
    # Analyze entities that improved in quality
    improved_entities = [
        entity for entity, data in entity_states.items()
        if data["initial_state"] in ["poor", "uncertain"] and data["current_state"] == "good"
    ]
    
    # Also track entities that improved from poor to uncertain
    partially_improved_entities = [
        entity for entity, data in entity_states.items()
        if data["initial_state"] == "poor" and data["current_state"] == "uncertain"
        and entity not in improved_entities
    ]
    
    # Find domain-relevant entities that improved
    domain_relevant_improvements = [
        entity for entity in improved_entities
        if entity_states[entity]["is_domain_relevant"]
    ]
    
    logger.info(f"\nEntities that improved to 'good' quality: {len(improved_entities)}")
    logger.info(f"Entities that improved from 'poor' to 'uncertain': {len(partially_improved_entities)}")
    logger.info(f"Domain-relevant entities that improved: {len(domain_relevant_improvements)}")
    
    # Show the most frequently appearing entities
    frequent_entities = sorted(
        [(entity, data["appearances"]) for entity, data in entity_states.items() if data["appearances"] > 1],
        key=lambda x: x[1],
        reverse=True
    )[:20]  # Top 20
    
    logger.info("\nMost frequently appearing entities:")
    for entity, appearances in frequent_entities:
        data = entity_states[entity]
        logger.info(f"  {entity} ({data['current_state']}): {appearances} appearances")
    
    # Detailed analysis of improved entities
    if improved_entities:
        logger.info("\nDetailed analysis of entities that improved to 'good' quality:")
        for entity in improved_entities:
            data = entity_states[entity]
            logger.info(f"  {entity} ({data['category']}): {' -> '.join(data['state_history'])}")
            logger.info(f"    Appearances: {data['appearances']}")
            logger.info(f"    Documents: {' -> '.join(data['document_history'])}")
            
            # Show metric improvements
            initial_metrics = data['metrics_history'][0]
            final_metrics = data['metrics_history'][-1]
            logger.info(f"    Initial metrics: coherence={initial_metrics.get('coherence', 0):.2f}, stability={initial_metrics.get('stability', 0):.2f}")
            logger.info(f"    Final metrics: coherence={final_metrics.get('coherence', 0):.2f}, stability={final_metrics.get('stability', 0):.2f}")
    
    # Analyze domain-relevant entities by category
    domain_entities_by_category = {}
    for category in ENTITY_CATEGORIES.keys():
        domain_entities_by_category[category] = [
            entity for entity, data in entity_states.items()
            if any(term.lower() in entity.lower() for term in ENTITY_CATEGORIES[category])
        ]
    
    logger.info("\nDomain-relevant entities by category:")
    for category, entities in domain_entities_by_category.items():
        good_count = sum(1 for e in entities if entity_states[e]["current_state"] == "good")
        uncertain_count = sum(1 for e in entities if entity_states[e]["current_state"] == "uncertain")
        poor_count = sum(1 for e in entities if entity_states[e]["current_state"] == "poor")
        
        logger.info(f"  {category}: {len(entities)} entities (good: {good_count}, uncertain: {uncertain_count}, poor: {poor_count})")
        
        # Show top entities in this category
        top_entities = sorted(
            [(e, entity_states[e]["appearances"]) for e in entities],
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5
        
        for entity, appearances in top_entities:
            state = entity_states[entity]["current_state"]
            logger.info(f"    {entity} ({state}): {appearances} appearances")
    
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
    logger.info("\nQuality state machine demo completed successfully.")
    
    return True

if __name__ == "__main__":
    success = run_quality_state_machine_demo()
    if success:
        logging.info("Quality state machine demo completed successfully")
    else:
        logging.error("Test failed")
