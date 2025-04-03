"""
Simplified test for context-aware pattern extraction.

This script tests if the components can be initialized and run a basic operation
and saves representative data to a log file for analysis.
"""

import logging
import sys
import json
import os
from pathlib import Path
from datetime import datetime

# Import the fix for handling both import styles
from .import_fix import *

from src.habitat_evolution.adaptive_core.emergence.context_aware_extraction.context_aware_extractor import ContextAwareExtractor
from src.habitat_evolution.pattern_aware_rag.context.quality_aware_context import QualityAwarePatternContext
from src.habitat_evolution.pattern_aware_rag.quality_rag.context_aware_rag import ContextAwareRAG
from src.habitat_evolution.pattern_aware_rag.quality_rag.quality_enhanced_retrieval import QualityEnhancedRetrieval
from src.habitat_evolution.adaptive_core.persistence.interfaces.repository_adapter import InMemoryPatternRepository

# Set up file logger
results_dir = Path(__file__).parent / "analysis_results"
results_dir.mkdir(exist_ok=True)
log_file = results_dir / f"context_extraction_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging to both console and file
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Remove root logger handlers to avoid duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

def run_simplified_test():
    """Run a test with representative climate risk data."""
    logger.info("Running context-aware extraction test with representative data")
    
    # Initialize components
    pattern_repository = InMemoryPatternRepository()
    context_aware_extractor = ContextAwareExtractor(window_sizes=[2, 3, 4, 5])
    quality_retrieval = QualityEnhancedRetrieval(quality_weight=0.7)
    
    # Initialize ContextAwareRAG
    context_aware_rag = ContextAwareRAG(
        pattern_repository=pattern_repository,
        window_sizes=[2, 3, 4, 5],
        quality_threshold=0.7
    )
    
    # Create a representative climate risk document
    test_document = """
    Climate Risk Assessment - Cape Cod National Seashore
    
    Executive Summary:
    The Cape Cod National Seashore faces significant climate risks including sea level rise, 
    coastal erosion, and increased storm intensity. This assessment identifies critical 
    vulnerabilities and recommends adaptation strategies.
    
    Key Findings:
    1. Sea level rise: Projected 1-3 feet by 2050, threatening low-lying areas
    2. Salt marsh degradation: 30% of marshes at risk of submergence
    3. Coastal erosion: Accelerating at 3-5 feet per year along outer beaches
    4. Habitat shifts: Migration of species and vegetation communities
    
    Recommended Actions:
    - Implement living shoreline projects at high-risk locations
    - Restore salt marsh complexes to enhance resilience
    - Develop managed retreat strategies for vulnerable infrastructure
    - Establish monitoring protocols for ecological changes
    
    This assessment provides a foundation for climate adaptation planning
    and will inform future management decisions for the Cape Cod National Seashore.
    """
    
    # Process the document
    logger.info("Processing climate risk document")
    result = context_aware_rag.process_document_for_patterns(test_document)
    
    # Save results to JSON file
    results_file = Path(log_file).with_suffix('.json')
    
    # Extract quality assessments
    quality_data = {
        'good_entities': [],
        'uncertain_entities': [],
        'poor_entities': [],
        'relationships': []
    }
    
    # Get quality context from result
    if result and hasattr(result, 'quality_assessments'):
        for entity, assessment in result.quality_assessments.items():
            entity_data = {
                'entity': entity,
                'quality_state': assessment.quality_state,
                'pattern_state': assessment.pattern_state.name if hasattr(assessment.pattern_state, 'name') else str(assessment.pattern_state),
                'metrics': assessment.metrics
            }
            
            if assessment.quality_state == 'good':
                quality_data['good_entities'].append(entity_data)
            elif assessment.quality_state == 'uncertain':
                quality_data['uncertain_entities'].append(entity_data)
            else:
                quality_data['poor_entities'].append(entity_data)
    
    # Save to JSON
    with open(results_file, 'w') as f:
        json.dump(quality_data, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Log summary statistics
    logger.info(f"Extraction summary: {len(quality_data['good_entities'])} good entities, "
                f"{len(quality_data['uncertain_entities'])} uncertain entities, "
                f"{len(quality_data['poor_entities'])} poor entities")
    
    # Check if result is not None
    if result is not None:
        logger.info("Test passed: Document processed successfully")
        return True
    else:
        logger.error("Test failed: Document processing returned None")
        return False

if __name__ == "__main__":
    success = run_simplified_test()
    if success:
        logger.info("All components initialized and functioning correctly")
    else:
        logger.error("Test failed")
