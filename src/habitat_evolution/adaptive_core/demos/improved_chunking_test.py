"""
Improved Entity Chunking Test for Habitat Evolution.

This script tests an improved sliding window approach for entity chunking
that better identifies complete entities in climate risk documents.
"""

import logging
import sys
import json
import os
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List, Any, Optional
import re

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
    log_file = log_dir / f"improved_chunking_test_{timestamp}.log"
    results_file = log_dir / f"improved_chunking_test_{timestamp}.json"
    
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

class ImprovedChunkingExtractor(ContextAwareExtractor):
    """Enhanced extractor with improved entity chunking capabilities."""
    
    def __init__(self, window_sizes=None, quality_threshold=0.7, data_dir=None):
        """Initialize with improved chunking capabilities."""
        super().__init__(window_sizes, quality_threshold, data_dir)
        
        # Add additional window sizes for better phrase capture
        if window_sizes is None:
            self.window_sizes = [2, 3, 4, 5, 7]  # Added 4 and 7-word windows
        
        # Patterns for identifying complete phrases
        self.phrase_patterns = [
            # Title patterns (e.g., "Climate Risk Assessment")
            r'[A-Z][a-z]+(?: [A-Z][a-z]+)+ (?:Assessment|Analysis|Report|Study)',
            
            # Location patterns (e.g., "Cape Cod, Massachusetts")
            r'[A-Z][a-z]+(?: [A-Z][a-z]+)*, [A-Z][a-z]+',
            
            # Section headers (e.g., "Key Findings:")
            r'[A-Z][a-z]+(?: [A-Z][a-z]+)+ *:',
            
            # Climate hazards (e.g., "Sea Level Rise")
            r'(?:Sea Level Rise|Coastal Erosion|Storm Surge|Extreme Precipitation)',
            
            # Ecosystem types (e.g., "Salt Marsh Complexes")
            r'(?:Salt Marsh|Barrier Beach|Coastal Dune|Freshwater Wetland)s?',
            
            # Complete noun phrases
            r'(?:the|a|an) [a-z]+(?: [a-z]+){1,3} (?:of|in|for|to) (?:the|a|an)? [a-z]+'
        ]
        
        # Incomplete phrase patterns to avoid
        self.incomplete_patterns = [
            # Fragments ending with prepositions or articles
            r'.+ (?:of|in|for|to|the|a|an)$',
            
            # Fragments starting with prepositions or articles
            r'^(?:of|in|for|to|the|a|an) .+',
            
            # Fragments with dangling punctuation
            r'.+[,:]$',
            r'^[,:].*'
        ]
        
        logging.info(f"Initialized ImprovedChunkingExtractor with window sizes {self.window_sizes}")
    
    def process_document(self, document: str) -> Dict[str, Any]:
        """Process document with improved chunking."""
        # First, use the standard processing
        results = super().process_document(document)
        
        # Then apply improved chunking
        improved_entities = self._apply_improved_chunking(document)
        
        # Merge improved entities with original results
        for entity, data in improved_entities.items():
            # Add to quality states if not already present or if quality is better
            for quality_state in ['good', 'uncertain', 'poor']:
                if entity in improved_entities.get(quality_state, {}):
                    if entity not in results.get('quality_states', {}).get(quality_state, {}):
                        if 'quality_states' not in results:
                            results['quality_states'] = {}
                        if quality_state not in results['quality_states']:
                            results['quality_states'][quality_state] = {}
                        results['quality_states'][quality_state][entity] = data
        
        return results
    
    def _apply_improved_chunking(self, document: str) -> Dict[str, Dict[str, Any]]:
        """Apply improved chunking to identify complete entities."""
        improved_entities = {'good': {}, 'uncertain': {}, 'poor': {}}
        
        # Split document into lines for better context
        lines = document.split('\n')
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Apply phrase patterns to identify complete entities
            for pattern in self.phrase_patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    entity = match.group(0).strip()
                    if len(entity) > 3:  # Skip very short entities
                        # Check if this is a complete entity
                        is_complete = True
                        for incomplete_pattern in self.incomplete_patterns:
                            if re.match(incomplete_pattern, entity):
                                is_complete = False
                                break
                        
                        if is_complete:
                            # Determine quality based on entity completeness
                            quality = self._assess_entity_quality(entity, line)
                            improved_entities[quality][entity] = {
                                'metrics': {
                                    'coherence': 0.7 if quality == 'good' else 0.5 if quality == 'uncertain' else 0.3,
                                    'stability': 0.6 if quality == 'good' else 0.4 if quality == 'uncertain' else 0.0,
                                    'energy': 0.8 if quality == 'good' else 0.6 if quality == 'uncertain' else 0.4
                                },
                                'source': 'improved_chunking',
                                'context': line
                            }
        
        return improved_entities
    
    def _assess_entity_quality(self, entity: str, context: str) -> str:
        """Assess the quality of an entity based on completeness and context."""
        # Check for domain relevance
        is_domain_relevant = False
        for category, terms in ENTITY_CATEGORIES.items():
            if any(term.lower() in entity.lower() for term in terms):
                is_domain_relevant = True
                break
        
        # Check for completeness (no dangling words)
        has_dangling_words = any(re.match(pattern, entity) for pattern in self.incomplete_patterns)
        
        # Check for proper capitalization
        proper_capitalization = bool(re.match(r'[A-Z][a-zA-Z ]*', entity))
        
        # Check for context relevance
        context_relevance = entity.lower() in context.lower() and len(context) > len(entity) * 2
        
        # Determine quality
        if is_domain_relevant and not has_dangling_words and proper_capitalization:
            return 'good'
        elif not has_dangling_words and (is_domain_relevant or proper_capitalization):
            return 'uncertain'
        else:
            return 'poor'

def load_document(file_path: str) -> str:
    """Load document from file."""
    with open(file_path, 'r') as f:
        return f.read()

def run_improved_chunking_test():
    """Run the improved chunking test with climate risk documents."""
    # Set up logging
    log_file, results_file = setup_logging()
    logger = logging.getLogger()
    
    logger.info("Running improved entity chunking test with climate risk documents")
    
    # Initialize components
    pattern_repository = InMemoryPatternRepository()
    
    # Create standard extractor for comparison
    standard_extractor = ContextAwareExtractor(
        window_sizes=[2, 3, 5],
        quality_threshold=0.7
    )
    
    # Create improved chunking extractor
    improved_extractor = ImprovedChunkingExtractor(
        window_sizes=[2, 3, 4, 5, 7],
        quality_threshold=0.7
    )
    
    # Define document paths
    data_dir = Path("/Users/prphillips/Documents/GitHub/habitat-windsurf/data/climate_risk")
    
    # Get all text files in the climate_risk directory
    documents = list(data_dir.glob("*.txt"))
    
    # Sort documents to ensure consistent processing order
    documents.sort()
    
    logger.info(f"Found {len(documents)} climate risk documents to process")
    
    # Initialize comparison metrics
    comparison_results = {
        'documents': [],
        'overall': {
            'standard': {
                'total_entities': 0,
                'good_entities': 0,
                'uncertain_entities': 0,
                'poor_entities': 0,
                'fragmented_entities': 0,
                'complete_entities': 0
            },
            'improved': {
                'total_entities': 0,
                'good_entities': 0,
                'uncertain_entities': 0,
                'poor_entities': 0,
                'fragmented_entities': 0,
                'complete_entities': 0
            }
        }
    }
    
    # Process each document and compare results
    for doc_path in documents:
        doc_name = doc_path.name
        logger.info(f"\n===== Processing document: {doc_name} =====")
        
        # Load document
        document_text = load_document(str(doc_path))
        
        # Process with standard extractor
        logger.info("Processing with standard extractor...")
        standard_results = standard_extractor.process_document(document_text)
        
        # Process with improved extractor
        logger.info("Processing with improved chunking extractor...")
        improved_results = improved_extractor.process_document(document_text)
        
        # Analyze standard results
        standard_quality_states = standard_results.get('quality_states', {})
        standard_good = len(standard_quality_states.get('good', {}))
        standard_uncertain = len(standard_quality_states.get('uncertain', {}))
        standard_poor = len(standard_quality_states.get('poor', {}))
        standard_total = standard_good + standard_uncertain + standard_poor
        
        # Count fragmented entities in standard results
        standard_fragmented = 0
        for quality, entities in standard_quality_states.items():
            for entity in entities:
                # Check if entity is likely fragmented
                if any(re.match(pattern, entity) for pattern in improved_extractor.incomplete_patterns):
                    standard_fragmented += 1
        
        standard_complete = standard_total - standard_fragmented
        
        # Analyze improved results
        improved_quality_states = improved_results.get('quality_states', {})
        improved_good = len(improved_quality_states.get('good', {}))
        improved_uncertain = len(improved_quality_states.get('uncertain', {}))
        improved_poor = len(improved_quality_states.get('poor', {}))
        improved_total = improved_good + improved_uncertain + improved_poor
        
        # Count fragmented entities in improved results
        improved_fragmented = 0
        for quality, entities in improved_quality_states.items():
            for entity in entities:
                # Check if entity is likely fragmented
                if any(re.match(pattern, entity) for pattern in improved_extractor.incomplete_patterns):
                    improved_fragmented += 1
        
        improved_complete = improved_total - improved_fragmented
        
        # Log comparison
        logger.info(f"\n----- Extraction Results Comparison for {doc_name} -----")
        logger.info(f"Standard Extractor: {standard_total} entities (good: {standard_good}, uncertain: {standard_uncertain}, poor: {standard_poor})")
        logger.info(f"  Complete entities: {standard_complete}, Fragmented entities: {standard_fragmented}")
        logger.info(f"  Fragmentation rate: {(standard_fragmented / max(1, standard_total)) * 100:.1f}%")
        
        logger.info(f"Improved Extractor: {improved_total} entities (good: {improved_good}, uncertain: {improved_uncertain}, poor: {improved_poor})")
        logger.info(f"  Complete entities: {improved_complete}, Fragmented entities: {improved_fragmented}")
        logger.info(f"  Fragmentation rate: {(improved_fragmented / max(1, improved_total)) * 100:.1f}%")
        
        # Calculate improvement
        complete_entity_improvement = improved_complete - standard_complete
        fragmentation_rate_improvement = ((standard_fragmented / max(1, standard_total)) - 
                                         (improved_fragmented / max(1, improved_total))) * 100
        
        logger.info(f"Improvement: {complete_entity_improvement} more complete entities")
        logger.info(f"Fragmentation rate improvement: {fragmentation_rate_improvement:.1f}%")
        
        # Add to comparison results
        comparison_results['documents'].append({
            'name': doc_name,
            'standard': {
                'total_entities': standard_total,
                'good_entities': standard_good,
                'uncertain_entities': standard_uncertain,
                'poor_entities': standard_poor,
                'fragmented_entities': standard_fragmented,
                'complete_entities': standard_complete,
                'fragmentation_rate': (standard_fragmented / max(1, standard_total)) * 100
            },
            'improved': {
                'total_entities': improved_total,
                'good_entities': improved_good,
                'uncertain_entities': improved_uncertain,
                'poor_entities': improved_poor,
                'fragmented_entities': improved_fragmented,
                'complete_entities': improved_complete,
                'fragmentation_rate': (improved_fragmented / max(1, improved_total)) * 100
            },
            'comparison': {
                'complete_entity_improvement': complete_entity_improvement,
                'fragmentation_rate_improvement': fragmentation_rate_improvement
            }
        })
        
        # Update overall metrics
        comparison_results['overall']['standard']['total_entities'] += standard_total
        comparison_results['overall']['standard']['good_entities'] += standard_good
        comparison_results['overall']['standard']['uncertain_entities'] += standard_uncertain
        comparison_results['overall']['standard']['poor_entities'] += standard_poor
        comparison_results['overall']['standard']['fragmented_entities'] += standard_fragmented
        comparison_results['overall']['standard']['complete_entities'] += standard_complete
        
        comparison_results['overall']['improved']['total_entities'] += improved_total
        comparison_results['overall']['improved']['good_entities'] += improved_good
        comparison_results['overall']['improved']['uncertain_entities'] += improved_uncertain
        comparison_results['overall']['improved']['poor_entities'] += improved_poor
        comparison_results['overall']['improved']['fragmented_entities'] += improved_fragmented
        comparison_results['overall']['improved']['complete_entities'] += improved_complete
    
    # Calculate overall improvement
    standard_overall_fragmentation_rate = (comparison_results['overall']['standard']['fragmented_entities'] / 
                                          max(1, comparison_results['overall']['standard']['total_entities'])) * 100
    
    improved_overall_fragmentation_rate = (comparison_results['overall']['improved']['fragmented_entities'] / 
                                          max(1, comparison_results['overall']['improved']['total_entities'])) * 100
    
    overall_complete_entity_improvement = (comparison_results['overall']['improved']['complete_entities'] - 
                                          comparison_results['overall']['standard']['complete_entities'])
    
    overall_fragmentation_rate_improvement = standard_overall_fragmentation_rate - improved_overall_fragmentation_rate
    
    comparison_results['overall']['comparison'] = {
        'complete_entity_improvement': overall_complete_entity_improvement,
        'fragmentation_rate_improvement': overall_fragmentation_rate_improvement
    }
    
    # Log overall results
    logger.info("\n\n===== Overall Improvement Results =====")
    logger.info(f"Standard Extractor: {comparison_results['overall']['standard']['total_entities']} entities")
    logger.info(f"  Complete entities: {comparison_results['overall']['standard']['complete_entities']}")
    logger.info(f"  Fragmented entities: {comparison_results['overall']['standard']['fragmented_entities']}")
    logger.info(f"  Fragmentation rate: {standard_overall_fragmentation_rate:.1f}%")
    
    logger.info(f"Improved Extractor: {comparison_results['overall']['improved']['total_entities']} entities")
    logger.info(f"  Complete entities: {comparison_results['overall']['improved']['complete_entities']}")
    logger.info(f"  Fragmented entities: {comparison_results['overall']['improved']['fragmented_entities']}")
    logger.info(f"  Fragmentation rate: {improved_overall_fragmentation_rate:.1f}%")
    
    logger.info(f"Overall Improvement: {overall_complete_entity_improvement} more complete entities")
    logger.info(f"Overall Fragmentation rate improvement: {overall_fragmentation_rate_improvement:.1f}%")
    
    # Save results to JSON for further analysis
    with open(results_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    logger.info("\nImproved chunking test completed successfully.")
    
    return True

if __name__ == "__main__":
    success = run_improved_chunking_test()
    if success:
        logging.info("Improved chunking test completed successfully")
    else:
        logging.error("Test failed")
