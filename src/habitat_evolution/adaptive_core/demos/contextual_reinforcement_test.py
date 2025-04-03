"""
Contextual Reinforcement Test for Habitat Evolution.

This script tests using surrounding entities to boost quality assessments,
focusing on how contextual reinforcement improves entity quality states.
"""

import logging
import sys
import json
import os
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List, Any, Optional, Set
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
    log_file = log_dir / f"contextual_reinforcement_test_{timestamp}.log"
    results_file = log_dir / f"contextual_reinforcement_test_{timestamp}.json"
    
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

class ContextualReinforcementExtractor(ContextAwareExtractor):
    """Enhanced extractor with contextual reinforcement capabilities."""
    
    def __init__(self, window_sizes=None, quality_threshold=0.7, data_dir=None):
        """Initialize with contextual reinforcement capabilities."""
        super().__init__(window_sizes, quality_threshold, data_dir)
        
        # Track entity categories
        self.entity_categories = {}
        
        # Track contextual relationships
        self.contextual_relationships = {}
        
        # Track entity clusters
        self.entity_clusters = {}
        
        # Track reinforcement events
        self.reinforcement_events = []
        
        logging.info(f"Initialized ContextualReinforcementExtractor with window sizes {self.window_sizes}")
    
    def process_document(self, document: str) -> Dict[str, Any]:
        """Process document with contextual reinforcement."""
        # First, use the standard processing
        results = super().process_document(document)
        
        # Then apply contextual reinforcement
        self._apply_contextual_reinforcement(document)
        
        # Add reinforcement events to results
        results['reinforcement_events'] = self.reinforcement_events
        
        # Add contextual relationships to results
        results['contextual_relationships'] = self.contextual_relationships
        
        return results
    
    def _apply_contextual_reinforcement(self, document: str) -> None:
        """Apply contextual reinforcement to improve entity quality."""
        # Get current quality states
        quality_states = self.get_quality_states()
        
        # Identify entity clusters in the document
        self._identify_entity_clusters(document)
        
        # Apply reinforcement based on clusters
        for cluster_id, cluster in self.entity_clusters.items():
            self._reinforce_entities_in_cluster(cluster)
    
    def _identify_entity_clusters(self, document: str) -> None:
        """Identify clusters of related entities in the document."""
        # Split document into paragraphs
        paragraphs = document.split('\n\n')
        
        # Process each paragraph as a potential cluster
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            
            # Create a cluster ID
            cluster_id = f"cluster_{i}"
            
            # Find entities in this paragraph
            entities_in_paragraph = self._find_entities_in_text(paragraph)
            
            # Only create clusters with at least 2 entities
            if len(entities_in_paragraph) >= 2:
                self.entity_clusters[cluster_id] = {
                    'entities': entities_in_paragraph,
                    'text': paragraph,
                    'domain_relevance': self._calculate_cluster_domain_relevance(entities_in_paragraph)
                }
                
                # Create contextual relationships between entities in this cluster
                self._create_contextual_relationships(entities_in_paragraph, paragraph, cluster_id)
    
    def _find_entities_in_text(self, text: str) -> List[str]:
        """Find all known entities in a text."""
        entities = []
        quality_states = self.get_quality_states()
        
        # Check all entities from all quality states
        for quality_state in ['good', 'uncertain', 'poor']:
            for entity in quality_states.get(quality_state, {}):
                if entity in text:
                    entities.append(entity)
        
        return entities
    
    def _calculate_cluster_domain_relevance(self, entities: List[str]) -> float:
        """Calculate the domain relevance of an entity cluster."""
        domain_relevant_count = 0
        
        for entity in entities:
            category = self._determine_entity_category(entity)
            if category:
                domain_relevant_count += 1
        
        return domain_relevant_count / max(1, len(entities))
    
    def _create_contextual_relationships(self, entities: List[str], context: str, cluster_id: str) -> None:
        """Create contextual relationships between entities in a cluster."""
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                source = entities[i]
                target = entities[j]
                
                # Create a relationship ID
                rel_id = f"{source}|contextually_related|{target}"
                
                # Store the relationship
                if rel_id not in self.contextual_relationships:
                    self.contextual_relationships[rel_id] = {
                        'source': source,
                        'target': target,
                        'clusters': [cluster_id],
                        'contexts': [context]
                    }
                else:
                    if cluster_id not in self.contextual_relationships[rel_id]['clusters']:
                        self.contextual_relationships[rel_id]['clusters'].append(cluster_id)
                    if context not in self.contextual_relationships[rel_id]['contexts']:
                        self.contextual_relationships[rel_id]['contexts'].append(context)
    
    def _reinforce_entities_in_cluster(self, cluster: Dict[str, Any]) -> None:
        """Reinforce entities based on their cluster membership."""
        entities = cluster['entities']
        domain_relevance = cluster['domain_relevance']
        
        # Get current quality states
        quality_states = self.get_quality_states()
        
        # Find the highest quality entity in the cluster
        highest_quality = 'poor'
        highest_quality_entity = None
        
        for entity in entities:
            # Find the current quality state of this entity
            current_quality = 'poor'
            for quality in ['good', 'uncertain', 'poor']:
                if entity in quality_states.get(quality, {}):
                    current_quality = quality
                    break
            
            # Update highest quality if needed
            if (current_quality == 'good' or 
                (current_quality == 'uncertain' and highest_quality == 'poor')):
                highest_quality = current_quality
                highest_quality_entity = entity
        
        # If there's a good or uncertain entity, use it to reinforce others
        if highest_quality_entity and highest_quality in ['good', 'uncertain']:
            for entity in entities:
                if entity != highest_quality_entity:
                    self._reinforce_entity(entity, highest_quality_entity, cluster)
    
    def _reinforce_entity(self, entity: str, reinforcing_entity: str, cluster: Dict[str, Any]) -> None:
        """Reinforce an entity using another higher-quality entity."""
        # Get current quality states
        quality_states = self.get_quality_states()
        
        # Find the current quality state and metrics of this entity
        current_quality = 'poor'
        current_metrics = None
        
        for quality in ['good', 'uncertain', 'poor']:
            if entity in quality_states.get(quality, {}):
                current_quality = quality
                current_metrics = quality_states[quality][entity]
                break
        
        if not current_metrics:
            return
        
        # Find the quality state of the reinforcing entity
        reinforcing_quality = 'poor'
        for quality in ['good', 'uncertain', 'poor']:
            if reinforcing_entity in quality_states.get(quality, {}):
                reinforcing_quality = quality
                break
        
        # Calculate reinforcement boost based on reinforcing entity quality
        coherence_boost = 0.1 if reinforcing_quality == 'good' else 0.05
        stability_boost = 0.1 if reinforcing_quality == 'good' else 0.05
        
        # Get current metrics with defaults for missing values
        current_coherence = current_metrics.get('coherence', 0.3)
        current_stability = current_metrics.get('stability', 0.0)
        
        # Apply boosts to metrics
        new_coherence = min(0.9, current_coherence + coherence_boost)
        new_stability = min(0.9, current_stability + stability_boost)
        
        # Determine new quality state based on updated metrics
        new_quality = self._determine_quality_state(new_coherence, new_stability)
        
        # If quality state would improve, apply the reinforcement
        if ((new_quality == 'good' and current_quality != 'good') or
            (new_quality == 'uncertain' and current_quality == 'poor')):
            
            # Remove from current quality state
            if entity in quality_states.get(current_quality, {}):
                del quality_states[current_quality][entity]
            
            # Add to new quality state with updated metrics
            if new_quality not in quality_states:
                quality_states[new_quality] = {}
            
            quality_states[new_quality][entity] = {
                'coherence': new_coherence,
                'stability': new_stability,
                'energy': current_metrics['energy']
            }
            
            # Record the reinforcement event
            self.reinforcement_events.append({
                'entity': entity,
                'reinforcing_entity': reinforcing_entity,
                'from_quality': current_quality,
                'to_quality': new_quality,
                'from_metrics': {
                    'coherence': current_metrics['coherence'],
                    'stability': current_metrics['stability']
                },
                'to_metrics': {
                    'coherence': new_coherence,
                    'stability': new_stability
                },
                'cluster_domain_relevance': cluster['domain_relevance'],
                'context': cluster['text'][:100] + '...' if len(cluster['text']) > 100 else cluster['text']
            })
            
            logging.info(f"Reinforced entity '{entity}' from {current_quality} to {new_quality} using '{reinforcing_entity}'")
    
    def _determine_quality_state(self, coherence: float, stability: float) -> str:
        """Determine quality state based on coherence and stability metrics."""
        if coherence >= 0.6 and stability >= 0.5:
            return 'good'
        elif coherence >= 0.4 and stability >= 0.2:
            return 'uncertain'
        else:
            return 'poor'
    
    def _determine_entity_category(self, entity: str) -> Optional[str]:
        """Determine the category of an entity."""
        # Check if we've already categorized this entity
        if entity in self.entity_categories:
            return self.entity_categories[entity]
        
        # Check each category for matches
        for category, terms in ENTITY_CATEGORIES.items():
            for term in terms:
                if term.lower() in entity.lower():
                    self.entity_categories[entity] = category
                    return category
        
        # No category found
        return None

def load_document(file_path: str) -> str:
    """Load document from file."""
    with open(file_path, 'r') as f:
        return f.read()

def run_contextual_reinforcement_test():
    """Run the contextual reinforcement test with climate risk documents."""
    # Set up logging
    log_file, results_file = setup_logging()
    logger = logging.getLogger()
    
    logger.info("Running contextual reinforcement test with climate risk documents")
    
    # Initialize components
    pattern_repository = InMemoryPatternRepository()
    
    # Create standard extractor for comparison
    standard_extractor = ContextAwareExtractor(
        window_sizes=[2, 3, 5],
        quality_threshold=0.7
    )
    
    # Create contextual reinforcement extractor
    contextual_extractor = ContextualReinforcementExtractor(
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
    
    # Initialize comparison results
    comparison_results = {
        'documents': [],
        'overall': {
            'standard': {
                'total_entities': 0,
                'good_entities': 0,
                'uncertain_entities': 0,
                'poor_entities': 0
            },
            'contextual': {
                'total_entities': 0,
                'good_entities': 0,
                'uncertain_entities': 0,
                'poor_entities': 0,
                'reinforcement_events': 0
            }
        }
    }
    
    # Process each document
    for doc_path in documents:
        doc_name = doc_path.name
        logger.info(f"\n===== Processing document: {doc_name} =====")
        
        # Load document
        document_text = load_document(str(doc_path))
        
        # Process with standard extractor
        logger.info("Processing with standard extractor...")
        standard_results = standard_extractor.process_document(document_text)
        
        # Process with contextual reinforcement extractor
        logger.info("Processing with contextual reinforcement extractor...")
        contextual_results = contextual_extractor.process_document(document_text)
        
        # Analyze standard results
        standard_quality_states = standard_results.get('quality_states', {})
        standard_good = len(standard_quality_states.get('good', {}))
        standard_uncertain = len(standard_quality_states.get('uncertain', {}))
        standard_poor = len(standard_quality_states.get('poor', {}))
        standard_total = standard_good + standard_uncertain + standard_poor
        
        # Analyze contextual results
        contextual_quality_states = contextual_results.get('quality_states', {})
        contextual_good = len(contextual_quality_states.get('good', {}))
        contextual_uncertain = len(contextual_quality_states.get('uncertain', {}))
        contextual_poor = len(contextual_quality_states.get('poor', {}))
        contextual_total = contextual_good + contextual_uncertain + contextual_poor
        
        # Count reinforcement events
        reinforcement_events = contextual_results.get('reinforcement_events', [])
        num_reinforcement_events = len(reinforcement_events)
        
        # Log comparison
        logger.info(f"\n----- Extraction Results Comparison for {doc_name} -----")
        logger.info(f"Standard Extractor: {standard_total} entities (good: {standard_good}, uncertain: {standard_uncertain}, poor: {standard_poor})")
        logger.info(f"Contextual Extractor: {contextual_total} entities (good: {contextual_good}, uncertain: {contextual_uncertain}, poor: {contextual_poor})")
        logger.info(f"Reinforcement events: {num_reinforcement_events}")
        
        # Calculate quality improvements
        good_improvement = contextual_good - standard_good
        uncertain_improvement = contextual_uncertain - standard_uncertain
        
        logger.info(f"Quality improvements: +{good_improvement} good entities, +{uncertain_improvement} uncertain entities")
        
        # Log reinforcement events
        if num_reinforcement_events > 0:
            logger.info("\nTop reinforcement events:")
            for i, event in enumerate(reinforcement_events[:5]):  # Show top 5 events
                logger.info(f"  {i+1}. '{event['entity']}' improved from {event['from_quality']} to {event['to_quality']} using '{event['reinforcing_entity']}'")
        
        # Add to comparison results
        comparison_results['documents'].append({
            'name': doc_name,
            'standard': {
                'total_entities': standard_total,
                'good_entities': standard_good,
                'uncertain_entities': standard_uncertain,
                'poor_entities': standard_poor
            },
            'contextual': {
                'total_entities': contextual_total,
                'good_entities': contextual_good,
                'uncertain_entities': contextual_uncertain,
                'poor_entities': contextual_poor,
                'reinforcement_events': num_reinforcement_events
            },
            'improvements': {
                'good_entities': good_improvement,
                'uncertain_entities': uncertain_improvement
            }
        })
        
        # Update overall metrics
        comparison_results['overall']['standard']['total_entities'] += standard_total
        comparison_results['overall']['standard']['good_entities'] += standard_good
        comparison_results['overall']['standard']['uncertain_entities'] += standard_uncertain
        comparison_results['overall']['standard']['poor_entities'] += standard_poor
        
        comparison_results['overall']['contextual']['total_entities'] += contextual_total
        comparison_results['overall']['contextual']['good_entities'] += contextual_good
        comparison_results['overall']['contextual']['uncertain_entities'] += contextual_uncertain
        comparison_results['overall']['contextual']['poor_entities'] += contextual_poor
        comparison_results['overall']['contextual']['reinforcement_events'] += num_reinforcement_events
    
    # Calculate overall improvements
    overall_good_improvement = (comparison_results['overall']['contextual']['good_entities'] - 
                               comparison_results['overall']['standard']['good_entities'])
    
    overall_uncertain_improvement = (comparison_results['overall']['contextual']['uncertain_entities'] - 
                                    comparison_results['overall']['standard']['uncertain_entities'])
    
    comparison_results['overall']['improvements'] = {
        'good_entities': overall_good_improvement,
        'uncertain_entities': overall_uncertain_improvement
    }
    
    # Log overall results
    logger.info("\n\n===== Overall Contextual Reinforcement Results =====")
    logger.info(f"Standard Extractor: {comparison_results['overall']['standard']['total_entities']} entities")
    logger.info(f"  Good entities: {comparison_results['overall']['standard']['good_entities']}")
    logger.info(f"  Uncertain entities: {comparison_results['overall']['standard']['uncertain_entities']}")
    logger.info(f"  Poor entities: {comparison_results['overall']['standard']['poor_entities']}")
    
    logger.info(f"Contextual Extractor: {comparison_results['overall']['contextual']['total_entities']} entities")
    logger.info(f"  Good entities: {comparison_results['overall']['contextual']['good_entities']}")
    logger.info(f"  Uncertain entities: {comparison_results['overall']['contextual']['uncertain_entities']}")
    logger.info(f"  Poor entities: {comparison_results['overall']['contextual']['poor_entities']}")
    logger.info(f"  Total reinforcement events: {comparison_results['overall']['contextual']['reinforcement_events']}")
    
    logger.info(f"Overall Improvements: +{overall_good_improvement} good entities, +{overall_uncertain_improvement} uncertain entities")
    
    # Save results to JSON for further analysis
    with open(results_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    logger.info("\nContextual reinforcement test completed successfully.")
    
    return True

if __name__ == "__main__":
    success = run_contextual_reinforcement_test()
    if success:
        logging.info("Contextual reinforcement test completed successfully")
    else:
        logging.error("Test failed")
