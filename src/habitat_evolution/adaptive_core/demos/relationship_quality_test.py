"""
Relationship Quality Test for Habitat Evolution.

This script tests the implementation of quality assessment for relationships
between entities, extending our quality state machine to include relationship quality.
"""

import logging
import sys
import json
import os
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List, Any, Optional, Tuple
import re

# Import the fix for handling both import styles
from .import_fix import *

from habitat_evolution.adaptive_core.emergence.context_aware_extraction.context_aware_extractor import ContextAwareExtractor
from habitat_evolution.adaptive_core.emergence.context_aware_extraction.quality_assessment import QualityAssessment
from habitat_evolution.adaptive_core.persistence.interfaces.repository_adapter import InMemoryPatternRepository

# Define relationship types for comprehensive testing
RELATIONSHIP_TYPES = [
    # Structural relationships
    'part_of', 'contains', 'component_of', 'located_in', 'adjacent_to',
    
    # Causal relationships
    'causes', 'affects', 'damages', 'mitigates', 'prevents',
    
    # Functional relationships
    'protects_against', 'analyzes', 'evaluates', 'monitors', 'implements',
    
    # Temporal relationships
    'precedes', 'follows', 'concurrent_with', 'during', 'after'
]

# Define entity categories for comprehensive testing
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

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path("src/habitat_evolution/adaptive_core/demos/analysis_results")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file and console
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"relationship_quality_test_{timestamp}.log"
    results_file = log_dir / f"relationship_quality_test_{timestamp}.json"
    
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

class Relationship:
    """Represents a relationship between two entities with quality assessment."""
    
    def __init__(self, source: str, target: str, relation_type: str, 
                 source_category: str = None, target_category: str = None):
        """Initialize a relationship with source and target entities."""
        self.source = source
        self.target = target
        self.relation_type = relation_type
        self.source_category = source_category
        self.target_category = target_category
        self.quality_state = "poor"  # Start with poor quality
        self.metrics = {
            "coherence": 0.0,
            "stability": 0.0,
            "energy": 0.0,
            "frequency": 0
        }
        self.appearances = 0
        self.contexts = []
    
    def add_appearance(self, context: str):
        """Record an appearance of this relationship in a document."""
        self.appearances += 1
        self.metrics["frequency"] += 1
        
        # Store context for analysis
        if context not in self.contexts:
            self.contexts.append(context)
        
        # Update stability based on number of appearances
        self.metrics["stability"] = min(0.9, self.appearances * 0.1)
        
        # Update quality state based on metrics
        self._update_quality_state()
    
    def update_coherence(self, coherence_score: float):
        """Update the coherence score for this relationship."""
        # Weighted average to smooth changes
        if self.metrics["coherence"] == 0.0:
            self.metrics["coherence"] = coherence_score
        else:
            self.metrics["coherence"] = (self.metrics["coherence"] * 0.7) + (coherence_score * 0.3)
        
        # Update quality state based on metrics
        self._update_quality_state()
    
    def _update_quality_state(self):
        """Update the quality state based on current metrics."""
        stability = self.metrics["stability"]
        coherence = self.metrics["coherence"]
        
        # Determine quality state based on thresholds
        if stability >= 0.5 and coherence >= 0.6:
            self.quality_state = "good"
        elif stability >= 0.2 and coherence >= 0.4:
            self.quality_state = "uncertain"
        else:
            self.quality_state = "poor"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary for serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "source_category": self.source_category,
            "target_category": self.target_category,
            "quality_state": self.quality_state,
            "metrics": self.metrics,
            "appearances": self.appearances,
            "contexts": self.contexts
        }

class RelationshipQualityExtractor(ContextAwareExtractor):
    """Enhanced extractor with relationship quality assessment capabilities."""
    
    def __init__(self, window_sizes=None, quality_threshold=0.7, data_dir=None):
        """Initialize with relationship quality assessment capabilities."""
        super().__init__(window_sizes, quality_threshold, data_dir)
        
        # Store relationships with their quality assessments
        self.relationships = {}
        
        # Track entity categories
        self.entity_categories = {}
        
        # Pre-compile patterns for relationship extraction
        self._compile_relationship_patterns()
        
        logging.info(f"Initialized RelationshipQualityExtractor with window sizes {self.window_sizes}")
    
    def _compile_relationship_patterns(self):
        """Compile patterns for relationship extraction."""
        self.relationship_patterns = {
            # Causal patterns
            'causes': [
                r'(.*?)\s+causes\s+(.*)',
                r'(.*?)\s+leads to\s+(.*)',
                r'(.*?)\s+results in\s+(.*)',
                r'(.*?)\s+triggers\s+(.*)'
            ],
            'affects': [
                r'(.*?)\s+affects\s+(.*)',
                r'(.*?)\s+impacts\s+(.*)',
                r'(.*?)\s+influences\s+(.*)'
            ],
            'damages': [
                r'(.*?)\s+damages\s+(.*)',
                r'(.*?)\s+harms\s+(.*)',
                r'(.*?)\s+degrades\s+(.*)'
            ],
            'mitigates': [
                r'(.*?)\s+mitigates\s+(.*)',
                r'(.*?)\s+reduces\s+(.*)',
                r'(.*?)\s+alleviates\s+(.*)'
            ],
            
            # Structural patterns
            'part_of': [
                r'(.*?)\s+is part of\s+(.*)',
                r'(.*?)\s+belongs to\s+(.*)',
                r'(.*?)\s+within\s+(.*)'
            ],
            'contains': [
                r'(.*?)\s+contains\s+(.*)',
                r'(.*?)\s+includes\s+(.*)',
                r'(.*?)\s+encompasses\s+(.*)'
            ],
            
            # Functional patterns
            'protects_against': [
                r'(.*?)\s+protects against\s+(.*)',
                r'(.*?)\s+defends against\s+(.*)',
                r'(.*?)\s+shields from\s+(.*)'
            ],
            'analyzes': [
                r'(.*?)\s+analyzes\s+(.*)',
                r'(.*?)\s+examines\s+(.*)',
                r'(.*?)\s+studies\s+(.*)'
            ],
            
            # Temporal patterns
            'precedes': [
                r'(.*?)\s+precedes\s+(.*)',
                r'(.*?)\s+comes before\s+(.*)',
                r'(.*?)\s+prior to\s+(.*)'
            ],
            'concurrent_with': [
                r'(.*?)\s+concurrent with\s+(.*)',
                r'(.*?)\s+simultaneous with\s+(.*)',
                r'(.*?)\s+along with\s+(.*)'
            ]
        }
    
    def process_document(self, document: str) -> Dict[str, Any]:
        """Process document with relationship quality assessment."""
        # First, use the standard processing
        results = super().process_document(document)
        
        # Extract and assess relationships
        relationships = self._extract_relationships(document)
        
        # Add relationships to results
        results['relationships'] = {
            'good': {},
            'uncertain': {},
            'poor': {}
        }
        
        for rel_id, relationship in relationships.items():
            quality = relationship.quality_state
            results['relationships'][quality][rel_id] = relationship.to_dict()
        
        return results
    
    def _extract_relationships(self, document: str) -> Dict[str, Relationship]:
        """Extract relationships from document text."""
        # Split document into lines for better context
        lines = document.split('\n')
        
        # Extract relationships from each line
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Apply relationship patterns
            for relation_type, patterns in self.relationship_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        source = match.group(1).strip()
                        target = match.group(2).strip()
                        
                        # Skip if source or target is too short
                        if len(source) < 3 or len(target) < 3:
                            continue
                        
                        # Create relationship ID
                        rel_id = f"{source}|{relation_type}|{target}"
                        
                        # Determine entity categories
                        source_category = self._determine_entity_category(source)
                        target_category = self._determine_entity_category(target)
                        
                        # Create or update relationship
                        if rel_id not in self.relationships:
                            self.relationships[rel_id] = Relationship(
                                source=source,
                                target=target,
                                relation_type=relation_type,
                                source_category=source_category,
                                target_category=target_category
                            )
                        
                        # Record appearance and update coherence
                        self.relationships[rel_id].add_appearance(line)
                        
                        # Calculate coherence based on context
                        coherence = self._calculate_relationship_coherence(source, target, relation_type, line)
                        self.relationships[rel_id].update_coherence(coherence)
        
        # Also extract co-occurrence relationships
        self._extract_cooccurrence_relationships(document)
        
        return self.relationships
    
    def _extract_cooccurrence_relationships(self, document: str):
        """Extract co-occurrence relationships between entities."""
        # Get all entities from the document
        entities = []
        quality_states = self.get_quality_states()
        for quality_state in ['good', 'uncertain', 'poor']:
            entities.extend(list(quality_states.get(quality_state, {}).keys()))
        
        # Create a window of 50 words to check for co-occurrences
        words = document.split()
        window_size = 50
        
        for i in range(len(words) - window_size + 1):
            window = ' '.join(words[i:i+window_size])
            
            # Find entities in this window
            window_entities = []
            for entity in entities:
                if entity in window:
                    window_entities.append(entity)
            
            # Create co-occurrence relationships for entities in this window
            for j in range(len(window_entities)):
                for k in range(j+1, len(window_entities)):
                    source = window_entities[j]
                    target = window_entities[k]
                    
                    # Skip if source or target is too short
                    if len(source) < 3 or len(target) < 3:
                        continue
                    
                    # Create relationship ID
                    rel_id = f"{source}|co_occurs_with|{target}"
                    
                    # Determine entity categories
                    source_category = self._determine_entity_category(source)
                    target_category = self._determine_entity_category(target)
                    
                    # Create or update relationship
                    if rel_id not in self.relationships:
                        self.relationships[rel_id] = Relationship(
                            source=source,
                            target=target,
                            relation_type="co_occurs_with",
                            source_category=source_category,
                            target_category=target_category
                        )
                    
                    # Record appearance and update coherence
                    self.relationships[rel_id].add_appearance(window)
                    
                    # Calculate coherence based on proximity
                    coherence = 0.5  # Base coherence for co-occurrence
                    self.relationships[rel_id].update_coherence(coherence)
                    
                    logging.info(f"Identified co-occurrence relationship: {source} <-> {target}")
    
    def _calculate_relationship_coherence(self, source: str, target: str, relation_type: str, context: str) -> float:
        """Calculate coherence score for a relationship based on context."""
        # Base coherence score
        coherence = 0.3
        
        # Check if relationship makes semantic sense based on entity categories
        source_category = self._determine_entity_category(source)
        target_category = self._determine_entity_category(target)
        
        # Certain relationships between specific categories are more coherent
        if relation_type == 'causes' and source_category == 'CLIMATE_HAZARD' and target_category == 'ECOSYSTEM':
            coherence += 0.3
        elif relation_type == 'mitigates' and source_category == 'ADAPTATION_STRATEGY' and target_category == 'CLIMATE_HAZARD':
            coherence += 0.3
        elif relation_type == 'part_of' and source_category == target_category:
            coherence += 0.2
        elif relation_type == 'analyzes' and source_category == 'ASSESSMENT_COMPONENT':
            coherence += 0.2
        
        # Check if entities are close to each other in the context
        source_pos = context.find(source)
        target_pos = context.find(target)
        if source_pos >= 0 and target_pos >= 0:
            distance = abs(source_pos - target_pos)
            if distance < 20:
                coherence += 0.2
            elif distance < 50:
                coherence += 0.1
        
        # Check if the relationship is explicitly stated in the context
        relation_phrase = f"{source} {relation_type.replace('_', ' ')} {target}"
        if relation_phrase.lower() in context.lower():
            coherence += 0.2
        
        return min(0.9, coherence)  # Cap at 0.9
    
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

def run_relationship_quality_test():
    """Run the relationship quality test with climate risk documents."""
    # Set up logging
    log_file, results_file = setup_logging()
    logger = logging.getLogger()
    
    logger.info("Running relationship quality test with climate risk documents")
    
    # Initialize components
    pattern_repository = InMemoryPatternRepository()
    
    # Create relationship quality extractor
    relationship_extractor = RelationshipQualityExtractor(
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
    
    # Initialize results
    test_results = {
        'documents': [],
        'overall': {
            'total_relationships': 0,
            'good_relationships': 0,
            'uncertain_relationships': 0,
            'poor_relationships': 0,
            'relationship_types': {},
            'cross_category_relationships': {}
        }
    }
    
    # Process each document
    for doc_path in documents:
        doc_name = doc_path.name
        logger.info(f"\n===== Processing document: {doc_name} =====")
        
        # Load document
        document_text = load_document(str(doc_path))
        
        # Process with relationship quality extractor
        results = relationship_extractor.process_document(document_text)
        
        # Analyze relationship quality
        relationships = results.get('relationships', {})
        good_relationships = relationships.get('good', {})
        uncertain_relationships = relationships.get('uncertain', {})
        poor_relationships = relationships.get('poor', {})
        
        total_relationships = len(good_relationships) + len(uncertain_relationships) + len(poor_relationships)
        
        # Count relationship types
        relationship_types = {}
        cross_category_relationships = {}
        
        for quality, rels in relationships.items():
            for rel_id, rel_data in rels.items():
                rel_type = rel_data['relation_type']
                source_category = rel_data['source_category']
                target_category = rel_data['target_category']
                
                # Count relationship types
                if rel_type not in relationship_types:
                    relationship_types[rel_type] = {
                        'total': 0,
                        'good': 0,
                        'uncertain': 0,
                        'poor': 0
                    }
                
                relationship_types[rel_type]['total'] += 1
                relationship_types[rel_type][quality] += 1
                
                # Count cross-category relationships
                if source_category and target_category:
                    category_pair = f"{source_category}|{target_category}"
                    if category_pair not in cross_category_relationships:
                        cross_category_relationships[category_pair] = {
                            'total': 0,
                            'good': 0,
                            'uncertain': 0,
                            'poor': 0
                        }
                    
                    cross_category_relationships[category_pair]['total'] += 1
                    cross_category_relationships[category_pair][quality] += 1
        
        # Log results for this document
        logger.info(f"Extracted {total_relationships} relationships (good: {len(good_relationships)}, uncertain: {len(uncertain_relationships)}, poor: {len(poor_relationships)})")
        
        # Log relationship types
        logger.info("\nRelationship types:")
        for rel_type, counts in relationship_types.items():
            logger.info(f"  {rel_type}: {counts['total']} relationships (good: {counts['good']}, uncertain: {counts['uncertain']}, poor: {counts['poor']})")
        
        # Log cross-category relationships
        logger.info("\nCross-category relationships:")
        for category_pair, counts in cross_category_relationships.items():
            source_cat, target_cat = category_pair.split('|')
            logger.info(f"  {source_cat} -> {target_cat}: {counts['total']} relationships (good: {counts['good']}, uncertain: {counts['uncertain']}, poor: {counts['poor']})")
        
        # Add to test results
        test_results['documents'].append({
            'name': doc_name,
            'total_relationships': total_relationships,
            'good_relationships': len(good_relationships),
            'uncertain_relationships': len(uncertain_relationships),
            'poor_relationships': len(poor_relationships),
            'relationship_types': relationship_types,
            'cross_category_relationships': cross_category_relationships
        })
        
        # Update overall metrics
        test_results['overall']['total_relationships'] += total_relationships
        test_results['overall']['good_relationships'] += len(good_relationships)
        test_results['overall']['uncertain_relationships'] += len(uncertain_relationships)
        test_results['overall']['poor_relationships'] += len(poor_relationships)
        
        # Update overall relationship types
        for rel_type, counts in relationship_types.items():
            if rel_type not in test_results['overall']['relationship_types']:
                test_results['overall']['relationship_types'][rel_type] = {
                    'total': 0,
                    'good': 0,
                    'uncertain': 0,
                    'poor': 0
                }
            
            test_results['overall']['relationship_types'][rel_type]['total'] += counts['total']
            test_results['overall']['relationship_types'][rel_type]['good'] += counts['good']
            test_results['overall']['relationship_types'][rel_type]['uncertain'] += counts['uncertain']
            test_results['overall']['relationship_types'][rel_type]['poor'] += counts['poor']
        
        # Update overall cross-category relationships
        for category_pair, counts in cross_category_relationships.items():
            if category_pair not in test_results['overall']['cross_category_relationships']:
                test_results['overall']['cross_category_relationships'][category_pair] = {
                    'total': 0,
                    'good': 0,
                    'uncertain': 0,
                    'poor': 0
                }
            
            test_results['overall']['cross_category_relationships'][category_pair]['total'] += counts['total']
            test_results['overall']['cross_category_relationships'][category_pair]['good'] += counts['good']
            test_results['overall']['cross_category_relationships'][category_pair]['uncertain'] += counts['uncertain']
            test_results['overall']['cross_category_relationships'][category_pair]['poor'] += counts['poor']
    
    # Log overall results
    logger.info("\n\n===== Overall Relationship Quality Results =====")
    logger.info(f"Total relationships: {test_results['overall']['total_relationships']}")
    logger.info(f"Good relationships: {test_results['overall']['good_relationships']} ({test_results['overall']['good_relationships'] / max(1, test_results['overall']['total_relationships']) * 100:.1f}%)")
    logger.info(f"Uncertain relationships: {test_results['overall']['uncertain_relationships']} ({test_results['overall']['uncertain_relationships'] / max(1, test_results['overall']['total_relationships']) * 100:.1f}%)")
    logger.info(f"Poor relationships: {test_results['overall']['poor_relationships']} ({test_results['overall']['poor_relationships'] / max(1, test_results['overall']['total_relationships']) * 100:.1f}%)")
    
    # Log overall relationship types
    logger.info("\nOverall relationship types:")
    for rel_type, counts in test_results['overall']['relationship_types'].items():
        logger.info(f"  {rel_type}: {counts['total']} relationships (good: {counts['good']}, uncertain: {counts['uncertain']}, poor: {counts['poor']})")
    
    # Log overall cross-category relationships
    logger.info("\nOverall cross-category relationships:")
    for category_pair, counts in test_results['overall']['cross_category_relationships'].items():
        source_cat, target_cat = category_pair.split('|')
        logger.info(f"  {source_cat} -> {target_cat}: {counts['total']} relationships (good: {counts['good']}, uncertain: {counts['uncertain']}, poor: {counts['poor']})")
    
    # Save results to JSON for further analysis
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    logger.info("\nRelationship quality test completed successfully.")
    
    return True

if __name__ == "__main__":
    success = run_relationship_quality_test()
    if success:
        logging.info("Relationship quality test completed successfully")
    else:
        logging.error("Test failed")
