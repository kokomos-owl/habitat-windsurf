"""
Context-Aware NER Evolution Test for Habitat Evolution.

This script demonstrates how domain-specific Named Entity Recognition (NER)
evolves through document ingestion, leveraging contextual reinforcement
and quality transitions to improve entity recognition over time.
"""

import logging
import sys
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional, Set
import uuid
import networkx as nx
import matplotlib.pyplot as plt

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(src_path))

# Import the semantic current observer
from src.habitat_evolution.adaptive_core.transformation.semantic_current_observer import SemanticCurrentObserver, SemanticRelationship

# Configure logging
def setup_logging():
    """Set up logging for the test."""
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent / "analysis_results"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file and console
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"context_aware_ner_evolution_{timestamp}.log"
    results_file = log_dir / f"context_aware_ner_evolution_{timestamp}.json"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger, log_file, results_file

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

class MockAdaptiveID:
    """Mock implementation of AdaptiveID."""
    
    def __init__(self, id_str, creator_id=None):
        """Initialize the mock adaptive ID."""
        self.id = id_str
        self.creator_id = creator_id or "mock_creator"

class MockEvent:
    """Mock implementation of Event."""
    
    def __init__(self, event_type, source, data):
        """Initialize the mock event."""
        self.event_type = event_type
        self.source = source
        self.data = data

class MockEventBus:
    """Mock implementation of EventBus."""
    
    def __init__(self):
        """Initialize the mock event bus."""
        self.subscribers = {}
    
    def subscribe(self, event_type, callback):
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def publish(self, event):
        """Publish an event."""
        event_type = event.event_type
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                callback(event)

class ContextAwareNEREvolutionTest:
    """Test for context-aware NER evolution through document ingestion."""
    
    def __init__(self, logger):
        """Initialize the test components."""
        self.logger = logger
        
        # Create mock event bus
        self.event_bus = MockEventBus()
        
        # Create mock adaptive ID
        self.observer_id = MockAdaptiveID("context_aware_ner_observer")
        
        # Create semantic observer
        self.semantic_observer = SemanticCurrentObserver(self.observer_id)
        
        # Initialize metrics tracking
        self.metrics = {
            'documents_processed': 0,
            'entities': {
                'total': 0,
                'by_quality': {'good': 0, 'uncertain': 0, 'poor': 0},
                'by_category': {category: 0 for category in ENTITY_CATEGORIES.keys()}
            },
            'relationships': {
                'total': 0,
                'by_quality': {'good': 0, 'uncertain': 0, 'poor': 0},
                'by_type': {rel_type: 0 for rel_type in RELATIONSHIP_TYPES}
            },
            'quality_transitions': {
                'poor_to_uncertain': 0,
                'uncertain_to_good': 0,
                'poor_to_good': 0
            },
            'domain_relevance': {
                'initial': 0.0,
                'final': 0.0
            }
        }
        
        # Track document processing results
        self.document_results = []
        
        # Initialize visualization data
        self.entity_network = nx.DiGraph()
        
        # Subscribe to quality transition events
        self.event_bus.subscribe("entity.quality.transition", self._on_quality_transition)
        
        self.logger.info("Initialized Context-Aware NER Evolution Test")
    
    def _on_quality_transition(self, event):
        """Handle entity quality transition events."""
        transition_data = event.data
        entity = transition_data.get('entity', '')
        from_quality = transition_data.get('from_quality', '')
        to_quality = transition_data.get('to_quality', '')
        
        # Update transition metrics
        transition_key = f"{from_quality}_to_{to_quality}"
        if transition_key in self.metrics['quality_transitions']:
            self.metrics['quality_transitions'][transition_key] += 1
        
        # Update entity in network
        if self.entity_network.has_node(entity):
            self.entity_network.nodes[entity]['quality'] = to_quality
        
        self.logger.info(f"Entity quality transition: {entity} from {from_quality} to {to_quality}")
    
    def _determine_entity_category(self, entity: str) -> Optional[str]:
        """Determine the category of an entity."""
        for category, terms in ENTITY_CATEGORIES.items():
            for term in terms:
                if term.lower() in entity.lower():
                    return category
        return None
    
    def _extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using a sliding window approach."""
        entities = []
        
        # Split text into sentences
        sentences = text.split('.')
        
        # Process each sentence
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Clean the sentence
            clean_sentence = sentence.strip()
            
            # Split into words
            words = clean_sentence.split()
            
            # Use sliding windows of different sizes to extract potential entities
            for window_size in [1, 2, 3, 4, 5]:
                for i in range(len(words) - window_size + 1):
                    # Extract potential entity
                    entity_text = ' '.join(words[i:i+window_size])
                    
                    # Check if this matches any known entity category
                    category = self._determine_entity_category(entity_text)
                    
                    if category:
                        # Determine initial quality based on exact match
                        quality = 'uncertain'
                        for term in ENTITY_CATEGORIES[category]:
                            if term.lower() == entity_text.lower():
                                quality = 'good'
                                break
                        
                        # Create entity
                        entity = {
                            'id': str(uuid.uuid4()),
                            'text': entity_text,
                            'category': category,
                            'quality': quality,
                            'context': {
                                'sentence': clean_sentence,
                                'position': i
                            }
                        }
                        
                        entities.append(entity)
        
        return entities
    
    def _extract_relationships_from_entities(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        relationships = []
        
        # Create a map of entities by their position in the text
        entity_map = {}
        for entity in entities:
            position = entity['context']['position']
            if position not in entity_map:
                entity_map[position] = []
            entity_map[position].append(entity)
        
        # Look for relationships between entities that are close to each other
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                entity1 = entities[i]
                entity2 = entities[j]
                
                # Skip if entities are the same
                if entity1['id'] == entity2['id']:
                    continue
                
                # Calculate distance between entities
                pos1 = entity1['context']['position']
                pos2 = entity2['context']['position']
                distance = abs(pos1 - pos2)
                
                # Only consider entities that are close to each other
                if distance <= 10:
                    # Determine relationship type based on entity categories
                    rel_type = self._determine_relationship_type(entity1, entity2)
                    
                    if rel_type:
                        # Create relationship
                        relationship = {
                            'id': str(uuid.uuid4()),
                            'source': entity1['text'],
                            'target': entity2['text'],
                            'relation_type': rel_type,
                            'source_category': entity1['category'],
                            'target_category': entity2['category'],
                            'quality': 'uncertain',
                            'context': {
                                'sentence': entity1['context']['sentence'],
                                'distance': distance
                            }
                        }
                        
                        relationships.append(relationship)
        
        return relationships
    
    def _determine_relationship_type(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> Optional[str]:
        """Determine the type of relationship between two entities."""
        # Define relationship patterns based on entity categories
        relationship_patterns = {
            ('CLIMATE_HAZARD', 'ECOSYSTEM'): ['affects', 'damages'],
            ('CLIMATE_HAZARD', 'INFRASTRUCTURE'): ['damages', 'affects'],
            ('ECOSYSTEM', 'INFRASTRUCTURE'): ['protects_against', 'adjacent_to'],
            ('ADAPTATION_STRATEGY', 'CLIMATE_HAZARD'): ['mitigates', 'prevents'],
            ('ADAPTATION_STRATEGY', 'ECOSYSTEM'): ['protects', 'enhances'],
            ('ASSESSMENT_COMPONENT', 'CLIMATE_HAZARD'): ['evaluates', 'analyzes'],
            ('ASSESSMENT_COMPONENT', 'ECOSYSTEM'): ['evaluates', 'analyzes'],
            ('ASSESSMENT_COMPONENT', 'INFRASTRUCTURE'): ['evaluates', 'analyzes'],
            ('ASSESSMENT_COMPONENT', 'ADAPTATION_STRATEGY'): ['evaluates', 'implements']
        }
        
        # Check if we have a defined relationship pattern
        cat1 = entity1['category']
        cat2 = entity2['category']
        
        if (cat1, cat2) in relationship_patterns:
            # Return the first relationship type
            return relationship_patterns[(cat1, cat2)][0]
        elif (cat2, cat1) in relationship_patterns:
            # Return the first relationship type for the reversed categories
            return relationship_patterns[(cat2, cat1)][0]
        
        # Default relationships based on general patterns
        if cat1 == cat2:
            return 'related_to'
        
        return 'affects'
    
    def _apply_contextual_reinforcement(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply contextual reinforcement to improve entity quality."""
        quality_transitions = []
        
        # Create a map of entities by their text
        entity_map = {entity['text']: entity for entity in entities}
        
        # Create a map of relationships by source and target
        relationship_map = {}
        for rel in relationships:
            source = rel['source']
            target = rel['target']
            if source not in relationship_map:
                relationship_map[source] = {}
            relationship_map[source][target] = rel
        
        # Apply reinforcement rules
        for entity in entities:
            entity_text = entity['text']
            current_quality = entity['quality']
            
            # Rule 1: If an entity has multiple relationships, increase its quality
            related_entities = 0
            if entity_text in relationship_map:
                related_entities = len(relationship_map[entity_text])
            
            # Count relationships where this entity is the target
            for source in relationship_map:
                if entity_text in relationship_map[source]:
                    related_entities += 1
            
            # Rule 2: If an entity is related to a good quality entity, increase its quality
            related_to_good = False
            if entity_text in relationship_map:
                for target in relationship_map[entity_text]:
                    if target in entity_map and entity_map[target]['quality'] == 'good':
                        related_to_good = True
                        break
            
            # Apply quality transitions
            new_quality = current_quality
            
            if current_quality == 'poor':
                if related_entities >= 2 or related_to_good:
                    new_quality = 'uncertain'
            elif current_quality == 'uncertain':
                if related_entities >= 3 and related_to_good:
                    new_quality = 'good'
            
            # Record quality transition if quality changed
            if new_quality != current_quality:
                entity['quality'] = new_quality
                
                # Create quality transition event
                transition = {
                    'entity': entity_text,
                    'from_quality': current_quality,
                    'to_quality': new_quality,
                    'timestamp': datetime.now().isoformat()
                }
                
                quality_transitions.append(transition)
                
                # Publish quality transition event
                event = MockEvent(
                    event_type="entity.quality.transition",
                    source=self.observer_id.id,
                    data=transition
                )
                
                self.event_bus.publish(event)
        
        return quality_transitions
    
    def _calculate_domain_relevance(self) -> float:
        """Calculate the domain relevance of detected entities."""
        if not self.entity_network.nodes():
            return 0.0
        
        categorized_entities = 0
        for node, data in self.entity_network.nodes(data=True):
            if data.get('category'):
                categorized_entities += 1
        
        return categorized_entities / max(1, len(self.entity_network.nodes()))
    
    def process_document(self, document_path: str) -> Dict[str, Any]:
        """Process a document to extract entities and relationships."""
        # Load document
        with open(document_path, 'r') as f:
            document_text = f.read()
        
        document_name = Path(document_path).name
        self.logger.info(f"\n===== Processing document: {document_name} =====")
        
        # Extract entities from text
        self.logger.info(f"Extracting entities from {document_name}")
        entities = self._extract_entities_from_text(document_text)
        
        # Extract relationships between entities
        self.logger.info(f"Extracting relationships from {document_name}")
        relationships = self._extract_relationships_from_entities(entities, document_text)
        
        # Apply contextual reinforcement
        self.logger.info(f"Applying contextual reinforcement for {document_name}")
        quality_transitions = self._apply_contextual_reinforcement(entities, relationships)
        
        # Update metrics
        self.metrics['documents_processed'] += 1
        
        # Update entity metrics
        for entity in entities:
            # Add to entity network
            self.entity_network.add_node(
                entity['text'],
                type='entity',
                quality=entity['quality'],
                category=entity['category'],
                weight=1
            )
            
            # Update metrics
            self.metrics['entities']['total'] += 1
            self.metrics['entities']['by_quality'][entity['quality']] += 1
            
            if entity['category'] in self.metrics['entities']['by_category']:
                self.metrics['entities']['by_category'][entity['category']] += 1
            
            # Add to semantic observer
            self.semantic_observer.update_entity_quality(entity['text'], entity['quality'])
        
        # Update relationship metrics
        for rel in relationships:
            # Add to entity network
            source = rel['source']
            target = rel['target']
            
            if source and target:
                # Ensure nodes exist
                if not self.entity_network.has_node(source):
                    self.entity_network.add_node(source, type='entity', quality='unknown')
                
                if not self.entity_network.has_node(target):
                    self.entity_network.add_node(target, type='entity', quality='unknown')
                
                # Add edge
                self.entity_network.add_edge(
                    source,
                    target,
                    type=rel['relation_type'],
                    quality=rel['quality'],
                    weight=1
                )
            
            # Update metrics
            self.metrics['relationships']['total'] += 1
            self.metrics['relationships']['by_quality'][rel['quality']] += 1
            
            if rel['relation_type'] in self.metrics['relationships']['by_type']:
                self.metrics['relationships']['by_type'][rel['relation_type']] += 1
            
            # Add to semantic observer
            self.semantic_observer.observe_relationship(
                source=rel['source'],
                predicate=rel['relation_type'],
                target=rel['target'],
                quality=rel['quality'],
                source_category=rel['source_category'],
                target_category=rel['target_category']
            )
        
        # Calculate domain relevance
        domain_relevance = self._calculate_domain_relevance()
        
        # If this is the first document, set initial domain relevance
        if self.metrics['documents_processed'] == 1:
            self.metrics['domain_relevance']['initial'] = domain_relevance
        
        # Always update final domain relevance
        self.metrics['domain_relevance']['final'] = domain_relevance
        
        # Create document result
        document_result = {
            'name': document_name,
            'entities': {
                'total': len(entities),
                'by_quality': {
                    'good': sum(1 for e in entities if e['quality'] == 'good'),
                    'uncertain': sum(1 for e in entities if e['quality'] == 'uncertain'),
                    'poor': sum(1 for e in entities if e['quality'] == 'poor')
                },
                'by_category': {}
            },
            'relationships': {
                'total': len(relationships),
                'by_quality': {
                    'good': sum(1 for r in relationships if r['quality'] == 'good'),
                    'uncertain': sum(1 for r in relationships if r['quality'] == 'uncertain'),
                    'poor': sum(1 for r in relationships if r['quality'] == 'poor')
                },
                'by_type': {}
            },
            'quality_transitions': len(quality_transitions),
            'domain_relevance': domain_relevance
        }
        
        # Count entities by category
        for category in ENTITY_CATEGORIES:
            document_result['entities']['by_category'][category] = sum(1 for e in entities if e['category'] == category)
        
        # Count relationships by type
        for rel_type in RELATIONSHIP_TYPES:
            document_result['relationships']['by_type'][rel_type] = sum(1 for r in relationships if r['relation_type'] == rel_type)
        
        self.document_results.append(document_result)
        
        self.logger.info(f"Completed processing document: {document_name}")
        self.logger.info(f"  Entities: {document_result['entities']['total']} (good: {document_result['entities']['by_quality']['good']}, uncertain: {document_result['entities']['by_quality']['uncertain']}, poor: {document_result['entities']['by_quality']['poor']})")
        self.logger.info(f"  Relationships: {document_result['relationships']['total']} (good: {document_result['relationships']['by_quality']['good']}, uncertain: {document_result['relationships']['by_quality']['uncertain']}, poor: {document_result['relationships']['by_quality']['poor']})")
        self.logger.info(f"  Quality transitions: {document_result['quality_transitions']}")
        self.logger.info(f"  Domain relevance: {document_result['domain_relevance']:.2f}")
        
        return document_result
    
    def process_all_documents(self, data_dir: str) -> Dict[str, Any]:
        """Process all documents in the specified directory."""
        # Get all text files in the directory
        documents = list(Path(data_dir).glob("*.txt"))
        
        # Sort documents to ensure consistent processing order
        documents.sort()
        
        self.logger.info(f"Found {len(documents)} documents to process in {data_dir}")
        
        # Process each document
        for doc_path in documents:
            self.process_document(str(doc_path))
        
        # Calculate final metrics
        final_metrics = {
            'documents_processed': self.metrics['documents_processed'],
            'entities': self.metrics['entities'],
            'relationships': self.metrics['relationships'],
            'quality_transitions': self.metrics['quality_transitions'],
            'domain_relevance': self.metrics['domain_relevance'],
            'document_results': self.document_results
        }
        
        return final_metrics
    
    def visualize_entity_network(self, output_path: str):
        """Visualize the entity network with quality states."""
        if not self.entity_network.nodes():
            self.logger.warning("No entities to visualize")
            return
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Create position layout
        pos = nx.spring_layout(self.entity_network, seed=42)
        
        # Define node colors by quality
        quality_colors = {
            'good': 'green',
            'uncertain': 'orange',
            'poor': 'red',
            'unknown': 'gray'
        }
        
        # Define node shapes by category
        category_shapes = {
            'CLIMATE_HAZARD': 'o',
            'ECOSYSTEM': 's',
            'INFRASTRUCTURE': '^',
            'ADAPTATION_STRATEGY': 'd',
            'ASSESSMENT_COMPONENT': 'p',
            None: 'o'
        }
        
        # Draw nodes by quality
        for quality, color in quality_colors.items():
            nodes = [n for n, d in self.entity_network.nodes(data=True) if d.get('quality') == quality]
            nx.draw_networkx_nodes(
                self.entity_network, 
                pos, 
                nodelist=nodes,
                node_color=color,
                node_size=300,
                alpha=0.8
            )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.entity_network,
            pos,
            width=1.0,
            alpha=0.5,
            arrows=True
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.entity_network,
            pos,
            font_size=8,
            font_family='sans-serif'
        )
        
        # Add legend for quality
        quality_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=quality) 
                          for quality, color in quality_colors.items()]
        
        plt.legend(handles=quality_patches, title="Entity Quality", loc='upper left')
        
        # Add title
        plt.title(f"Entity Network Evolution (Entities: {len(self.entity_network.nodes)}, Relationships: {len(self.entity_network.edges)})")
        
        # Remove axis
        plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Saved entity network visualization to {output_path}")
    
    def run_test(self, data_dir: str) -> Dict[str, Any]:
        """Run the test with all climate risk documents."""
        self.logger.info(f"Running Context-Aware NER Evolution Test with documents in {data_dir}")
        
        # Process all documents
        results = self.process_all_documents(data_dir)
        
        # Create output directory
        output_dir = Path(__file__).parent / "analysis_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results to JSON
        results_file = output_dir / f"context_aware_ner_evolution_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Visualize entity network
        network_file = output_dir / f"context_aware_ner_network_{timestamp}.png"
        self.visualize_entity_network(str(network_file))
        
        self.logger.info(f"\nContext-Aware NER Evolution Test completed successfully")
        self.logger.info(f"Results saved to {results_file}")
        self.logger.info(f"Entity network visualization saved to {network_file}")
        
        # Log summary statistics
        self.logger.info("\n===== Summary Statistics =====")
        self.logger.info(f"Documents processed: {results['documents_processed']}")
        self.logger.info(f"Total entities: {results['entities']['total']}")
        self.logger.info(f"  Good: {results['entities']['by_quality']['good']} ({results['entities']['by_quality']['good'] / max(1, results['entities']['total']) * 100:.1f}%)")
        self.logger.info(f"  Uncertain: {results['entities']['by_quality']['uncertain']} ({results['entities']['by_quality']['uncertain'] / max(1, results['entities']['total']) * 100:.1f}%)")
        self.logger.info(f"  Poor: {results['entities']['by_quality']['poor']} ({results['entities']['by_quality']['poor'] / max(1, results['entities']['total']) * 100:.1f}%)")
        self.logger.info(f"Total relationships: {results['relationships']['total']}")
        self.logger.info(f"Quality transitions:")
        self.logger.info(f"  Poor → Uncertain: {results['quality_transitions']['poor_to_uncertain']}")
        self.logger.info(f"  Uncertain → Good: {results['quality_transitions']['uncertain_to_good']}")
        self.logger.info(f"  Poor → Good: {results['quality_transitions']['poor_to_good']}")
        self.logger.info(f"Domain relevance improvement: {results['domain_relevance']['initial']:.2f} → {results['domain_relevance']['final']:.2f} ({(results['domain_relevance']['final'] - results['domain_relevance']['initial']) * 100:.1f}%)")
        
        return results

def run_context_aware_ner_evolution_test():
    """Run the context-aware NER evolution test."""
    # Set up logging
    logger, log_file, results_file = setup_logging()
    
    # Create test instance
    test = ContextAwareNEREvolutionTest(logger)
    
    # Define data directory
    data_dir = str(Path(__file__).parent.parent.parent.parent.parent / "data" / "climate_risk")
    
    # Run test
    results = test.run_test(data_dir)
    
    return results

if __name__ == "__main__":
    results = run_context_aware_ner_evolution_test()
    print("Context-Aware NER Evolution Test completed successfully")
