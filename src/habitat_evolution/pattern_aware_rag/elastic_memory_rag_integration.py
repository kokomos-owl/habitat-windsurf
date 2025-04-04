"""
Elastic Semantic Memory Integration with Pattern-Aware RAG.

This module implements the integration between the elastic semantic memory system
and the pattern-aware RAG system, creating a complete RAG↔Evolution↔Persistence loop.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
import networkx as nx
from datetime import datetime
import os
from pathlib import Path

from src.habitat_evolution.pattern_aware_rag.quality_rag.quality_enhanced_retrieval import QualityEnhancedRetrieval, RetrievalResult
from src.habitat_evolution.pattern_aware_rag.context.quality_aware_context import QualityAwarePatternContext
from src.habitat_evolution.adaptive_core.quality.predicate_quality_tracker import PredicateQualityTracker
from src.habitat_evolution.adaptive_core.persistence.semantic_memory_persistence import SemanticMemoryPersistence
from src.habitat_evolution.adaptive_core.models import Pattern, Relationship
from src.habitat_evolution.core.pattern import PatternState

logger = logging.getLogger(__name__)

class ElasticMemoryRAGIntegration:
    """
    Integration between elastic semantic memory and pattern-aware RAG.
    
    This class implements the complete RAG↔Evolution↔Persistence loop,
    enabling bidirectional entity-predicate evolution and quality-enhanced
    retrieval based on persisted semantic memory.
    """
    
    def __init__(self, 
                 predicate_quality_tracker: Optional[PredicateQualityTracker] = None,
                 persistence_layer: Optional[SemanticMemoryPersistence] = None,
                 quality_retrieval: Optional[QualityEnhancedRetrieval] = None,
                 event_bus = None,
                 persistence_base_dir: str = None):
        """
        Initialize the elastic memory RAG integration.
        
        Args:
            predicate_quality_tracker: Optional predicate quality tracker
            persistence_layer: Optional semantic memory persistence layer
            quality_retrieval: Optional quality-enhanced retrieval component
            event_bus: Optional event bus for publishing events
            persistence_base_dir: Base directory for persistence files
        """
        self.event_bus = event_bus
        
        # Initialize predicate quality tracker
        self.predicate_quality_tracker = predicate_quality_tracker or PredicateQualityTracker(event_bus, logger)
        
        # Initialize persistence layer
        persistence_dir = persistence_base_dir or os.path.join(os.getcwd(), 'persistence')
        self.persistence_layer = persistence_layer or SemanticMemoryPersistence(persistence_dir, logger)
        
        # Initialize quality-enhanced retrieval
        self.quality_retrieval = quality_retrieval or QualityEnhancedRetrieval(
            predicate_quality_tracker=self.predicate_quality_tracker,
            persistence_layer=self.persistence_layer,
            event_bus=self.event_bus
        )
        
        # Entity network for tracking relationships
        self.entity_network = nx.DiGraph()
        
        # Track entity and predicate quality states
        self.entity_quality = {}
        self.entity_confidence = {}
        self.entity_transition_history = {}
        
        # Field metrics
        self.field_metrics = {
            'local_density': 0.0,
            'global_density': 0.0,
            'stability': 0.5,
            'coherence': 0.5
        }
        
        logger.info("Initialized Elastic Memory RAG Integration")
    
    def retrieve_with_quality(self, query: str, context: QualityAwarePatternContext, 
                             max_results: int = 10) -> RetrievalResult:
        """
        Retrieve patterns with quality awareness and persistence.
        
        Args:
            query: Query string
            context: Quality-aware pattern context
            max_results: Maximum number of results to return
            
        Returns:
            RetrievalResult with retrieved patterns and persistence info
        """
        # Use quality-enhanced retrieval
        result = self.quality_retrieval.retrieve_with_quality(
            query=query,
            context=context,
            max_results=max_results,
            use_persistence=True
        )
        
        # Update entity network with relationships from result
        self._update_entity_network(result.entity_relationships)
        
        # Update entity quality states
        self._update_entity_quality_from_patterns(result.patterns)
        
        # Update predicate quality based on relationships
        self._update_predicate_quality(result.entity_relationships)
        
        logger.info(f"Retrieved {len(result.patterns)} patterns with elastic memory integration")
        
        return result
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save current state to persistence layer.
        
        Returns:
            Dictionary with persistence information
        """
        persistence_paths = self.persistence_layer.save_complete_state(
            entity_quality=self.entity_quality,
            entity_confidence=self.entity_confidence,
            entity_transition_history=self.entity_transition_history,
            predicate_quality={pred: self.predicate_quality_tracker.get_predicate_quality(pred) 
                              for pred in self.predicate_quality_tracker.predicate_quality},
            predicate_confidence={pred: self.predicate_quality_tracker.get_predicate_confidence(pred) 
                                for pred in self.predicate_quality_tracker.predicate_quality},
            domain_predicate_specialization=self.predicate_quality_tracker.domain_predicate_specialization,
            predicate_transition_history=self.predicate_quality_tracker.quality_transition_history,
            field_metrics=self.field_metrics,
            field_stability=self.field_metrics.get('stability', 0.5),
            field_coherence=self.field_metrics.get('coherence', 0.5),
            field_density={'global': self.field_metrics.get('global_density', 0.0),
                          'local': self.field_metrics.get('local_density', 0.0)},
            entity_network=self.entity_network
        )
        
        logger.info(f"Saved elastic memory state with {len(self.entity_quality)} entities and {self.entity_network.number_of_edges()} relationships")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'persistence_paths': persistence_paths,
            'entity_count': len(self.entity_quality),
            'predicate_count': len(self.predicate_quality_tracker.predicate_quality),
            'relationship_count': self.entity_network.number_of_edges()
        }
    
    def load_state(self) -> Dict[str, Any]:
        """
        Load state from persistence layer.
        
        Returns:
            Dictionary with loaded state information
        """
        # Load latest state
        latest_state = self.persistence_layer.get_latest_state()
        
        # Update entity quality
        entity_quality_data = latest_state.get('entity_quality', {})
        if 'entity_quality' in entity_quality_data:
            self.entity_quality = entity_quality_data['entity_quality']
        
        if 'entity_confidence' in entity_quality_data:
            self.entity_confidence = entity_quality_data['entity_confidence']
        
        if 'quality_transition_history' in entity_quality_data:
            self.entity_transition_history = entity_quality_data['quality_transition_history']
        
        # Load entity network if available
        entity_network = self.persistence_layer.load_entity_network()
        if entity_network and entity_network.number_of_nodes() > 0:
            self.entity_network = entity_network
        
        # Load predicate quality tracker state if available
        predicate_quality_data = latest_state.get('predicate_quality', {})
        if predicate_quality_data:
            # Create a new predicate quality tracker with loaded data
            new_tracker = PredicateQualityTracker(self.event_bus, logger)
            
            if 'predicate_quality' in predicate_quality_data:
                new_tracker.predicate_quality = predicate_quality_data['predicate_quality']
            
            if 'predicate_confidence' in predicate_quality_data:
                new_tracker.predicate_confidence = predicate_quality_data['predicate_confidence']
            
            if 'quality_transition_history' in predicate_quality_data:
                new_tracker.quality_transition_history = predicate_quality_data['quality_transition_history']
            
            # Replace current tracker
            self.predicate_quality_tracker = new_tracker
        
        # Load field metrics if available
        vector_field_data = latest_state.get('vector_field', {})
        if 'field_metrics' in vector_field_data:
            self.field_metrics = vector_field_data['field_metrics']
        
        logger.info(f"Loaded elastic memory state with {len(self.entity_quality)} entities and {self.entity_network.number_of_edges()} relationships")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'loaded_from': latest_state.get('timestamp', 'unknown'),
            'entity_count': len(self.entity_quality),
            'predicate_count': len(self.predicate_quality_tracker.predicate_quality),
            'relationship_count': self.entity_network.number_of_edges()
        }
    
    def transition_entity_quality(self, entity: str, from_quality: str, to_quality: str, 
                                 evidence: Optional[str] = None) -> bool:
        """
        Transition an entity to a new quality state.
        
        Args:
            entity: The entity to transition
            from_quality: The current quality state
            to_quality: The target quality state
            evidence: Optional evidence supporting this transition
            
        Returns:
            True if transition was successful, False otherwise
        """
        # Update entity quality
        self.entity_quality[entity] = to_quality
        
        # Update confidence based on quality
        quality_confidence = {
            'poor': 0.2,
            'uncertain': 0.5,
            'good': 0.8
        }
        self.entity_confidence[entity] = quality_confidence.get(to_quality, 0.5)
        
        # Record transition in history
        transition = {
            'timestamp': datetime.now().isoformat(),
            'from_quality': from_quality,
            'to_quality': to_quality,
            'evidence': evidence
        }
        
        if entity not in self.entity_transition_history:
            self.entity_transition_history[entity] = []
        
        self.entity_transition_history[entity].append(transition)
        
        # Update entity in network
        if entity in self.entity_network:
            self.entity_network.nodes[entity]['quality'] = to_quality
            self.entity_network.nodes[entity]['confidence'] = self.entity_confidence[entity]
        
        # Publish event if event bus is available
        if self.event_bus:
            event_data = {
                'entity': entity,
                'from_quality': from_quality,
                'to_quality': to_quality,
                'confidence': self.entity_confidence[entity],
                'evidence': evidence
            }
            
            event = {
                'event_type': 'entity.quality.transition',
                'source': 'elastic_memory_rag_integration',
                'data': event_data
            }
            
            self.event_bus.publish(event)
        
        logger.info(f"Entity quality transition: {entity} from {from_quality} to {to_quality}")
        
        return True
    
    def transition_predicate_quality(self, predicate: str, to_quality: str,
                                    source_domain: Optional[str] = None,
                                    target_domain: Optional[str] = None,
                                    evidence: Optional[str] = None) -> bool:
        """
        Transition a predicate to a new quality state.
        
        Args:
            predicate: The predicate to transition
            to_quality: The target quality state
            source_domain: Optional source domain for domain-specific transitions
            target_domain: Optional target domain for domain-specific transitions
            evidence: Optional evidence supporting this transition
            
        Returns:
            True if transition was successful, False otherwise
        """
        return self.predicate_quality_tracker.transition_predicate_quality(
            predicate=predicate,
            to_quality=to_quality,
            source_domain=source_domain,
            target_domain=target_domain,
            evidence=evidence
        )
    
    def apply_contextual_reinforcement(self, entities: List[str], 
                                      relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply contextual reinforcement to improve entity and relationship quality.
        
        Args:
            entities: List of entities to reinforce
            relationships: List of relationships to reinforce
            
        Returns:
            Dictionary with reinforcement results
        """
        reinforced_entities = []
        reinforced_predicates = []
        
        # Reinforce entities
        for entity in entities:
            current_quality = self.entity_quality.get(entity, 'uncertain')
            
            if current_quality == 'uncertain':
                # Count relationships involving this entity
                entity_relationships = [
                    rel for rel in relationships 
                    if rel.get('source') == entity or rel.get('target') == entity
                ]
                
                # If entity has multiple relationships, improve its quality
                if len(entity_relationships) >= 2:
                    success = self.transition_entity_quality(
                        entity=entity,
                        from_quality='uncertain',
                        to_quality='good',
                        evidence=f"Contextual reinforcement: entity involved in {len(entity_relationships)} relationships"
                    )
                    
                    if success:
                        reinforced_entities.append(entity)
        
        # Reinforce predicates
        predicate_counts = {}
        for rel in relationships:
            predicate = rel.get('predicate')
            if predicate:
                predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1
        
        for predicate, count in predicate_counts.items():
            current_quality = self.predicate_quality_tracker.get_predicate_quality(predicate)
            
            if current_quality == 'uncertain' and count >= 3:
                # If predicate occurs frequently, improve its quality
                success = self.transition_predicate_quality(
                    predicate=predicate,
                    to_quality='good',
                    evidence=f"Contextual reinforcement: predicate used in {count} relationships"
                )
                
                if success:
                    reinforced_predicates.append(predicate)
        
        logger.info(f"Applied contextual reinforcement: improved {len(reinforced_entities)} entities and {len(reinforced_predicates)} predicates")
        
        return {
            'reinforced_entities': reinforced_entities,
            'reinforced_predicates': reinforced_predicates,
            'entity_count': len(entities),
            'relationship_count': len(relationships)
        }
    
    def _update_entity_network(self, relationships: List[Dict[str, Any]]):
        """
        Update entity network with relationships.
        
        Args:
            relationships: List of relationship dictionaries
        """
        for rel in relationships:
            source = rel.get('source')
            target = rel.get('target')
            predicate = rel.get('predicate')
            quality = rel.get('quality', 'uncertain')
            confidence = rel.get('confidence', 0.5)
            
            if source and target and predicate:
                # Add nodes if they don't exist
                if source not in self.entity_network:
                    self.entity_network.add_node(
                        source, 
                        quality=self.entity_quality.get(source, 'uncertain'),
                        confidence=self.entity_confidence.get(source, 0.5)
                    )
                
                if target not in self.entity_network:
                    self.entity_network.add_node(
                        target, 
                        quality=self.entity_quality.get(target, 'uncertain'),
                        confidence=self.entity_confidence.get(target, 0.5)
                    )
                
                # Add or update edge
                self.entity_network.add_edge(
                    source, target, 
                    predicate=predicate, 
                    quality=quality,
                    confidence=confidence,
                    timestamp=datetime.now().isoformat()
                )
    
    def _update_entity_quality_from_patterns(self, patterns: List[Pattern]):
        """
        Update entity quality states from patterns.
        
        Args:
            patterns: List of patterns
        """
        for pattern in patterns:
            entity = pattern.metadata.get('entity')
            quality = pattern.metadata.get('quality_state', 'uncertain')
            confidence = pattern.metadata.get('confidence', 0.5)
            
            if entity:
                # Update entity quality if it's an improvement
                current_quality = self.entity_quality.get(entity, 'uncertain')
                
                if quality == 'good' and current_quality != 'good':
                    self.transition_entity_quality(
                        entity=entity,
                        from_quality=current_quality,
                        to_quality=quality,
                        evidence=f"Quality improvement from pattern: {pattern.text[:50]}..."
                    )
                else:
                    # Just update the quality without a formal transition
                    self.entity_quality[entity] = quality
                    self.entity_confidence[entity] = confidence
    
    def _update_predicate_quality(self, relationships: List[Dict[str, Any]]):
        """
        Update predicate quality based on relationships.
        
        Args:
            relationships: List of relationship dictionaries
        """
        # Group relationships by predicate
        predicate_groups = {}
        for rel in relationships:
            predicate = rel.get('predicate')
            if predicate:
                if predicate not in predicate_groups:
                    predicate_groups[predicate] = []
                predicate_groups[predicate].append(rel)
        
        # Analyze each predicate group
        for predicate, rels in predicate_groups.items():
            current_quality = self.predicate_quality_tracker.get_predicate_quality(predicate)
            
            # Check for domain-specific patterns
            domain_pairs = set()
            for rel in rels:
                source = rel.get('source')
                target = rel.get('target')
                source_domain = self._get_entity_domain(source)
                target_domain = self._get_entity_domain(target)
                
                if source_domain and target_domain:
                    domain_pairs.add((source_domain, target_domain))
            
            # If predicate consistently connects specific domains, reinforce it
            if len(rels) >= 3 and len(domain_pairs) <= 2:
                # Predicate has consistent domain usage
                if current_quality == 'uncertain':
                    # Improve predicate quality
                    self.predicate_quality_tracker.transition_predicate_quality(
                        predicate=predicate,
                        to_quality='good',
                        source_domain=list(domain_pairs)[0][0] if domain_pairs else None,
                        target_domain=list(domain_pairs)[0][1] if domain_pairs else None,
                        evidence=f"Consistent domain usage across {len(rels)} relationships"
                    )
    
    def _get_entity_domain(self, entity: str) -> Optional[str]:
        """
        Get the domain of an entity based on naming patterns.
        
        Args:
            entity: Entity name
            
        Returns:
            Domain name or None if unknown
        """
        # Simple domain detection based on entity name patterns
        entity_lower = entity.lower()
        
        if any(term in entity_lower for term in ['sea level', 'flood', 'storm', 'erosion', 'precipitation']):
            return 'CLIMATE_HAZARD'
        elif any(term in entity_lower for term in ['marsh', 'wetland', 'beach', 'estuary', 'ecosystem']):
            return 'ECOSYSTEM'
        elif any(term in entity_lower for term in ['culvert', 'stormwater', 'wastewater', 'infrastructure']):
            return 'INFRASTRUCTURE'
        elif any(term in entity_lower for term in ['retreat', 'shoreline', 'adaptation', 'resilience']):
            return 'ADAPTATION_STRATEGY'
        elif any(term in entity_lower for term in ['assessment', 'vulnerability', 'metric', 'evaluation']):
            return 'ASSESSMENT_COMPONENT'
        
        return None


def run_elastic_memory_rag_integration_test():
    """
    Run a test of the elastic memory RAG integration.
    
    Returns:
        Dictionary with test results
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logger.info("Starting Elastic Memory RAG Integration Test")
    
    # Create mock event bus
    class MockEventBus:
        def __init__(self):
            self.subscribers = {}
        
        def subscribe(self, event_type, callback):
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(callback)
        
        def publish(self, event):
            event_type = event.get('event_type')
            if event_type in self.subscribers:
                for callback in self.subscribers[event_type]:
                    callback(event)
    
    event_bus = MockEventBus()
    
    # Create test directory for persistence
    test_dir = os.path.join(os.getcwd(), 'test_persistence')
    os.makedirs(test_dir, exist_ok=True)
    
    # Initialize integration
    integration = ElasticMemoryRAGIntegration(
        event_bus=event_bus,
        persistence_base_dir=test_dir
    )
    
    # Create mock patterns
    patterns = []
    for i in range(10):
        pattern = Pattern(
            id=f"pattern_{i}",
            text=f"Test pattern {i} about climate adaptation",
            metadata={
                'entity': f"entity_{i}",
                'quality_state': 'uncertain',
                'confidence': 0.5,
                'relationships': [
                    {
                        'source': f"entity_{i}",
                        'target': f"entity_{(i+1) % 10}",
                        'predicate': 'affects',
                        'quality': 'uncertain',
                        'confidence': 0.5
                    },
                    {
                        'source': f"entity_{i}",
                        'target': f"entity_{(i+2) % 10}",
                        'predicate': 'part_of',
                        'quality': 'uncertain',
                        'confidence': 0.5
                    }
                ]
            }
        )
        patterns.append(pattern)
    
    # Create mock context
    class MockContext:
        def __init__(self):
            self.coherence_level = 0.7
            self.pattern_state_distribution = {PatternState.ACTIVE: 7, PatternState.EMERGING: 3}
        
        def prioritize_patterns_by_quality(self):
            return patterns
    
    context = MockContext()
    
    # Test retrieval
    logger.info("Testing retrieval with quality")
    result = integration.retrieve_with_quality(
        query="climate adaptation",
        context=context,
        max_results=5
    )
    
    logger.info(f"Retrieved {len(result.patterns)} patterns")
    
    # Test contextual reinforcement
    logger.info("Testing contextual reinforcement")
    entities = [f"entity_{i}" for i in range(10)]
    relationships = []
    for i in range(10):
        relationships.append({
            'source': f"entity_{i}",
            'target': f"entity_{(i+1) % 10}",
            'predicate': 'affects',
            'quality': 'uncertain',
            'confidence': 0.5
        })
        relationships.append({
            'source': f"entity_{i}",
            'target': f"entity_{(i+2) % 10}",
            'predicate': 'part_of',
            'quality': 'uncertain',
            'confidence': 0.5
        })
    
    reinforcement_result = integration.apply_contextual_reinforcement(
        entities=entities,
        relationships=relationships
    )
    
    logger.info(f"Reinforced {len(reinforcement_result['reinforced_entities'])} entities and {len(reinforcement_result['reinforced_predicates'])} predicates")
    
    # Test persistence
    logger.info("Testing state persistence")
    save_result = integration.save_state()
    
    logger.info(f"Saved state with {save_result['entity_count']} entities and {save_result['relationship_count']} relationships")
    
    # Test loading
    logger.info("Testing state loading")
    load_result = integration.load_state()
    
    logger.info(f"Loaded state with {load_result['entity_count']} entities and {load_result['relationship_count']} relationships")
    
    # Return test results
    return {
        'retrieval_result': {
            'pattern_count': len(result.patterns),
            'quality_distribution': result.quality_distribution,
            'confidence': result.confidence
        },
        'reinforcement_result': reinforcement_result,
        'persistence_result': {
            'save': save_result,
            'load': load_result
        },
        'entity_count': len(integration.entity_quality),
        'relationship_count': integration.entity_network.number_of_edges(),
        'predicate_count': len(integration.predicate_quality_tracker.predicate_quality)
    }


if __name__ == "__main__":
    results = run_elastic_memory_rag_integration_test()
    print(f"Test completed with {results['entity_count']} entities and {results['relationship_count']} relationships")
