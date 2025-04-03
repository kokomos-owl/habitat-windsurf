"""
Context-Aware NER with Vector-Tonic-Window Integration Test.

This script demonstrates the integration between context-aware NER evolution
and the vector-tonic-window system, enabling topological and temporal analysis
of entity relationships as they evolve through document ingestion.
"""

import logging
import sys
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional, Set, Tuple
import uuid
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(src_path))

# Import components
from habitat_evolution.adaptive_core.transformation.semantic_current_observer import SemanticCurrentObserver, SemanticRelationship
from habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import VectorTonicWindowIntegrator, create_vector_tonic_window_integrator
from habitat_evolution.adaptive_core.emergence.tonic_harmonic_integration import TonicHarmonicPatternDetector
from habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService as BaseHarmonicIOService, OperationType

# Create a mock version of HarmonicIOService with the process_operation method
class MockHarmonicIOService(BaseHarmonicIOService):
    """Mock implementation of HarmonicIOService with process_operation method."""
    
    def __init__(self, event_bus):
        """Initialize the mock service."""
        super().__init__(event_bus)
        self.operations = []
    
    def process_operation(self, operation):
        """Process a field operation."""
        self.operations.append(operation)
        
        # Publish a field update event
        event = MockEvent(
            event_type="vector.field.updated",
            source="harmonic_io_service",
            data={
                'operation': operation,
                'stability': 0.7,  # Mock stability value
                'coherence': 0.75  # Mock coherence value
            }
        )
        self.event_bus.publish(event)
        
        return True
from habitat_evolution.field.harmonic_field_io_bridge import HarmonicFieldIOBridge
from habitat_evolution.adaptive_core.resonance.tonic_harmonic_metrics import TonicHarmonicMetrics as BaseTonicHarmonicMetrics

# Create an extended version of TonicHarmonicMetrics with additional methods for our integration
class TonicHarmonicMetrics(BaseTonicHarmonicMetrics):
    """Extended version of TonicHarmonicMetrics with additional methods for integration testing."""
    
    def __init__(self):
        """Initialize the extended metrics."""
        super().__init__()
        self.coherence_scores = [0.5]  # Start with a neutral coherence
        self.stability_scores = [0.5]  # Start with a neutral stability
    
    def get_average_coherence(self):
        """Get the average coherence score."""
        return sum(self.coherence_scores) / max(1, len(self.coherence_scores))
    
    def get_field_stability(self):
        """Get the current field stability score."""
        return self.stability_scores[-1] if self.stability_scores else 0.5
    
    def record_coherence(self, coherence_score):
        """Record a new coherence score."""
        self.coherence_scores.append(coherence_score)
        
    def record_stability(self, stability_score):
        """Record a new stability score."""
        self.stability_scores.append(stability_score)
from habitat_evolution.pattern_aware_rag.learning.learning_control import WindowState, BackPressureController, LearningWindow

# Configure logging
def setup_logging():
    """Set up logging for the test."""
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent / "analysis_results"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file and console
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"context_vector_tonic_integration_{timestamp}.log"
    results_file = log_dir / f"context_vector_tonic_integration_{timestamp}.json"
    
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

# Define domain-specific entity categories
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

# Define relationship types
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

class MockPatternEventPublisher:
    """Mock implementation of PatternEventPublisher."""
    
    def __init__(self, event_bus):
        """Initialize the mock pattern event publisher."""
        self.event_bus = event_bus
    
    def publish_pattern_detected(self, pattern_data):
        """Publish a pattern detected event."""
        event = MockEvent("pattern.detected", "pattern_publisher", pattern_data)
        self.event_bus.publish(event)

class ContextAwareVectorTonicIntegration:
    """Integration between context-aware NER and vector-tonic-window systems."""
    
    def __init__(self, logger):
        """Initialize the integration components."""
        self.logger = logger
        
        # Create mock event bus
        self.event_bus = MockEventBus()
        
        # Create mock adaptive IDs
        self.observer_id = MockAdaptiveID("context_aware_ner_observer")
        self.integrator_id = MockAdaptiveID("vector_tonic_integrator")
        
        # Create semantic observer for context-aware NER
        self.semantic_observer = SemanticCurrentObserver(self.observer_id)
        
        # Create harmonic services for vector-tonic integration
        self.harmonic_io_service = MockHarmonicIOService(self.event_bus)
        self.metrics = TonicHarmonicMetrics()
        self.field_bridge = HarmonicFieldIOBridge(self.harmonic_io_service)
        
        # Create pattern publisher
        self.pattern_publisher = MockPatternEventPublisher(self.event_bus)
        
        # Create vector-tonic window integrator
        self._create_vector_tonic_integrator()
        
        # Initialize metrics tracking
        self.metrics_data = {
            'documents_processed': 0,
            'entities': {
                'total': 0,
                'by_quality': {'good': 0, 'uncertain': 0, 'poor': 0},
                'by_category': {category: 0 for category in ENTITY_CATEGORIES.keys()}
            },
            'relationships': {
                'total': 0,
                'by_quality': {'good': 0, 'uncertain': 0, 'poor': 0},
                'by_type': {rel_type: 0 for rel_type in RELATIONSHIP_TYPES},
                'cross_category': {}
            },
            'quality_transitions': {
                'poor_to_uncertain': 0,
                'uncertain_to_good': 0,
                'poor_to_good': 0
            },
            'vector_tonic_metrics': {
                'window_states': [],
                'coherence_scores': [],
                'pattern_emergence': [],
                'topological_stability': [],
                'temporal_evolution': []
            },
            'domain_relevance': {
                'initial': 0.0,
                'final': 0.0
            }
        }
        
        # Track document processing results
        self.document_results = []
        
        # Initialize entity network
        self.entity_network = nx.DiGraph()
        
        # Initialize learning window with parameters aligned with our quality assessment state machine
        self.learning_window = LearningWindow(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=30),
            stability_threshold=0.65,  # Threshold for considering the window stable
            coherence_threshold=0.70,   # Threshold for considering patterns coherent
            max_changes_per_window=50   # Maximum number of changes allowed per window
        )
        
        # Set initial window state
        self.learning_window.window_state = WindowState.OPENING
        
        # Create back pressure controller
        self.back_pressure_controller = BackPressureController()
        
        # Subscribe to events
        self._setup_event_subscriptions()
        
        self.logger.info("Initialized Context-Aware Vector-Tonic Integration")
    
    def _create_vector_tonic_integrator(self):
        """Create the vector-tonic window integrator."""
        # Create a basic tonic-harmonic detector
        self.tonic_detector = TonicHarmonicPatternDetector(
            base_detector=None,  # We'll use semantic observer instead
            harmonic_io_service=self.harmonic_io_service,
            event_bus=self.event_bus,
            field_bridge=self.field_bridge,
            metrics=self.metrics
        )
        
        # Create the vector-tonic window integrator
        self.integrator = create_vector_tonic_window_integrator(
            tonic_detector=self.tonic_detector,
            event_bus=self.event_bus,
            harmonic_io_service=self.harmonic_io_service,
            metrics=self.metrics,
            adaptive_soak_period=True
        )
        
        self.logger.info("Created vector-tonic window integrator")
    
    def _setup_event_subscriptions(self):
        """Set up event subscriptions."""
        # Subscribe to entity quality transition events
        self.event_bus.subscribe("entity.quality.transition", self._on_quality_transition)
        
        # Subscribe to vector-tonic window events
        self.event_bus.subscribe("window.state.changed", self._on_window_state_changed)
        self.event_bus.subscribe("pattern.detected", self._on_pattern_detected)
        self.event_bus.subscribe("vector.field.updated", self._on_vector_field_updated)
        
        self.logger.info("Set up event subscriptions")
    
    def _on_quality_transition(self, event):
        """Handle entity quality transition events."""
        transition_data = event.data
        entity = transition_data.get('entity', '')
        from_quality = transition_data.get('from_quality', '')
        to_quality = transition_data.get('to_quality', '')
        
        # Update transition metrics
        transition_key = f"{from_quality}_to_{to_quality}"
        if transition_key in self.metrics_data['quality_transitions']:
            self.metrics_data['quality_transitions'][transition_key] += 1
        
        # Update entity in network
        if self.entity_network.has_node(entity):
            self.entity_network.nodes[entity]['quality'] = to_quality
        
        # Notify vector-tonic integrator of quality change
        self._notify_vector_tonic_quality_change(entity, from_quality, to_quality)
        
        self.logger.info(f"Entity quality transition: {entity} from {from_quality} to {to_quality}")
    
    def _on_window_state_changed(self, event):
        """Handle window state change events."""
        window_data = event.data
        state = window_data.get('state', '')
        metrics = window_data.get('metrics', {})
        
        # Update window state metrics
        self.metrics_data['vector_tonic_metrics']['window_states'].append({
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'metrics': metrics
        })
        
        self.logger.info(f"Window state changed to {state} with metrics: {metrics}")
    
    def _on_pattern_detected(self, event):
        """Handle pattern detected events."""
        pattern_data = event.data
        pattern_type = pattern_data.get('type', '')
        pattern_entities = pattern_data.get('entities', [])
        pattern_score = pattern_data.get('score', 0.0)
        
        # Update pattern emergence metrics
        self.metrics_data['vector_tonic_metrics']['pattern_emergence'].append({
            'timestamp': datetime.now().isoformat(),
            'type': pattern_type,
            'entities': pattern_entities,
            'score': pattern_score
        })
        
        self.logger.info(f"Pattern detected: {pattern_type} with score {pattern_score}")
    
    def _on_vector_field_updated(self, event):
        """Handle vector field update events."""
        field_data = event.data
        stability = field_data.get('stability', 0.0)
        coherence = field_data.get('coherence', 0.0)
        
        # Update topological stability metrics
        self.metrics_data['vector_tonic_metrics']['topological_stability'].append({
            'timestamp': datetime.now().isoformat(),
            'stability': stability
        })
        
        # Update coherence scores
        self.metrics_data['vector_tonic_metrics']['coherence_scores'].append({
            'timestamp': datetime.now().isoformat(),
            'coherence': coherence
        })
        
        self.logger.info(f"Vector field updated with stability {stability} and coherence {coherence}")
    
    def _notify_vector_tonic_quality_change(self, entity, from_quality, to_quality):
        """Notify the vector-tonic integrator of entity quality changes."""
        # Create a vector field update based on quality change
        quality_values = {'poor': 0.2, 'uncertain': 0.5, 'good': 0.8}
        from_value = quality_values.get(from_quality, 0.0)
        to_value = quality_values.get(to_quality, 0.0)
        
        # Calculate the quality change vector
        quality_change = to_value - from_value
        
        # Create a field operation
        operation = {
            'type': OperationType.UPDATE,
            'entity': entity,
            'quality_change': quality_change,
            'field_impact': quality_change * 0.5  # Scale the impact
        }
        
        # Send to harmonic IO service
        self.harmonic_io_service.process_operation(operation)
        
        # Update metrics based on quality change
        # Higher quality transitions lead to higher coherence and stability
        if quality_change > 0:
            # Improvement in quality increases coherence and stability
            new_coherence = min(0.95, self.metrics.get_average_coherence() + quality_change * 0.1)
            new_stability = min(0.95, self.metrics.get_field_stability() + quality_change * 0.05)
            
            self.metrics.record_coherence(new_coherence)
            self.metrics.record_stability(new_stability)
        
        self.logger.info(f"Notified vector-tonic integrator of quality change for {entity}")
    
    def _determine_entity_category(self, entity: str) -> Optional[str]:
        """Determine the category of an entity."""
        for category, terms in ENTITY_CATEGORIES.items():
            for term in terms:
                if entity.lower() == term.lower() or entity.lower() in term.lower() or term.lower() in entity.lower():
                    return category
        return None
    
    def _determine_relationship_type(self, source: str, target: str, text: str) -> str:
        """Determine the most likely relationship type between two entities."""
        # Simple heuristic: choose a relationship type based on entity categories
        source_category = self._determine_entity_category(source)
        target_category = self._determine_entity_category(target)
        
        if not source_category or not target_category:
            return 'related_to'  # Default relationship
        
        # Structural relationships for same category
        if source_category == target_category:
            return np.random.choice(['part_of', 'contains', 'component_of', 'adjacent_to'])
        
        # Causal relationships from CLIMATE_HAZARD to others
        if source_category == 'CLIMATE_HAZARD' and target_category in ['ECOSYSTEM', 'INFRASTRUCTURE']:
            return np.random.choice(['affects', 'damages', 'causes'])
        
        # Functional relationships from ADAPTATION_STRATEGY to others
        if source_category == 'ADAPTATION_STRATEGY' and target_category in ['ECOSYSTEM', 'INFRASTRUCTURE']:
            return np.random.choice(['protects_against', 'mitigates', 'prevents'])
        
        # Assessment relationships
        if source_category == 'ASSESSMENT_COMPONENT':
            return np.random.choice(['analyzes', 'evaluates', 'monitors'])
        
        # Default to a random relationship type
        return np.random.choice(RELATIONSHIP_TYPES)
    
    def process_document(self, doc_name: str, doc_text: str) -> Dict[str, Any]:
        """Process a document for entity extraction and relationship detection."""
        self.logger.info(f"Processing document: {doc_name}")
        
        # Update window state to OPENING
        self.learning_window.window_state = WindowState.OPENING
        self._publish_window_state_change()
        
        # Extract entities from document
        entities = self._extract_entities(doc_text)
        
        # Add entities to network
        for entity, data in entities.items():
            if entity not in self.entity_network:
                self.entity_network.add_node(
                    entity,
                    category=data['category'],
                    quality=data['quality'],
                    first_seen=doc_name
                )
        
        # Detect relationships between entities
        relationships = self._detect_relationships(entities, doc_text)
        
        # Update window state to OPEN
        self.learning_window.window_state = WindowState.OPEN
        self._publish_window_state_change()
        
        # Add relationships to network
        for rel in relationships:
            source = rel['source']
            target = rel['target']
            rel_type = rel['type']
            
            # Add edge to network if it doesn't exist
            if not self.entity_network.has_edge(source, target):
                self.entity_network.add_edge(
                    source,
                    target,
                    type=rel_type,
                    quality=rel['quality'],
                    first_seen=doc_name
                )
            
            # Observe relationship in semantic observer
            self.semantic_observer.observe_relationship(
                source=source,
                predicate=rel_type,
                target=target,
                context={
                    'document': doc_name,
                    'quality': rel['quality'],
                    'source_category': self.entity_network.nodes[source]['category'],
                    'target_category': self.entity_network.nodes[target]['category']
                }
            )
        
        # Apply contextual reinforcement
        self._apply_contextual_reinforcement()
        
        # Update metrics
        self.metrics_data['documents_processed'] += 1
        self.metrics_data['entities']['total'] = len(self.entity_network.nodes)
        self.metrics_data['relationships']['total'] = len(self.entity_network.edges)
        
        # Update quality counts
        quality_counts = {'good': 0, 'uncertain': 0, 'poor': 0}
        for node, data in self.entity_network.nodes(data=True):
            quality = data.get('quality', 'uncertain')
            quality_counts[quality] += 1
        self.metrics_data['entities']['by_quality'] = quality_counts
        
        # Update category counts
        category_counts = {category: 0 for category in ENTITY_CATEGORIES.keys()}
        for node, data in self.entity_network.nodes(data=True):
            category = data.get('category')
            if category in category_counts:
                category_counts[category] += 1
        self.metrics_data['entities']['by_category'] = category_counts
        
        # Update relationship type counts
        rel_type_counts = {rel_type: 0 for rel_type in RELATIONSHIP_TYPES}
        for source, target, data in self.entity_network.edges(data=True):
            rel_type = data.get('type', 'related_to')
            if rel_type in rel_type_counts:
                rel_type_counts[rel_type] += 1
        self.metrics_data['relationships']['by_type'] = rel_type_counts
        
        # Update cross-category relationship counts
        cross_category = {}
        for source, target, data in self.entity_network.edges(data=True):
            source_category = self.entity_network.nodes[source].get('category', 'UNKNOWN')
            target_category = self.entity_network.nodes[target].get('category', 'UNKNOWN')
            key = f"{source_category}_to_{target_category}"
            
            if key not in cross_category:
                cross_category[key] = 0
            cross_category[key] += 1
        self.metrics_data['relationships']['cross_category'] = cross_category
        
        # Create document result summary
        result = {
            'document': doc_name,
            'entities': {
                'total': len(entities),
                'by_category': {category: sum(1 for e in entities.values() if e['category'] == category) for category in ENTITY_CATEGORIES.keys()}
            },
            'relationships': {
                'total': len(relationships),
                'by_type': {rel_type: sum(1 for r in relationships if r['type'] == rel_type) for rel_type in RELATIONSHIP_TYPES}
            },
            'vector_tonic_metrics': {
                'window_state': self.learning_window.window_state.value,
                'coherence': self.metrics.get_average_coherence(),
                'stability': self.metrics.get_field_stability()
            }
        }
        
        self.document_results.append(result)
        
        # Update window state to CLOSED
        self.learning_window.window_state = WindowState.CLOSED
        self._publish_window_state_change()
        
        self.logger.info(f"Completed processing document: {doc_name}")
        return result
    
    def _extract_entities(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extract entities from text using a simple keyword-based approach."""
        entities = {}
        
        # Extract entities from each category
        for category, terms in ENTITY_CATEGORIES.items():
            for term in terms:
                if term.lower() in text.lower():
                    # Found a match
                    entity_key = term
                    entities[entity_key] = {
                        'category': category,
                        'quality': 'uncertain',  # Start with uncertain quality
                        'mentions': text.lower().count(term.lower())
                    }
        
        self.logger.info(f"Extracted {len(entities)} entities from text")
        return entities
    
    def _detect_relationships(self, entities: Dict[str, Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """Detect relationships between entities."""
        relationships = []
        
        # Get list of entity names
        entity_names = list(entities.keys())
        
        # Generate relationships between entities
        for i, source in enumerate(entity_names):
            for target in entity_names[i+1:]:
                # Skip self-relationships
                if source == target:
                    continue
                
                # Determine relationship type
                rel_type = self._determine_relationship_type(source, target, text)
                
                # Create relationship
                relationship = {
                    'source': source,
                    'target': target,
                    'type': rel_type,
                    'quality': 'uncertain'  # Start with uncertain quality
                }
                
                relationships.append(relationship)
        
        self.logger.info(f"Detected {len(relationships)} relationships between entities")
        return relationships
    
    def _apply_contextual_reinforcement(self):
        """Apply contextual reinforcement to improve entity and relationship quality."""
        # Count relationships per entity
        relationship_counts = {}
        for source, target in self.entity_network.edges():
            if source not in relationship_counts:
                relationship_counts[source] = 0
            if target not in relationship_counts:
                relationship_counts[target] = 0
            
            relationship_counts[source] += 1
            relationship_counts[target] += 1
        
        # Improve quality based on relationship count
        for entity, count in relationship_counts.items():
            if entity in self.entity_network.nodes:
                current_quality = self.entity_network.nodes[entity].get('quality', 'uncertain')
                
                # Entities with many relationships are more likely to be good quality
                if count >= 5 and current_quality == 'uncertain':
                    self._transition_entity_quality(entity, 'uncertain', 'good')
                elif count >= 2 and current_quality == 'poor':
                    self._transition_entity_quality(entity, 'poor', 'uncertain')
        
        # Improve quality based on connections to good quality entities
        for source, target in self.entity_network.edges():
            source_quality = self.entity_network.nodes[source].get('quality', 'uncertain')
            target_quality = self.entity_network.nodes[target].get('quality', 'uncertain')
            
            # If one entity is good, improve the other if it's uncertain
            if source_quality == 'good' and target_quality == 'uncertain':
                self._transition_entity_quality(target, 'uncertain', 'good')
            elif target_quality == 'good' and source_quality == 'uncertain':
                self._transition_entity_quality(source, 'uncertain', 'good')
        
        # Improve relationship quality based on entity quality
        for source, target, data in self.entity_network.edges(data=True):
            source_quality = self.entity_network.nodes[source].get('quality', 'uncertain')
            target_quality = self.entity_network.nodes[target].get('quality', 'uncertain')
            rel_quality = data.get('quality', 'uncertain')
            
            # If both entities are good, relationship is good
            if source_quality == 'good' and target_quality == 'good' and rel_quality != 'good':
                self.entity_network.edges[source, target]['quality'] = 'good'
                
                # Update relationship metrics
                self.metrics_data['relationships']['by_quality']['uncertain'] -= 1
                self.metrics_data['relationships']['by_quality']['good'] += 1
        
        self.logger.info("Applied contextual reinforcement to improve entity and relationship quality")
    
    def _transition_entity_quality(self, entity: str, from_quality: str, to_quality: str):
        """Transition an entity's quality state and publish an event."""
        if entity in self.entity_network.nodes:
            # Update entity quality
            self.entity_network.nodes[entity]['quality'] = to_quality
            
            # Update quality metrics
            self.metrics_data['entities']['by_quality'][from_quality] -= 1
            self.metrics_data['entities']['by_quality'][to_quality] += 1
            
            # Publish quality transition event
            event = MockEvent(
                event_type="entity.quality.transition",
                source="semantic_observer",
                data={
                    'entity': entity,
                    'from_quality': from_quality,
                    'to_quality': to_quality,
                    'category': self.entity_network.nodes[entity].get('category', 'UNKNOWN')
                }
            )
            self.event_bus.publish(event)
    
    def _publish_window_state_change(self):
        """Publish a window state change event."""
        # Create metrics for the window state
        metrics = {
            'local_density': len(self.entity_network.edges) / max(1, len(self.entity_network.nodes)),
            'global_density': self.metrics_data['entities']['by_quality']['good'] / max(1, self.metrics_data['entities']['total']),
            'coherence': self.metrics.get_average_coherence(),
            'stability': self.metrics.get_field_stability()
        }
        
        # Publish window state change event
        event = MockEvent(
            event_type="window.state.changed",
            source="learning_window",
            data={
                'state': self.learning_window.window_state.value,
                'metrics': metrics
            }
        )
        self.event_bus.publish(event)
    
    def visualize_entity_network(self, output_dir: str = "visualizations"):
        """Visualize the entity network with vector-tonic metrics."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize by quality
        self._visualize_by_quality(output_dir)
        
        # Visualize by category
        self._visualize_by_category(output_dir)
        
        # Visualize quality distribution
        self._visualize_quality_distribution(output_dir)
        
        # Visualize vector-tonic metrics
        self._visualize_vector_tonic_metrics(output_dir)
        
        # Visualize temporal evolution
        self._visualize_temporal_evolution(output_dir)
        
        self.logger.info(f"Generated visualizations in {output_dir}")
    
    def _visualize_by_quality(self, output_dir: str):
        """Visualize the entity network colored by quality."""
        plt.figure(figsize=(12, 10))
        
        # Create a copy of the graph for visualization
        G = self.entity_network.copy()
        
        # Define node colors by quality
        quality_colors = {'good': 'green', 'uncertain': 'orange', 'poor': 'red'}
        node_colors = [quality_colors.get(G.nodes[node].get('quality', 'uncertain'), 'gray') for node in G.nodes()]
        
        # Draw the network
        pos = nx.spring_layout(G, seed=42)  # Consistent layout
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        plt.title(f"Entity Network by Quality (Entities: {len(G.nodes)}, Relationships: {len(G.edges)})")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/entity_network_by_quality.png", dpi=300)
        plt.close()
    
    def _visualize_by_category(self, output_dir: str):
        """Visualize the entity network colored by category."""
        plt.figure(figsize=(12, 10))
        
        # Create a copy of the graph for visualization
        G = self.entity_network.copy()
        
        # Define node colors by category
        category_colors = {
            'CLIMATE_HAZARD': 'tab:blue',
            'ECOSYSTEM': 'tab:green',
            'INFRASTRUCTURE': 'tab:red',
            'ADAPTATION_STRATEGY': 'tab:purple',
            'ASSESSMENT_COMPONENT': 'tab:orange'
        }
        
        node_colors = [category_colors.get(G.nodes[node].get('category', 'UNKNOWN'), 'gray') for node in G.nodes()]
        
        # Draw the network
        pos = nx.spring_layout(G, seed=42)  # Consistent layout
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                    label=category, markersize=10) 
                        for category, color in category_colors.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f"Entity Network by Category (Entities: {len(G.nodes)}, Relationships: {len(G.edges)})")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/entity_network_by_category.png", dpi=300)
        plt.close()
    
    def _visualize_quality_distribution(self, output_dir: str):
        """Visualize the quality distribution of entities by category."""
        plt.figure(figsize=(10, 6))
        
        # Collect quality data by category
        categories = list(ENTITY_CATEGORIES.keys())
        quality_data = {category: {'good': 0, 'uncertain': 0, 'poor': 0} for category in categories}
        
        for node, data in self.entity_network.nodes(data=True):
            category = data.get('category', 'UNKNOWN')
            quality = data.get('quality', 'uncertain')
            
            if category in quality_data and quality in quality_data[category]:
                quality_data[category][quality] += 1
        
        # Create stacked bar chart
        good_counts = [quality_data[category]['good'] for category in categories]
        uncertain_counts = [quality_data[category]['uncertain'] for category in categories]
        poor_counts = [quality_data[category]['poor'] for category in categories]
        
        x = range(len(categories))
        width = 0.35
        
        plt.bar(x, good_counts, width, label='Good', color='green')
        plt.bar(x, uncertain_counts, width, bottom=good_counts, label='Uncertain', color='orange')
        plt.bar(x, poor_counts, width, bottom=[g+u for g, u in zip(good_counts, uncertain_counts)], label='Poor', color='red')
        
        plt.xlabel('Category')
        plt.ylabel('Number of Entities')
        plt.title('Entity Quality Distribution by Category')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/quality_distribution.png", dpi=300)
        plt.close()
    
    def _visualize_vector_tonic_metrics(self, output_dir: str):
        """Visualize the vector-tonic metrics over time."""
        plt.figure(figsize=(12, 8))
        
        # Extract coherence scores and timestamps
        coherence_data = self.metrics_data['vector_tonic_metrics']['coherence_scores']
        stability_data = self.metrics_data['vector_tonic_metrics']['topological_stability']
        
        if coherence_data and stability_data:
            # Convert timestamps to relative time points
            coherence_times = list(range(len(coherence_data)))
            coherence_values = [d['coherence'] for d in coherence_data]
            
            stability_times = list(range(len(stability_data)))
            stability_values = [d['stability'] for d in stability_data]
            
            # Plot metrics
            plt.plot(coherence_times, coherence_values, 'b-', label='Coherence')
            plt.plot(stability_times, stability_values, 'r-', label='Topological Stability')
            
            plt.xlabel('Time Point')
            plt.ylabel('Metric Value')
            plt.title('Vector-Tonic Metrics Over Time')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/vector_tonic_metrics.png", dpi=300)
        else:
            plt.text(0.5, 0.5, 'Insufficient data for visualization', 
                    horizontalalignment='center', verticalalignment='center')
            plt.savefig(f"{output_dir}/vector_tonic_metrics.png", dpi=300)
        
        plt.close()
    
    def _visualize_temporal_evolution(self, output_dir: str):
        """Visualize the temporal evolution of the entity network."""
        plt.figure(figsize=(10, 6))
        
        # Extract document processing results
        doc_names = [result['document'] for result in self.document_results]
        entity_counts = [result['entities']['total'] for result in self.document_results]
        relationship_counts = [result['relationships']['total'] for result in self.document_results]
        
        if doc_names:
            x = range(len(doc_names))
            
            # Plot entity and relationship growth
            plt.plot(x, entity_counts, 'b-o', label='Entities')
            plt.plot(x, relationship_counts, 'r-o', label='Relationships')
            
            plt.xlabel('Document Processing Sequence')
            plt.ylabel('Count')
            plt.title('Temporal Evolution of Entity Network')
            plt.xticks(x, doc_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/temporal_evolution.png", dpi=300)
        else:
            plt.text(0.5, 0.5, 'No documents processed yet', 
                    horizontalalignment='center', verticalalignment='center')
            plt.savefig(f"{output_dir}/temporal_evolution.png", dpi=300)
        
        plt.close()
    
    def save_results(self, output_file: str):
        """Save test results to a JSON file."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics_data,
            'document_results': self.document_results,
            'entity_count': len(self.entity_network.nodes),
            'relationship_count': len(self.entity_network.edges),
            'vector_tonic_metrics': {
                'coherence': self.metrics.get_average_coherence(),
                'stability': self.metrics.get_field_stability(),
                'window_states': self.metrics_data['vector_tonic_metrics']['window_states']
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved results to {output_file}")
        return results

def run_context_aware_vector_tonic_integration_test():
    """Run the context-aware NER with vector-tonic-window integration test."""
    # Set up logging
    logger, log_file, results_file = setup_logging()
    logger.info("Starting Context-Aware NER with Vector-Tonic-Window Integration Test")
    
    # Create test instance
    test = ContextAwareVectorTonicIntegration(logger)
    
    # Define test documents
    test_docs = {
        "basic_test_doc_cape_code.txt": """
        Cape Cod faces increasing threats from sea level rise and coastal erosion.
        Salt marsh complexes provide natural protection against storm surge.
        Culverts and stormwater systems need upgrades to handle extreme precipitation.
        Vulnerability assessment shows high risk to coastal infrastructure.
        Living shorelines and managed retreat are recommended adaptation strategies.
        """,
        
        "climate_risk_marthas_vineyard.txt": """
        Martha's Vineyard is experiencing accelerated coastal erosion due to sea level rise.
        Barrier beaches protect inland infrastructure from storm surge damage.
        Roads and bridges on the island require climate resilience upgrades.
        Stakeholder engagement has identified ecosystem protection as a priority.
        Green infrastructure solutions can mitigate flooding while preserving natural systems.
        """,
        
        "complex_test_doc_boston_harbor_islands.txt": """
        Boston Harbor Islands National Park faces multiple climate hazards including sea level rise,
        extreme precipitation, and storm surge. The islands contain diverse ecosystems including
        salt marsh complexes, coastal dunes, and upland forests. Critical infrastructure such as
        docks, visitor facilities, and historic structures are at risk. A comprehensive vulnerability
        assessment has been completed, analyzing both natural and built environments. Adaptation
        planning recommends a combination of living shorelines, managed retreat from low-lying areas,
        and infrastructure hardening where appropriate. Implementation timeline spans 10 years with
        monitoring protocols to track effectiveness.
        """,
        
        "recovery_test_doc_nantucket_sound.txt": """
        Nantucket Sound's ecosystems are showing signs of recovery following implementation
        of adaptation strategies. Salt marsh restoration projects have improved coastal
        resilience against storm surge and sea level rise. Green infrastructure installations
        have reduced stormwater impacts on marine environments. Ongoing monitoring protocols
        track ecosystem health and adaptation effectiveness. Stakeholder engagement has
        increased community support for nature-based solutions.
        """,
        
        "temporal_analysis_plum_island.txt": """
        Temporal analysis of Plum Island shows accelerating coastal erosion precedes major
        infrastructure damage. Beach nourishment projects temporarily mitigate erosion but
        require frequent renewal. Salt marsh degradation follows increased flooding frequency
        from sea level rise. Adaptation planning must consider these temporal relationships
        when evaluating solutions. Funding mechanisms need to account for long-term maintenance
        requirements of adaptation measures.
        """,
        
        "vineyard_sound_structure_meaning_test_doc.txt": """
        Vineyard Sound's structural complexity contains multiple interconnected systems.
        Barrier beaches are part of larger coastal defense complexes that protect inland areas.
        Culverts connect freshwater wetlands to estuaries, maintaining ecological balance.
        Wastewater treatment facilities affect water quality which impacts ecosystem health.
        Resilience metrics evaluate how these structural relationships respond to climate stressors.
        """
    }
    
    # Process each document
    for doc_name, doc_text in test_docs.items():
        test.process_document(doc_name, doc_text)
    
    # Generate visualizations
    test.visualize_entity_network()
    
    # Save results
    results = test.save_results(results_file)
    
    logger.info("Context-Aware NER with Vector-Tonic-Window Integration Test completed successfully")
    logger.info(f"Processed {len(test_docs)} documents")
    logger.info(f"Extracted {results['entity_count']} entities with {results['relationship_count']} relationships")
    logger.info(f"Results saved to {results_file}")
    
    return results

if __name__ == "__main__":
    results = run_context_aware_vector_tonic_integration_test()
    print("Context-Aware NER with Vector-Tonic-Window Integration Test completed successfully")
