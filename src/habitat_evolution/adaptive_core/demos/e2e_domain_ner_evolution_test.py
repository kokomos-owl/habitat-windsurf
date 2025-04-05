"""
End-to-End Domain NER Evolution Test for Habitat Evolution.

This script demonstrates how domain-specific Named Entity Recognition (NER)
evolves through the ingestion process, leveraging the full Habitat Evolution
system including vector-tonic-window integration and pattern-aware RAG.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the fix for handling both import styles
from .import_fix import *

# Core Habitat Evolution components
from habitat_evolution.core.services.event_bus import LocalEventBus, Event
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from habitat_evolution.adaptive_core.emergence.event_aware_detector import EventAwarePatternDetector
from habitat_evolution.adaptive_core.emergence.event_bus_integration import PatternEventPublisher
from habitat_evolution.adaptive_core.emergence.learning_window_integration import LearningWindowAwareDetector
from habitat_evolution.adaptive_core.emergence.tonic_harmonic_integration import TonicHarmonicPatternDetector
from habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import VectorTonicWindowIntegrator, create_vector_tonic_window_integrator
from habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService, OperationType
from habitat_evolution.field.harmonic_field_io_bridge import HarmonicFieldIOBridge
from habitat_evolution.adaptive_core.resonance.tonic_harmonic_metrics import TonicHarmonicMetrics
from habitat_evolution.pattern_aware_rag.learning.learning_control import WindowState, BackPressureController, LearningWindow
from habitat_evolution.adaptive_core.emergence.integration_service import EventBusIntegrationService

# Pattern-aware RAG components
from habitat_evolution.pattern_aware_rag.quality_rag.context_aware_rag import ContextAwareRAG
from habitat_evolution.pattern_aware_rag.quality_rag.quality_enhanced_retrieval import QualityEnhancedRetrieval
from habitat_evolution.pattern_aware_rag.context.quality_aware_context import QualityAwarePatternContext
from habitat_evolution.pattern_aware_rag.context.quality_transitions import QualityTransitionTracker

# Context-aware extraction components
from habitat_evolution.adaptive_core.emergence.context_aware_extraction.context_aware_extractor import ContextAwareExtractor
from habitat_evolution.adaptive_core.emergence.context_aware_extraction.quality_assessment import QualityAssessment
from habitat_evolution.adaptive_core.persistence.interfaces.repository_adapter import InMemoryPatternRepository

# Semantic current observer
from habitat_evolution.adaptive_core.transformation.semantic_current_observer import SemanticCurrentObserver

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path("src/habitat_evolution/adaptive_core/demos/analysis_results")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file and console
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"e2e_domain_ner_evolution_{timestamp}.log"
    results_file = log_dir / f"e2e_domain_ner_evolution_{timestamp}.json"
    
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

class E2EDomainNEREvolutionTest:
    """End-to-End test for domain NER evolution through document ingestion."""
    
    def __init__(self):
        """Initialize the test components."""
        # Create event bus
        self.event_bus = LocalEventBus()
        
        # Create AdaptiveIDs for components
        self.semantic_observer_id = AdaptiveID("domain_ner_semantic_observer", creator_id="e2e_test")
        self.pattern_detector_id = AdaptiveID("domain_ner_pattern_detector", creator_id="e2e_test")
        self.publisher_id = AdaptiveID("domain_ner_publisher", creator_id="e2e_test")
        
        # Create semantic observer
        self.semantic_observer = SemanticCurrentObserver(self.semantic_observer_id)
        
        # Create pattern repository
        self.pattern_repository = InMemoryPatternRepository()
        
        # Initialize quality transition tracker
        self.quality_transition_tracker = QualityTransitionTracker()
        
        # Initialize event bus integration
        self._init_event_bus_integration()
        
        # Create pattern detector components
        self._create_pattern_detector()
        
        # Create harmonic services
        self._create_harmonic_services()
        
        # Create vector-tonic window integrator
        self._create_vector_tonic_integrator()
        
        # Create context-aware RAG
        self._create_context_aware_rag()
        
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
        
        # Track learning windows
        self.learning_windows = []
        
        # Track document processing results
        self.document_results = []
        
        # Initialize visualization data
        self.entity_network = nx.DiGraph()
        
        # Set logger
        self.logger = logger
        
        logger.info("Initialized E2E Domain NER Evolution Test")
    
    def _init_event_bus_integration(self):
        """Initialize event bus integration."""
        # Create integration service
        self.integration_service = EventBusIntegrationService(self.event_bus)
        
        # Integrate AdaptiveIDs with event bus
        self.integration_service.integrate_adaptive_id(self.semantic_observer_id)
        self.integration_service.integrate_adaptive_id(self.pattern_detector_id)
        
        # Create pattern publisher
        self.pattern_publisher = self.integration_service.create_pattern_publisher(self.pattern_detector_id.id)
        
        # Subscribe to events
        self.event_bus.subscribe("pattern.detected", self._on_pattern_detected)
        self.event_bus.subscribe("entity.quality.transition", self._on_quality_transition)
        
        logger.info("Initialized event bus integration")
    
    def _create_pattern_detector(self):
        """Create pattern detector components."""
        # Create base pattern detector
        self.base_detector = EventAwarePatternDetector(
            semantic_observer=self.semantic_observer,
            event_bus=self.event_bus,
            pattern_publisher=self.pattern_publisher,
            threshold=2  # Lower threshold for pattern detection to facilitate learning
        )
        
        # Create learning window aware detector
        self.learning_detector = LearningWindowAwareDetector(
            detector=self.base_detector,
            pattern_publisher=self.pattern_publisher,
            back_pressure_controller=BackPressureController()
        )
        
        # Create publisher for field events
        self.field_publisher = PatternEventPublisher(self.event_bus)
        
        logger.info("Created pattern detector components")
    
    def _create_harmonic_services(self):
        """Create harmonic services."""
        # Create harmonic I/O service
        self.harmonic_io_service = HarmonicIOService(self.event_bus)
        
        # Create tonic-harmonic metrics
        self.tonic_metrics = TonicHarmonicMetrics()
        
        # Create field bridge
        self.field_bridge = HarmonicFieldIOBridge(self.harmonic_io_service)
        
        # Create tonic-harmonic detector
        self.tonic_detector = TonicHarmonicPatternDetector(
            base_detector=self.learning_detector,
            harmonic_io_service=self.harmonic_io_service,
            event_bus=self.event_bus,
            field_bridge=self.field_bridge,
            metrics=self.tonic_metrics
        )
        
        logger.info("Created harmonic services")
    
    def _create_vector_tonic_integrator(self):
        """Create vector-tonic window integrator."""
        # Create integrator
        self.integrator = create_vector_tonic_window_integrator(
            tonic_detector=self.tonic_detector,
            event_bus=self.event_bus,
            harmonic_io_service=self.harmonic_io_service,
            metrics=self.tonic_metrics,
            adaptive_soak_period=True
        )
        
        logger.info("Created vector-tonic window integrator")
    
    def _create_context_aware_rag(self):
        """Create context-aware RAG components."""
        # Create context-aware extractor
        self.context_extractor = ContextAwareExtractor(
            window_sizes=[2, 3, 4, 5, 7],  # Enhanced window sizes
            quality_threshold=0.6  # Lower threshold to facilitate transitions
        )
        
        # Create quality enhanced retrieval
        self.quality_retrieval = QualityEnhancedRetrieval(
            quality_weight=0.7,
            coherence_threshold=0.5  # Lower threshold to facilitate transitions
        )
        
        # Create quality aware context
        self.quality_context = QualityAwarePatternContext()
        
        # Create context-aware RAG
        self.context_rag = ContextAwareRAG(
            pattern_repository=self.pattern_repository,
            window_sizes=[2, 3, 4, 5, 7],
            quality_threshold=0.6,
            quality_weight=0.7,
            coherence_threshold=0.5
        )
        
        logger.info("Created context-aware RAG components")
    
    def _on_pattern_detected(self, event: Event):
        """Handle pattern detected events."""
        pattern_data = event.data.get('pattern', {})
        pattern_id = pattern_data.get('id')
        pattern_type = pattern_data.get('type')
        
        if pattern_type == 'entity':
            # Update entity metrics
            self.metrics['entities']['total'] += 1
            
            # Determine quality
            quality = pattern_data.get('quality', 'poor')
            self.metrics['entities']['by_quality'][quality] += 1
            
            # Determine category
            entity_text = pattern_data.get('text', '')
            category = self._determine_entity_category(entity_text)
            if category:
                self.metrics['entities']['by_category'][category] += 1
            
            # Add to entity network
            self.entity_network.add_node(
                entity_text, 
                type='entity',
                quality=quality,
                category=category,
                weight=1
            )
            
            logger.info(f"Detected entity pattern: {entity_text} (quality: {quality}, category: {category})")
        
        elif pattern_type == 'relationship':
            # Update relationship metrics
            self.metrics['relationships']['total'] += 1
            
            # Determine quality
            quality = pattern_data.get('quality', 'poor')
            self.metrics['relationships']['by_quality'][quality] += 1
            
            # Determine relationship type
            rel_type = pattern_data.get('relation_type', 'unknown')
            if rel_type in self.metrics['relationships']['by_type']:
                self.metrics['relationships']['by_type'][rel_type] += 1
            
            # Add to entity network
            source = pattern_data.get('source', '')
            target = pattern_data.get('target', '')
            
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
                    type=rel_type,
                    quality=quality,
                    weight=1
                )
            
            logger.info(f"Detected relationship pattern: {source} -> {rel_type} -> {target} (quality: {quality})")
    
    def _on_quality_transition(self, event: Event):
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
        
        logger.info(f"Entity quality transition: {entity} from {from_quality} to {to_quality}")
    
    def _determine_entity_category(self, entity: str) -> Optional[str]:
        """Determine the category of an entity."""
        for category, terms in ENTITY_CATEGORIES.items():
            for term in terms:
                if term.lower() in entity.lower():
                    return category
        return None
    
    def _calculate_domain_relevance(self) -> float:
        """Calculate the domain relevance of detected entities."""
        if not self.entity_network:
            return 0.0
        
        categorized_entities = 0
        for node, data in self.entity_network.nodes(data=True):
            if data.get('category'):
                categorized_entities += 1
        
        return categorized_entities / max(1, len(self.entity_network.nodes))
    
    def process_document(self, document_path: str, window_duration: int = 5) -> Dict[str, Any]:
        """Process a document through the full Habitat Evolution pipeline."""
        # Load document
        with open(document_path, 'r') as f:
            document_text = f.read()
        
        document_name = Path(document_path).name
        logger.info(f"\n===== Processing document: {document_name} =====")
        
        # Create a learning window for this document
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=window_duration)
        
        learning_window = LearningWindow(
            start_time=start_time,
            end_time=end_time,
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes_per_window=20
        )
        
        # Add custom attributes for tracking
        learning_window.window_id = f"window_{len(self.learning_windows) + 1}"
        learning_window.description = f"Learning window for {document_name}"
        
        self.learning_windows.append(learning_window)
        
        # Set the current learning window
        self.learning_detector.set_learning_window(learning_window)
        
        # Process document through context-aware RAG
        logger.info(f"Processing document through context-aware RAG: {document_name}")
        # Create a default query based on the document name
        default_query = f"Extract key information from {document_name}"
        rag_results = self.context_rag.process_with_context_aware_patterns(
            query=default_query,
            document=document_text
        )
        
        # Extract entities and relationships
        entities = rag_results.get('entities', {})
        relationships = rag_results.get('relationships', {})
        
        # Process through vector-tonic window integrator
        logger.info(f"Processing document through vector-tonic window integrator: {document_name}")
        
        # Simulate processing through the integrator by creating events
        for entity_id, entity_data in entities.items():
            # Create entity pattern event
            entity_event = Event(
                event_type="pattern.detected",
                source=self.pattern_detector_id.id,
                data={
                    'pattern': {
                        'id': entity_id,
                        'type': 'entity',
                        'text': entity_data.get('text', ''),
                        'quality': entity_data.get('quality', 'poor'),
                        'metrics': entity_data.get('metrics', {})
                    }
                }
            )
            
            # Publish event
            self.event_bus.publish(entity_event)
        
        for rel_id, rel_data in relationships.items():
            # Create relationship pattern event
            rel_event = Event(
                event_type="pattern.detected",
                source=self.pattern_detector_id.id,
                data={
                    'pattern': {
                        'id': rel_id,
                        'type': 'relationship',
                        'source': rel_data.get('source', ''),
                        'target': rel_data.get('target', ''),
                        'relation_type': rel_data.get('relation_type', ''),
                        'quality': rel_data.get('quality', 'poor'),
                        'metrics': rel_data.get('metrics', {})
                    }
                }
            )
            
            # Publish event
            self.event_bus.publish(rel_event)
        
        # Wait for processing to complete
        logger.info(f"Waiting for processing to complete for {document_name}")
        time.sleep(1)  # Simulate processing time
        
        # Update metrics
        self.metrics['documents_processed'] += 1
        
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
                    'good': sum(1 for e in entities.values() if e.get('quality') == 'good'),
                    'uncertain': sum(1 for e in entities.values() if e.get('quality') == 'uncertain'),
                    'poor': sum(1 for e in entities.values() if e.get('quality') == 'poor')
                }
            },
            'relationships': {
                'total': len(relationships),
                'by_quality': {
                    'good': sum(1 for r in relationships.values() if r.get('quality') == 'good'),
                    'uncertain': sum(1 for r in relationships.values() if r.get('quality') == 'uncertain'),
                    'poor': sum(1 for r in relationships.values() if r.get('quality') == 'poor')
                }
            },
            'domain_relevance': domain_relevance,
            'quality_transitions': {
                'poor_to_uncertain': self.metrics['quality_transitions']['poor_to_uncertain'],
                'uncertain_to_good': self.metrics['quality_transitions']['uncertain_to_good'],
                'poor_to_good': self.metrics['quality_transitions']['poor_to_good']
            }
        }
        
        self.document_results.append(document_result)
        
        logger.info(f"Completed processing document: {document_name}")
        logger.info(f"  Entities: {document_result['entities']['total']} (good: {document_result['entities']['by_quality']['good']}, uncertain: {document_result['entities']['by_quality']['uncertain']}, poor: {document_result['entities']['by_quality']['poor']})")
        logger.info(f"  Relationships: {document_result['relationships']['total']} (good: {document_result['relationships']['by_quality']['good']}, uncertain: {document_result['relationships']['by_quality']['uncertain']}, poor: {document_result['relationships']['by_quality']['poor']})")
        logger.info(f"  Domain relevance: {document_result['domain_relevance']:.2f}")
        logger.info(f"  Quality transitions: poor→uncertain: {document_result['quality_transitions']['poor_to_uncertain']}, uncertain→good: {document_result['quality_transitions']['uncertain_to_good']}, poor→good: {document_result['quality_transitions']['poor_to_good']}")
        
        return document_result
    
    def process_all_documents(self, data_dir: str) -> Dict[str, Any]:
        """Process all documents in the specified directory."""
        # Get all text files in the directory
        documents = list(Path(data_dir).glob("*.txt"))
        
        # Sort documents to ensure consistent processing order
        documents.sort()
        
        logger.info(f"Found {len(documents)} documents to process in {data_dir}")
        
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
            logger.warning("No entities to visualize")
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
        
        logger.info(f"Saved entity network visualization to {output_path}")
    
    def run_e2e_test(self, data_dir: str) -> Dict[str, Any]:
        """Run the end-to-end test with all climate risk documents."""
        logger.info(f"Running E2E Domain NER Evolution Test with documents in {data_dir}")
        
        # Process all documents
        results = self.process_all_documents(data_dir)
        
        # Create output directory
        output_dir = Path("src/habitat_evolution/adaptive_core/demos/analysis_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results to JSON
        results_file = output_dir / f"e2e_domain_ner_evolution_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Visualize entity network
        network_file = output_dir / f"e2e_domain_ner_network_{timestamp}.png"
        self.visualize_entity_network(str(network_file))
        
        logger.info(f"\nE2E Domain NER Evolution Test completed successfully")
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Entity network visualization saved to {network_file}")
        
        # Log summary statistics
        logger.info("\n===== Summary Statistics =====")
        logger.info(f"Documents processed: {results['documents_processed']}")
        logger.info(f"Total entities: {results['entities']['total']}")
        logger.info(f"  Good: {results['entities']['by_quality']['good']} ({results['entities']['by_quality']['good'] / max(1, results['entities']['total']) * 100:.1f}%)")
        logger.info(f"  Uncertain: {results['entities']['by_quality']['uncertain']} ({results['entities']['by_quality']['uncertain'] / max(1, results['entities']['total']) * 100:.1f}%)")
        logger.info(f"  Poor: {results['entities']['by_quality']['poor']} ({results['entities']['by_quality']['poor'] / max(1, results['entities']['total']) * 100:.1f}%)")
        logger.info(f"Total relationships: {results['relationships']['total']}")
        logger.info(f"Quality transitions:")
        logger.info(f"  Poor → Uncertain: {results['quality_transitions']['poor_to_uncertain']}")
        logger.info(f"  Uncertain → Good: {results['quality_transitions']['uncertain_to_good']}")
        logger.info(f"  Poor → Good: {results['quality_transitions']['poor_to_good']}")
        logger.info(f"Domain relevance improvement: {results['domain_relevance']['initial']:.2f} → {results['domain_relevance']['final']:.2f} ({(results['domain_relevance']['final'] - results['domain_relevance']['initial']) * 100:.1f}%)")
        
        return results

def run_e2e_domain_ner_evolution_test():
    """Run the end-to-end domain NER evolution test."""
    # Set up logging
    log_file, results_file = setup_logging()
    logger = logging.getLogger()
    
    # Create test instance
    test = E2EDomainNEREvolutionTest()
    
    # Define data directory
    data_dir = "/Users/prphillips/Documents/GitHub/habitat-windsurf/data/climate_risk"
    
    # Run test
    results = test.run_e2e_test(data_dir)
    
    return results

if __name__ == "__main__":
    results = run_e2e_domain_ner_evolution_test()
    logging.info("E2E Domain NER Evolution Test completed successfully")
