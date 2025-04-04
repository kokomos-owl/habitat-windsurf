#!/usr/bin/env python
"""
Process climate risk data with PatternAwareRAG and output verbose logging.

This script loads climate risk data, processes it through the PatternAwareRAG system,
and logs the pattern evolution, quality state transitions, and emergence points.
"""
import os
import sys
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import json

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pattern_evolution.log')
    ]
)
logger = logging.getLogger('habitat_evolution')

# Import PatternAwareRAG components
from habitat_evolution.pattern_aware_rag.pattern_aware_rag import (
    PatternAwareRAG,
    RAGPatternContext,
    PatternMetrics,
    LearningWindowState
)
from habitat_evolution.pattern_aware_rag.state.test_states import (
    GraphStateSnapshot,
    ConceptNode,
    PatternState,
    ConceptRelation
)
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

# Mock services implementation
import unittest.mock as mock

class PatternEvolutionTracker:
    """Track pattern evolution through quality states."""
    
    def __init__(self):
        self.emergence_points = []
        self.transitions = []
        self.patterns = {}
        
    def record_emergence_point(self, stage: str, data: Dict[str, Any]):
        """Record an emergence point during processing."""
        self.emergence_points.append({
            'stage': stage,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data': {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                    for k, v in data.items()}
        })
        logger.info(f"Emergence point recorded: {stage}")
        
    def record_transition(self, pattern_id: str, initial_state: str, 
                         context: RAGPatternContext):
        """Record a quality state transition."""
        transition = {
            'pattern_id': pattern_id,
            'initial_state': initial_state,
            'coherence_level': context.coherence_level,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'evolution_metrics': {
                'coherence': context.evolution_metrics.coherence,
                'emergence_rate': context.evolution_metrics.emergence_rate,
                'stability': context.evolution_metrics.stability,
                'cross_pattern_flow': context.evolution_metrics.cross_pattern_flow
            } if context.evolution_metrics else None
        }
        self.transitions.append(transition)
        logger.info(f"Transition recorded for pattern {pattern_id}: {initial_state} -> {self.determine_new_state(transition)}")
        
    def determine_new_state(self, transition: Dict[str, Any]) -> str:
        """Determine the new quality state based on metrics."""
        if not transition.get('evolution_metrics'):
            return transition['initial_state']
            
        metrics = transition['evolution_metrics']
        coherence = metrics.get('coherence', 0)
        stability = metrics.get('stability', 0)
        
        if coherence >= 0.85 and stability >= 0.9:
            return 'stable'
        elif coherence >= 0.7 and stability >= 0.7:
            return 'good'
        elif coherence >= 0.5 and stability >= 0.5:
            return 'emerging'
        else:
            return 'uncertain'
    
    def save_results(self, filename: str):
        """Save tracking results to a file."""
        results = {
            'emergence_points': self.emergence_points,
            'transitions': self.transitions,
            'summary': {
                'total_patterns': len(self.patterns),
                'total_transitions': len(self.transitions),
                'emergence_points': len(self.emergence_points)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filename}")

async def process_document(doc_path: str, rag: PatternAwareRAG, tracker: PatternEvolutionTracker):
    """Process a document through PatternAwareRAG."""
    logger.info(f"Processing document: {doc_path}")
    
    # Read the document
    with open(doc_path, 'r') as f:
        content = f.read()
    
    # Extract sections (simple approach - split by newlines with some filtering)
    sections = [s.strip() for s in content.split('\n\n') if s.strip() and len(s.strip()) > 100]
    logger.info(f"Extracted {len(sections)} sections from document")
    
    # Process each section
    for i, section in enumerate(sections):
        # Create a pattern state for the section
        pattern_id = str(AdaptiveID(base_concept=f"section_{i}", creator_id="climate_risk"))
        initial_state = "uncertain"  # Start all patterns as uncertain
        
        pattern = PatternState(
            id=pattern_id,
            content=section,
            metadata={"quality_state": initial_state, "source": os.path.basename(doc_path)},
            timestamp=datetime.now(timezone.utc),
            confidence=0.3  # Low initial confidence for uncertain state
        )
        
        # Store the pattern
        tracker.patterns[pattern_id] = pattern
        
        # Process the pattern
        logger.info(f"Processing pattern {pattern_id} (section {i+1}/{len(sections)})")
        result = await rag.process_with_patterns(
            query=section,
            context={
                'pattern': pattern,
                'mode': 'document_processing',
                'initial_quality_state': initial_state,
                'document': os.path.basename(doc_path)
            }
        )
        
        response, pattern_context = result
        
        # Record the transition
        tracker.record_transition(pattern_id, initial_state, pattern_context)
        
        # Record emergence point
        tracker.record_emergence_point(f'document_processing_{i}', {
            'pattern_id': pattern_id,
            'content_preview': section[:100] + "...",
            'initial_state': initial_state,
            'context': str(pattern_context)
        })
        
        # Log some details about the pattern processing
        logger.info(f"Pattern {pattern_id} processed:")
        logger.info(f"  - Coherence level: {pattern_context.coherence_level}")
        if pattern_context.evolution_metrics:
            logger.info(f"  - Emergence rate: {pattern_context.evolution_metrics.emergence_rate}")
            logger.info(f"  - Stability: {pattern_context.evolution_metrics.stability}")
        
        # Short pause between processing sections to avoid overwhelming the system
        await asyncio.sleep(0.1)
    
    return {
        'document': os.path.basename(doc_path),
        'sections_processed': len(sections),
        'patterns_created': len(sections)
    }

async def main():
    """Main function to process climate risk data."""
    logger.info("Starting climate risk data processing with PatternAwareRAG")
    
    # Initialize PatternAwareRAG with mock services
    # Create our own mock services
    services = create_mock_services()
    
    # Mock the PatternGraphService
    import unittest.mock as mock
    from datetime import datetime, timezone
    
    # Create a properly initialized GraphStateSnapshot
    graph_snapshot = GraphStateSnapshot(
        id=str(AdaptiveID(base_concept='graph', creator_id='test')),
        nodes={
            'node1': ConceptNode(id='node1', content='test node', confidence=0.8),
            'node2': ConceptNode(id='node2', content='test node 2', confidence=0.7)
        },
        relations={
            'rel1': ConceptRelation(id='rel1', source='node1', target='node2', relation_type='connects_to', confidence=0.8)
        },
        patterns={
            'pattern1': PatternState(id='pattern1', content='test pattern', metadata={}, timestamp=datetime.now(timezone.utc), confidence=0.8)
        },
        timestamp=datetime.now(timezone.utc)
    )
    
    # Mock PatternGraphService
    mock_graph_service = mock.MagicMock()
    mock_graph_service.get_current_state.return_value = graph_snapshot
    
    # Add the mock graph service to our services
    services['graph_service'] = mock_graph_service
    
    # Create RAG instance with mocks
    rag = PatternAwareRAG(**services)
    
    # Set config attribute
    rag.config = services['settings']
    
    # Mock methods
    async def mock_field_state(context):
        field_id = str(AdaptiveID(base_concept='field', creator_id='test'))
        return mock.MagicMock(
            id=field_id,
            position={'x': 0.0, 'y': 0.0},
            state='active',
            metrics={
                'density': 0.8,
                'coherence': 0.9,
                'stability': 0.7
            },
            patterns=[
                {
                    'id': str(AdaptiveID(base_concept='pattern', creator_id='test')),
                    'content': 'test pattern',
                    'confidence': 0.8
                }
            ]
        )
    rag._get_current_field_state = mock_field_state
    
    # Mock pattern extraction
    async def mock_extract_patterns(query):
        return ["pattern_1", "pattern_2"]
    rag._extract_query_patterns = mock_extract_patterns
    
    # Initialize pattern evolution tracker
    tracker = PatternEvolutionTracker()
    
    # Get climate risk documents
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'climate_risk')
    documents = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                if f.endswith('.txt') and not f.startswith('.')]
    
    logger.info(f"Found {len(documents)} climate risk documents to process")
    
    # Process each document
    results = []
    for doc_path in documents:
        result = await process_document(doc_path, rag, tracker)
        results.append(result)
        logger.info(f"Completed processing document: {os.path.basename(doc_path)}")
        logger.info(f"  - Sections processed: {result['sections_processed']}")
        logger.info(f"  - Patterns created: {result['patterns_created']}")
    
    # Save results
    tracker.save_results('climate_risk_pattern_evolution.json')
    
    # Print summary
    logger.info("=== Processing Summary ===")
    logger.info(f"Documents processed: {len(results)}")
    total_sections = sum(r['sections_processed'] for r in results)
    logger.info(f"Total sections processed: {total_sections}")
    logger.info(f"Total patterns created: {sum(r['patterns_created'] for r in results)}")
    logger.info(f"Total transitions recorded: {len(tracker.transitions)}")
    logger.info(f"Total emergence points: {len(tracker.emergence_points)}")
    
    # Quality state distribution
    quality_states = {}
    for transition in tracker.transitions:
        new_state = tracker.determine_new_state(transition)
        quality_states[new_state] = quality_states.get(new_state, 0) + 1
    
    logger.info("Quality state distribution:")
    for state, count in quality_states.items():
        percentage = (count / len(tracker.transitions)) * 100
        logger.info(f"  - {state}: {count} ({percentage:.1f}%)")
    
    logger.info("Processing complete. Results saved to climate_risk_pattern_evolution.json")

def create_mock_services():
    """Create mock services for PatternAwareRAG."""
    class MockService:
        def __init__(self):
            self.pattern_store = {}
            self.relationship_store = {}
            self.event_bus = None
            self.context = mock.MagicMock(
                state_space={
                    "density": 0.7,
                    "coherence": 0.8,
                    "stability": 0.9
                }
            )
        
        def get_flow_state(self):
            return {
                "density": 0.7,
                "coherence": 0.8,
                "stability": 0.9
            }
        
        async def process_with_patterns(self, query, context):
            # Create metrics based on the content length and complexity
            query_length = len(query) if query else 0
            has_climate_terms = any(term in query.lower() for term in 
                                   ['climate', 'risk', 'flood', 'drought', 'sea level'])
            
            # Adjust coherence based on content
            coherence = 0.6 + (min(query_length, 1000) / 2000) + (0.1 if has_climate_terms else 0)
            coherence = min(coherence, 0.95)  # Cap at 0.95
            
            # Determine quality state based on coherence
            if coherence >= 0.85:
                quality_state = "stable"
                confidence = 0.95
            elif coherence >= 0.7:
                quality_state = "good"
                confidence = 0.8
            elif coherence >= 0.5:
                quality_state = "emerging"
                confidence = 0.6
            else:
                quality_state = "uncertain"
                confidence = 0.3
                
            # Create metrics with some randomness for realistic variation
            import random
            variation = lambda: random.uniform(-0.05, 0.05)
            
            metrics = PatternMetrics(
                coherence=coherence + variation(),
                emergence_rate=0.5 + (coherence * 0.3) + variation(),
                cross_pattern_flow=0.6 + variation(),
                energy_state=0.7 + variation(),
                adaptation_rate=0.6 + variation(),
                stability=0.7 + (coherence * 0.2) + variation()
            )
            
            # Create context with quality state information
            pattern_context = RAGPatternContext(
                query_patterns=[query[:100]],  # Truncate for readability
                retrieval_patterns=[f"climate_pattern_{i}" for i in range(3)],
                augmentation_patterns=[f"augmented_pattern_{i}" for i in range(2)],
                coherence_level=coherence,
                evolution_metrics=metrics,
                temporal_context={
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "window_state": "OPEN",
                    "sequence_id": "climate_risk_sequence"
                },
                state_space={
                    "density": 0.7 + variation(),
                    "coherence": coherence,
                    "stability": metrics.stability,
                    "quality_state": quality_state,
                    "confidence": confidence
                }
            )
            
            # Create a mock response
            response = f"Processed content about {query[:50]}... with {quality_state} quality state"
            
            return response, pattern_context
        
        async def get_current_state(self, context=None):
            return {
                'id': str(AdaptiveID(base_concept='field', creator_id='test')),
                'state': 'active',
                'metrics': {
                    'density': 0.8,
                    'coherence': 0.9,
                    'stability': 0.7
                },
                'patterns': [
                    {
                        'id': str(AdaptiveID(base_concept='pattern', creator_id='test')),
                        'content': 'test pattern',
                        'confidence': 0.8
                    }
                ]
            }
        
        async def calculate_local_density(self, field_id, position=None):
            return 0.8
        
        async def calculate_global_density(self):
            return 0.7
        
        async def calculate_coherence(self, field_id):
            return 0.9
        
        async def get_cross_pattern_paths(self, field_id):
            return ["test_path_1", "test_path_2"]
        
        async def calculate_back_pressure(self, field_id: str, position: Dict[str, float] = None):
            return 0.3
        
        async def calculate_flow_stability(self, field_id: str):
            return 0.85
        
        def subscribe(self, event, handler):
            pass
    
    class MockSettings:
        def __init__(self):
            self.VECTOR_STORE_DIR = "/tmp/test_vector_store"
            self.thresholds = {
                "density": 0.5,
                "coherence": 0.6,
                "stability": 0.7,
                "back_pressure": 0.8
            }
            
        def __getitem__(self, key):
            return getattr(self, key)
    
    return {
        'pattern_evolution_service': MockService(),
        'field_state_service': MockService(),
        'gradient_service': MockService(),
        'flow_dynamics_service': MockService(),
        'metrics_service': MockService(),
        'quality_metrics_service': MockService(),
        'event_service': MockService(),
        'coherence_analyzer': MockService(),
        'emergence_flow': MockService(),
        'settings': MockSettings()
    }

if __name__ == "__main__":
    asyncio.run(main())
