"""
Test for bidirectional pattern processing with quality state transitions.

This test demonstrates the complete bidirectional flow between PatternAwareRAG
and the vector-tonic system using real climate risk data from Martha's Vineyard.
It tracks pattern quality state transitions (poor → uncertain → good → stable)
as patterns receive contextual reinforcement.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from habitat_evolution.pattern_aware_rag.pattern_aware_rag import (
    PatternAwareRAG, RAGPatternContext, LearningWindowState, WindowMetrics
)
from habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector import (
    VectorTonicPersistenceConnector
)
from habitat_evolution.services.event_management_service import EventManagementService
from habitat_evolution.services.pattern_evolution_service import PatternEvolutionService
from habitat_evolution.services.field_state_service import FieldStateService
from habitat_evolution.services.gradient_service import GradientService
from habitat_evolution.services.flow_dynamics_service import FlowDynamicsService
from habitat_evolution.services.metrics_service import MetricsService
from habitat_evolution.services.quality_metrics_service import QualityMetricsService
from habitat_evolution.core.models.pattern import Pattern
from habitat_evolution.field.persistence.semantic_potential_calculator import SemanticPotentialCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mock classes for testing
class MockCoherenceAnalyzer:
    def __init__(self):
        self.precision_weight = 0.5
        
    async def extract_patterns(self, doc: str) -> Dict[str, Any]:
        """Extract patterns from document."""
        # Simple pattern extraction based on keywords
        patterns = []
        domain_strengths = {}
        
        if "extreme precipitation" in doc.lower():
            patterns.append({
                "id": f"pattern_{uuid.uuid4().hex[:8]}",
                "text": "Extreme precipitation events are increasing in frequency and intensity.",
                "confidence": 0.6,
                "quality": "poor"  # Initial quality state
            })
        
        if "sea level rise" in doc.lower():
            patterns.append({
                "id": f"pattern_{uuid.uuid4().hex[:8]}",
                "text": "Sea level rise threatens coastal communities with increased flooding.",
                "confidence": 0.65,
                "quality": "poor"  # Initial quality state
            })
        
        if "drought" in doc.lower():
            patterns.append({
                "id": f"pattern_{uuid.uuid4().hex[:8]}",
                "text": "Drought conditions are expected to worsen in many regions.",
                "confidence": 0.55,
                "quality": "poor"  # Initial quality state
            })
        
        return {
            "patterns": patterns,
            "domain_strengths": {
                "climate": 0.8,
                "environment": 0.7,
                "infrastructure": 0.5
            }
        }
        
    def set_precision_weight(self, weight: float):
        """Set precision weight for retrieval."""
        self.precision_weight = weight


class MockEmergenceFlow:
    def __init__(self):
        self.exploration_weight = 0.5
        
    def set_exploration_weight(self, weight: float):
        """Set exploration weight."""
        self.exploration_weight = weight


class MockEvolutionManager:
    def __init__(self):
        self.learning_window_interface = MockLearningWindowInterface()


class MockLearningWindowInterface:
    def __init__(self):
        self.windows = []
        
    def register_window(self, window_data: Dict[str, Any]):
        """Register a learning window."""
        self.windows.append(window_data)
        
    def get_density_analysis(self):
        """Get density analysis."""
        return {
            "local_density": 0.75,
            "global_density": 0.65,
            "density_centers": [
                {"x": 0.2, "y": 0.3, "strength": 0.8},
                {"x": 0.7, "y": 0.6, "strength": 0.6}
            ]
        }
    
# Test function
async def test_bidirectional_pattern_processor():
    """Test bidirectional pattern processing with quality state transitions."""
    logger.info("Initializing test environment...")
    
    # Create shared event bus
    event_bus = EventManagementService()
    
    # Create services
    pattern_evolution_service = PatternEvolutionService(event_bus=event_bus)
    field_state_service = FieldStateService(event_bus=event_bus)
    gradient_service = GradientService()
    flow_dynamics_service = FlowDynamicsService()
    metrics_service = MetricsService()
    quality_metrics_service = QualityMetricsService()
    
    # Create semantic potential calculator
    semantic_potential_calculator = SemanticPotentialCalculator()
    
    # Create mock components
    coherence_analyzer = MockCoherenceAnalyzer()
    emergence_flow = MockEmergenceFlow()
    
    # Create settings
    settings = {
        "VECTOR_STORE_DIR": "./vector_store",
        "thresholds": {
            "density": 0.5,
            "coherence": 0.6,
            "back_pressure": 0.7
        }
    }
    
    # Create PatternAwareRAG
    pattern_aware_rag = PatternAwareRAG(
        pattern_evolution_service=pattern_evolution_service,
        field_state_service=field_state_service,
        gradient_service=gradient_service,
        flow_dynamics_service=flow_dynamics_service,
        metrics_service=metrics_service,
        quality_metrics_service=quality_metrics_service,
        event_service=event_bus,
        coherence_analyzer=coherence_analyzer,
        emergence_flow=emergence_flow,
        settings=settings
    )
    
    # Initialize evolution manager (normally done internally)
    pattern_aware_rag.evolution_manager = MockEvolutionManager()
    
    # Create VectorTonicPersistenceConnector
    vector_tonic_connector = VectorTonicPersistenceConnector(
        event_bus=event_bus,
        pattern_store=pattern_evolution_service.pattern_store,
        relationship_store=pattern_evolution_service.relationship_store,
        field_state_service=field_state_service
    )
    
    # Initialize connector
    vector_tonic_connector.initialize()
    
    # Track patterns and quality transitions
    tracked_patterns = {}
    quality_transitions = {}
    
    # Subscribe to pattern events
    def on_pattern_detected(event):
        pattern_data = event.data.get("pattern", {})
        pattern_id = pattern_data.get("id")
        if pattern_id:
            tracked_patterns[pattern_id] = pattern_data
            logger.info(f"Pattern detected: {pattern_id} with quality {pattern_data.get('quality', 'unknown')}")
            if pattern_id not in quality_transitions:
                quality_transitions[pattern_id] = []
    
    def on_pattern_quality_changed(event):
        pattern_id = event.data.get("pattern_id")
        old_quality = event.data.get("old_quality")
        new_quality = event.data.get("new_quality")
        if pattern_id and pattern_id in tracked_patterns:
            quality_transitions[pattern_id].append({
                "timestamp": datetime.now(),
                "old_quality": old_quality,
                "new_quality": new_quality
            })
            logger.info(f"Pattern quality changed: {pattern_id} from {old_quality} to {new_quality}")
    
    event_bus.subscribe("pattern.detected", on_pattern_detected)
    event_bus.subscribe("pattern.quality.changed", on_pattern_quality_changed)
    
    # Load climate risk document
    logger.info("Loading climate risk document...")
    climate_risk_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "data",
        "climate_risk",
        "climate_risk_marthas_vineyard.txt"
    )
    
    with open(climate_risk_path, "r") as f:
        climate_risk_doc = f.read()
    
    # Process document with PatternAwareRAG
    logger.info("Processing document with PatternAwareRAG...")
    context = RAGPatternContext(
        query_patterns=[],
        retrieval_patterns=[],
        augmentation_patterns=[],
        coherence_level=0.0
    )
    
    result = await pattern_aware_rag.process_document(climate_risk_doc, context)
    logger.info(f"Document processing result: {result}")
    
    # Wait for events to propagate
    await asyncio.sleep(1)
    
    # Simulate vector-tonic system sending field state update
    logger.info("Simulating vector-tonic field state update...")
    event_bus.publish(
        "field.state.updated",
        {
            "field_id": "climate_risk_field",
            "state": {
                "coherence": 0.85,
                "density": 0.78,
                "stability": 0.72,
                "cross_paths": ["climate_adaptation", "economic_impact"],
                "back_pressure": 0.45
            },
            "direction": "retrieval",
            "source": "vector_tonic"
        }
    )
    
    # Wait for events to propagate
    await asyncio.sleep(1)
    
    # Process a query
    logger.info("Processing query with updated field state...")
    query = "What are the projected impacts of extreme precipitation on Martha's Vineyard?"
    
    result = await pattern_aware_rag.process_with_patterns(
        query,
        context={"region": "Martha's Vineyard"}
    )
    
    logger.info(f"Query processing result: {result}")
    
    # Simulate vector-tonic system sending pattern evolution event
    logger.info("Simulating vector-tonic pattern evolution event...")
    pattern_id = "pattern_123"
    pattern = Pattern(
        id=pattern_id,
        base_concept="extreme_precipitation",
        creator_id="vector_tonic",
        weight=1.0,
        confidence=0.88,
        uncertainty=0.12,
        coherence=0.85,
        phase_stability=0.75,
        signal_strength=0.82,
        properties={
            "source": "vector_tonic",
            "timestamp": datetime.now().isoformat(),
            "quality": "good"
        }
    )
    
    event_bus.publish(
        "pattern.evolved",
        {
            "pattern_id": pattern_id,
            "pattern": pattern,
            "old_quality": "uncertain",
            "new_quality": "good",
            "direction": "retrieval",
            "source": "vector_tonic"
        }
    )
    
    # Wait for events to propagate
    await asyncio.sleep(1)
    
    # Process another query to see adaptation
    logger.info("Processing another query after pattern evolution...")
    query = "How will drought and wildfire danger change on Martha's Vineyard?"
    
    result = await pattern_aware_rag.process_with_patterns(
        query,
        context={"region": "Martha's Vineyard"}
    )
    
    logger.info(f"Second query processing result: {result}")
    
    # Check for quality transitions
    logger.info("Checking for quality transitions...")
    for pattern_id, transitions in quality_transitions.items():
        if transitions:
            logger.info(f"Pattern {pattern_id} quality transitions:")
            for transition in transitions:
                logger.info(f"  - {transition['old_quality']} → {transition['new_quality']}")
    
    # Return results for verification
    return {
        "tracked_patterns": tracked_patterns,
        "quality_transitions": quality_transitions
    }

# Run the test
if __name__ == "__main__":
    asyncio.run(test_bidirectional_pattern_processor())
