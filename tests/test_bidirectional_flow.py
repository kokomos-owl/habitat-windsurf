"""
Test script for bidirectional flow between PatternAwareRAG and vector-tonic system.

This test uses climate risk data from Martha's Vineyard to demonstrate the
bidirectional flow of information, patterns, and state changes.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any, List
from datetime import datetime

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
            patterns.append("extreme precipitation")
            domain_strengths["extreme precipitation"] = 0.85
            
        if "drought" in doc.lower():
            patterns.append("drought")
            domain_strengths["drought"] = 0.78
            
        if "wildfire" in doc.lower():
            patterns.append("wildfire")
            domain_strengths["wildfire"] = 0.72
            
        if "storm" in doc.lower():
            patterns.append("storm risk")
            domain_strengths["storm risk"] = 0.68
            
        if "climate change" in doc.lower():
            patterns.append("climate change")
            domain_strengths["climate change"] = 0.92
            
        if "flood" in doc.lower():
            patterns.append("flood risk")
            domain_strengths["flood risk"] = 0.88
            
        return {
            "patterns": patterns,
            "domain_strengths": domain_strengths
        }
        
    def set_precision_weight(self, weight: float):
        """Set precision weight for retrieval."""
        logger.info(f"Setting precision weight to {weight}")
        self.precision_weight = weight

class MockEmergenceFlow:
    def __init__(self):
        self.exploration_weight = 0.5
        
    def set_exploration_weight(self, weight: float):
        """Set exploration weight."""
        logger.info(f"Setting exploration weight to {weight}")
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
        logger.info(f"Registered window: {window_data['score']:.2f} score, {window_data['potential']:.2f} potential")
        
    def get_density_analysis(self) -> Dict[str, Any]:
        """Get density analysis."""
        return {
            "density_centers": [
                {"domain": "flood risk", "density": 0.85, "alignments": ["extreme precipitation", "storm risk"]},
                {"domain": "climate change", "density": 0.92, "alignments": ["drought", "wildfire", "flood risk"]}
            ],
            "cross_domain_paths": [
                {"source": "flood risk", "target": "climate change", "strength": 0.78},
                {"source": "drought", "target": "wildfire", "strength": 0.82}
            ],
            "global_density": 0.75
        }

# Test function
async def test_bidirectional_flow():
    """Test bidirectional flow between PatternAwareRAG and vector-tonic system."""
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
        content="Martha's Vineyard faces increasing flood risks with the historical 100-year rainfall event becoming 5 times more likely by late-century.",
        metadata={
            "source": "vector_tonic",
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "coherence": 0.88,
                "density": 0.82,
                "stability": 0.75
            }
        }
    )
    
    event_bus.publish(
        "pattern.evolved",
        {
            "pattern_id": pattern_id,
            "pattern": pattern,
            "direction": "retrieval",
            "source": "vector_tonic"
        }
    )
    
    # Wait for events to propagate
    await asyncio.sleep(1)
    
    # Simulate window state change
    logger.info("Simulating vector-tonic window state change...")
    event_bus.publish(
        "window.state.changed",
        {
            "window_id": "climate_risk_window",
            "old_state": "CLOSED",
            "new_state": "OPENING",
            "direction": "retrieval",
            "source": "vector_tonic"
        }
    )
    
    # Wait for events to propagate
    await asyncio.sleep(1)
    
    # Process another query to see adaptation
    logger.info("Processing another query after window state change...")
    query = "How will drought and wildfire danger change on Martha's Vineyard?"
    
    result = await pattern_aware_rag.process_with_patterns(
        query,
        context={"region": "Martha's Vineyard"}
    )
    
    logger.info(f"Second query processing result: {result}")
    
    # Check final state
    logger.info(f"Final window state: {pattern_aware_rag.current_window_state}")
    logger.info(f"Final coherence level: {pattern_aware_rag.window_metrics.coherence}")
    logger.info(f"Final exploration weight: {emergence_flow.exploration_weight}")
    logger.info(f"Final precision weight: {coherence_analyzer.precision_weight}")
    
    logger.info("Test completed successfully!")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_bidirectional_flow())
