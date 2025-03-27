"""
Test script for Pattern-Aware RAG integration with dynamic pattern detection.
"""
import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from habitat_evolution.pattern_aware_rag.pattern_aware_rag import (
    PatternAwareRAG, RAGPatternContext, PatternMetrics
)
from habitat_evolution.adaptive_core.emergence.emergent_pattern_detector import EmergentPatternDetector
from habitat_evolution.adaptive_core.emergence.semantic_current_observer import SemanticCurrentObserver
from habitat_evolution.adaptive_core.emergence.resonance_trail_observer import ResonanceTrailObserver
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from habitat_evolution.adaptive_core.emergence.climate_data_loader import ClimateDataLoader

# Mock services for testing
class MockService:
    def __init__(self, name):
        self.name = name
        self.pattern_store = {}
        self.relationship_store = {}
        self.event_bus = None
        
    def __str__(self):
        return f"MockService({self.name})"

def main():
    """Run the integration test."""
    logger.info("Starting Pattern-Aware RAG integration test")
    
    # Create mock services
    pattern_evolution_service = MockService("PatternEvolution")
    field_state_service = MockService("FieldState")
    gradient_service = MockService("Gradient")
    flow_dynamics_service = MockService("FlowDynamics")
    metrics_service = MockService("Metrics")
    quality_metrics_service = MockService("QualityMetrics")
    event_service = MockService("EventManagement")
    coherence_analyzer = MockService("CoherenceAnalyzer")
    emergence_flow = MockService("EmergenceFlow")
    
    # Settings mock
    class Settings:
        VECTOR_STORE_DIR = "/tmp/vector_store"
    
    settings = Settings()
    
    try:
        # Initialize Pattern-Aware RAG
        logger.info("Initializing Pattern-Aware RAG")
        rag = PatternAwareRAG(
            pattern_evolution_service=pattern_evolution_service,
            field_state_service=field_state_service,
            gradient_service=gradient_service,
            flow_dynamics_service=flow_dynamics_service,
            metrics_service=metrics_service,
            quality_metrics_service=quality_metrics_service,
            event_service=event_service,
            coherence_analyzer=coherence_analyzer,
            emergence_flow=emergence_flow,
            settings=settings
        )
        
        logger.info("Pattern-Aware RAG initialized successfully")
        
        # Initialize dynamic pattern detection components
        logger.info("Initializing dynamic pattern detection components")
        adaptive_id = AdaptiveID(entity_id="test_entity")
        semantic_observer = SemanticCurrentObserver(adaptive_id=adaptive_id)
        pattern_detector = EmergentPatternDetector(adaptive_id=adaptive_id)
        resonance_observer = ResonanceTrailObserver(adaptive_id=adaptive_id)
        
        logger.info("Dynamic pattern detection components initialized successfully")
        
        # Load climate data
        logger.info("Loading climate data")
        data_loader = ClimateDataLoader()
        
        # Process a sample query
        logger.info("Processing sample query")
        query = "How will sea level rise affect coastal communities?"
        context = {"domain": "climate_risk", "location": "coastal"}
        
        # This will fail because we haven't integrated the components yet
        # But it will show us what we need to do
        try:
            result = rag.process_with_patterns(query, context)
            logger.info(f"Result: {result}")
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.info("This is expected as we haven't integrated the components yet")
        
        # Get the current evolution state
        try:
            evolution_state = rag.get_evolution_state()
            logger.info(f"Evolution state: {evolution_state}")
        except Exception as e:
            logger.error(f"Error getting evolution state: {e}")
        
        logger.info("Integration test completed")
        
    except Exception as e:
        logger.error(f"Error in integration test: {e}")
        raise

if __name__ == "__main__":
    main()
