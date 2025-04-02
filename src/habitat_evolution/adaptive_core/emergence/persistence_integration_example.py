"""
Example usage of the Vector-Tonic Persistence Integration.

This module demonstrates how to use the Vector-Tonic Persistence Integration
to connect the vector-tonic-window system with the ArangoDB persistence layer.
"""

import logging
import json
from datetime import datetime

from src.habitat_evolution.adaptive_core.event_bus import LocalEventBus, Event
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import VectorTonicWindowIntegrator
from src.habitat_evolution.adaptive_core.emergence.persistence_integration import VectorTonicPersistenceIntegration
from src.habitat_evolution.adaptive_core.persistence.arangodb.connection import ArangoDBConnectionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_integration():
    """Set up the integration between vector-tonic-window and persistence.
    
    Returns:
        Tuple containing the event bus, vector-tonic integrator, and persistence integration
    """
    # Create event bus
    event_bus = LocalEventBus()
    
    # Create vector-tonic window integrator
    vector_tonic_integrator = VectorTonicWindowIntegrator(event_bus=event_bus)
    
    # Create persistence integration
    persistence_integration = VectorTonicPersistenceIntegration(event_bus=event_bus)
    
    # Initialize both integrators
    vector_tonic_integrator.initialize()
    persistence_integration.initialize()
    
    logger.info("Integration setup complete")
    
    return event_bus, vector_tonic_integrator, persistence_integration

def process_document(persistence_integration, document_data):
    """Process a document through the integration pipeline.
    
    Args:
        persistence_integration: The persistence integration service
        document_data: Document data to process
        
    Returns:
        The document ID
    """
    # Process the document
    document_id = persistence_integration.process_document(document_data)
    
    logger.info(f"Document processed with ID: {document_id}")
    
    return document_id

def publish_pattern_detected(event_bus, pattern_id, pattern_data):
    """Publish a pattern detected event.
    
    Args:
        event_bus: The event bus
        pattern_id: ID of the detected pattern
        pattern_data: Data about the pattern
    """
    # Create event data
    event_data = {
        "pattern_id": pattern_id,
        "pattern_data": pattern_data,
        "confidence": 0.85,
        "timestamp": datetime.now().isoformat()
    }
    
    # Create and publish event
    event = Event("pattern.detected", event_data, source="pattern_detector")
    event_bus.publish(event)
    
    logger.info(f"Published pattern detected event for {pattern_id}")

def publish_relationship_detected(event_bus, source, predicate, target):
    """Publish a relationship detected event.
    
    Args:
        event_bus: The event bus
        source: Source concept
        predicate: Relationship predicate
        target: Target concept
    """
    # Create event data
    event_data = {
        "source": source,
        "predicate": predicate,
        "target": target,
        "confidence": 0.75,
        "observation_count": 1,
        "harmonic_properties": {
            "resonance": 0.65,
            "coherence": 0.7,
            "stability": 0.8
        }
    }
    
    # Create and publish event
    event = Event("relationship.detected", event_data, source="relationship_detector")
    event_bus.publish(event)
    
    logger.info(f"Published relationship detected event: {source} -{predicate}-> {target}")

def publish_field_state_updated(event_bus, field_state_id):
    """Publish a field state updated event.
    
    Args:
        event_bus: The event bus
        field_state_id: ID of the field state
    """
    # Create field state data
    field_state_data = {
        "id": field_state_id,
        "density": 0.65,
        "turbulence": 0.35,
        "coherence": 0.75,
        "stability": 0.8,
        "metrics": {
            "pattern_count": 12,
            "meta_pattern_count": 3,
            "resonance_density": 0.45,
            "interference_complexity": 0.55,
            "flow_coherence": 0.7,
            "stability_trend": 0.65,
            "effective_dimensionality": 4.2,
            "eigenspace_stability": 0.75,
            "pattern_coherence": 0.8,
            "resonance_level": 0.6,
            "system_load": 0.4
        }
    }
    
    # Create event data
    event_data = {
        "field_state": field_state_data,
        "timestamp": datetime.now().isoformat()
    }
    
    # Create and publish event
    event = Event("field.state.updated", event_data, source="field_state_monitor")
    event_bus.publish(event)
    
    logger.info(f"Published field state updated event for {field_state_id}")

def run_example():
    """Run a complete example of the integration."""
    # Set up integration
    event_bus, vector_tonic_integrator, persistence_integration = setup_integration()
    
    # Process a document
    document_data = {
        "id": "climate_risk_report_2025",
        "content": """
        Climate Risk Assessment Report 2025
        
        This report analyzes the impact of climate change on global food security,
        with a focus on vulnerable regions. Rising temperatures and changing precipitation
        patterns are affecting crop yields and water availability. Adaptation strategies
        must be implemented to ensure food security for future generations.
        
        Key findings:
        1. Temperature increases of 1.5°C will reduce crop yields by 5-10% in tropical regions
        2. Water scarcity will affect 40% more agricultural land by 2040
        3. Sustainable farming practices can mitigate 30% of climate-related yield losses
        """
    }
    
    document_id = process_document(persistence_integration, document_data)
    
    # Publish pattern detected events
    publish_pattern_detected(
        event_bus,
        "pattern_16_Climate Change_impacts_Food Security",
        {
            "description": "Climate change negatively impacts food security",
            "confidence": 0.85,
            "supporting_evidence": ["temperature increases", "crop yields", "water scarcity"],
            "entities": ["Climate Change", "Food Security"],
            "relationship": "impacts"
        }
    )
    
    publish_pattern_detected(
        event_bus,
        "pattern_17_Sustainable Farming_mitigates_Climate Impact",
        {
            "description": "Sustainable farming practices mitigate climate change impacts",
            "confidence": 0.8,
            "supporting_evidence": ["sustainable farming", "mitigate", "yield losses"],
            "entities": ["Sustainable Farming", "Climate Impact"],
            "relationship": "mitigates"
        }
    )
    
    # Publish relationship detected events
    publish_relationship_detected(
        event_bus,
        "Climate Change",
        "impacts",
        "Food Security"
    )
    
    publish_relationship_detected(
        event_bus,
        "Sustainable Farming",
        "mitigates",
        "Climate Impact"
    )
    
    publish_relationship_detected(
        event_bus,
        "Food Security",
        "depends_on",
        "Water Availability"
    )
    
    # Publish field state updated event
    publish_field_state_updated(
        event_bus,
        "field_state_2025_04_01"
    )
    
    logger.info("Example completed successfully")

if __name__ == "__main__":
    run_example()
