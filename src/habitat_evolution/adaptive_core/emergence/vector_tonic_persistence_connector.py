"""
Vector-Tonic Persistence Connector.

This module connects the vector-tonic-window system's events with the persistence layer,
ensuring that patterns, field states, and relationships are properly captured and persisted
in ArangoDB. It serves as the bridge between the pattern detection system and the
persistence infrastructure.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.habitat_evolution.core.services.event_bus import Event, LocalEventBus
from src.habitat_evolution.adaptive_core.emergence.persistence_integration import (
    VectorTonicPersistenceIntegration,
    PatternPersistenceService,
    FieldStatePersistenceService,
    RelationshipPersistenceService
)
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import VectorTonicWindowIntegrator
from src.habitat_evolution.adaptive_core.persistence.arangodb.connection import ArangoDBConnectionManager

logger = logging.getLogger(__name__)

class VectorTonicPersistenceConnector:
    """
    Connector between the vector-tonic-window system and the persistence layer.
    
    This class ensures that events from the vector-tonic-window system are properly
    captured and persisted in ArangoDB. It handles the bidirectional flow of information
    between the pattern detection system and the persistence infrastructure.
    """
    
    def __init__(self, event_bus=None, db=None):
        """Initialize the connector.
        
        Args:
            event_bus: Optional event bus. If not provided, a new event bus will be created.
            db: Optional database connection. If not provided, a new connection will be created.
        """
        self.event_bus = event_bus or LocalEventBus()
        self.db = db or ArangoDBConnectionManager().get_db()
        
        # Create the persistence integration
        self.persistence_integration = VectorTonicPersistenceIntegration(self.event_bus, self.db)
        
        # Create direct references to persistence services for convenience
        self.pattern_service = self.persistence_integration.pattern_service
        self.field_state_service = self.persistence_integration.field_state_service
        self.relationship_service = self.persistence_integration.relationship_service
        
        # Track initialization state
        self.initialized = False
    
    def initialize(self):
        """Initialize the connector and all persistence services."""
        if self.initialized:
            logger.warning("VectorTonicPersistenceConnector already initialized")
            return
            
        # Initialize persistence integration
        self.persistence_integration.initialize()
        
        # Subscribe to additional events for bidirectional flow
        self.event_bus.subscribe("document.processed", self._on_document_processed)
        self.event_bus.subscribe("vector.gradient.updated", self._on_vector_gradient_updated)
        self.event_bus.subscribe("learning.window.closed", self._on_learning_window_closed)
        
        self.initialized = True
        logger.info("VectorTonicPersistenceConnector initialized")
    
    def connect_to_integrator(self, integrator: VectorTonicWindowIntegrator):
        """Connect to a VectorTonicWindowIntegrator.
        
        This ensures that events from the integrator are properly captured and persisted.
        
        Args:
            integrator: The VectorTonicWindowIntegrator to connect to
        """
        if not self.initialized:
            self.initialize()
            
        # Ensure the integrator is initialized
        if not integrator.initialized:
            integrator.initialize()
            
        logger.info(f"Connected to VectorTonicWindowIntegrator {id(integrator)}")
        
        # Publish a connection event
        self.event_bus.publish(Event(
            "persistence.connected",
            {
                "integrator_id": id(integrator),
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_connector"
        ))
    
    def process_document(self, document: Dict[str, Any]) -> str:
        """Process a document through the persistence integration.
        
        Args:
            document: The document to process
            
        Returns:
            The document ID
        """
        if not self.initialized:
            self.initialize()
            
        return self.persistence_integration.process_document(document)
    
    def _on_document_processed(self, event: Event):
        """Handle document processed events.
        
        Args:
            event: The document processed event
        """
        document_data = event.data
        if not document_data or "document_id" not in document_data:
            logger.warning("Invalid document processed data in event")
            return
            
        document_id = document_data["document_id"]
        logger.info(f"Document processed: {document_id}")
        
        # Extract entities and relationships if available
        if "entities" in document_data:
            for entity in document_data["entities"]:
                # Publish entity detected event
                self.event_bus.publish(Event(
                    "entity.detected",
                    {
                        "entity_id": entity.get("id"),
                        "entity_type": entity.get("type"),
                        "entity_text": entity.get("text"),
                        "document_id": document_id,
                        "confidence": entity.get("confidence", 0.5),
                        "timestamp": datetime.now().isoformat()
                    },
                    source="document_processor"
                ))
                
        if "relationships" in document_data:
            for relationship in document_data["relationships"]:
                # Publish relationship detected event
                self.event_bus.publish(Event(
                    "relationship.detected",
                    {
                        "source": relationship.get("source"),
                        "predicate": relationship.get("predicate"),
                        "target": relationship.get("target"),
                        "document_id": document_id,
                        "confidence": relationship.get("confidence", 0.5),
                        "timestamp": datetime.now().isoformat()
                    },
                    source="document_processor"
                ))
    
    def _on_vector_gradient_updated(self, event: Event):
        """Handle vector gradient updated events.
        
        Args:
            event: The vector gradient updated event
        """
        gradient_data = event.data.get("gradient", {})
        if not gradient_data:
            logger.warning("No gradient data found in event")
            return
            
        logger.info("Vector gradient updated")
        
        # Check if this gradient update should trigger a field state update
        if "field_state_id" in gradient_data:
            field_state_id = gradient_data["field_state_id"]
            
            # Get current field state metrics
            metrics = gradient_data.get("metrics", {})
            
            # Publish field state updated event if metrics are available
            if metrics:
                self.event_bus.publish(Event(
                    "field.state.updated",
                    {
                        "field_state": {
                            "id": field_state_id,
                            "metrics": metrics
                        },
                        "timestamp": datetime.now().isoformat()
                    },
                    source="gradient_monitor"
                ))
    
    def _on_learning_window_closed(self, event: Event):
        """Handle learning window closed events.
        
        Args:
            event: The learning window closed event
        """
        window_data = event.data
        if not window_data or "window_id" not in window_data:
            logger.warning("Invalid learning window data in event")
            return
            
        window_id = window_data["window_id"]
        logger.info(f"Learning window closed: {window_id}")
        
        # Extract patterns detected during this learning window
        patterns = window_data.get("patterns", [])
        
        for pattern in patterns:
            # Publish pattern detected event for each pattern
            self.event_bus.publish(Event(
                "pattern.detected",
                {
                    "pattern_id": pattern.get("id"),
                    "pattern_data": pattern,
                    "confidence": pattern.get("confidence", 0.5),
                    "window_id": window_id,
                    "timestamp": datetime.now().isoformat()
                },
                source="learning_window"
            ))
            
        # Extract field state at window closure
        if "field_state" in window_data:
            field_state = window_data["field_state"]
            
            # Publish field state updated event
            self.event_bus.publish(Event(
                "field.state.updated",
                {
                    "field_state": field_state,
                    "window_id": window_id,
                    "timestamp": datetime.now().isoformat()
                },
                source="learning_window"
            ))


def create_connector(event_bus=None, db=None) -> VectorTonicPersistenceConnector:
    """Create and initialize a VectorTonicPersistenceConnector.
    
    Args:
        event_bus: Optional event bus. If not provided, a new event bus will be created.
        db: Optional database connection. If not provided, a new connection will be created.
        
    Returns:
        An initialized VectorTonicPersistenceConnector
    """
    connector = VectorTonicPersistenceConnector(event_bus, db)
    connector.initialize()
    return connector
