"""
Bidirectional flow service for Habitat Evolution.

This module provides the implementation of the BidirectionalFlowInterface,
enabling bidirectional communication between components in the Habitat Evolution system.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

from src.habitat_evolution.infrastructure.interfaces.services.bidirectional_flow_interface import BidirectionalFlowInterface
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.pattern_aware_rag_interface import PatternAwareRAGInterface
from src.habitat_evolution.infrastructure.interfaces.persistence.arangodb_connection_interface import ArangoDBConnectionInterface

logger = logging.getLogger(__name__)


class BidirectionalFlowService(BidirectionalFlowInterface):
    """
    Implementation of the BidirectionalFlowInterface.
    
    This service enables bidirectional communication between components in the
    Habitat Evolution system, facilitating the exchange of patterns, field states,
    and relationships. It uses the event service for communication and integrates
    with the pattern-aware RAG system.
    """
    
    def __init__(
        self,
        event_service: EventServiceInterface,
        pattern_aware_rag_service: PatternAwareRAGInterface,
        arangodb_connection: ArangoDBConnectionInterface
    ):
        """
        Initialize the bidirectional flow service.
        
        Args:
            event_service: The event service to use for communication
            pattern_aware_rag_service: The pattern-aware RAG service to integrate with
        """
        self.event_service = event_service
        self.pattern_aware_rag_service = pattern_aware_rag_service
        self.arangodb_connection = arangodb_connection
        self.pattern_handlers = []
        self.field_state_handlers = []
        self.relationship_handlers = []
        self.running = False
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the bidirectional flow service.
        
        This method sets up the necessary collections and subscriptions for
        bidirectional communication between components.
        
        Args:
            config: Optional configuration dictionary for the service
        """
        logger.info("Initializing BidirectionalFlowService")
        
        # Initialize collections for pattern evolution tracking
        self._initialize_collections()
        
        self.running = True
        logger.info("BidirectionalFlowService initialized")
        
    def shutdown(self) -> None:
        """
        Shutdown the bidirectional flow service.
        
        This method cleans up resources and unsubscribes from events.
        """
        logger.info("Shutting down BidirectionalFlowService")
        
        # Unsubscribe from events
        # TODO: Implement unsubscribe logic if needed
        
        self.running = False
        logger.info("BidirectionalFlowService shut down")
        
        # Initialize collections for pattern evolution tracking
        self._initialize_collections()
        
        # Subscribe to events from the event service
        self._setup_event_subscriptions()
        
        logger.info("Initialized BidirectionalFlowService with ArangoDB integration")
        
    def _setup_event_subscriptions(self) -> None:
        """
        Set up subscriptions to events from the event service.
        """
        self.event_service.subscribe("pattern.detected", self._handle_pattern_event)
        self.event_service.subscribe("field_state.updated", self._handle_field_state_event)
        self.event_service.subscribe("relationship.created", self._handle_relationship_event)
        
    def _handle_pattern_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle a pattern event from the event service.
        
        Args:
            event_data: The pattern event data
        """
        logger.debug(f"Received pattern event: {event_data}")
        for handler in self.pattern_handlers:
            try:
                handler(event_data)
            except Exception as e:
                logger.error(f"Error in pattern handler: {e}")
                
    def _handle_field_state_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle a field state event from the event service.
        
        Args:
            event_data: The field state event data
        """
        logger.debug(f"Received field state event: {event_data}")
        for handler in self.field_state_handlers:
            try:
                handler(event_data)
            except Exception as e:
                logger.error(f"Error in field state handler: {e}")
                
    def _handle_relationship_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle a relationship event from the event service.
        
        Args:
            event_data: The relationship event data
        """
        logger.debug(f"Received relationship event: {event_data}")
        for handler in self.relationship_handlers:
            try:
                handler(event_data)
            except Exception as e:
                logger.error(f"Error in relationship handler: {e}")
    
    def register_pattern_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a handler for pattern events.
        
        Args:
            handler: The handler function to call when a pattern event occurs
        """
        self.pattern_handlers.append(handler)
        logger.debug(f"Registered pattern handler: {handler}")
        
    def register_field_state_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a handler for field state events.
        
        Args:
            handler: The handler function to call when a field state event occurs
        """
        self.field_state_handlers.append(handler)
        logger.debug(f"Registered field state handler: {handler}")
        
    def register_relationship_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a handler for relationship events.
        
        Args:
            handler: The handler function to call when a relationship event occurs
        """
        self.relationship_handlers.append(handler)
        logger.debug(f"Registered relationship handler: {handler}")
        
    def _initialize_collections(self) -> None:
        """
        Initialize collections for pattern evolution tracking.
        """
        try:
            # Create collections if they don't exist
            if not self.arangodb_connection.collection_exists("patterns"):
                self.arangodb_connection.create_collection("patterns")
                logger.info("Created patterns collection")
                
            if not self.arangodb_connection.collection_exists("pattern_transitions"):
                self.arangodb_connection.create_collection("pattern_transitions")
                logger.info("Created pattern_transitions collection")
                
            if not self.arangodb_connection.collection_exists("pattern_relationships"):
                self.arangodb_connection.create_collection("pattern_relationships", edge=True)
                logger.info("Created pattern_relationships collection")
                
            # Create indexes for efficient querying
            try:
                self.arangodb_connection.create_index(
                    collection_name="patterns",
                    index_type="persistent",
                    fields=["id"]
                )
                logger.info("Created index on patterns collection")
            except Exception as e:
                logger.error(f"Error creating index on patterns collection: {e}")
            
            try:
                self.arangodb_connection.create_index(
                    collection_name="pattern_transitions",
                    index_type="persistent",
                    fields=["pattern_id", "timestamp"]
                )
                logger.info("Created index on pattern_transitions collection")
            except Exception as e:
                logger.error(f"Error creating index on pattern_transitions collection: {e}")
        except Exception as e:
            logger.error(f"Error initializing collections: {e}")
    
    def publish_pattern(self, pattern: Dict[str, Any]) -> None:
        """
        Publish a pattern event.
        
        Args:
            pattern: The pattern data to publish
        """
        # Generate a unique ID if not provided
        if "id" not in pattern:
            pattern["id"] = str(uuid.uuid4())
            
        # Add timestamp if not present
        if "timestamp" not in pattern:
            pattern["timestamp"] = datetime.now().isoformat()
        
        # Publish the pattern event
        self.event_service.publish("pattern.detected", pattern)
        logger.debug(f"Published pattern event: {pattern['id']}")
        
        # Store or update the pattern in ArangoDB
        try:
            # Check if the pattern already exists
            existing_pattern = self._get_pattern_from_db(pattern["id"])
            
            if existing_pattern:
                # Update the existing pattern
                self._update_pattern_in_db(pattern)
                
                # Track the pattern transition
                self._track_pattern_transition(
                    pattern["id"],
                    existing_pattern.get("quality_state", "unknown"),
                    pattern.get("quality_state", "unknown")
                )
            else:
                # Store the new pattern
                self._store_pattern_in_db(pattern)
        except Exception as e:
            logger.error(f"Error storing pattern in ArangoDB: {e}")
        
        # Update the pattern in the pattern-aware RAG service
        try:
            self.pattern_aware_rag_service.add_pattern(pattern)
        except Exception as e:
            logger.error(f"Error updating pattern in RAG service: {e}")
        
    def publish_field_state(self, field_state: Dict[str, Any]) -> None:
        """
        Publish a field state event.
        
        Args:
            field_state: The field state data to publish
        """
        self.event_service.publish("field_state.updated", field_state)
        logger.debug(f"Published field state event: {field_state}")
        
        # Update the field state in the pattern-aware RAG service
        try:
            self.pattern_aware_rag_service.update_field_state(field_state)
        except Exception as e:
            logger.error(f"Error updating field state in RAG service: {e}")
        
    def publish_relationship(self, relationship: Dict[str, Any]) -> None:
        """
        Publish a relationship event.
        
        Args:
            relationship: The relationship data to publish
        """
        self.event_service.publish("relationship.created", relationship)
        logger.debug(f"Published relationship event: {relationship}")
        
        # Create the relationship in the pattern-aware RAG service
        try:
            self.pattern_aware_rag_service.create_relationship(
                relationship["source_id"],
                relationship["target_id"],
                relationship["type"],
                relationship.get("properties", {})
            )
        except Exception as e:
            logger.error(f"Error creating relationship in RAG service: {e}")
        
    def start(self) -> None:
        """
        Start the bidirectional flow manager.
        """
        self.running = True
        logger.info("Started BidirectionalFlowService")
        
    def stop(self) -> None:
        """
        Stop the bidirectional flow manager.
        """
        self.running = False
        logger.info("Stopped BidirectionalFlowService")
        
    def is_running(self) -> bool:
        """
        Check if the bidirectional flow manager is running.
        
        Returns:
            True if the bidirectional flow manager is running, False otherwise
        """
        return self.running
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the bidirectional flow manager.
        
        Returns:
            A dictionary containing the status information
        """
        return {
            "running": self.running,
            "pattern_handlers": len(self.pattern_handlers),
            "field_state_handlers": len(self.field_state_handlers),
            "relationship_handlers": len(self.relationship_handlers)
        }
        
    def _get_pattern_from_db(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a pattern from ArangoDB.
        
        Args:
            pattern_id: The ID of the pattern
            
        Returns:
            The pattern document, or None if not found
        """
        try:
            query = """
            FOR p IN patterns
                FILTER p.id == @pattern_id
                RETURN p
            """
            bind_vars = {"pattern_id": pattern_id}
            
            result = list(self.arangodb_connection.execute_aql(query, bind_vars))
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting pattern from ArangoDB: {e}")
            return None
    
    def _store_pattern_in_db(self, pattern: Dict[str, Any]) -> None:
        """
        Store a pattern in ArangoDB.
        
        Args:
            pattern: The pattern to store
        """
        try:
            # Prepare the pattern document
            pattern_doc = pattern.copy()
            pattern_doc["_key"] = pattern["id"]
            
            # Store the pattern in ArangoDB
            self.arangodb_connection.create_document("patterns", pattern_doc)
            logger.debug(f"Stored pattern in ArangoDB: {pattern['id']}")
        except Exception as e:
            logger.error(f"Error storing pattern in ArangoDB: {e}")
    
    def _update_pattern_in_db(self, pattern: Dict[str, Any]) -> None:
        """
        Update a pattern in ArangoDB.
        
        Args:
            pattern: The pattern to update
        """
        try:
            # Prepare the pattern document
            pattern_doc = pattern.copy()
            pattern_doc["_key"] = pattern["id"]
            
            # Update the pattern in ArangoDB
            self.arangodb_connection.update_document("patterns", pattern["id"], pattern_doc)
            logger.debug(f"Updated pattern in ArangoDB: {pattern['id']}")
        except Exception as e:
            logger.error(f"Error updating pattern in ArangoDB: {e}")
    
    def _track_pattern_transition(self, pattern_id: str, old_state: str, new_state: str) -> None:
        """
        Track a pattern quality state transition in ArangoDB.
        
        Args:
            pattern_id: The ID of the pattern
            old_state: The old quality state
            new_state: The new quality state
        """
        try:
            # Skip if there's no state change
            if old_state == new_state:
                return
                
            # Create the transition document
            transition = {
                "_key": str(uuid.uuid4()),
                "pattern_id": pattern_id,
                "old_state": old_state,
                "new_state": new_state,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store the transition in ArangoDB
            self.arangodb_connection.create_document("pattern_transitions", transition)
            logger.debug(f"Tracked pattern transition for {pattern_id}: {old_state} -> {new_state}")
        except Exception as e:
            logger.error(f"Error tracking pattern transition: {e}")
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document through the bidirectional flow.
        
        This method processes a document through the pattern-aware RAG service,
        extracts patterns, and publishes them to the event service. It then
        returns the processed document with additional information.
        
        Args:
            document: The document to process
            
        Returns:
            The processed document with additional information
        """
        # Add document ID if not present
        if "id" not in document:
            document["id"] = str(uuid.uuid4())
        
        # Process the document through the pattern-aware RAG service
        result = self.pattern_aware_rag_service.process_document(document)
        
        # Extract patterns from the result
        patterns = result.get("patterns", [])
        
        # Publish patterns to the event service and store in ArangoDB
        for pattern in patterns:
            self.publish_pattern(pattern)
        
        # Create relationships between patterns
        self._create_pattern_relationships(patterns)
            
        # Update the result with bidirectional flow information
        result["bidirectional_flow"] = {
            "pattern_count": len(patterns),
            "status": self.get_status(),
            "document_id": document["id"]
        }
        
        return result
        
    def _create_pattern_relationships(self, patterns: List[Dict[str, Any]]) -> None:
        """
        Create relationships between patterns in ArangoDB.
        
        Args:
            patterns: The patterns to create relationships between
        """
        try:
            # Create relationships between patterns based on similarity or other criteria
            for i, pattern1 in enumerate(patterns):
                for pattern2 in patterns[i+1:]:
                    # Calculate similarity or relationship strength
                    # For now, we'll use a simple placeholder
                    relationship_strength = 0.5
                    
                    # Create the relationship if strength is above threshold
                    if relationship_strength > 0.3:
                        self._create_pattern_relationship(
                            pattern1["id"],
                            pattern2["id"],
                            "related",
                            {"strength": relationship_strength}
                        )
        except Exception as e:
            logger.error(f"Error creating pattern relationships: {e}")
    
    def _create_pattern_relationship(self, from_id: str, to_id: str, rel_type: str, properties: Dict[str, Any]) -> None:
        """
        Create a relationship between two patterns in ArangoDB.
        
        Args:
            from_id: The ID of the source pattern
            to_id: The ID of the target pattern
            rel_type: The type of relationship
            properties: Additional properties for the relationship
        """
        try:
            # Create the relationship document
            relationship = {
                "_key": f"{from_id}_{to_id}_{rel_type}",
                "_from": f"patterns/{from_id}",
                "_to": f"patterns/{to_id}",
                "type": rel_type,
                "properties": properties,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store the relationship in ArangoDB
            self.arangodb_connection.create_document("pattern_relationships", relationship)
            logger.debug(f"Created pattern relationship: {from_id} -> {to_id} ({rel_type})")
            
            # Create the relationship in the pattern-aware RAG service
            self.pattern_aware_rag_service.create_relationship(from_id, to_id, rel_type, properties)
        except Exception as e:
            logger.error(f"Error creating pattern relationship: {e}")
    
    def query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the pattern-aware RAG service through the bidirectional flow.
        
        This method queries the pattern-aware RAG service, extracts patterns
        from the result, and publishes them to the event service. It then
        returns the query result with additional information.
        
        Args:
            query: The query to process
            context: Optional context for the query
            
        Returns:
            The query result with additional information
        """
        # Generate a query ID if not in context
        if context is None:
            context = {}
        
        if "query_id" not in context:
            context["query_id"] = str(uuid.uuid4())
        
        # Retrieve relevant patterns from ArangoDB
        relevant_patterns = self._get_relevant_patterns_for_query(query)
        
        # Add relevant patterns to context
        if relevant_patterns:
            context["relevant_patterns"] = relevant_patterns
        
        # Query the pattern-aware RAG service
        result = self.pattern_aware_rag_service.query(query, context)
        
        # Extract patterns from the result
        patterns = result.get("patterns", [])
        
        # Publish patterns to the event service and store in ArangoDB
        for pattern in patterns:
            self.publish_pattern(pattern)
        
        # Create relationships between patterns
        self._create_pattern_relationships(patterns)
            
        # Update the result with bidirectional flow information
        result["bidirectional_flow"] = {
            "query_id": context["query_id"],
            "pattern_count": len(patterns),
            "relevant_patterns_count": len(relevant_patterns),
            "status": self.get_status()
        }
        
        return result
