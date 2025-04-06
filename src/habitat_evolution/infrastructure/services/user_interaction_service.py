"""
User interaction service for Habitat Evolution.

This module provides the implementation of the UserInteractionInterface,
enabling user-driven interactions that drive the bidirectional flow of patterns
and insights in the Habitat Evolution system.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.habitat_evolution.infrastructure.interfaces.services.user_interaction_interface import UserInteractionInterface
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.pattern_aware_rag_interface import PatternAwareRAGInterface
from src.habitat_evolution.infrastructure.interfaces.services.bidirectional_flow_interface import BidirectionalFlowInterface
from src.habitat_evolution.infrastructure.interfaces.persistence.arangodb_connection_interface import ArangoDBConnectionInterface

logger = logging.getLogger(__name__)


class UserInteractionService(UserInteractionInterface):
    """
    Implementation of the UserInteractionInterface.
    
    This service enables user-driven interactions that drive the bidirectional flow
    of patterns and insights in the Habitat Evolution system. It integrates with
    the pattern-aware RAG system, the bidirectional flow service, and ArangoDB
    to create a complete functional loop.
    """
    
    def __init__(
        self,
        event_service: EventServiceInterface,
        pattern_aware_rag_service: PatternAwareRAGInterface,
        bidirectional_flow_service: BidirectionalFlowInterface,
        arangodb_connection: ArangoDBConnectionInterface
    ):
        """
        Initialize the user interaction service.
        
        Args:
            event_service: The event service to use for communication
            pattern_aware_rag_service: The pattern-aware RAG service to integrate with
            bidirectional_flow_service: The bidirectional flow service to drive pattern evolution
            arangodb_connection: The ArangoDB connection for pattern persistence
        """
        self.event_service = event_service
        self.pattern_aware_rag_service = pattern_aware_rag_service
        self.bidirectional_flow_service = bidirectional_flow_service
        self.arangodb_connection = arangodb_connection
        self.running = False
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the user interaction service.
        
        This method sets up the necessary collections and subscriptions for
        user interaction tracking.
        
        Args:
            config: Optional configuration dictionary for the service
        """
        logger.info("Initializing UserInteractionService")
        
        # Initialize collections for user interaction tracking
        self._initialize_collections()
        
        # Subscribe to events from the event service
        self._setup_event_subscriptions()
        
        self.running = True
        logger.info("UserInteractionService initialized")
        
    def shutdown(self) -> None:
        """
        Shutdown the user interaction service.
        
        This method cleans up resources and unsubscribes from events.
        """
        logger.info("Shutting down UserInteractionService")
        
        # Unsubscribe from events
        # TODO: Implement unsubscribe logic if needed
        
        self.running = False
        logger.info("UserInteractionService shut down")
        
    def _initialize_collections(self) -> None:
        """
        Initialize collections for user interaction tracking.
        
        This method ensures that the necessary collections exist in ArangoDB
        for tracking user interactions, including queries, documents, and feedback.
        """
        # Ensure the user_interactions collection exists
        if not self.arangodb_connection.collection_exists("user_interactions"):
            self.arangodb_connection.create_collection("user_interactions")
            logger.info("Created user_interactions collection")
            
        # Ensure the user_feedback collection exists
        if not self.arangodb_connection.collection_exists("user_feedback"):
            self.arangodb_connection.create_collection("user_feedback")
            logger.info("Created user_feedback collection")
            
    def _setup_event_subscriptions(self) -> None:
        """
        Set up event subscriptions for the user interaction service.
        
        This method subscribes to relevant events from the event service,
        such as pattern updates and field state changes.
        """
        # Subscribe to pattern events
        self.bidirectional_flow_service.register_pattern_handler(self._handle_pattern_event)
        
        # Subscribe to field state events
        self.bidirectional_flow_service.register_field_state_handler(self._handle_field_state_event)
        
        logger.info("Set up event subscriptions for UserInteractionService")
        
    def _handle_pattern_event(self, event: Dict[str, Any]) -> None:
        """
        Handle a pattern event.
        
        This method is called when a pattern event is received from the
        bidirectional flow service.
        
        Args:
            event: The pattern event to handle
        """
        logger.debug(f"Handling pattern event: {event}")
        # TODO: Implement pattern event handling
        
    def _handle_field_state_event(self, event: Dict[str, Any]) -> None:
        """
        Handle a field state event.
        
        This method is called when a field state event is received from the
        bidirectional flow service.
        
        Args:
            event: The field state event to handle
        """
        logger.debug(f"Handling field state event: {event}")
        # TODO: Implement field state event handling
        
    def _initialize_collections(self) -> None:
        """
        Initialize collections for storing interaction history and patterns.
        """
        try:
            # Create collections if they don't exist
            if not self.arangodb_connection.collection_exists("user_interactions"):
                self.arangodb_connection.create_collection("user_interactions")
                logger.info("Created user_interactions collection")
                
            if not self.arangodb_connection.collection_exists("pattern_evolution"):
                self.arangodb_connection.create_collection("pattern_evolution")
                logger.info("Created pattern_evolution collection")
                
            # Create indexes for efficient querying
            self.arangodb_connection.create_index(
                "user_interactions",
                {
                    "type": "persistent",
                    "fields": ["user_id", "timestamp"]
                }
            )
            logger.info("Created index on user_interactions collection")
            
            self.arangodb_connection.create_index(
                "pattern_evolution",
                {
                    "type": "persistent",
                    "fields": ["pattern_id", "timestamp"]
                }
            )
            logger.info("Created index on pattern_evolution collection")
        except Exception as e:
            logger.error(f"Error initializing collections: {e}")
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user query and return a response.
        
        This method takes a user query, processes it through the pattern-aware RAG system,
        and returns a response. It also initiates the bidirectional flow by retrieving
        patterns from ArangoDB and updating them based on the interaction.
        
        Args:
            query: The user query to process
            context: Optional context for the query
            
        Returns:
            A dictionary containing the response and additional information
        """
        # Generate a unique ID for this interaction
        interaction_id = str(uuid.uuid4())
        
        # Prepare context if not provided
        if context is None:
            context = {}
        
        # Add interaction ID to context
        context["interaction_id"] = interaction_id
        
        # Log the start of the interaction
        logger.info(f"Processing query: {query} (interaction_id: {interaction_id})")
        
        try:
            # Retrieve relevant patterns from ArangoDB based on the query
            # This is where we would use vector search or other retrieval methods
            # For now, we'll use the pattern_aware_rag_service to get patterns
            patterns = self.pattern_aware_rag_service.get_patterns(query, context)
            
            # Add patterns to context for the RAG service
            context["patterns"] = patterns
            
            # Process the query through the pattern-aware RAG service
            rag_result = self.pattern_aware_rag_service.query(query, context)
            
            # Extract patterns from the RAG result
            result_patterns = rag_result.get("patterns", [])
            
            # Update patterns in ArangoDB based on the interaction
            for pattern in result_patterns:
                # Get the current state of the pattern
                current_pattern = self.pattern_aware_rag_service.get_pattern(pattern["id"])
                
                # Update the pattern with new information
                updated_pattern = self._update_pattern_with_interaction(
                    current_pattern,
                    interaction_id,
                    query,
                    rag_result
                )
                
                # Publish the updated pattern through the bidirectional flow
                self.bidirectional_flow_service.publish_pattern(updated_pattern)
            
            # Store the interaction in ArangoDB
            self._store_interaction(
                interaction_id,
                "query",
                query,
                context.get("user_id"),
                rag_result,
                result_patterns
            )
            
            # Add interaction ID to the result
            rag_result["interaction_id"] = interaction_id
            
            return rag_result
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "interaction_id": interaction_id,
                "error": str(e),
                "status": "error"
            }
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document and extract patterns.
        
        This method takes a document, processes it through the pattern-aware RAG system,
        extracts patterns, and stores them in ArangoDB. It also initiates the bidirectional
        flow by updating existing patterns based on the new information.
        
        Args:
            document: The document to process
            
        Returns:
            A dictionary containing the processing results and additional information
        """
        # Generate a unique ID for this interaction
        interaction_id = str(uuid.uuid4())
        
        # Log the start of the document processing
        logger.info(f"Processing document: {document.get('id', 'unknown')} (interaction_id: {interaction_id})")
        
        try:
            # Process the document through the pattern-aware RAG service
            rag_result = self.pattern_aware_rag_service.process_document(document)
            
            # Extract patterns from the RAG result
            result_patterns = rag_result.get("patterns", [])
            
            # Store or update patterns in ArangoDB
            for pattern in result_patterns:
                # Check if the pattern already exists
                existing_pattern = self.pattern_aware_rag_service.get_pattern(pattern["id"])
                
                if existing_pattern:
                    # Update the existing pattern
                    updated_pattern = self._update_pattern_with_document(
                        existing_pattern,
                        interaction_id,
                        document,
                        rag_result
                    )
                    
                    # Publish the updated pattern through the bidirectional flow
                    self.bidirectional_flow_service.publish_pattern(updated_pattern)
                else:
                    # Add document reference to the pattern
                    pattern["document_references"] = [
                        {
                            "document_id": document.get("id"),
                            "interaction_id": interaction_id,
                            "timestamp": datetime.now().isoformat()
                        }
                    ]
                    
                    # Add the new pattern to the RAG service
                    self.pattern_aware_rag_service.add_pattern(pattern)
                    
                    # Publish the new pattern through the bidirectional flow
                    self.bidirectional_flow_service.publish_pattern(pattern)
            
            # Store the interaction in ArangoDB
            self._store_interaction(
                interaction_id,
                "document",
                document.get("content", ""),
                document.get("user_id"),
                rag_result,
                result_patterns
            )
            
            # Add interaction ID to the result
            rag_result["interaction_id"] = interaction_id
            
            return rag_result
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {
                "interaction_id": interaction_id,
                "error": str(e),
                "status": "error"
            }
    
    def provide_feedback(self, query_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide feedback on a previous query response.
        
        This method takes feedback on a previous query response and uses it to update
        the quality states of the patterns involved. It also initiates the bidirectional
        flow by updating patterns in ArangoDB based on the feedback.
        
        Args:
            query_id: The ID of the query to provide feedback for
            feedback: The feedback to provide
            
        Returns:
            A dictionary containing the feedback processing results
        """
        # Log the feedback
        logger.info(f"Processing feedback for query: {query_id}")
        
        try:
            # Retrieve the interaction from ArangoDB
            interaction = self._get_interaction(query_id)
            
            if not interaction:
                return {
                    "error": f"Interaction {query_id} not found",
                    "status": "error"
                }
            
            # Extract patterns from the interaction
            patterns = interaction.get("patterns", [])
            
            # Update pattern quality based on feedback
            for pattern in patterns:
                # Get the current state of the pattern
                current_pattern = self.pattern_aware_rag_service.get_pattern(pattern["id"])
                
                if not current_pattern:
                    continue
                
                # Update pattern quality based on feedback
                updated_pattern = self._update_pattern_with_feedback(
                    current_pattern,
                    query_id,
                    feedback
                )
                
                # Publish the updated pattern through the bidirectional flow
                self.bidirectional_flow_service.publish_pattern(updated_pattern)
            
            # Store the feedback in ArangoDB
            self._store_feedback(query_id, feedback)
            
            return {
                "interaction_id": query_id,
                "feedback_processed": True,
                "patterns_updated": len(patterns),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return {
                "interaction_id": query_id,
                "error": str(e),
                "status": "error"
            }
    
    def get_interaction_history(self, user_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the interaction history for a user.
        
        This method retrieves the interaction history for a user, including queries,
        responses, and feedback. It can be used to provide context for future interactions.
        
        Args:
            user_id: Optional user ID to get history for (if None, get all history)
            limit: Maximum number of interactions to return
            
        Returns:
            A list of dictionaries containing the interaction history
        """
        try:
            # Build the AQL query
            if user_id:
                query = """
                FOR i IN user_interactions
                    FILTER i.user_id == @user_id
                    SORT i.timestamp DESC
                    LIMIT @limit
                    RETURN i
                """
                bind_vars = {"user_id": user_id, "limit": limit}
            else:
                query = """
                FOR i IN user_interactions
                    SORT i.timestamp DESC
                    LIMIT @limit
                    RETURN i
                """
                bind_vars = {"limit": limit}
            
            # Execute the query
            result = self.arangodb_connection.execute_aql(query, bind_vars)
            
            return list(result)
        except Exception as e:
            logger.error(f"Error getting interaction history: {e}")
            return []
    
    def get_pattern_evolution(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get the evolution history for a pattern.
        
        This method retrieves the evolution history for a pattern, including quality
        state transitions, usage statistics, and relationship changes. It can be used
        to understand how patterns evolve over time.
        
        Args:
            pattern_id: The ID of the pattern to get evolution history for
            
        Returns:
            A dictionary containing the pattern evolution history
        """
        try:
            # Get the current pattern
            current_pattern = self.pattern_aware_rag_service.get_pattern(pattern_id)
            
            if not current_pattern:
                return {
                    "error": f"Pattern {pattern_id} not found",
                    "status": "error"
                }
            
            # Get the pattern evolution history
            query = """
            FOR e IN pattern_evolution
                FILTER e.pattern_id == @pattern_id
                SORT e.timestamp ASC
                RETURN e
            """
            bind_vars = {"pattern_id": pattern_id}
            
            # Execute the query
            evolution_history = list(self.arangodb_connection.execute_aql(query, bind_vars))
            
            # Get related patterns
            related_patterns = self.pattern_aware_rag_service.get_related_patterns(pattern_id)
            
            # Compile the evolution data
            evolution_data = {
                "pattern_id": pattern_id,
                "current_state": current_pattern,
                "evolution_history": evolution_history,
                "related_patterns": related_patterns,
                "status": "success"
            }
            
            return evolution_data
        except Exception as e:
            logger.error(f"Error getting pattern evolution: {e}")
            return {
                "pattern_id": pattern_id,
                "error": str(e),
                "status": "error"
            }
    
    def _update_pattern_with_interaction(
        self,
        pattern: Dict[str, Any],
        interaction_id: str,
        query: str,
        rag_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update a pattern based on a user interaction.
        
        Args:
            pattern: The pattern to update
            interaction_id: The ID of the interaction
            query: The user query
            rag_result: The RAG result
            
        Returns:
            The updated pattern
        """
        # Clone the pattern to avoid modifying the original
        updated_pattern = pattern.copy()
        
        # Add the interaction to the pattern's history
        if "interaction_history" not in updated_pattern:
            updated_pattern["interaction_history"] = []
            
        updated_pattern["interaction_history"].append({
            "interaction_id": interaction_id,
            "type": "query",
            "query": query,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update pattern metrics based on the RAG result
        if "metrics" not in updated_pattern:
            updated_pattern["metrics"] = {}
            
        # Update usage count
        updated_pattern["metrics"]["usage_count"] = updated_pattern["metrics"].get("usage_count", 0) + 1
        
        # Update last used timestamp
        updated_pattern["metrics"]["last_used"] = datetime.now().isoformat()
        
        # Store the pattern evolution event
        self._store_pattern_evolution(
            pattern["id"],
            "interaction",
            updated_pattern,
            {
                "interaction_id": interaction_id,
                "query": query
            }
        )
        
        return updated_pattern
    
    def _update_pattern_with_document(
        self,
        pattern: Dict[str, Any],
        interaction_id: str,
        document: Dict[str, Any],
        rag_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update a pattern based on a document processing.
        
        Args:
            pattern: The pattern to update
            interaction_id: The ID of the interaction
            document: The document
            rag_result: The RAG result
            
        Returns:
            The updated pattern
        """
        # Clone the pattern to avoid modifying the original
        updated_pattern = pattern.copy()
        
        # Add the document reference to the pattern
        if "document_references" not in updated_pattern:
            updated_pattern["document_references"] = []
            
        updated_pattern["document_references"].append({
            "document_id": document.get("id"),
            "interaction_id": interaction_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update pattern metrics based on the document
        if "metrics" not in updated_pattern:
            updated_pattern["metrics"] = {}
            
        # Update document count
        updated_pattern["metrics"]["document_count"] = updated_pattern["metrics"].get("document_count", 0) + 1
        
        # Update last updated timestamp
        updated_pattern["metrics"]["last_updated"] = datetime.now().isoformat()
        
        # Store the pattern evolution event
        self._store_pattern_evolution(
            pattern["id"],
            "document",
            updated_pattern,
            {
                "interaction_id": interaction_id,
                "document_id": document.get("id")
            }
        )
        
        return updated_pattern
    
    def _update_pattern_with_feedback(
        self,
        pattern: Dict[str, Any],
        interaction_id: str,
        feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update a pattern based on user feedback.
        
        Args:
            pattern: The pattern to update
            interaction_id: The ID of the interaction
            feedback: The feedback
            
        Returns:
            The updated pattern
        """
        # Clone the pattern to avoid modifying the original
        updated_pattern = pattern.copy()
        
        # Add the feedback to the pattern's history
        if "feedback_history" not in updated_pattern:
            updated_pattern["feedback_history"] = []
            
        updated_pattern["feedback_history"].append({
            "interaction_id": interaction_id,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update pattern quality based on feedback
        quality_rating = feedback.get("quality_rating")
        if quality_rating is not None:
            # Update pattern quality metrics
            if "quality" not in updated_pattern:
                updated_pattern["quality"] = {}
                
            # Calculate new quality score as a weighted average
            current_score = updated_pattern["quality"].get("score", 0.5)
            current_count = updated_pattern["quality"].get("feedback_count", 0)
            
            new_count = current_count + 1
            new_score = ((current_score * current_count) + quality_rating) / new_count
            
            updated_pattern["quality"]["score"] = new_score
            updated_pattern["quality"]["feedback_count"] = new_count
            updated_pattern["quality"]["last_feedback"] = datetime.now().isoformat()
        
        # Store the pattern evolution event
        self._store_pattern_evolution(
            pattern["id"],
            "feedback",
            updated_pattern,
            {
                "interaction_id": interaction_id,
                "feedback": feedback
            }
        )
        
        return updated_pattern
    
    def _store_interaction(
        self,
        interaction_id: str,
        interaction_type: str,
        content: str,
        user_id: Optional[str],
        result: Dict[str, Any],
        patterns: List[Dict[str, Any]]
    ) -> None:
        """
        Store an interaction in ArangoDB.
        
        Args:
            interaction_id: The ID of the interaction
            interaction_type: The type of interaction (query or document)
            content: The content of the interaction
            user_id: The ID of the user (if applicable)
            result: The result of the interaction
            patterns: The patterns involved in the interaction
        """
        try:
            # Create the interaction document
            interaction = {
                "_key": interaction_id,
                "type": interaction_type,
                "content": content,
                "user_id": user_id,
                "result": result,
                "patterns": patterns,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store the interaction in ArangoDB
            self.arangodb_connection.create_document("user_interactions", interaction)
            
            logger.debug(f"Stored interaction: {interaction_id}")
        except Exception as e:
            logger.error(f"Error storing interaction: {e}")
    
    def _store_feedback(self, interaction_id: str, feedback: Dict[str, Any]) -> None:
        """
        Store feedback for an interaction in ArangoDB.
        
        Args:
            interaction_id: The ID of the interaction
            feedback: The feedback to store
        """
        try:
            # Get the interaction
            interaction = self._get_interaction(interaction_id)
            
            if not interaction:
                logger.error(f"Interaction {interaction_id} not found")
                return
            
            # Add feedback to the interaction
            interaction["feedback"] = feedback
            interaction["feedback_timestamp"] = datetime.now().isoformat()
            
            # Update the interaction in ArangoDB
            self.arangodb_connection.update_document(
                "user_interactions",
                interaction_id,
                interaction
            )
            
            logger.debug(f"Stored feedback for interaction: {interaction_id}")
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
    
    def _store_pattern_evolution(
        self,
        pattern_id: str,
        event_type: str,
        pattern: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        """
        Store a pattern evolution event in ArangoDB.
        
        Args:
            pattern_id: The ID of the pattern
            event_type: The type of event (interaction, document, or feedback)
            pattern: The updated pattern
            context: Additional context for the event
        """
        try:
            # Create the evolution event document
            evolution_event = {
                "_key": str(uuid.uuid4()),
                "pattern_id": pattern_id,
                "event_type": event_type,
                "pattern_state": pattern,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store the evolution event in ArangoDB
            self.arangodb_connection.create_document("pattern_evolution", evolution_event)
            
            logger.debug(f"Stored pattern evolution event for pattern: {pattern_id}")
        except Exception as e:
            logger.error(f"Error storing pattern evolution event: {e}")
    
    def _get_interaction(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an interaction from ArangoDB.
        
        Args:
            interaction_id: The ID of the interaction
            
        Returns:
            The interaction document, or None if not found
        """
        try:
            # Get the interaction from ArangoDB
            return self.arangodb_connection.get_document("user_interactions", interaction_id)
        except Exception as e:
            logger.error(f"Error getting interaction: {e}")
            return None
