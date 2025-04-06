"""
Pattern evolution service for Habitat Evolution.

This module provides the implementation of the PatternEvolutionInterface,
enabling patterns to evolve based on usage and feedback in the Habitat Evolution system.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.habitat_evolution.infrastructure.interfaces.services.pattern_evolution_interface import PatternEvolutionInterface
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.bidirectional_flow_interface import BidirectionalFlowInterface
from src.habitat_evolution.infrastructure.interfaces.persistence.arangodb_connection_interface import ArangoDBConnectionInterface

logger = logging.getLogger(__name__)


class PatternEvolutionService(PatternEvolutionInterface):
    """
    Implementation of the PatternEvolutionInterface.
    
    This service enables patterns to evolve based on usage and feedback in the
    Habitat Evolution system. It tracks pattern usage, processes feedback, and
    updates pattern quality states based on evidence.
    """
    
    def __init__(
        self,
        event_service: EventServiceInterface,
        bidirectional_flow_service: BidirectionalFlowInterface,
        arangodb_connection: ArangoDBConnectionInterface
    ):
        """
        Initialize the pattern evolution service.
        
        Args:
            event_service: The event service to use for communication
            bidirectional_flow_service: The bidirectional flow service to integrate with
            arangodb_connection: The ArangoDB connection for pattern persistence
        """
        self.event_service = event_service
        self.bidirectional_flow_service = bidirectional_flow_service
        self.arangodb_connection = arangodb_connection
        self.running = False
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the pattern evolution service.
        
        This method sets up the necessary collections and subscriptions for
        pattern evolution tracking.
        
        Args:
            config: Optional configuration dictionary for the service
        """
        logger.info("Initializing PatternEvolutionService")
        
        # Initialize collections for pattern evolution tracking
        self._initialize_collections()
        
        # Subscribe to events from the event service
        self._setup_event_subscriptions()
        
        self.running = True
        logger.info("PatternEvolutionService initialized")
        
    def shutdown(self) -> None:
        """
        Shutdown the pattern evolution service.
        
        This method cleans up resources and unsubscribes from events.
        """
        logger.info("Shutting down PatternEvolutionService")
        
        # Unsubscribe from events
        # TODO: Implement unsubscribe logic if needed
        
        self.running = False
        logger.info("PatternEvolutionService shut down")
        
    def _initialize_collections(self) -> None:
        """
        Initialize collections for pattern evolution tracking.
        """
        try:
            # Create collections if they don't exist
            if not self.arangodb_connection.collection_exists("pattern_usage"):
                self.arangodb_connection.create_collection("pattern_usage")
                logger.info("Created pattern_usage collection")
                
            if not self.arangodb_connection.collection_exists("pattern_feedback"):
                self.arangodb_connection.create_collection("pattern_feedback")
                logger.info("Created pattern_feedback collection")
                
            if not self.arangodb_connection.collection_exists("pattern_quality_transitions"):
                self.arangodb_connection.create_collection("pattern_quality_transitions")
                logger.info("Created pattern_quality_transitions collection")
                
            # Create indexes for efficient querying
            self.arangodb_connection.create_index(
                "pattern_usage",
                {
                    "type": "persistent",
                    "fields": ["pattern_id", "timestamp"]
                }
            )
            logger.info("Created index on pattern_usage collection")
            
            self.arangodb_connection.create_index(
                "pattern_feedback",
                {
                    "type": "persistent",
                    "fields": ["pattern_id", "timestamp"]
                }
            )
            logger.info("Created index on pattern_feedback collection")
            
            self.arangodb_connection.create_index(
                "pattern_quality_transitions",
                {
                    "type": "persistent",
                    "fields": ["pattern_id", "timestamp"]
                }
            )
            logger.info("Created index on pattern_quality_transitions collection")
        except Exception as e:
            logger.error(f"Error initializing collections: {e}")
            
    def _setup_event_subscriptions(self) -> None:
        """
        Set up subscriptions to events from the event service.
        """
        self.event_service.subscribe("pattern.detected", self._handle_pattern_event)
        self.event_service.subscribe("pattern.used", self._handle_pattern_usage_event)
        self.event_service.subscribe("pattern.feedback", self._handle_pattern_feedback_event)
        
    def _handle_pattern_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle a pattern event from the event service.
        
        Args:
            event_data: The pattern event data
        """
        logger.debug(f"Received pattern event: {event_data}")
        
        # Check if the pattern exists in the database
        pattern_id = event_data.get("id")
        if not pattern_id:
            logger.error("Pattern event missing ID")
            return
            
        # Get the pattern from the database
        pattern = self._get_pattern(pattern_id)
        
        # If the pattern doesn't exist, create it
        if not pattern:
            self._create_pattern(event_data)
        else:
            # Update the pattern with the new data
            self._update_pattern(pattern_id, event_data)
            
    def _handle_pattern_usage_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle a pattern usage event from the event service.
        
        Args:
            event_data: The pattern usage event data
        """
        logger.debug(f"Received pattern usage event: {event_data}")
        
        # Extract pattern ID and context
        pattern_id = event_data.get("pattern_id")
        context = event_data.get("context", {})
        
        if not pattern_id:
            logger.error("Pattern usage event missing pattern_id")
            return
            
        # Track the pattern usage
        self.track_pattern_usage(pattern_id, context)
        
    def _handle_pattern_feedback_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle a pattern feedback event from the event service.
        
        Args:
            event_data: The pattern feedback event data
        """
        logger.debug(f"Received pattern feedback event: {event_data}")
        
        # Extract pattern ID and feedback
        pattern_id = event_data.get("pattern_id")
        feedback = event_data.get("feedback", {})
        
        if not pattern_id:
            logger.error("Pattern feedback event missing pattern_id")
            return
            
        # Track the pattern feedback
        self.track_pattern_feedback(pattern_id, feedback)
        
    def _get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
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
            
    def _create_pattern(self, pattern_data: Dict[str, Any]) -> None:
        """
        Create a pattern in ArangoDB.
        
        Args:
            pattern_data: The pattern data to create
        """
        try:
            # Ensure the pattern has an ID
            if "id" not in pattern_data:
                pattern_data["id"] = str(uuid.uuid4())
                
            # Ensure the pattern has a timestamp
            if "timestamp" not in pattern_data:
                pattern_data["timestamp"] = datetime.now().isoformat()
                
            # Ensure the pattern has a quality state
            if "quality_state" not in pattern_data:
                pattern_data["quality_state"] = "emerging"
                
            # Ensure the pattern has quality metrics
            if "quality" not in pattern_data:
                pattern_data["quality"] = {
                    "score": 0.5,
                    "feedback_count": 0,
                    "usage_count": 0,
                    "last_used": pattern_data["timestamp"],
                    "last_feedback": None
                }
                
            # Prepare the pattern document
            pattern_doc = pattern_data.copy()
            pattern_doc["_key"] = pattern_data["id"]
            
            # Store the pattern in ArangoDB
            self.arangodb_connection.create_document("patterns", pattern_doc)
            logger.debug(f"Created pattern in ArangoDB: {pattern_data['id']}")
            
            # Track the quality state transition
            self._track_quality_transition(
                pattern_data["id"],
                None,
                pattern_data["quality_state"],
                "creation"
            )
        except Exception as e:
            logger.error(f"Error creating pattern in ArangoDB: {e}")
            
    def _update_pattern(self, pattern_id: str, pattern_data: Dict[str, Any]) -> None:
        """
        Update a pattern in ArangoDB.
        
        Args:
            pattern_id: The ID of the pattern to update
            pattern_data: The new pattern data
        """
        try:
            # Get the current pattern
            current_pattern = self._get_pattern(pattern_id)
            
            if not current_pattern:
                logger.error(f"Pattern not found for update: {pattern_id}")
                return
                
            # Update the pattern with the new data
            updated_pattern = current_pattern.copy()
            
            # Update fields from pattern_data
            for key, value in pattern_data.items():
                if key not in ["_key", "_id", "_rev"]:
                    updated_pattern[key] = value
                    
            # Update the timestamp
            updated_pattern["timestamp"] = datetime.now().isoformat()
            
            # Check if the quality state has changed
            old_state = current_pattern.get("quality_state")
            new_state = updated_pattern.get("quality_state")
            
            if old_state != new_state:
                # Track the quality state transition
                self._track_quality_transition(
                    pattern_id,
                    old_state,
                    new_state,
                    "update"
                )
                
            # Update the pattern in ArangoDB
            self.arangodb_connection.update_document("patterns", pattern_id, updated_pattern)
            logger.debug(f"Updated pattern in ArangoDB: {pattern_id}")
        except Exception as e:
            logger.error(f"Error updating pattern in ArangoDB: {e}")
            
    def _track_quality_transition(
        self,
        pattern_id: str,
        old_state: Optional[str],
        new_state: str,
        reason: str
    ) -> None:
        """
        Track a pattern quality state transition in ArangoDB.
        
        Args:
            pattern_id: The ID of the pattern
            old_state: The old quality state (None for new patterns)
            new_state: The new quality state
            reason: The reason for the transition
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
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store the transition in ArangoDB
            self.arangodb_connection.create_document("pattern_quality_transitions", transition)
            logger.debug(f"Tracked pattern quality transition for {pattern_id}: {old_state} -> {new_state}")
            
            # Publish the transition event
            self.event_service.publish("pattern.quality_transition", transition)
        except Exception as e:
            logger.error(f"Error tracking pattern quality transition: {e}")
            
    def track_pattern_usage(self, pattern_id: str, context: Dict[str, Any]) -> None:
        """
        Track the usage of a pattern.
        
        This method records when a pattern is used in a query or document processing,
        updating its usage statistics and potentially its quality state.
        
        Args:
            pattern_id: The ID of the pattern to track
            context: The context in which the pattern was used
        """
        try:
            # Get the current pattern
            pattern = self._get_pattern(pattern_id)
            
            if not pattern:
                logger.error(f"Pattern not found for usage tracking: {pattern_id}")
                return
                
            # Create the usage document
            usage = {
                "_key": str(uuid.uuid4()),
                "pattern_id": pattern_id,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store the usage in ArangoDB
            self.arangodb_connection.create_document("pattern_usage", usage)
            logger.debug(f"Tracked pattern usage for {pattern_id}")
            
            # Update the pattern's usage statistics
            updated_pattern = pattern.copy()
            
            # Ensure the pattern has quality metrics
            if "quality" not in updated_pattern:
                updated_pattern["quality"] = {
                    "score": 0.5,
                    "feedback_count": 0,
                    "usage_count": 0,
                    "last_used": None,
                    "last_feedback": None
                }
                
            # Update usage count
            updated_pattern["quality"]["usage_count"] = updated_pattern["quality"].get("usage_count", 0) + 1
            
            # Update last used timestamp
            updated_pattern["quality"]["last_used"] = datetime.now().isoformat()
            
            # Check if the pattern should transition to a new quality state
            self._check_quality_transition(updated_pattern)
            
            # Update the pattern in ArangoDB
            self._update_pattern(pattern_id, updated_pattern)
            
            # Publish the updated pattern
            self.bidirectional_flow_service.publish_pattern(updated_pattern)
        except Exception as e:
            logger.error(f"Error tracking pattern usage: {e}")
            
    def track_pattern_feedback(self, pattern_id: str, feedback: Dict[str, Any]) -> None:
        """
        Track feedback for a pattern.
        
        This method records feedback for a pattern, updating its quality metrics
        and potentially its quality state.
        
        Args:
            pattern_id: The ID of the pattern to track
            feedback: The feedback to record
        """
        try:
            # Get the current pattern
            pattern = self._get_pattern(pattern_id)
            
            if not pattern:
                logger.error(f"Pattern not found for feedback tracking: {pattern_id}")
                return
                
            # Create the feedback document
            feedback_doc = {
                "_key": str(uuid.uuid4()),
                "pattern_id": pattern_id,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store the feedback in ArangoDB
            self.arangodb_connection.create_document("pattern_feedback", feedback_doc)
            logger.debug(f"Tracked pattern feedback for {pattern_id}")
            
            # Update the pattern's quality metrics
            updated_pattern = pattern.copy()
            
            # Ensure the pattern has quality metrics
            if "quality" not in updated_pattern:
                updated_pattern["quality"] = {
                    "score": 0.5,
                    "feedback_count": 0,
                    "usage_count": 0,
                    "last_used": None,
                    "last_feedback": None
                }
                
            # Update feedback count
            updated_pattern["quality"]["feedback_count"] = updated_pattern["quality"].get("feedback_count", 0) + 1
            
            # Update last feedback timestamp
            updated_pattern["quality"]["last_feedback"] = datetime.now().isoformat()
            
            # Update quality score based on feedback
            quality_rating = feedback.get("quality_rating")
            if quality_rating is not None:
                # Calculate new quality score as a weighted average
                current_score = updated_pattern["quality"].get("score", 0.5)
                current_count = updated_pattern["quality"].get("feedback_count", 0)
                
                new_count = current_count + 1
                new_score = ((current_score * current_count) + quality_rating) / new_count
                
                updated_pattern["quality"]["score"] = new_score
                
            # Check if the pattern should transition to a new quality state
            self._check_quality_transition(updated_pattern)
            
            # Update the pattern in ArangoDB
            self._update_pattern(pattern_id, updated_pattern)
            
            # Publish the updated pattern
            self.bidirectional_flow_service.publish_pattern(updated_pattern)
        except Exception as e:
            logger.error(f"Error tracking pattern feedback: {e}")
            
    def _check_quality_transition(self, pattern: Dict[str, Any]) -> None:
        """
        Check if a pattern should transition to a new quality state.
        
        Args:
            pattern: The pattern to check
        """
        # Get the current quality state
        current_state = pattern.get("quality_state", "emerging")
        
        # Get the quality metrics
        quality = pattern.get("quality", {})
        score = quality.get("score", 0.5)
        usage_count = quality.get("usage_count", 0)
        feedback_count = quality.get("feedback_count", 0)
        
        # Determine the new quality state based on metrics
        new_state = current_state
        
        if current_state == "emerging":
            # Transition to "established" if the pattern has been used enough
            # and has a good quality score
            if usage_count >= 5 and feedback_count >= 3 and score >= 0.7:
                new_state = "established"
        elif current_state == "established":
            # Transition to "validated" if the pattern has been used extensively
            # and has a high quality score
            if usage_count >= 20 and feedback_count >= 10 and score >= 0.8:
                new_state = "validated"
            # Transition back to "emerging" if the quality score drops
            elif score < 0.5:
                new_state = "emerging"
        elif current_state == "validated":
            # Transition back to "established" if the quality score drops
            if score < 0.7:
                new_state = "established"
                
        # Update the quality state if it has changed
        if new_state != current_state:
            pattern["quality_state"] = new_state
            
    def get_pattern_evolution(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get the evolution history for a pattern.
        
        This method retrieves the evolution history for a pattern, including quality
        state transitions, usage statistics, and relationship changes.
        
        Args:
            pattern_id: The ID of the pattern to get evolution history for
            
        Returns:
            A dictionary containing the pattern evolution history
        """
        try:
            # Get the current pattern
            pattern = self._get_pattern(pattern_id)
            
            if not pattern:
                return {
                    "error": f"Pattern {pattern_id} not found",
                    "status": "error"
                }
                
            # Get the quality state transitions
            transitions_query = """
            FOR t IN pattern_quality_transitions
                FILTER t.pattern_id == @pattern_id
                SORT t.timestamp ASC
                RETURN t
            """
            transitions = list(self.arangodb_connection.execute_aql(
                transitions_query,
                {"pattern_id": pattern_id}
            ))
            
            # Get the usage history
            usage_query = """
            FOR u IN pattern_usage
                FILTER u.pattern_id == @pattern_id
                SORT u.timestamp ASC
                RETURN u
            """
            usage = list(self.arangodb_connection.execute_aql(
                usage_query,
                {"pattern_id": pattern_id}
            ))
            
            # Get the feedback history
            feedback_query = """
            FOR f IN pattern_feedback
                FILTER f.pattern_id == @pattern_id
                SORT f.timestamp ASC
                RETURN f
            """
            feedback = list(self.arangodb_connection.execute_aql(
                feedback_query,
                {"pattern_id": pattern_id}
            ))
            
            # Get related patterns
            related_query = """
            FOR r IN pattern_relationships
                FILTER r._from == CONCAT('patterns/', @pattern_id) OR r._to == CONCAT('patterns/', @pattern_id)
                RETURN r
            """
            relationships = list(self.arangodb_connection.execute_aql(
                related_query,
                {"pattern_id": pattern_id}
            ))
            
            # Compile the evolution data
            evolution_data = {
                "pattern_id": pattern_id,
                "current_state": pattern,
                "quality_transitions": transitions,
                "usage_history": usage,
                "feedback_history": feedback,
                "relationships": relationships,
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
            
    def get_pattern_quality(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get the quality metrics for a pattern.
        
        This method retrieves the quality metrics for a pattern, including its
        current quality state, score, and feedback count.
        
        Args:
            pattern_id: The ID of the pattern to get quality metrics for
            
        Returns:
            A dictionary containing the pattern quality metrics
        """
        try:
            # Get the current pattern
            pattern = self._get_pattern(pattern_id)
            
            if not pattern:
                return {
                    "error": f"Pattern {pattern_id} not found",
                    "status": "error"
                }
                
            # Extract the quality metrics
            quality_state = pattern.get("quality_state", "unknown")
            quality_metrics = pattern.get("quality", {})
            
            # Compile the quality data
            quality_data = {
                "pattern_id": pattern_id,
                "quality_state": quality_state,
                "quality_metrics": quality_metrics,
                "status": "success"
            }
            
            return quality_data
        except Exception as e:
            logger.error(f"Error getting pattern quality: {e}")
            return {
                "pattern_id": pattern_id,
                "error": str(e),
                "status": "error"
            }
            
    def update_pattern_quality(self, pattern_id: str, quality_metrics: Dict[str, Any]) -> None:
        """
        Update the quality metrics for a pattern.
        
        This method updates the quality metrics for a pattern, potentially
        transitioning it to a new quality state.
        
        Args:
            pattern_id: The ID of the pattern to update
            quality_metrics: The new quality metrics
        """
        try:
            # Get the current pattern
            pattern = self._get_pattern(pattern_id)
            
            if not pattern:
                logger.error(f"Pattern not found for quality update: {pattern_id}")
                return
                
            # Update the pattern's quality metrics
            updated_pattern = pattern.copy()
            
            # Ensure the pattern has quality metrics
            if "quality" not in updated_pattern:
                updated_pattern["quality"] = {
                    "score": 0.5,
                    "feedback_count": 0,
                    "usage_count": 0,
                    "last_used": None,
                    "last_feedback": None
                }
                
            # Update the quality metrics
            for key, value in quality_metrics.items():
                updated_pattern["quality"][key] = value
                
            # Check if the pattern should transition to a new quality state
            self._check_quality_transition(updated_pattern)
            
            # Update the pattern in ArangoDB
            self._update_pattern(pattern_id, updated_pattern)
            
            # Publish the updated pattern
            self.bidirectional_flow_service.publish_pattern(updated_pattern)
        except Exception as e:
            logger.error(f"Error updating pattern quality: {e}")
            
    def identify_emerging_patterns(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Identify emerging patterns based on usage and quality metrics.
        
        This method identifies patterns that are emerging as important based on
        their usage and quality metrics.
        
        Args:
            threshold: The threshold for identifying emerging patterns
            
        Returns:
            A list of emerging patterns
        """
        try:
            # Query for patterns with high quality scores but still in the emerging state
            query = """
            FOR p IN patterns
                FILTER p.quality_state == "emerging" AND p.quality.score >= @threshold
                SORT p.quality.usage_count DESC
                LIMIT 10
                RETURN p
            """
            bind_vars = {"threshold": threshold}
            
            # Execute the query
            emerging_patterns = list(self.arangodb_connection.execute_aql(query, bind_vars))
            
            return emerging_patterns
        except Exception as e:
            logger.error(f"Error identifying emerging patterns: {e}")
            return []
