"""
Integration between PKM Repository and BidirectionalFlow for Habitat Evolution.

This module provides the integration between the PKM Repository and the 
BidirectionalFlow service, enabling pattern-driven query generation and
knowledge capture in PKM files.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime

from src.habitat_evolution.infrastructure.interfaces.services.bidirectional_flow_interface import BidirectionalFlowInterface
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter
from src.habitat_evolution.pkm.pkm_repository import PKMRepository, PKMFile, create_pkm_from_claude_response

logger = logging.getLogger(__name__)

class PKMBidirectionalIntegration:
    """
    Integration between PKM Repository and BidirectionalFlow.
    
    This class provides the integration between the PKM Repository and the
    BidirectionalFlow service, enabling:
    
    1. Pattern-driven query generation
    2. Knowledge capture in PKM files
    3. Relationship creation between patterns and knowledge
    4. Bidirectional flow between patterns and knowledge
    """
    
    def __init__(
        self,
        pkm_repository: PKMRepository,
        bidirectional_flow_service: BidirectionalFlowInterface,
        event_service: EventServiceInterface,
        claude_adapter: ClaudeAdapter,
        creator_id: Optional[str] = None
    ):
        """
        Initialize the PKM Bidirectional Integration.
        
        Args:
            pkm_repository: The PKM repository to use for storing knowledge
            bidirectional_flow_service: The bidirectional flow service to integrate with
            event_service: The event service to use for communication
            claude_adapter: The Claude adapter to use for query processing
            creator_id: Optional creator ID for PKM files
        """
        self.pkm_repository = pkm_repository
        self.bidirectional_flow_service = bidirectional_flow_service
        self.event_service = event_service
        self.claude_adapter = claude_adapter
        self.creator_id = creator_id or "system"
        
        # Pattern cache for query generation
        self.pattern_cache = {}
        
        # PKM file cache for relationship creation
        self.pkm_file_cache = {}
        
        # Register handlers with bidirectional flow service
        self._register_handlers()
        
        logger.info("Initialized PKM Bidirectional Integration")
    
    def _register_handlers(self) -> None:
        """Register handlers with the bidirectional flow service."""
        # Register pattern handler
        self.bidirectional_flow_service.register_pattern_handler(self._handle_pattern_event)
        
        # Register relationship handler
        self.bidirectional_flow_service.register_relationship_handler(self._handle_relationship_event)
        
        # Register field state handler
        self.bidirectional_flow_service.register_field_state_handler(self._handle_field_state_event)
        
        # Subscribe to PKM events
        self.event_service.subscribe("pkm.created", self._handle_pkm_created_event)
        self.event_service.subscribe("pkm.updated", self._handle_pkm_updated_event)
        
        logger.info("Registered handlers with bidirectional flow service")
    
    def _handle_pattern_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle a pattern event from the bidirectional flow service.
        
        This method is called when a new pattern is detected or updated.
        It updates the pattern cache and potentially generates a query.
        
        Args:
            event_data: The pattern event data
        """
        pattern = event_data
        pattern_id = pattern.get("id")
        
        if not pattern_id:
            logger.warning("Received pattern event without ID")
            return
        
        # Update pattern cache
        self.pattern_cache[pattern_id] = pattern
        
        # Check if we should generate a query based on this pattern
        if self._should_generate_query(pattern):
            query = self._generate_query_from_pattern(pattern)
            
            if query:
                logger.info(f"Generated query from pattern {pattern_id}: {query}")
                
                # Process the query with Claude
                self._process_query(query, pattern)
    
    def _handle_relationship_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle a relationship event from the bidirectional flow service.
        
        This method is called when a new relationship is created.
        It creates a relationship between PKM files if relevant.
        
        Args:
            event_data: The relationship event data
        """
        relationship = event_data
        source_id = relationship.get("source_id")
        target_id = relationship.get("target_id")
        
        if not source_id or not target_id:
            logger.warning("Received relationship event without source or target ID")
            return
        
        # Check if we have PKM files for these patterns
        source_pkm_id = self._get_pkm_id_for_pattern(source_id)
        target_pkm_id = self._get_pkm_id_for_pattern(target_id)
        
        if source_pkm_id and target_pkm_id:
            # Create relationship between PKM files
            relationship_id = self.pkm_repository.create_pkm_relationship(
                from_pkm_id=source_pkm_id,
                to_pkm_id=target_pkm_id,
                relationship_type=relationship.get("type", "related"),
                metadata=relationship.get("properties", {})
            )
            
            logger.info(f"Created PKM relationship: {source_pkm_id} -> {target_pkm_id}")
    
    def _handle_field_state_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle a field state event from the bidirectional flow service.
        
        This method is called when a field state is updated.
        It updates the context for query generation.
        
        Args:
            event_data: The field state event data
        """
        # Store field state for context in query generation
        field_state = event_data
        field_id = field_state.get("id", str(uuid.uuid4()))
        
        # TODO: Implement field state handling for context
        logger.debug(f"Received field state event: {field_id}")
    
    def _handle_pkm_created_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle a PKM created event.
        
        This method is called when a new PKM file is created.
        It updates the PKM file cache and publishes patterns.
        
        Args:
            event_data: The PKM created event data
        """
        pkm_id = event_data.get("pkm_id")
        
        if not pkm_id:
            logger.warning("Received PKM created event without ID")
            return
        
        # Get the PKM file
        pkm_file = self.pkm_repository.get_pkm_file(pkm_id)
        
        if not pkm_file:
            logger.warning(f"PKM file not found: {pkm_id}")
            return
        
        # Update PKM file cache
        self.pkm_file_cache[pkm_id] = pkm_file
        
        # Extract patterns from PKM file and publish them
        for pattern in pkm_file.patterns:
            if pattern.get("type") == "claude_response":
                # Don't publish Claude responses as patterns
                continue
                
            # Publish pattern to bidirectional flow service
            self.bidirectional_flow_service.publish_pattern(pattern)
            
            logger.info(f"Published pattern from PKM file: {pattern.get('id')}")
    
    def _handle_pkm_updated_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle a PKM updated event.
        
        This method is called when a PKM file is updated.
        It updates the PKM file cache and publishes new patterns.
        
        Args:
            event_data: The PKM updated event data
        """
        pkm_id = event_data.get("pkm_id")
        
        if not pkm_id:
            logger.warning("Received PKM updated event without ID")
            return
        
        # Get the PKM file
        pkm_file = self.pkm_repository.get_pkm_file(pkm_id)
        
        if not pkm_file:
            logger.warning(f"PKM file not found: {pkm_id}")
            return
        
        # Get the old PKM file from cache
        old_pkm_file = self.pkm_file_cache.get(pkm_id)
        
        if not old_pkm_file:
            # Handle as if it's a new PKM file
            self._handle_pkm_created_event(event_data)
            return
        
        # Update PKM file cache
        self.pkm_file_cache[pkm_id] = pkm_file
        
        # Find new patterns
        old_pattern_ids = {p.get("id") for p in old_pkm_file.patterns}
        new_patterns = [p for p in pkm_file.patterns if p.get("id") not in old_pattern_ids]
        
        # Publish new patterns
        for pattern in new_patterns:
            if pattern.get("type") == "claude_response":
                # Don't publish Claude responses as patterns
                continue
                
            # Publish pattern to bidirectional flow service
            self.bidirectional_flow_service.publish_pattern(pattern)
            
            logger.info(f"Published new pattern from updated PKM file: {pattern.get('id')}")
    
    def _should_generate_query(self, pattern: Dict[str, Any]) -> bool:
        """
        Determine if a query should be generated from a pattern.
        
        Args:
            pattern: The pattern to check
            
        Returns:
            True if a query should be generated, False otherwise
        """
        # Check pattern quality
        quality = pattern.get("quality", 0.0)
        if quality < 0.7:
            return False
        
        # Check pattern type
        pattern_type = pattern.get("type")
        if pattern_type not in ["semantic", "statistical"]:
            return False
        
        # Check if we've already generated a query for this pattern
        pattern_id = pattern.get("id")
        if pattern_id in self.pkm_file_cache:
            return False
        
        return True
    
    def _generate_query_from_pattern(self, pattern: Dict[str, Any]) -> Optional[str]:
        """
        Generate a query from a pattern.
        
        Args:
            pattern: The pattern to generate a query from
            
        Returns:
            The generated query, or None if no query could be generated
        """
        pattern_type = pattern.get("type")
        pattern_content = pattern.get("content")
        
        if not pattern_content:
            return None
        
        if pattern_type == "semantic":
            # For semantic patterns, create a query about the pattern
            return f"What are the implications of {pattern_content}?"
        elif pattern_type == "statistical":
            # For statistical patterns, create a query about the pattern
            return f"What does the statistical pattern '{pattern_content}' indicate?"
        else:
            # For other pattern types, create a generic query
            return f"What insights can be derived from the pattern: {pattern_content}?"
    
    def _process_query(self, query: str, pattern: Dict[str, Any]) -> None:
        """
        Process a query with Claude and store the result as a PKM file.
        
        Args:
            query: The query to process
            pattern: The pattern that generated the query
        """
        # Create context from pattern
        context = {
            "pattern_id": pattern.get("id"),
            "pattern_type": pattern.get("type"),
            "pattern_content": pattern.get("content"),
            "pattern_metadata": pattern.get("metadata", {})
        }
        
        try:
            # Process query with Claude
            response = self.claude_adapter.query(query, context)
            
            # Create PKM file from response
            pkm_file = create_pkm_from_claude_response(
                response=response,
                query=query,
                source_documents=[],  # TODO: Get source documents from pattern
                patterns=[pattern],
                creator_id=self.creator_id
            )
            
            # Store PKM file in repository
            pkm_id = self.pkm_repository.create_pkm_file(pkm_file)
            
            # Update PKM file cache
            self.pkm_file_cache[pkm_id] = pkm_file
            
            # Create mapping from pattern ID to PKM ID
            self._map_pattern_to_pkm(pattern.get("id"), pkm_id)
            
            # Publish PKM created event
            self.event_service.publish("pkm.created", {
                "pkm_id": pkm_id,
                "pattern_id": pattern.get("id"),
                "query": query
            })
            
            logger.info(f"Created PKM file from query: {query} (ID: {pkm_id})")
        except Exception as e:
            logger.error(f"Error processing query with Claude: {e}")
    
    def _get_pkm_id_for_pattern(self, pattern_id: str) -> Optional[str]:
        """
        Get the PKM ID for a pattern.
        
        Args:
            pattern_id: The pattern ID
            
        Returns:
            The PKM ID, or None if not found
        """
        # TODO: Implement proper mapping from pattern ID to PKM ID
        for pkm_id, pkm_file in self.pkm_file_cache.items():
            for pattern in pkm_file.patterns:
                if pattern.get("id") == pattern_id:
                    return pkm_id
        
        return None
    
    def _map_pattern_to_pkm(self, pattern_id: str, pkm_id: str) -> None:
        """
        Map a pattern ID to a PKM ID.
        
        Args:
            pattern_id: The pattern ID
            pkm_id: The PKM ID
        """
        # TODO: Implement proper mapping from pattern ID to PKM ID
        logger.debug(f"Mapped pattern {pattern_id} to PKM file {pkm_id}")
    
    def generate_query_from_patterns(self, patterns: List[Dict[str, Any]]) -> Optional[str]:
        """
        Generate a query from multiple patterns.
        
        This method can be called directly to generate a query from a set of patterns,
        rather than waiting for pattern events.
        
        Args:
            patterns: The patterns to generate a query from
            
        Returns:
            The generated query, or None if no query could be generated
        """
        if not patterns:
            return None
        
        # For now, just use the first pattern
        # TODO: Implement more sophisticated query generation from multiple patterns
        return self._generate_query_from_pattern(patterns[0])
    
    def process_query_with_patterns(self, query: str, patterns: List[Dict[str, Any]]) -> Optional[str]:
        """
        Process a query with Claude using patterns as context.
        
        This method can be called directly to process a query with patterns as context,
        rather than generating the query from patterns.
        
        Args:
            query: The query to process
            patterns: The patterns to use as context
            
        Returns:
            The PKM ID of the created PKM file, or None if an error occurred
        """
        if not patterns:
            logger.warning("No patterns provided for query processing")
            return None
        
        try:
            # Create context from patterns
            context = {
                "patterns": [
                    {
                        "id": p.get("id"),
                        "type": p.get("type"),
                        "content": p.get("content"),
                        "metadata": p.get("metadata", {})
                    }
                    for p in patterns
                ]
            }
            
            # Process query with Claude
            response = self.claude_adapter.query(query, context)
            
            # Create PKM file from response
            pkm_file = create_pkm_from_claude_response(
                response=response,
                query=query,
                source_documents=[],  # TODO: Get source documents from patterns
                patterns=patterns,
                creator_id=self.creator_id
            )
            
            # Store PKM file in repository
            pkm_id = self.pkm_repository.create_pkm_file(pkm_file)
            
            # Update PKM file cache
            self.pkm_file_cache[pkm_id] = pkm_file
            
            # Create mappings from pattern IDs to PKM ID
            for pattern in patterns:
                self._map_pattern_to_pkm(pattern.get("id"), pkm_id)
            
            # Publish PKM created event
            self.event_service.publish("pkm.created", {
                "pkm_id": pkm_id,
                "pattern_ids": [p.get("id") for p in patterns],
                "query": query
            })
            
            logger.info(f"Created PKM file from query with patterns: {query} (ID: {pkm_id})")
            
            return pkm_id
        except Exception as e:
            logger.error(f"Error processing query with Claude: {e}")
            return None
