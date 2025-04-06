"""
User interaction interface for Habitat Evolution.

This module defines the interface for user interaction in the Habitat Evolution system,
providing a consistent approach to handling user queries and feedback.
"""

from typing import Protocol, Dict, List, Any, Optional
from abc import abstractmethod

from src.habitat_evolution.infrastructure.interfaces.service_interface import ServiceInterface


class UserInteractionInterface(ServiceInterface, Protocol):
    """
    Interface for user interaction in Habitat Evolution.
    
    User interaction provides a consistent approach to handling user queries and feedback,
    enabling the system to respond to user requests while collecting information for
    pattern evolution. This supports the pattern evolution and co-evolution principles
    of Habitat by enabling user interactions to drive pattern evolution.
    """
    
    @abstractmethod
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
        ...
        
    @abstractmethod
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
        ...
        
    @abstractmethod
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
        ...
        
    @abstractmethod
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
        ...
        
    @abstractmethod
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
        ...
