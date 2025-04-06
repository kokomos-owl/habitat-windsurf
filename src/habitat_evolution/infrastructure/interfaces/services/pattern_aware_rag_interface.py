"""
Pattern-aware RAG interface for Habitat Evolution.

This module defines the interface for pattern-aware retrieval augmented generation (RAG)
in the Habitat Evolution system, providing a consistent approach to pattern-aware
information retrieval and generation.
"""

from typing import Protocol, Dict, List, Any, Optional, Union
from abc import abstractmethod

from src.habitat_evolution.infrastructure.interfaces.service_interface import ServiceInterface


class PatternAwareRAGInterface(ServiceInterface, Protocol):
    """
    Interface for pattern-aware RAG in Habitat Evolution.
    
    Pattern-aware RAG provides a consistent approach to retrieval augmented generation
    that is aware of patterns in the semantic field. This supports the pattern evolution
    and co-evolution principles of Habitat by enabling the retrieval and generation
    of information that is sensitive to the evolving patterns in the system.
    """
    
    @abstractmethod
    def process_document(self, document: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document through the pattern-aware RAG system.
        
        Args:
            document: The document to process
            metadata: Optional metadata for the document
            
        Returns:
            Processing results including extracted patterns and metrics
        """
        ...
        
    @abstractmethod
    def query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the pattern-aware RAG system.
        
        Args:
            query: The query to process
            context: Optional context for the query
            
        Returns:
            Query results including relevant patterns and generated response
        """
        ...
        
    @abstractmethod
    def get_patterns(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get patterns from the pattern-aware RAG system.
        
        Args:
            filter_criteria: Optional criteria to filter patterns by
            
        Returns:
            A list of matching patterns
        """
        ...
        
    @abstractmethod
    def get_field_state(self) -> Dict[str, Any]:
        """
        Get the current state of the semantic field.
        
        Returns:
            The current field state
        """
        ...
        
    @abstractmethod
    def add_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a pattern to the pattern-aware RAG system.
        
        Args:
            pattern: The pattern to add
            
        Returns:
            The added pattern with any generated IDs or timestamps
        """
        ...
        
    @abstractmethod
    def update_pattern(self, pattern_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a pattern in the pattern-aware RAG system.
        
        Args:
            pattern_id: The ID of the pattern to update
            updates: The updates to apply to the pattern
            
        Returns:
            The updated pattern
        """
        ...
        
    @abstractmethod
    def delete_pattern(self, pattern_id: str) -> bool:
        """
        Delete a pattern from the pattern-aware RAG system.
        
        Args:
            pattern_id: The ID of the pattern to delete
            
        Returns:
            True if the pattern was deleted, False otherwise
        """
        ...
        
    @abstractmethod
    def create_relationship(self, source_id: str, target_id: str, 
                           relationship_type: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a relationship between two patterns.
        
        Args:
            source_id: The ID of the source pattern
            target_id: The ID of the target pattern
            relationship_type: The type of relationship
            metadata: Optional metadata for the relationship
            
        Returns:
            The ID of the created relationship
        """
        ...
        
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from the pattern-aware RAG system.
        
        Returns:
            Current system metrics
        """
        ...
