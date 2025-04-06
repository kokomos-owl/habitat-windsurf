"""
Document repository interface for Habitat Evolution.

This module defines the interface for document repositories in the Habitat Evolution system,
providing a consistent approach to document persistence across the application.
"""

from typing import Protocol, Dict, List, Any, Optional, TypeVar, Generic
from abc import abstractmethod
from datetime import datetime

from src.habitat_evolution.infrastructure.interfaces.repositories.repository_interface import RepositoryInterface

T = TypeVar('T')


class DocumentRepositoryInterface(RepositoryInterface[T], Protocol):
    """
    Interface for document repositories in Habitat Evolution.
    
    Document repositories provide a consistent approach to document persistence,
    abstracting the details of document storage and retrieval. This supports the
    pattern evolution and co-evolution principles of Habitat by enabling flexible
    document persistence mechanisms while maintaining a consistent interface.
    """
    
    @abstractmethod
    def find_by_title(self, title: str, exact_match: bool = True) -> List[T]:
        """
        Find documents by title.
        
        Args:
            title: The title to search for
            exact_match: Whether to require an exact match
            
        Returns:
            A list of matching documents
        """
        ...
        
    @abstractmethod
    def find_by_content(self, content_query: str) -> List[T]:
        """
        Find documents by content.
        
        Args:
            content_query: The content query to search for
            
        Returns:
            A list of matching documents
        """
        ...
        
    @abstractmethod
    def find_by_metadata(self, metadata_criteria: Dict[str, Any]) -> List[T]:
        """
        Find documents by metadata criteria.
        
        Args:
            metadata_criteria: The metadata criteria to filter by
            
        Returns:
            A list of matching documents
        """
        ...
        
    @abstractmethod
    def find_by_timestamp(self, start_time: datetime, end_time: Optional[datetime] = None) -> List[T]:
        """
        Find documents by timestamp.
        
        Args:
            start_time: The start of the time range
            end_time: The end of the time range (defaults to now)
            
        Returns:
            A list of documents created in the time range
        """
        ...
        
    @abstractmethod
    def find_by_pattern(self, pattern_id: str) -> List[T]:
        """
        Find documents containing a specific pattern.
        
        Args:
            pattern_id: The ID of the pattern to find documents for
            
        Returns:
            A list of documents containing the pattern
        """
        ...
        
    @abstractmethod
    def add_pattern_to_document(self, document_id: str, pattern_id: str, 
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Associate a pattern with a document.
        
        Args:
            document_id: The ID of the document
            pattern_id: The ID of the pattern
            metadata: Optional metadata for the association
            
        Returns:
            True if the association was created, False otherwise
        """
        ...
        
    @abstractmethod
    def remove_pattern_from_document(self, document_id: str, pattern_id: str) -> bool:
        """
        Remove a pattern association from a document.
        
        Args:
            document_id: The ID of the document
            pattern_id: The ID of the pattern
            
        Returns:
            True if the association was removed, False otherwise
        """
        ...
        
    @abstractmethod
    def get_document_patterns(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get patterns associated with a document.
        
        Args:
            document_id: The ID of the document
            
        Returns:
            A list of patterns associated with the document
        """
        ...
        
    @abstractmethod
    def update_document_metadata(self, document_id: str, metadata: Dict[str, Any]) -> T:
        """
        Update a document's metadata.
        
        Args:
            document_id: The ID of the document
            metadata: The new metadata
            
        Returns:
            The updated document
        """
        ...
