"""
Document service interface for Habitat Evolution.

This module defines the interface for document services in the Habitat Evolution system,
providing a consistent approach to document management across the application.
"""

from typing import Protocol, Dict, List, Any, Optional, Union, BinaryIO
from abc import abstractmethod
from datetime import datetime

from src.habitat_evolution.infrastructure.interfaces.service_interface import ServiceInterface


class DocumentServiceInterface(ServiceInterface, Protocol):
    """
    Interface for document services in Habitat Evolution.
    
    Document services provide a consistent approach to document management,
    abstracting the details of document processing, storage, and retrieval.
    This supports the pattern evolution and co-evolution principles of Habitat
    by enabling flexible document handling while maintaining a consistent interface.
    """
    
    @abstractmethod
    def process_document(self, content: Union[str, BinaryIO], 
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document.
        
        Args:
            content: The document content as text or file-like object
            metadata: Optional metadata for the document
            
        Returns:
            Processing results including document ID and extracted information
        """
        ...
        
    @abstractmethod
    def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Get a document by ID.
        
        Args:
            document_id: The ID of the document to get
            
        Returns:
            The document
        """
        ...
        
    @abstractmethod
    def find_documents(self, filter_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find documents matching the filter criteria.
        
        Args:
            filter_criteria: Criteria to filter documents by
            
        Returns:
            A list of matching documents
        """
        ...
        
    @abstractmethod
    def search_documents(self, query: str, 
                        filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search documents by content.
        
        Args:
            query: The search query
            filter_criteria: Optional criteria to filter documents by
            
        Returns:
            A list of matching documents
        """
        ...
        
    @abstractmethod
    def update_document(self, document_id: str, 
                       updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a document.
        
        Args:
            document_id: The ID of the document to update
            updates: The updates to apply to the document
            
        Returns:
            The updated document
        """
        ...
        
    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            document_id: The ID of the document to delete
            
        Returns:
            True if the document was deleted, False otherwise
        """
        ...
        
    @abstractmethod
    def extract_patterns(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Extract patterns from a document.
        
        Args:
            document_id: The ID of the document to extract patterns from
            
        Returns:
            A list of extracted patterns
        """
        ...
        
    @abstractmethod
    def get_document_patterns(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get patterns associated with a document.
        
        Args:
            document_id: The ID of the document to get patterns for
            
        Returns:
            A list of patterns associated with the document
        """
        ...
        
    @abstractmethod
    def get_document_metrics(self, document_id: str) -> Dict[str, Any]:
        """
        Get metrics for a document.
        
        Args:
            document_id: The ID of the document to get metrics for
            
        Returns:
            Metrics for the document
        """
        ...
