"""
ArangoDB connection interface for Habitat Evolution.

This module defines the interface for ArangoDB connections in the Habitat Evolution system,
providing a consistent approach to ArangoDB access across the application.
"""

from typing import Protocol, Dict, List, Any, Optional, Union
from abc import abstractmethod

from src.habitat_evolution.infrastructure.interfaces.persistence.database_connection_interface import DatabaseConnectionInterface


class ArangoDBConnectionInterface(DatabaseConnectionInterface, Protocol):
    """
    Interface for ArangoDB connections in Habitat Evolution.
    
    ArangoDB connections provide a consistent approach to ArangoDB access,
    abstracting the details of the connection and providing methods specific
    to ArangoDB operations. This supports the pattern evolution and co-evolution
    principles of Habitat by enabling flexible graph operations while maintaining
    a consistent interface.
    """
    
    @abstractmethod
    def create_collection(self, collection_name: str, edge: bool = False) -> Any:
        """
        Create a collection in ArangoDB.
        
        Args:
            collection_name: The name of the collection to create
            edge: Whether the collection is an edge collection
            
        Returns:
            The created collection
        """
        ...
        
    @abstractmethod
    def get_collection(self, collection_name: str) -> Any:
        """
        Get a collection from ArangoDB.
        
        Args:
            collection_name: The name of the collection to get
            
        Returns:
            The collection
        """
        ...
        
    @abstractmethod
    def create_graph(self, graph_name: str, edge_definitions: Optional[List[Dict[str, Any]]] = None) -> Any:
        """
        Create a graph in ArangoDB.
        
        Args:
            graph_name: The name of the graph to create
            edge_definitions: Optional edge definitions for the graph
            
        Returns:
            The created graph
        """
        ...
        
    @abstractmethod
    def get_graph(self, graph_name: str) -> Any:
        """
        Get a graph from ArangoDB.
        
        Args:
            graph_name: The name of the graph to get
            
        Returns:
            The graph
        """
        ...
        
    @abstractmethod
    def execute_aql(self, query: str, bind_vars: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute an AQL query.
        
        Args:
            query: The AQL query to execute
            bind_vars: Optional bind variables for the query
            
        Returns:
            The query results
        """
        ...
        
    @abstractmethod
    def create_index(self, collection_name: str, index_type: str, 
                    fields: List[str], unique: bool = False) -> Dict[str, Any]:
        """
        Create an index on a collection.
        
        Args:
            collection_name: The name of the collection
            index_type: The type of index to create
            fields: The fields to index
            unique: Whether the index should enforce uniqueness
            
        Returns:
            The created index
        """
        ...
        
    @abstractmethod
    def create_vertex(self, collection_name: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a vertex in a collection.
        
        Args:
            collection_name: The name of the collection
            document: The document to insert
            
        Returns:
            The created vertex
        """
        ...
        
    @abstractmethod
    def create_edge(self, collection_name: str, from_vertex: str, 
                   to_vertex: str, document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create an edge between two vertices.
        
        Args:
            collection_name: The name of the edge collection
            from_vertex: The ID of the source vertex
            to_vertex: The ID of the target vertex
            document: Optional properties for the edge
            
        Returns:
            The created edge
        """
        ...
        
    @abstractmethod
    def begin_transaction(self) -> Any:
        """
        Begin a transaction.
        
        Returns:
            The transaction object
        """
        ...
        
    @abstractmethod
    def commit_transaction(self, transaction: Any) -> bool:
        """
        Commit a transaction.
        
        Args:
            transaction: The transaction to commit
            
        Returns:
            True if the transaction was committed, False otherwise
        """
        ...
        
    @abstractmethod
    def abort_transaction(self, transaction: Any) -> bool:
        """
        Abort a transaction.
        
        Args:
            transaction: The transaction to abort
            
        Returns:
            True if the transaction was aborted, False otherwise
        """
        ...
