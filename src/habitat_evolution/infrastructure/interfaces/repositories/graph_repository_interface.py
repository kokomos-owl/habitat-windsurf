"""
Graph repository interface for Habitat Evolution.

This module defines the interface for graph repositories in the Habitat Evolution system,
providing a consistent approach to graph data access across the application.
"""

from typing import Protocol, Dict, List, Any, Optional, TypeVar, Generic
from abc import abstractmethod

from src.habitat_evolution.infrastructure.interfaces.repositories.repository_interface import RepositoryInterface

T = TypeVar('T')


class GraphRepositoryInterface(RepositoryInterface[T], Protocol):
    """
    Interface for graph repositories in Habitat Evolution.
    
    Graph repositories provide a consistent approach to graph data access,
    abstracting the underlying graph database technology and providing
    domain-specific methods for retrieving and persisting graph entities.
    This supports the pattern evolution and co-evolution principles of Habitat
    by enabling flexible graph storage mechanisms while maintaining a consistent interface.
    """
    
    @abstractmethod
    def create_node(self, node_type: str, properties: Dict[str, Any]) -> str:
        """
        Create a node in the graph.
        
        Args:
            node_type: The type of node to create
            properties: The properties of the node
            
        Returns:
            The ID of the created node
        """
        ...
        
    @abstractmethod
    def create_edge(self, edge_type: str, from_node_id: str, to_node_id: str, 
                   properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Create an edge between two nodes in the graph.
        
        Args:
            edge_type: The type of edge to create
            from_node_id: The ID of the source node
            to_node_id: The ID of the target node
            properties: Optional properties of the edge
            
        Returns:
            The ID of the created edge
        """
        ...
        
    @abstractmethod
    def get_node(self, node_id: str) -> Dict[str, Any]:
        """
        Get a node by its ID.
        
        Args:
            node_id: The ID of the node to get
            
        Returns:
            The node as a dictionary
        """
        ...
        
    @abstractmethod
    def get_nodes_by_type(self, node_type: str, 
                         properties: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get nodes by type and optional properties.
        
        Args:
            node_type: The type of nodes to get
            properties: Optional properties to filter by
            
        Returns:
            A list of matching nodes
        """
        ...
        
    @abstractmethod
    def get_edges(self, from_node_id: Optional[str] = None, 
                 to_node_id: Optional[str] = None,
                 edge_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get edges by source node, target node, and/or edge type.
        
        Args:
            from_node_id: Optional ID of the source node
            to_node_id: Optional ID of the target node
            edge_type: Optional type of edges to get
            
        Returns:
            A list of matching edges
        """
        ...
        
    @abstractmethod
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a node's properties.
        
        Args:
            node_id: The ID of the node to update
            properties: The new properties of the node
            
        Returns:
            The updated node
        """
        ...
        
    @abstractmethod
    def update_edge(self, edge_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an edge's properties.
        
        Args:
            edge_id: The ID of the edge to update
            properties: The new properties of the edge
            
        Returns:
            The updated edge
        """
        ...
        
    @abstractmethod
    def delete_node(self, node_id: str) -> bool:
        """
        Delete a node from the graph.
        
        Args:
            node_id: The ID of the node to delete
            
        Returns:
            True if the node was deleted, False otherwise
        """
        ...
        
    @abstractmethod
    def delete_edge(self, edge_id: str) -> bool:
        """
        Delete an edge from the graph.
        
        Args:
            edge_id: The ID of the edge to delete
            
        Returns:
            True if the edge was deleted, False otherwise
        """
        ...
        
    @abstractmethod
    def execute_graph_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a graph query.
        
        Args:
            query: The query to execute
            parameters: Optional parameters for the query
            
        Returns:
            The query results
        """
        ...
