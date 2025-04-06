"""
ArangoDB graph repository implementation for Habitat Evolution.

This module provides a concrete implementation of the GraphRepositoryInterface
using ArangoDB as the persistence layer, supporting the pattern evolution and
co-evolution principles of Habitat Evolution.
"""

import logging
from typing import Dict, List, Any, Optional, TypeVar, Generic, Type, Union
from datetime import datetime
import uuid

from src.habitat_evolution.infrastructure.interfaces.repositories.graph_repository_interface import GraphRepositoryInterface
from src.habitat_evolution.infrastructure.interfaces.persistence.arangodb_connection_interface import ArangoDBConnectionInterface
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_repository import ArangoDBRepository

T = TypeVar('T')
logger = logging.getLogger(__name__)


class ArangoDBGraphRepository(ArangoDBRepository[T], GraphRepositoryInterface[T], Generic[T]):
    """
    ArangoDB implementation of the GraphRepositoryInterface.
    
    This repository provides a consistent approach to graph operations using ArangoDB
    as the persistence layer, supporting the pattern evolution and co-evolution
    principles of Habitat Evolution.
    """
    
    def __init__(self, 
                 node_collection_name: str,
                 edge_collection_name: str,
                 graph_name: str,
                 db_connection: ArangoDBConnectionInterface,
                 event_service: EventServiceInterface,
                 entity_class: Type[T]):
        """
        Initialize a new ArangoDB graph repository.
        
        Args:
            node_collection_name: The name of the node collection
            edge_collection_name: The name of the edge collection
            graph_name: The name of the graph
            db_connection: The ArangoDB connection to use
            event_service: The event service for publishing events
            entity_class: The class to use for entity instantiation
        """
        super().__init__(node_collection_name, db_connection, event_service, entity_class)
        self._edge_collection_name = edge_collection_name
        self._graph_name = graph_name
        logger.debug(f"ArangoDBGraphRepository created for graph: {graph_name}")
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the repository with the specified configuration.
        
        Args:
            config: Optional configuration for the repository
        """
        if self._initialized:
            logger.warning(f"ArangoDBGraphRepository for {self._graph_name} already initialized")
            return
            
        logger.info(f"Initializing ArangoDBGraphRepository for {self._graph_name}")
        
        # Ensure collections exist
        self._db_connection.ensure_collection(self._collection_name)
        self._db_connection.ensure_edge_collection(self._edge_collection_name)
        
        # Ensure graph exists
        self._db_connection.ensure_graph(
            self._graph_name,
            edge_definitions=[
                {
                    "collection": self._edge_collection_name,
                    "from": [self._collection_name],
                    "to": [self._collection_name]
                }
            ]
        )
        
        self._initialized = True
        logger.info(f"ArangoDBGraphRepository for {self._graph_name} initialized")
        
        # Publish initialization event
        self._event_service.publish("repository.initialized", {
            "graph": self._graph_name,
            "repository_type": "ArangoDBGraphRepository"
        })
    
    def create_node(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a node in the graph.
        
        Args:
            node_data: The data for the node
            
        Returns:
            The created node
        """
        if not self._initialized:
            self.initialize()
            
        # Generate node ID if not provided
        if '_key' not in node_data:
            node_data['_key'] = str(uuid.uuid4())
            
        # Add metadata if not present
        if 'metadata' not in node_data:
            node_data['metadata'] = {}
            
        # Add timestamps
        current_time = datetime.utcnow().isoformat()
        node_data['created_at'] = current_time
        node_data['updated_at'] = current_time
        
        # Insert node
        result = self._db_connection.insert(self._collection_name, node_data)
        
        # Publish node created event
        self._event_service.publish("graph.node_created", {
            "graph": self._graph_name,
            "node_id": result['_id'],
            "node_type": node_data.get('type', 'unknown')
        })
        
        return result
    
    def create_edge(self, from_node_id: str, to_node_id: str, 
                   edge_type: str, edge_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create an edge between two nodes.
        
        Args:
            from_node_id: The ID of the source node
            to_node_id: The ID of the target node
            edge_type: The type of the edge
            edge_data: Optional data for the edge
            
        Returns:
            The created edge
        """
        if not self._initialized:
            self.initialize()
            
        # Prepare edge document
        edge = edge_data or {}
        edge['_from'] = from_node_id
        edge['_to'] = to_node_id
        edge['type'] = edge_type
        
        # Add metadata if not present
        if 'metadata' not in edge:
            edge['metadata'] = {}
            
        # Add timestamps
        current_time = datetime.utcnow().isoformat()
        edge['created_at'] = current_time
        edge['updated_at'] = current_time
        
        # Insert edge
        result = self._db_connection.insert(self._edge_collection_name, edge)
        
        # Publish edge created event
        self._event_service.publish("graph.edge_created", {
            "graph": self._graph_name,
            "edge_id": result['_id'],
            "edge_type": edge_type,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id
        })
        
        return result
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a node by ID.
        
        Args:
            node_id: The ID of the node
            
        Returns:
            The node, or None if not found
        """
        if not self._initialized:
            self.initialize()
            
        try:
            return self._db_connection.get_document(self._collection_name, node_id)
        except Exception as e:
            logger.error(f"Error getting node {node_id}: {str(e)}")
            return None
    
    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an edge by ID.
        
        Args:
            edge_id: The ID of the edge
            
        Returns:
            The edge, or None if not found
        """
        if not self._initialized:
            self.initialize()
            
        try:
            return self._db_connection.get_document(self._edge_collection_name, edge_id)
        except Exception as e:
            logger.error(f"Error getting edge {edge_id}: {str(e)}")
            return None
    
    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """
        Get nodes by type.
        
        Args:
            node_type: The type of nodes to get
            
        Returns:
            A list of matching nodes
        """
        if not self._initialized:
            self.initialize()
            
        query = f"""
        FOR node IN {self._collection_name}
        FILTER node.type == @node_type
        RETURN node
        """
        
        return self._db_connection.execute_query(query, {"node_type": node_type})
    
    def get_edges(self, from_node_id: Optional[str] = None, 
                 to_node_id: Optional[str] = None,
                 edge_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get edges matching the specified criteria.
        
        Args:
            from_node_id: Optional ID of the source node
            to_node_id: Optional ID of the target node
            edge_type: Optional type of the edge
            
        Returns:
            A list of matching edges
        """
        if not self._initialized:
            self.initialize()
            
        query = f"FOR edge IN {self._edge_collection_name}"
        filters = []
        bind_vars = {}
        
        if from_node_id:
            filters.append("edge._from == @from_node_id")
            bind_vars["from_node_id"] = from_node_id
            
        if to_node_id:
            filters.append("edge._to == @to_node_id")
            bind_vars["to_node_id"] = to_node_id
            
        if edge_type:
            filters.append("edge.type == @edge_type")
            bind_vars["edge_type"] = edge_type
            
        if filters:
            query += " FILTER " + " AND ".join(filters)
            
        query += " RETURN edge"
        
        return self._db_connection.execute_query(query, bind_vars)
    
    def get_neighbors(self, node_id: str, 
                     direction: str = "any", 
                     edge_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get neighbors of a node.
        
        Args:
            node_id: The ID of the node
            direction: The direction of the edges (outbound, inbound, or any)
            edge_type: Optional type of the edges
            
        Returns:
            A list of neighboring nodes
        """
        if not self._initialized:
            self.initialize()
            
        direction = direction.upper()
        if direction not in ["OUTBOUND", "INBOUND", "ANY"]:
            direction = "ANY"
            
        query = f"""
        FOR vertex, edge IN 1..1 {direction} @node_id GRAPH @graph_name
        """
        
        bind_vars = {
            "node_id": node_id,
            "graph_name": self._graph_name
        }
        
        if edge_type:
            query += " FILTER edge.type == @edge_type"
            bind_vars["edge_type"] = edge_type
            
        query += " RETURN vertex"
        
        return self._db_connection.execute_query(query, bind_vars)
    
    def find_path(self, from_node_id: str, to_node_id: str, 
                 max_depth: int = 5) -> List[Dict[str, Any]]:
        """
        Find a path between two nodes.
        
        Args:
            from_node_id: The ID of the source node
            to_node_id: The ID of the target node
            max_depth: The maximum path depth
            
        Returns:
            A list of edges forming the path, or an empty list if no path exists
        """
        if not self._initialized:
            self.initialize()
            
        query = f"""
        FOR path IN 1..@max_depth ANY SHORTEST_PATH @from_node_id TO @to_node_id GRAPH @graph_name
        RETURN path
        """
        
        bind_vars = {
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "max_depth": max_depth,
            "graph_name": self._graph_name
        }
        
        results = self._db_connection.execute_query(query, bind_vars)
        return results[0] if results else []
    
    def execute_graph_query(self, query: str, 
                          params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom graph query.
        
        Args:
            query: The AQL query to execute
            params: Optional parameters for the query
            
        Returns:
            The query results
        """
        if not self._initialized:
            self.initialize()
            
        # Add graph name to params if it's not already there
        if params is None:
            params = {}
            
        if "graph_name" not in params:
            params["graph_name"] = self._graph_name
            
        return self._db_connection.execute_query(query, params)
    
    def update_node(self, node_id: str, 
                   updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update a node.
        
        Args:
            node_id: The ID of the node to update
            updates: The updates to apply to the node
            
        Returns:
            The updated node, or None if not found
        """
        if not self._initialized:
            self.initialize()
            
        try:
            # Get current node
            node = self.get_node(node_id)
            if not node:
                return None
                
            # Apply updates
            for key, value in updates.items():
                if key == "metadata":
                    # Merge metadata
                    if "metadata" not in node:
                        node["metadata"] = {}
                    node["metadata"].update(value)
                else:
                    node[key] = value
                    
            # Update timestamp
            node["updated_at"] = datetime.utcnow().isoformat()
            
            # Update node
            result = self._db_connection.update_document(self._collection_name, node_id, node)
            
            # Publish node updated event
            self._event_service.publish("graph.node_updated", {
                "graph": self._graph_name,
                "node_id": node_id,
                "node_type": node.get("type", "unknown"),
                "updates": list(updates.keys())
            })
            
            return result
        except Exception as e:
            logger.error(f"Error updating node {node_id}: {str(e)}")
            return None
    
    def update_edge(self, edge_id: str, 
                   updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an edge.
        
        Args:
            edge_id: The ID of the edge to update
            updates: The updates to apply to the edge
            
        Returns:
            The updated edge, or None if not found
        """
        if not self._initialized:
            self.initialize()
            
        try:
            # Get current edge
            edge = self.get_edge(edge_id)
            if not edge:
                return None
                
            # Apply updates
            for key, value in updates.items():
                if key == "metadata":
                    # Merge metadata
                    if "metadata" not in edge:
                        edge["metadata"] = {}
                    edge["metadata"].update(value)
                elif key not in ["_from", "_to"]:  # Don't update from/to
                    edge[key] = value
                    
            # Update timestamp
            edge["updated_at"] = datetime.utcnow().isoformat()
            
            # Update edge
            result = self._db_connection.update_document(self._edge_collection_name, edge_id, edge)
            
            # Publish edge updated event
            self._event_service.publish("graph.edge_updated", {
                "graph": self._graph_name,
                "edge_id": edge_id,
                "edge_type": edge.get("type", "unknown"),
                "updates": list(updates.keys())
            })
            
            return result
        except Exception as e:
            logger.error(f"Error updating edge {edge_id}: {str(e)}")
            return None
    
    def delete_node(self, node_id: str, delete_edges: bool = True) -> bool:
        """
        Delete a node.
        
        Args:
            node_id: The ID of the node to delete
            delete_edges: Whether to delete connected edges
            
        Returns:
            True if the node was deleted, False otherwise
        """
        if not self._initialized:
            self.initialize()
            
        try:
            # Delete connected edges if requested
            if delete_edges:
                query = f"""
                FOR edge IN {self._edge_collection_name}
                FILTER edge._from == @node_id OR edge._to == @node_id
                REMOVE edge IN {self._edge_collection_name}
                """
                self._db_connection.execute_query(query, {"node_id": node_id})
            
            # Delete node
            self._db_connection.delete_document(self._collection_name, node_id)
            
            # Publish node deleted event
            self._event_service.publish("graph.node_deleted", {
                "graph": self._graph_name,
                "node_id": node_id
            })
            
            return True
        except Exception as e:
            logger.error(f"Error deleting node {node_id}: {str(e)}")
            return False
    
    def delete_edge(self, edge_id: str) -> bool:
        """
        Delete an edge.
        
        Args:
            edge_id: The ID of the edge to delete
            
        Returns:
            True if the edge was deleted, False otherwise
        """
        if not self._initialized:
            self.initialize()
            
        try:
            # Delete edge
            self._db_connection.delete_document(self._edge_collection_name, edge_id)
            
            # Publish edge deleted event
            self._event_service.publish("graph.edge_deleted", {
                "graph": self._graph_name,
                "edge_id": edge_id
            })
            
            return True
        except Exception as e:
            logger.error(f"Error deleting edge {edge_id}: {str(e)}")
            return False
