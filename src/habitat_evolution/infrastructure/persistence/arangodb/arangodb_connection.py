"""
ArangoDB connection implementation for Habitat Evolution.

This module provides a concrete implementation of the ArangoDBConnectionInterface,
enabling consistent access to ArangoDB throughout the application.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import os
from arango import ArangoClient
from arango.database import Database
from arango.collection import Collection
from arango.graph import Graph
from arango.exceptions import (
    ArangoServerError,
    DocumentGetError,
    DocumentInsertError,
    DocumentUpdateError,
    DocumentDeleteError,
    GraphCreateError
)

from src.habitat_evolution.infrastructure.interfaces.persistence.arangodb_connection_interface import ArangoDBConnectionInterface
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface

logger = logging.getLogger(__name__)


class ArangoDBConnection(ArangoDBConnectionInterface):
    """
    Implementation of the ArangoDBConnectionInterface for Habitat Evolution.
    
    This class provides a consistent approach to ArangoDB access,
    abstracting the details of connection management and query execution.
    It supports the pattern evolution and co-evolution principles of Habitat
    by enabling flexible data persistence while maintaining a consistent interface.
    
    The connection also supports vector-tonic field persistence for statistical pattern analysis,
    enabling the storage and retrieval of time-series patterns, vector field representations,
    and cross-domain resonance metrics between semantic and statistical patterns.
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 8529,
                 username: str = "root",
                 password: str = "",
                 database_name: str = "habitat_evolution",
                 event_service: Optional[EventServiceInterface] = None):
        """
        Initialize a new ArangoDB connection.
        
        Args:
            host: The ArangoDB host
            port: The ArangoDB port
            username: The ArangoDB username
            password: The ArangoDB password
            database_name: The name of the database to connect to
            event_service: Optional event service for publishing events
        """
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._database_name = database_name
        self._event_service = event_service
        self._client = None
        self._db = None
        self._initialized = False
        logger.debug("ArangoDBConnection created")
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the ArangoDB connection with the specified configuration.
        
        Args:
            config: Optional configuration for the connection
        """
        if self._initialized:
            logger.warning("ArangoDBConnection already initialized")
            return
            
        logger.info("Initializing ArangoDBConnection")
        
        # Apply configuration if provided
        if config:
            if "host" in config:
                self._host = config["host"]
            if "port" in config:
                self._port = config["port"]
            if "username" in config:
                self._username = config["username"]
            if "password" in config:
                self._password = config["password"]
            if "database_name" in config:
                self._database_name = config["database_name"]
        
        # Create client
        self._client = ArangoClient(
            hosts=f"http://{self._host}:{self._port}"
        )
        
        # Connect to system database to ensure our database exists
        sys_db = self._client.db(
            name="_system",
            username=self._username,
            password=self._password
        )
        
        # Create database if it doesn't exist
        if not sys_db.has_database(self._database_name):
            logger.info(f"Creating database: {self._database_name}")
            sys_db.create_database(self._database_name)
        
        # Connect to our database
        self._db = self._client.db(
            name=self._database_name,
            username=self._username,
            password=self._password
        )
        
        self._initialized = True
        logger.info("ArangoDBConnection initialized")
        
        if self._event_service:
            self._event_service.publish("arangodb.connection_initialized", {
                "database": self._database_name,
                "host": self._host,
                "port": self._port
            })
    
    def shutdown(self) -> None:
        """
        Release resources when shutting down the connection.
        """
        if not self._initialized:
            logger.warning("ArangoDBConnection not initialized")
            return
            
        logger.info("Shutting down ArangoDBConnection")
        self._client = None
        self._db = None
        self._initialized = False
        logger.info("ArangoDBConnection shut down")
        
        if self._event_service:
            self._event_service.publish("arangodb.connection_shutdown", {
                "database": self._database_name
            })
    
    def get_database(self) -> Database:
        """
        Get the ArangoDB database object.
        
        Returns:
            The ArangoDB database object
        """
        if not self._initialized:
            self.initialize()
            
        return self._db
    
    def ensure_collection(self, collection_name: str) -> Collection:
        """
        Ensure that a collection exists, creating it if necessary.
        
        Args:
            collection_name: The name of the collection
            
        Returns:
            The collection object
        """
        if not self._initialized:
            self.initialize()
            
        if not self._db.has_collection(collection_name):
            logger.info(f"Creating collection: {collection_name}")
            self._db.create_collection(collection_name)
            
            if self._event_service:
                self._event_service.publish("arangodb.collection_created", {
                    "collection": collection_name
                })
                
        return self._db.collection(collection_name)
    
    def ensure_edge_collection(self, collection_name: str) -> Collection:
        """
        Ensure that an edge collection exists, creating it if necessary.
        
        Args:
            collection_name: The name of the edge collection
            
        Returns:
            The edge collection object
        """
        if not self._initialized:
            self.initialize()
            
        if not self._db.has_collection(collection_name):
            logger.info(f"Creating edge collection: {collection_name}")
            self._db.create_collection(collection_name, edge=True)
            
            if self._event_service:
                self._event_service.publish("arangodb.edge_collection_created", {
                    "collection": collection_name
                })
                
        return self._db.collection(collection_name)
    
    def ensure_graph(self, graph_name: str, 
                    edge_definitions: List[Dict[str, Any]]) -> Graph:
        """
        Ensure that a graph exists, creating it if necessary.
        
        Args:
            graph_name: The name of the graph
            edge_definitions: The edge definitions for the graph
            
        Returns:
            The graph object
        """
        if not self._initialized:
            self.initialize()
            
        if not self._db.has_graph(graph_name):
            logger.info(f"Creating graph: {graph_name}")
            
            # Ensure all collections exist
            for edge_def in edge_definitions:
                self.ensure_edge_collection(edge_def["collection"])
                
                for from_col in edge_def["from"]:
                    self.ensure_collection(from_col)
                    
                for to_col in edge_def["to"]:
                    self.ensure_collection(to_col)
            
            # Create graph
            self._db.create_graph(graph_name, edge_definitions)
            
            if self._event_service:
                self._event_service.publish("arangodb.graph_created", {
                    "graph": graph_name,
                    "edge_definitions": edge_definitions
                })
                
        return self._db.graph(graph_name)
    
    def get_document(self, collection_name: str, document_id: str) -> Dict[str, Any]:
        """
        Get a document by ID.
        
        Args:
            collection_name: The name of the collection
            document_id: The ID of the document
            
        Returns:
            The document
            
        Raises:
            DocumentGetError: If the document doesn't exist
        """
        if not self._initialized:
            self.initialize()
            
        collection = self._db.collection(collection_name)
        
        # Handle both full IDs and keys
        if "/" in document_id:
            key = document_id.split("/")[1]
        else:
            key = document_id
            
        return collection.get(key)
    
    def insert(self, collection_name: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert a document into a collection.
        
        Args:
            collection_name: The name of the collection
            document: The document to insert
            
        Returns:
            The inserted document
            
        Raises:
            DocumentInsertError: If the document couldn't be inserted
        """
        if not self._initialized:
            self.initialize()
            
        collection = self._db.collection(collection_name)
        meta = collection.insert(document, return_new=True)
        return meta["new"]
    
    def update_document(self, collection_name: str, document_id: str, 
                       document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a document.
        
        Args:
            collection_name: The name of the collection
            document_id: The ID of the document
            document: The updated document
            
        Returns:
            The updated document
            
        Raises:
            DocumentUpdateError: If the document couldn't be updated
        """
        if not self._initialized:
            self.initialize()
            
        collection = self._db.collection(collection_name)
        
        # Handle both full IDs and keys
        if "/" in document_id:
            key = document_id.split("/")[1]
        else:
            key = document_id
            
        meta = collection.update(document, return_new=True)
        return meta["new"]
    
    def delete_document(self, collection_name: str, document_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            collection_name: The name of the collection
            document_id: The ID of the document
            
        Returns:
            True if the document was deleted, False otherwise
            
        Raises:
            DocumentDeleteError: If the document couldn't be deleted
        """
        if not self._initialized:
            self.initialize()
            
        collection = self._db.collection(collection_name)
        
        # Handle both full IDs and keys
        if "/" in document_id:
            key = document_id.split("/")[1]
        else:
            key = document_id
            
        collection.delete(key)
        return True
    
    def execute_query(self, query: str, 
                     bind_vars: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute an AQL query.
        
        Args:
            query: The AQL query to execute
            bind_vars: Optional bind variables for the query
            
        Returns:
            The query results
        """
        if not self._initialized:
            self.initialize()
            
        cursor = self._db.aql.execute(query, bind_vars=bind_vars or {})
        return [doc for doc in cursor]
    
    def begin_transaction(self) -> str:
        """
        Begin a transaction.
        
        Returns:
            The transaction ID
        """
        if not self._initialized:
            self.initialize()
            
        transaction = self._db.begin_transaction()
        return transaction.id
    
    def commit_transaction(self, transaction_id: str) -> bool:
        """
        Commit a transaction.
        
        Args:
            transaction_id: The ID of the transaction to commit
            
        Returns:
            True if the transaction was committed, False otherwise
        """
        if not self._initialized:
            self.initialize()
            
        transaction = self._db.transaction(transaction_id)
        transaction.commit()
        return True
    
    def abort_transaction(self, transaction_id: str) -> bool:
        """
        Abort a transaction.
        
        Args:
            transaction_id: The ID of the transaction to abort
            
        Returns:
            True if the transaction was aborted, False otherwise
        """
        if not self._initialized:
            self.initialize()
            
        transaction = self._db.transaction(transaction_id)
        transaction.abort()
        return True
        
    def connect(self) -> bool:
        """
        Connect to the ArangoDB server.
        
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            if self._initialized:
                logger.debug("Already connected to ArangoDB")
                return True
                
            self.initialize()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to ArangoDB: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the ArangoDB server.
        
        Returns:
            True if disconnected successfully, False otherwise
        """
        try:
            self.shutdown()
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from ArangoDB: {e}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if connected to the ArangoDB server.
        
        Returns:
            True if connected, False otherwise
        """
        if not self._initialized or not self._db:
            return False
            
        try:
            # Simple ping to check connection
            self._db.ping()
            return True
        except Exception:
            return False
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in ArangoDB.
        
        Args:
            collection_name: The name of the collection
            
        Returns:
            True if the collection exists, False otherwise
        """
        if not self._initialized:
            self.initialize()
            
        return self._db.has_collection(collection_name)
        
    def graph_exists(self, graph_name: str) -> bool:
        """
        Check if a graph exists in ArangoDB.
        
        Args:
            graph_name: The name of the graph
            
        Returns:
            True if the graph exists, False otherwise
        """
        if not self._initialized:
            self.initialize()
            
        try:
            return self._db.has_graph(graph_name)
        except Exception as e:
            logger.error(f"Error checking if graph exists: {e}")
            return False
    
    def create_collection(self, collection_name: str, edge: bool = False) -> Any:
        """
        Create a collection in ArangoDB.
        
        Args:
            collection_name: The name of the collection to create
            edge: Whether the collection is an edge collection
            
        Returns:
            The created collection
        """
        if not self._initialized:
            self.initialize()
            
        if edge:
            return self.ensure_edge_collection(collection_name)
        else:
            return self.ensure_collection(collection_name)
    
    def get_collection(self, collection_name: str) -> Any:
        """
        Get a collection from ArangoDB.
        
        Args:
            collection_name: The name of the collection to get
            
        Returns:
            The collection
        """
        if not self._initialized:
            self.initialize()
            
        return self._db.collection(collection_name)
    
    def create_index(self, collection_name: str, index_type: str, fields: List[str], 
                     unique: bool = False, sparse: bool = False, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create an index on a collection.
        
        Args:
            collection_name: The name of the collection
            index_type: The type of index (e.g., 'hash', 'skiplist', 'fulltext')
            fields: The fields to index
            unique: Whether the index is unique
            sparse: Whether the index is sparse
            name: Optional name for the index
            
        Returns:
            The index information
        """
        if not self._initialized:
            self.initialize()
            
        collection = self._db.collection(collection_name)
        
        if index_type == "hash":
            return collection.add_hash_index(fields, unique=unique, sparse=sparse, name=name)
        elif index_type == "skiplist":
            return collection.add_skiplist_index(fields, unique=unique, sparse=sparse, name=name)
        elif index_type == "fulltext":
            return collection.add_fulltext_index(fields, name=name)
        elif index_type == "geo":
            return collection.add_geo_index(fields, name=name)
        elif index_type == "persistent":
            return collection.add_persistent_index(fields, unique=unique, sparse=sparse, name=name)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
    
    def create_graph(self, graph_name: str, edge_definitions: List[Dict[str, Any]]) -> Any:
        """
        Create a graph in ArangoDB.
        
        Args:
            graph_name: The name of the graph to create
            edge_definitions: The edge definitions for the graph
            
        Returns:
            The created graph
        """
        return self.ensure_graph(graph_name, edge_definitions)
    
    def get_graph(self, graph_name: str) -> Any:
        """
        Get a graph from ArangoDB.
        
        Args:
            graph_name: The name of the graph to get
            
        Returns:
            The graph
        """
        if not self._initialized:
            self.initialize()
            
        return self._db.graph(graph_name)
    
    def create_vertex(self, graph_name: str, collection_name: str, vertex: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a vertex in a graph.
        
        Args:
            graph_name: The name of the graph
            collection_name: The name of the vertex collection
            vertex: The vertex data
            
        Returns:
            The created vertex
        """
        if not self._initialized:
            self.initialize()
            
        graph = self._db.graph(graph_name)
        vertex_collection = graph.vertex_collection(collection_name)
        meta = vertex_collection.insert(vertex, return_new=True)
        return meta["new"]
    
    def create_edge(self, graph_name: str, edge_collection: str, from_id: str, to_id: str, 
                   edge: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an edge in a graph.
        
        Args:
            graph_name: The name of the graph
            edge_collection: The name of the edge collection
            from_id: The ID of the source vertex
            to_id: The ID of the target vertex
            edge: The edge data
            
        Returns:
            The created edge
        """
        if not self._initialized:
            self.initialize()
            
        graph = self._db.graph(graph_name)
        edge_col = graph.edge_collection(edge_collection)
        
        # Add _from and _to if not present
        if "_from" not in edge:
            edge["_from"] = from_id
        if "_to" not in edge:
            edge["_to"] = to_id
            
        meta = edge_col.insert(edge, return_new=True)
        return meta["new"]
    
    def execute_aql(self, query: str, bind_vars: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute an AQL query.
        
        Args:
            query: The AQL query to execute
            bind_vars: Optional bind variables for the query
            
        Returns:
            The query results
        """
        return self.execute_query(query, bind_vars)
