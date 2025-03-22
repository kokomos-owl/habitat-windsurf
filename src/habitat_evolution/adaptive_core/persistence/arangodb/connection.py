"""
ArangoDB connection management for the Adaptive Core system.
"""

import os
from typing import Optional, Dict, Any
from arango import ArangoClient
from dotenv import load_dotenv

class ArangoDBConnectionManager:
    """
    Manages ArangoDB database connections and sessions.
    Implements singleton pattern to ensure single connection pool.
    """
    _instance: Optional['ArangoDBConnectionManager'] = None
    _client: Optional[ArangoClient] = None
    _db = None  # ArangoDB database instance

    def __new__(cls) -> 'ArangoDBConnectionManager':
        if cls._instance is None:
            cls._instance = super(ArangoDBConnectionManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the ArangoDB connection"""
        load_dotenv()
        
        # Get connection details from environment variables or use defaults
        host = os.getenv('ARANGO_HOST', 'http://localhost:8529')
        username = os.getenv('ARANGO_USER', 'root')
        password = os.getenv('ARANGO_PASSWORD', 'habitat')
        db_name = os.getenv('ARANGO_DB', 'habitat')
        
        try:
            # Initialize the client
            self._client = ArangoClient(hosts=host)
            
            # Connect to the database
            self._db = self._client.db(db_name, username=username, password=password)
            
            # Verify connection by making a simple query
            self._db.properties()
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to ArangoDB: {str(e)}")

    def get_db(self):
        """Get the ArangoDB database instance"""
        if not self._db:
            raise ConnectionError("ArangoDB database not initialized")
        return self._db
    
    def get_collection(self, collection_name: str):
        """Get a collection by name"""
        if not self._db:
            raise ConnectionError("ArangoDB database not initialized")
        
        if not self._db.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist")
            
        return self._db.collection(collection_name)
    
    def get_edge_collection(self, collection_name: str):
        """Get an edge collection by name"""
        if not self._db:
            raise ConnectionError("ArangoDB database not initialized")
        
        if not self._db.has_collection(collection_name):
            raise ValueError(f"Edge collection {collection_name} does not exist")
            
        return self._db.collection(collection_name)
    
    def execute_query(self, query: str, bind_vars: Optional[Dict[str, Any]] = None):
        """Execute an AQL query and return the results"""
        if not self._db:
            raise ConnectionError("ArangoDB database not initialized")
            
        cursor = self._db.aql.execute(query, bind_vars=bind_vars or {})
        return cursor
    
    def close(self) -> None:
        """Close the client connection"""
        # ArangoDB Python driver automatically manages connection pooling
        # No explicit close needed, but we'll reset our references
        self._client = None
        self._db = None

    def __enter__(self) -> 'ArangoDBConnectionManager':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
