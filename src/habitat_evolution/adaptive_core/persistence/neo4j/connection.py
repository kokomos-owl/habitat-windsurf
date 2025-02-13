"""
Neo4j connection management for the Adaptive Core system.
"""

import os
from typing import Optional
from neo4j import GraphDatabase, Driver, Session
from dotenv import load_dotenv

class Neo4jConnectionManager:
    """
    Manages Neo4j database connections and sessions.
    Implements singleton pattern to ensure single connection pool.
    """
    _instance: Optional['Neo4jConnectionManager'] = None
    _driver: Optional[Driver] = None

    def __new__(cls) -> 'Neo4jConnectionManager':
        if cls._instance is None:
            cls._instance = super(Neo4jConnectionManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the Neo4j connection"""
        load_dotenv()
        
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        user = os.getenv('NEO4J_USER', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'testpass')
        
        try:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            # Verify connection
            self._driver.verify_connectivity()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {str(e)}")

    def get_session(self) -> Session:
        """Get a new session from the connection pool"""
        if not self._driver:
            raise ConnectionError("Neo4j driver not initialized")
        return self._driver.session()

    def close(self) -> None:
        """Close the driver and all associated sessions"""
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> 'Neo4jConnectionManager':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
