"""
ArangoDB Connection Manager for Habitat Evolution.

This module provides a connection manager for ArangoDB, handling connection pooling
and configuration for the Habitat Evolution system.
"""

import os
import logging
from arango import ArangoClient
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ArangoDBConnectionManager:
    """
    Connection manager for ArangoDB.
    
    Handles connection pooling and configuration for ArangoDB connections.
    Uses environment variables for connection settings.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ArangoDBConnectionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the connection manager if not already initialized."""
        if self._initialized:
            return
        
        # Load environment variables
        load_dotenv()
        
        # Get connection details from environment variables
        self.host = os.getenv("ARANGO_HOST", "http://localhost:8529")
        self.username = os.getenv("ARANGO_USER", "root")
        self.password = os.getenv("ARANGO_PASSWORD", "")
        self.db_name = os.getenv("ARANGO_DB", "habitat_evolution")
        
        # Initialize client
        self.client = ArangoClient(hosts=self.host)
        
        # Initialize database connection
        self._db = None
        self._sys_db = None
        
        self._initialized = True
        
        logger.info(f"Initialized ArangoDB connection manager for {self.host}")
    
    def get_sys_db(self):
        """
        Get a connection to the system database.
        
        Returns:
            ArangoDB system database connection
        """
        if not self._sys_db:
            self._sys_db = self.client.db(
                "_system", 
                username=self.username, 
                password=self.password
            )
        return self._sys_db
    
    def get_db(self):
        """
        Get a connection to the application database.
        
        Creates the database if it doesn't exist.
        
        Returns:
            ArangoDB database connection
        """
        if not self._db:
            # Connect to system database first
            sys_db = self.get_sys_db()
            
            # Create database if it doesn't exist
            if not sys_db.has_database(self.db_name):
                logger.info(f"Creating database: {self.db_name}")
                sys_db.create_database(self.db_name)
            
            # Connect to application database
            self._db = self.client.db(
                self.db_name, 
                username=self.username, 
                password=self.password
            )
        
        return self._db
    
    def close(self):
        """Close all database connections."""
        self._db = None
        self._sys_db = None
        logger.info("Closed all ArangoDB connections")
