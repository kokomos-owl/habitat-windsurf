"""
Database connection interface for Habitat Evolution.

This module defines the interface for database connections in the Habitat Evolution system,
providing a consistent approach to database access across the application.
"""

from typing import Protocol, Dict, List, Any, Optional
from abc import abstractmethod


class DatabaseConnectionInterface(Protocol):
    """
    Interface for database connections in Habitat Evolution.
    
    Database connections provide a consistent approach to database access,
    abstracting the underlying database technology and providing methods for
    connecting to and interacting with the database. This supports the pattern
    evolution and co-evolution principles of Habitat by enabling flexible
    database mechanisms while maintaining a consistent interface.
    """
    
    @abstractmethod
    def connect(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Connect to the database.
        
        Args:
            config: Optional configuration for the connection
        """
        ...
        
    @abstractmethod
    def disconnect(self) -> None:
        """
        Disconnect from the database.
        """
        ...
        
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if the connection is active.
        
        Returns:
            True if connected, False otherwise
        """
        ...
        
    @abstractmethod
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a query on the database.
        
        Args:
            query: The query to execute
            parameters: Optional parameters for the query
            
        Returns:
            The query results
        """
        ...
        
    @abstractmethod
    def get_database(self) -> Any:
        """
        Get the underlying database object.
        
        Returns:
            The database object
        """
        ...
