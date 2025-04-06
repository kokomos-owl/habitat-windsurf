"""
Configuration interface for Habitat Evolution.

This module defines the interface for configuration in the Habitat Evolution system,
providing a consistent approach to service configuration across the application.
"""

from typing import Protocol, Dict, Any, Optional, List, TypeVar, Generic
from abc import abstractmethod

T = TypeVar('T')


class ConfigurationInterface(Protocol):
    """
    Interface for configuration in Habitat Evolution.
    
    Configuration provides a consistent approach to service configuration,
    abstracting the details of configuration storage and retrieval.
    This supports the pattern evolution and co-evolution principles of Habitat
    by enabling flexible configuration while maintaining a consistent interface.
    """
    
    @abstractmethod
    def get(self, key: str, default: Optional[T] = None) -> T:
        """
        Get a configuration value by key.
        
        Args:
            key: The key of the configuration value to get
            default: The default value to return if the key is not found
            
        Returns:
            The configuration value
        """
        ...
        
    @abstractmethod
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a configuration section.
        
        Args:
            section: The name of the section to get
            
        Returns:
            The configuration section
        """
        ...
        
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: The key of the configuration value to set
            value: The value to set
        """
        ...
        
    @abstractmethod
    def has(self, key: str) -> bool:
        """
        Check if a configuration key exists.
        
        Args:
            key: The key to check
            
        Returns:
            True if the key exists, False otherwise
        """
        ...
        
    @abstractmethod
    def load(self, source: Any) -> None:
        """
        Load configuration from a source.
        
        Args:
            source: The source to load configuration from
        """
        ...
        
    @abstractmethod
    def save(self, destination: Any) -> None:
        """
        Save configuration to a destination.
        
        Args:
            destination: The destination to save configuration to
        """
        ...
        
    @abstractmethod
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            All configuration values
        """
        ...
