"""
Storage interfaces for adaptive core.

This module defines the core storage interfaces used by the adaptive system.
These interfaces abstract away specific storage implementations, allowing
for different backends while maintaining consistent behavior.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TypeVar, Generic
from datetime import datetime
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class StorageMetadata:
    """Metadata for stored items."""
    created_at: datetime
    updated_at: datetime
    version: str
    tags: List[str]

class StorageResult(Generic[T]):
    """Result from a storage operation."""
    def __init__(self, 
                 success: bool,
                 data: Optional[T] = None,
                 error: Optional[str] = None,
                 metadata: Optional[StorageMetadata] = None):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata

class StateStore(ABC):
    """Interface for state persistence."""
    
    @abstractmethod
    async def save_state(self, 
                        id: str,
                        state: Dict[str, Any],
                        metadata: Optional[Dict[str, Any]] = None) -> StorageResult[str]:
        """Save state for an entity.
        
        Args:
            id: Entity identifier
            state: State to save
            metadata: Optional metadata about the state
            
        Returns:
            StorageResult with operation status
        """
        pass
    
    @abstractmethod
    async def load_state(self,
                        id: str,
                        version: Optional[str] = None) -> StorageResult[Dict[str, Any]]:
        """Load state for an entity.
        
        Args:
            id: Entity identifier
            version: Optional specific version to load
            
        Returns:
            StorageResult containing the state if found
        """
        pass
    
    @abstractmethod
    async def list_versions(self,
                          id: str) -> StorageResult[List[StorageMetadata]]:
        """List available versions for an entity.
        
        Args:
            id: Entity identifier
            
        Returns:
            StorageResult containing list of version metadata
        """
        pass

class PatternStore(ABC):
    """Interface for pattern persistence."""
    
    @abstractmethod
    async def save_pattern(self,
                          pattern: Dict[str, Any],
                          metadata: Optional[Dict[str, Any]] = None) -> StorageResult[str]:
        """Save a pattern.
        
        Args:
            pattern: Pattern to save
            metadata: Optional metadata about the pattern
            
        Returns:
            StorageResult with operation status
        """
        pass
    
    @abstractmethod
    async def find_patterns(self,
                          query: Dict[str, Any],
                          limit: Optional[int] = None,
                          offset: Optional[int] = None) -> StorageResult[List[Dict[str, Any]]]:
        """Find patterns matching query.
        
        Args:
            query: Search criteria
            limit: Optional maximum results
            offset: Optional starting offset
            
        Returns:
            StorageResult containing matching patterns
        """
        pass
    
    @abstractmethod
    async def delete_pattern(self,
                           id: str) -> StorageResult[bool]:
        """Delete a pattern.
        
        Args:
            id: Pattern identifier
            
        Returns:
            StorageResult indicating success
        """
        pass

class RelationshipStore(ABC):
    """Interface for relationship persistence."""
    
    @abstractmethod
    async def save_relationship(self,
                              source_id: str,
                              target_id: str,
                              type: str,
                              properties: Dict[str, Any],
                              metadata: Optional[Dict[str, Any]] = None) -> StorageResult[str]:
        """Save a relationship.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            type: Relationship type
            properties: Relationship properties
            metadata: Optional metadata
            
        Returns:
            StorageResult with operation status
        """
        pass
    
    @abstractmethod
    async def find_relationships(self,
                               query: Dict[str, Any],
                               limit: Optional[int] = None,
                               offset: Optional[int] = None) -> StorageResult[List[Dict[str, Any]]]:
        """Find relationships matching query.
        
        Args:
            query: Search criteria
            limit: Optional maximum results
            offset: Optional starting offset
            
        Returns:
            StorageResult containing matching relationships
        """
        pass
    
    @abstractmethod
    async def get_related(self,
                         id: str,
                         type: Optional[str] = None,
                         direction: Optional[str] = None) -> StorageResult[List[Dict[str, Any]]]:
        """Get entities related to an ID.
        
        Args:
            id: Entity identifier
            type: Optional relationship type filter
            direction: Optional 'incoming' or 'outgoing' filter
            
        Returns:
            StorageResult containing related entities
        """
        pass
