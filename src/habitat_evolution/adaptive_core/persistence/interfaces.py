"""
Core persistence interfaces for the Adaptive Core system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TypeVar, Generic

T = TypeVar('T')

class Repository(Generic[T], ABC):
    """Generic repository interface"""
    
    @abstractmethod
    def create(self, entity: T) -> str:
        """Create a new entity"""
        pass
    
    @abstractmethod
    def read(self, entity_id: str) -> Optional[T]:
        """Read an entity by ID"""
        pass
    
    @abstractmethod
    def update(self, entity: T) -> None:
        """Update an entity"""
        pass
    
    @abstractmethod
    def delete(self, entity_id: str) -> None:
        """Delete an entity"""
        pass
    
    @abstractmethod
    def list(self, filter_params: Optional[Dict[str, Any]] = None) -> List[T]:
        """List entities with optional filtering"""
        pass

class PatternRepository(Repository['Pattern'], ABC):
    """Pattern-specific repository interface"""
    
    @abstractmethod
    def get_by_concept(self, base_concept: str) -> List['Pattern']:
        """Get patterns by base concept"""
        pass
    
    @abstractmethod
    def get_by_creator(self, creator_id: str) -> List['Pattern']:
        """Get patterns by creator"""
        pass
    
    @abstractmethod
    def get_by_coherence_range(self, min_coherence: float, max_coherence: float) -> List['Pattern']:
        """Get patterns within a coherence range"""
        pass

class RelationshipRepository(Repository['Relationship'], ABC):
    """Relationship-specific repository interface"""
    
    @abstractmethod
    def get_by_source(self, source_id: str) -> List['Relationship']:
        """Get relationships by source pattern"""
        pass
    
    @abstractmethod
    def get_by_target(self, target_id: str) -> List['Relationship']:
        """Get relationships by target pattern"""
        pass
    
    @abstractmethod
    def get_by_type(self, relationship_type: str) -> List['Relationship']:
        """Get relationships by type"""
        pass

class MetricsRepository(Repository['PatternMetrics'], ABC):
    """Metrics-specific repository interface"""
    
    @abstractmethod
    def get_metrics_history(self, pattern_id: str) -> List['PatternMetrics']:
        """Get metrics history for a pattern"""
        pass
    
    @abstractmethod
    def get_latest_metrics(self, pattern_id: str) -> Optional['PatternMetrics']:
        """Get latest metrics for a pattern"""
        pass

class StateRepository(ABC):
    """State management repository interface"""
    
    @abstractmethod
    def create_version(self, entity_id: str, state: Dict[str, Any]) -> str:
        """Create a new version"""
        pass
    
    @abstractmethod
    def get_version(self, entity_id: str, version_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific version"""
        pass
    
    @abstractmethod
    def list_versions(self, entity_id: str) -> List[str]:
        """List all versions for an entity"""
        pass
    
    @abstractmethod
    def get_latest_version(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest version for an entity"""
        pass

class EventRepository(Repository['Event'], ABC):
    """Event-specific repository interface"""
    
    @abstractmethod
    def get_by_type(self, event_type: str) -> List['Event']:
        """Get events by type"""
        pass
    
    @abstractmethod
    def get_by_entity(self, entity_id: str) -> List['Event']:
        """Get events by entity"""
        pass
    
    @abstractmethod
    def get_in_timerange(self, start_time: str, end_time: str) -> List['Event']:
        """Get events within a time range"""
        pass
