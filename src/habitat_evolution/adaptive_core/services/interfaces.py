"""
Core service interfaces for the Adaptive Core system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

@dataclass
class PatternMetrics:
    """Pattern metrics data structure"""
    coherence: float
    signal_strength: float
    phase_stability: float
    flow_metrics: Dict[str, float]

@dataclass
class StabilityMetrics:
    """Stability metrics data structure"""
    overall_stability: float
    phase_stability: float
    amplitude_stability: float
    coherence_stability: float
    flow_stability: Dict[str, float]

@dataclass
class Relationship:
    """Relationship data structure"""
    id: str
    source_id: str
    target_id: str
    type: str
    strength: float
    properties: Dict[str, Any]

@dataclass
class Event:
    """Event data structure"""
    id: str
    type: str
    entity_id: str
    timestamp: str
    payload: Dict[str, Any]

class PatternEvolutionService(ABC):
    @abstractmethod
    def register_pattern(self, pattern_data: Dict[str, Any]) -> str:
        """Register a new pattern and return its ID"""
        pass
        
    @abstractmethod
    def calculate_coherence(self, pattern_id: str) -> float:
        """Calculate coherence for a pattern"""
        pass
        
    @abstractmethod
    def update_pattern_state(self, pattern_id: str, new_state: Dict[str, Any]) -> None:
        """Update pattern state"""
        pass
        
    @abstractmethod
    def get_pattern_metrics(self, pattern_id: str) -> PatternMetrics:
        """Get pattern metrics"""
        pass

class MetricsService(ABC):
    @abstractmethod
    def calculate_wave_metrics(self, pattern_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate wave mechanics metrics"""
        pass
        
    @abstractmethod
    def calculate_field_metrics(self, pattern_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate field theory metrics"""
        pass
        
    @abstractmethod
    def calculate_information_metrics(self, pattern_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate information theory metrics"""
        pass
        
    @abstractmethod
    def calculate_flow_dynamics(self, pattern_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate flow dynamics metrics"""
        pass

class StateManagementService(ABC):
    @abstractmethod
    def create_version(self, entity_id: str, state: Dict[str, Any]) -> str:
        """Create a new version and return version ID"""
        pass
        
    @abstractmethod
    def get_state(self, entity_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
        """Get state for entity at specific version"""
        pass
        
    @abstractmethod
    def list_versions(self, entity_id: str) -> List[str]:
        """List all versions for an entity"""
        pass
        
    @abstractmethod
    def compare_versions(self, entity_id: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions of an entity"""
        pass

class RelationshipService(ABC):
    @abstractmethod
    def create_relationship(self, source_id: str, target_id: str, type: str) -> str:
        """Create a new relationship"""
        pass
        
    @abstractmethod
    def get_relationships(self, entity_id: str) -> List[Relationship]:
        """Get all relationships for an entity"""
        pass
        
    @abstractmethod
    def update_relationship(self, relationship_id: str, properties: Dict[str, Any]) -> None:
        """Update relationship properties"""
        pass
        
    @abstractmethod
    def calculate_relationship_strength(self, relationship_id: str) -> float:
        """Calculate relationship strength"""
        pass

class QualityMetricsService(ABC):
    @abstractmethod
    def calculate_signal_strength(self, pattern_id: str) -> float:
        """Calculate signal strength for a pattern"""
        pass
        
    @abstractmethod
    def measure_coherence_quality(self, pattern_id: str) -> Dict[str, float]:
        """Measure coherence quality metrics"""
        pass
        
    @abstractmethod
    def evaluate_flow_dynamics(self, pattern_id: str) -> Dict[str, float]:
        """Evaluate flow dynamics quality"""
        pass
        
    @abstractmethod
    def assess_pattern_stability(self, pattern_id: str) -> StabilityMetrics:
        """Assess pattern stability"""
        pass

class EventManagementService(ABC):
    @abstractmethod
    def publish_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Publish an event"""
        pass
        
    @abstractmethod
    def subscribe_to_events(self, event_type: str, callback: Callable) -> str:
        """Subscribe to events of a specific type"""
        pass
        
    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events"""
        pass
        
    @abstractmethod
    def get_event_history(self, entity_id: str) -> List[Event]:
        """Get event history for an entity"""
        pass
