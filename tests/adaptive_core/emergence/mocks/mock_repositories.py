"""
Mock repository implementations for testing.

This module provides mock implementations of the repository interfaces
used by the VectorTonicPersistenceConnector for testing purposes.
"""

from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock

from src.habitat_evolution.adaptive_core.emergence.interfaces.field_state_repository import FieldStateRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.pattern_repository import PatternRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.relationship_repository import RelationshipRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.topology_repository import TopologyRepositoryInterface


class MockFieldStateRepository(FieldStateRepositoryInterface):
    """Mock implementation of FieldStateRepositoryInterface for testing."""
    
    def __init__(self):
        """Initialize the mock repository."""
        self.field_states = {}
        
        # Create mock methods that can be spied on
        self.save = MagicMock(side_effect=self._save)
        self.get_by_id = MagicMock(side_effect=self._get_by_id)
        self.update_coherence = MagicMock(side_effect=self._update_coherence)
        self.update_stability = MagicMock(side_effect=self._update_stability)
        self.update_density_centers = MagicMock(side_effect=self._update_density_centers)
        self.update_eigenspace = MagicMock(side_effect=self._update_eigenspace)
    
    def _save(self, field_state: Dict[str, Any]) -> str:
        """Save a field state to the repository."""
        field_id = field_state.get("id")
        if not field_id:
            raise ValueError("Field state must have an id")
        
        self.field_states[field_id] = field_state
        return field_id
    
    def _get_by_id(self, field_id: str) -> Optional[Dict[str, Any]]:
        """Get a field state by its ID."""
        return self.field_states.get(field_id)
    
    def _update_coherence(self, field_id: str, coherence: float) -> bool:
        """Update the coherence of a field state."""
        if field_id not in self.field_states:
            return False
        
        self.field_states[field_id]["coherence"] = coherence
        return True
    
    def _update_stability(self, field_id: str, stability: float) -> bool:
        """Update the stability of a field state."""
        if field_id not in self.field_states:
            return False
        
        self.field_states[field_id]["stability"] = stability
        return True
    
    def _update_density_centers(self, field_id: str, density_centers: List[List[float]]) -> bool:
        """Update the density centers of a field state."""
        if field_id not in self.field_states:
            return False
        
        self.field_states[field_id]["density_centers"] = density_centers
        return True
    
    def _update_eigenspace(self, field_id: str, eigenspace: Dict[str, Any]) -> bool:
        """Update the eigenspace of a field state."""
        if field_id not in self.field_states:
            return False
        
        self.field_states[field_id]["eigenspace"] = eigenspace
        return True


class MockPatternRepository(PatternRepositoryInterface):
    """Mock implementation of PatternRepositoryInterface for testing."""
    
    def __init__(self):
        """Initialize the mock repository."""
        self.patterns = {}
        
        # Create mock methods that can be spied on
        self.save = MagicMock(side_effect=self._save)
        self.get_by_id = MagicMock(side_effect=self._get_by_id)
        self.update_quality = MagicMock(side_effect=self._update_quality)
        self.get_all = MagicMock(side_effect=self._get_all)
    
    def _save(self, pattern: Dict[str, Any]) -> str:
        """Save a pattern to the repository."""
        pattern_id = pattern.get("id")
        if not pattern_id:
            raise ValueError("Pattern must have an id")
        
        self.patterns[pattern_id] = pattern
        return pattern_id
    
    def _get_by_id(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get a pattern by its ID."""
        return self.patterns.get(pattern_id)
    
    def _update_quality(self, pattern_id: str, quality_data: Dict[str, Any]) -> bool:
        """Update the quality metrics of a pattern."""
        if pattern_id not in self.patterns:
            return False
        
        for key, value in quality_data.items():
            self.patterns[pattern_id][key] = value
        
        return True
    
    def _get_all(self) -> List[Dict[str, Any]]:
        """Get all patterns."""
        return list(self.patterns.values())


class MockRelationshipRepository(RelationshipRepositoryInterface):
    """Mock implementation of RelationshipRepositoryInterface for testing."""
    
    def __init__(self):
        """Initialize the mock repository."""
        self.relationships = {}
        
        # Create mock methods that can be spied on
        self.save = MagicMock(side_effect=self._save)
        self.get_by_id = MagicMock(side_effect=self._get_by_id)
        self.get_by_source = MagicMock(side_effect=self._get_by_source)
        self.get_by_target = MagicMock(side_effect=self._get_by_target)
    
    def _save(self, relationship: Dict[str, Any]) -> str:
        """Save a relationship to the repository."""
        source_id = relationship.get("source_id")
        target_id = relationship.get("target_id")
        
        if not source_id or not target_id:
            raise ValueError("Relationship must have source_id and target_id")
        
        relationship_id = f"{source_id}_{target_id}"
        relationship["id"] = relationship_id
        self.relationships[relationship_id] = relationship
        
        return relationship_id
    
    def _get_by_id(self, relationship_id: str) -> Optional[Dict[str, Any]]:
        """Get a relationship by its ID."""
        return self.relationships.get(relationship_id)
    
    def _get_by_source(self, source_id: str) -> List[Dict[str, Any]]:
        """Get relationships by source ID."""
        return [r for r in self.relationships.values() if r.get("source_id") == source_id]
    
    def _get_by_target(self, target_id: str) -> List[Dict[str, Any]]:
        """Get relationships by target ID."""
        return [r for r in self.relationships.values() if r.get("target_id") == target_id]


class MockTopologyRepository(TopologyRepositoryInterface):
    """Mock implementation of TopologyRepositoryInterface for testing."""
    
    def __init__(self):
        """Initialize the mock repository."""
        self.topologies = {}
        
        # Create mock methods that can be spied on
        self.save = MagicMock(side_effect=self._save)
        self.get_by_field_id = MagicMock(side_effect=self._get_by_field_id)
    
    def _save(self, topology_data: Dict[str, Any]) -> str:
        """Save topology data to the repository."""
        field_id = topology_data.get("field_id")
        if not field_id:
            raise ValueError("Topology data must have a field_id")
        
        topology_id = topology_data.get("id", f"topology_{field_id}")
        topology_data["id"] = topology_id
        self.topologies[topology_id] = topology_data
        
        return topology_id
    
    def _get_by_field_id(self, field_id: str) -> Optional[Dict[str, Any]]:
        """Get topology data by field ID."""
        for topology in self.topologies.values():
            if topology.get("field_id") == field_id:
                return topology
        
        return None
