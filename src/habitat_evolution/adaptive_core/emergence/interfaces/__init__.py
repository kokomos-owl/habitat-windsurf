"""
Vector-Tonic Persistence Integration Interfaces.

This package contains interfaces for the Vector-Tonic Persistence Integration,
which connects the Vector-Tonic Window system with the ArangoDB persistence layer.
These interfaces define the contracts for repositories, observers, and services
that support the pattern evolution and co-evolution principles of Habitat Evolution.

NOTE: The repository interfaces have been moved to habitat_evolution.adaptive_core.persistence.interfaces.
This module now imports from there for backward compatibility.
"""

# Import repository interfaces from their new location
from habitat_evolution.adaptive_core.persistence.interfaces.field_state_repository import FieldStateRepositoryInterface
from habitat_evolution.adaptive_core.persistence.interfaces.pattern_repository import PatternRepositoryInterface
from habitat_evolution.adaptive_core.persistence.interfaces.relationship_repository import RelationshipRepositoryInterface
from habitat_evolution.adaptive_core.persistence.interfaces.topology_repository import TopologyRepositoryInterface

# Import observer interfaces from their current location
from src.habitat_evolution.adaptive_core.emergence.interfaces.learning_window_observer import LearningWindowObserverInterface, LearningWindowState
from src.habitat_evolution.adaptive_core.emergence.interfaces.pattern_observer import PatternObserverInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.field_observer import FieldObserverInterface

__all__ = [
    'FieldStateRepositoryInterface',
    'PatternRepositoryInterface',
    'RelationshipRepositoryInterface',
    'TopologyRepositoryInterface',
    'LearningWindowObserverInterface',
    'LearningWindowState',
    'PatternObserverInterface',
    'FieldObserverInterface',
]
