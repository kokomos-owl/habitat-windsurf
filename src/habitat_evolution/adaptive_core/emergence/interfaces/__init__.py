"""
Vector-Tonic Persistence Integration Interfaces.

This package contains interfaces for the Vector-Tonic Persistence Integration,
which connects the Vector-Tonic Window system with the ArangoDB persistence layer.
These interfaces define the contracts for repositories, observers, and services
that support the pattern evolution and co-evolution principles of Habitat Evolution.
"""

from src.habitat_evolution.adaptive_core.emergence.interfaces.field_state_repository import FieldStateRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.pattern_repository import PatternRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.relationship_repository import RelationshipRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.topology_repository import TopologyRepositoryInterface
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
