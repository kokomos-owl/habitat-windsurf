"""
Adapters for Vector-Tonic Persistence Integration.

This package contains adapters that bridge the gap between the Vector-Tonic Window system
and the ArangoDB persistence layer.
"""

from src.habitat_evolution.adaptive_core.emergence.adapters.field_state_repository_adapter import FieldStateRepositoryAdapter
from src.habitat_evolution.adaptive_core.emergence.adapters.pattern_repository_adapter import PatternRepositoryAdapter
from src.habitat_evolution.adaptive_core.emergence.adapters.relationship_repository_adapter import RelationshipRepositoryAdapter
from src.habitat_evolution.adaptive_core.emergence.adapters.topology_repository_adapter import TopologyRepositoryAdapter

__all__ = [
    'FieldStateRepositoryAdapter',
    'PatternRepositoryAdapter',
    'RelationshipRepositoryAdapter',
    'TopologyRepositoryAdapter'
]
