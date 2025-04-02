"""
Adapters for Vector-Tonic Persistence Integration.

This package contains adapters that bridge the gap between the Vector-Tonic Window system
and the ArangoDB persistence layer.

NOTE: The repository adapters have been moved to habitat_evolution.adaptive_core.persistence.adapters.
This module now imports from there for backward compatibility.
"""

# Import adapters from their new location
from habitat_evolution.adaptive_core.persistence.adapters.field_state_repository_adapter import FieldStateRepositoryAdapter
from habitat_evolution.adaptive_core.persistence.adapters.pattern_repository_adapter import PatternRepositoryAdapter
from habitat_evolution.adaptive_core.persistence.adapters.relationship_repository_adapter import RelationshipRepositoryAdapter
from habitat_evolution.adaptive_core.persistence.adapters.topology_repository_adapter import TopologyRepositoryAdapter

__all__ = [
    'FieldStateRepositoryAdapter',
    'PatternRepositoryAdapter',
    'RelationshipRepositoryAdapter',
    'TopologyRepositoryAdapter'
]
