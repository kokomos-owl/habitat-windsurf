"""
Pattern Topology module for detecting, managing, and articulating topological features
in the semantic landscape of the pattern co-evolution system.

This package provides tools for identifying frequency domains, boundaries, resonance points,
and field dynamics that emerge from pattern interactions, as well as persisting and
retrieving these topological constructs.
"""

from habitat_evolution.pattern_aware_rag.topology.models import (
    FrequencyDomain, Boundary, ResonancePoint, FieldMetrics, TopologyState, TopologyDiff
)
from habitat_evolution.pattern_aware_rag.topology.detector import TopologyDetector
from habitat_evolution.pattern_aware_rag.topology.manager import TopologyManager

__all__ = [
    'FrequencyDomain', 
    'Boundary', 
    'ResonancePoint', 
    'FieldMetrics', 
    'TopologyState', 
    'TopologyDiff',
    'TopologyDetector',
    'TopologyManager'
]
