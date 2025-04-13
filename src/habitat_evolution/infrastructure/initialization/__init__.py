"""
Initialization package for Habitat Evolution.

This package provides utilities for initializing components in the correct order
and tracking dependencies between components.
"""

from .dependency_tracker import (
    DependencyTracker,
    get_dependency_tracker,
    verify_initialization,
    verify_dependencies,
    initialize_with_dependencies
)

__all__ = [
    'DependencyTracker',
    'get_dependency_tracker',
    'verify_initialization',
    'verify_dependencies',
    'initialize_with_dependencies'
]
