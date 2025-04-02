"""
Persistence layer for the Habitat Evolution system.

This package provides the persistence layer for the Habitat Evolution system,
including repository interfaces, adapters, and implementations for various
database technologies.
"""

from habitat_evolution.adaptive_core.persistence.factory import create_repositories

__all__ = ['create_repositories']
