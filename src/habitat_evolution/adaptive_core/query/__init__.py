"""
Query module for the Habitat Evolution system.

This module implements a query interaction system that treats queries as first-class
actants, allowing them to participate in semantic relationships and transformations
across modalities and AI systems.
"""

from .query_actant import QueryActant
from .query_interaction import QueryInteraction

__all__ = ['QueryActant', 'QueryInteraction']
