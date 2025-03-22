"""
ArangoDB persistence layer for Habitat Evolution.
Provides database access for domain-predicate tracking and pattern evolution.
"""

from .connection import ArangoDBConnectionManager
from .base_repository import ArangoDBBaseRepository
