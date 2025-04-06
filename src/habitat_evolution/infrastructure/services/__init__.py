"""
Service implementations for Habitat Evolution.

This package contains concrete implementations of the service interfaces
defined in the infrastructure.interfaces.services package, providing the
actual functionality that supports the pattern evolution and co-evolution
principles of Habitat Evolution.
"""

from .event_service import EventService

__all__ = ['EventService']
