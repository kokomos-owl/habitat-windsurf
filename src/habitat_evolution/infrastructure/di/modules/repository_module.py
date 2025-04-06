"""
Repository module for Habitat Evolution DI framework.

This module provides registrations for repository implementations in Habitat Evolution,
organizing them in a clean, consistent way that aligns with the principles
of pattern evolution and co-evolution.
"""

from typing import Dict, Any, Optional

from src.habitat_evolution.infrastructure.di.module import Module
from src.habitat_evolution.infrastructure.interfaces.repositories.repository_interface import RepositoryInterface
from src.habitat_evolution.infrastructure.interfaces.repositories.graph_repository_interface import GraphRepositoryInterface
from src.habitat_evolution.infrastructure.interfaces.persistence.arangodb_connection_interface import ArangoDBConnectionInterface
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_repository import ArangoDBRepository
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_graph_repository import ArangoDBGraphRepository
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_pattern_repository import ArangoDBPatternRepository, Pattern


def create_repository_module() -> Module:
    """
    Create a module for registering repository implementations.
    
    Returns:
        A module configured with repository registrations
    """
    module = Module("Repositories")
    
    # Register the ArangoDBPatternRepository as a singleton
    # This is a factory registration that creates the repository with dependencies
    def create_pattern_repository(container):
        db_connection = container.resolve(ArangoDBConnectionInterface)
        event_service = container.resolve(EventServiceInterface)
        return ArangoDBPatternRepository(db_connection, event_service)
    
    module.register_singleton(
        ArangoDBPatternRepository,
        factory=create_pattern_repository
    )
    
    # Register additional repositories as they are implemented
    
    return module
