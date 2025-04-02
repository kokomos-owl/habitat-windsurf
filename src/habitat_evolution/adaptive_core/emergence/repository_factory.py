"""
Factory for creating repository instances.

This module provides factory methods for creating repository instances
that implement the repository interfaces defined in the interfaces package.
"""

import logging
from typing import Optional, Dict, Any

from src.habitat_evolution.adaptive_core.emergence.interfaces.field_state_repository import FieldStateRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.pattern_repository import PatternRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.relationship_repository import RelationshipRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.topology_repository import TopologyRepositoryInterface

logger = logging.getLogger(__name__)


def create_field_state_repository(db_connection: Any, config: Optional[Dict[str, Any]] = None) -> FieldStateRepositoryInterface:
    """
    Create a field state repository instance.
    
    Args:
        db_connection: The database connection to use.
        config: Optional configuration for the repository.
        
    Returns:
        A field state repository instance.
    
    Raises:
        ImportError: If the repository implementation cannot be imported.
        ValueError: If the repository implementation cannot be created.
    """
    try:
        # Import the repository implementation
        from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.field_state_repository import FieldStateRepository
        
        # Create and return the repository instance
        return FieldStateRepository(db_connection, config)
    except ImportError as e:
        logger.error(f"Failed to import FieldStateRepository: {str(e)}")
        raise ImportError(f"Failed to import FieldStateRepository: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to create FieldStateRepository: {str(e)}")
        raise ValueError(f"Failed to create FieldStateRepository: {str(e)}")


def create_pattern_repository(db_connection: Any, config: Optional[Dict[str, Any]] = None) -> PatternRepositoryInterface:
    """
    Create a pattern repository instance.
    
    Args:
        db_connection: The database connection to use.
        config: Optional configuration for the repository.
        
    Returns:
        A pattern repository instance.
    
    Raises:
        ImportError: If the repository implementation cannot be imported.
        ValueError: If the repository implementation cannot be created.
    """
    try:
        # Import the repository implementation
        from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.pattern_repository import PatternRepository
        
        # Create and return the repository instance
        return PatternRepository(db_connection, config)
    except ImportError as e:
        logger.error(f"Failed to import PatternRepository: {str(e)}")
        raise ImportError(f"Failed to import PatternRepository: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to create PatternRepository: {str(e)}")
        raise ValueError(f"Failed to create PatternRepository: {str(e)}")


def create_relationship_repository(db_connection: Any, config: Optional[Dict[str, Any]] = None) -> RelationshipRepositoryInterface:
    """
    Create a relationship repository instance.
    
    Args:
        db_connection: The database connection to use.
        config: Optional configuration for the repository.
        
    Returns:
        A relationship repository instance.
    
    Raises:
        ImportError: If the repository implementation cannot be imported.
        ValueError: If the repository implementation cannot be created.
    """
    try:
        # Import the repository implementation
        from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.relationship_repository import RelationshipRepository
        
        # Create and return the repository instance
        return RelationshipRepository(db_connection, config)
    except ImportError as e:
        logger.error(f"Failed to import RelationshipRepository: {str(e)}")
        raise ImportError(f"Failed to import RelationshipRepository: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to create RelationshipRepository: {str(e)}")
        raise ValueError(f"Failed to create RelationshipRepository: {str(e)}")


def create_topology_repository(db_connection: Any, config: Optional[Dict[str, Any]] = None) -> TopologyRepositoryInterface:
    """
    Create a topology repository instance.
    
    Args:
        db_connection: The database connection to use.
        config: Optional configuration for the repository.
        
    Returns:
        A topology repository instance.
    
    Raises:
        ImportError: If the repository implementation cannot be imported.
        ValueError: If the repository implementation cannot be created.
    """
    try:
        # Import the repository implementation
        from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.topology_repository import TopologyRepository
        
        # Create and return the repository instance
        return TopologyRepository(db_connection, config)
    except ImportError as e:
        logger.error(f"Failed to import TopologyRepository: {str(e)}")
        raise ImportError(f"Failed to import TopologyRepository: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to create TopologyRepository: {str(e)}")
        raise ValueError(f"Failed to create TopologyRepository: {str(e)}")


def create_repositories(db_connection: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create all repository instances.
    
    Args:
        db_connection: The database connection to use.
        config: Optional configuration for the repositories.
        
    Returns:
        A dictionary containing all repository instances.
    """
    repositories = {}
    
    try:
        repositories["field_state_repository"] = create_field_state_repository(db_connection, config)
    except Exception as e:
        logger.warning(f"Failed to create field_state_repository: {str(e)}")
    
    try:
        repositories["pattern_repository"] = create_pattern_repository(db_connection, config)
    except Exception as e:
        logger.warning(f"Failed to create pattern_repository: {str(e)}")
    
    try:
        repositories["relationship_repository"] = create_relationship_repository(db_connection, config)
    except Exception as e:
        logger.warning(f"Failed to create relationship_repository: {str(e)}")
    
    try:
        repositories["topology_repository"] = create_topology_repository(db_connection, config)
    except Exception as e:
        logger.warning(f"Failed to create topology_repository: {str(e)}")
    
    return repositories
