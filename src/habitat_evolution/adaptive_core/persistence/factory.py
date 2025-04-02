"""
Factory for creating repository instances.

This module provides factory methods for creating repository instances
that implement the repository interfaces defined in the interfaces package.
"""

import logging
from typing import Optional, Dict, Any

from src.habitat_evolution.adaptive_core.persistence.interfaces.field_state_repository import FieldStateRepositoryInterface
from src.habitat_evolution.adaptive_core.persistence.interfaces.pattern_repository import PatternRepositoryInterface
from src.habitat_evolution.adaptive_core.persistence.interfaces.relationship_repository import RelationshipRepositoryInterface
from src.habitat_evolution.adaptive_core.persistence.interfaces.topology_repository import TopologyRepositoryInterface

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
        # Import the adapter implementation
        from src.habitat_evolution.adaptive_core.persistence.adapters.field_state_repository_adapter import FieldStateRepositoryAdapter
        
        # Create and return the repository instance
        return FieldStateRepositoryAdapter(db_connection, config)
    except ImportError as e:
        logger.error(f"Failed to import FieldStateRepositoryAdapter: {str(e)}")
        raise ImportError(f"Failed to import FieldStateRepositoryAdapter: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to create FieldStateRepositoryAdapter: {str(e)}")
        raise ValueError(f"Failed to create FieldStateRepositoryAdapter: {str(e)}")


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
        # Import the adapter implementation
        from src.habitat_evolution.adaptive_core.persistence.adapters.pattern_repository_adapter import PatternRepositoryAdapter
        
        # Create and return the repository instance
        return PatternRepositoryAdapter(db_connection, config)
    except ImportError as e:
        logger.error(f"Failed to import PatternRepositoryAdapter: {str(e)}")
        raise ImportError(f"Failed to import PatternRepositoryAdapter: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to create PatternRepositoryAdapter: {str(e)}")
        raise ValueError(f"Failed to create PatternRepositoryAdapter: {str(e)}")


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
        # Import the adapter implementation
        from src.habitat_evolution.adaptive_core.persistence.adapters.relationship_repository_adapter import RelationshipRepositoryAdapter
        
        # Create and return the repository instance
        return RelationshipRepositoryAdapter(db_connection, config)
    except ImportError as e:
        logger.error(f"Failed to import RelationshipRepositoryAdapter: {str(e)}")
        raise ImportError(f"Failed to import RelationshipRepositoryAdapter: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to create RelationshipRepositoryAdapter: {str(e)}")
        raise ValueError(f"Failed to create RelationshipRepositoryAdapter: {str(e)}")


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
        # Import the adapter implementation
        from src.habitat_evolution.adaptive_core.persistence.adapters.topology_repository_adapter import TopologyRepositoryAdapter
        
        # Create and return the repository instance
        return TopologyRepositoryAdapter(db_connection, config)
    except ImportError as e:
        logger.error(f"Failed to import TopologyRepositoryAdapter: {str(e)}")
        raise ImportError(f"Failed to import TopologyRepositoryAdapter: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to create TopologyRepositoryAdapter: {str(e)}")
        raise ValueError(f"Failed to create TopologyRepositoryAdapter: {str(e)}")


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
