"""
Topology Repository Adapter for Vector-Tonic Persistence Integration.

This module provides an adapter that implements the TopologyRepositoryInterface
using the existing topology repository implementation.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.habitat_evolution.adaptive_core.emergence.interfaces.topology_repository import TopologyRepositoryInterface
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.topology_repository import TopologyRepository

logger = logging.getLogger(__name__)


class TopologyRepositoryAdapter(TopologyRepositoryInterface):
    """
    Adapter that implements TopologyRepositoryInterface using TopologyRepository.
    
    This adapter bridges the gap between the Vector-Tonic Window system and
    the ArangoDB persistence layer for topology constructs.
    """
    
    def __init__(self, db_connection=None, config=None):
        """
        Initialize the adapter with a database connection and configuration.
        
        Args:
            db_connection: The database connection to use.
            config: Optional configuration for the repository.
        """
        self.db_connection = db_connection
        self.config = config or {}
        
        # Create the underlying repository
        self.repository = TopologyRepository()
        if db_connection:
            self.repository.connection_manager.db = db_connection
    
    def save(self, topology_data: Dict[str, Any]) -> str:
        """
        Save topology data to the repository.
        
        Args:
            topology_data: The topology data to save.
            
        Returns:
            The ID of the saved topology data.
        """
        try:
            # Ensure the topology data has a field ID
            if "field_id" not in topology_data:
                raise ValueError("Topology data must have a field_id")
            
            # Ensure the topology data has an ID
            if "id" not in topology_data:
                topology_data["id"] = f"topology_{topology_data['field_id']}"
            
            # Save the topology data
            return self.repository.save(topology_data)
        except Exception as e:
            logger.error(f"Failed to save topology data: {str(e)}")
            raise
    
    def find_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Find topology data by its ID.
        
        Args:
            id: The ID of the topology data to find.
            
        Returns:
            The topology data if found, None otherwise.
        """
        try:
            return self.repository.find_by_id(id)
        except Exception as e:
            logger.error(f"Failed to find topology data with ID {id}: {str(e)}")
            return None
    
    def find_by_field_id(self, field_id: str) -> Optional[Dict[str, Any]]:
        """
        Find topology data by field ID.
        
        Args:
            field_id: The ID of the field.
            
        Returns:
            The topology data if found, None otherwise.
        """
        try:
            return self.repository.find_by_field_id(field_id)
        except Exception as e:
            logger.error(f"Failed to find topology data for field {field_id}: {str(e)}")
            return None
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all topology data.
        
        Returns:
            A list of all topology data.
        """
        try:
            return self.repository.get_all()
        except Exception as e:
            logger.error(f"Failed to get all topology data: {str(e)}")
            return []
    
    def delete(self, id: str) -> bool:
        """
        Delete topology data by its ID.
        
        Args:
            id: The ID of the topology data to delete.
            
        Returns:
            True if the deletion was successful, False otherwise.
        """
        try:
            return self.repository.delete(id)
        except Exception as e:
            logger.error(f"Failed to delete topology data {id}: {str(e)}")
            return False
