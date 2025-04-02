"""
Field State Repository Adapter for Vector-Tonic Persistence Integration.

This module provides an adapter that implements the FieldStateRepositoryInterface
using the existing TonicHarmonicFieldStateRepository implementation.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.habitat_evolution.adaptive_core.emergence.interfaces.field_state_repository import FieldStateRepositoryInterface
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.field_state_repository import TonicHarmonicFieldStateRepository

logger = logging.getLogger(__name__)


class FieldStateRepositoryAdapter(FieldStateRepositoryInterface):
    """
    Adapter that implements FieldStateRepositoryInterface using TonicHarmonicFieldStateRepository.
    
    This adapter bridges the gap between the Vector-Tonic Window system and
    the ArangoDB persistence layer for field states.
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
        self.repository = TonicHarmonicFieldStateRepository()
        if db_connection:
            self.repository.connection_manager.db = db_connection
    
    def save(self, field_state: Dict[str, Any]) -> str:
        """
        Save a field state to the repository.
        
        Args:
            field_state: The field state to save.
            
        Returns:
            The ID of the saved field state.
        """
        try:
            # Ensure the field state has an ID
            if "id" not in field_state:
                field_state["id"] = str(uuid.uuid4())
            
            # Save the field state
            return self.repository.save(field_state)
        except Exception as e:
            logger.error(f"Failed to save field state: {str(e)}")
            raise
    
    def find_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Find a field state by its ID.
        
        Args:
            id: The ID of the field state to find.
            
        Returns:
            The field state if found, None otherwise.
        """
        try:
            return self.repository.find_by_id(id)
        except Exception as e:
            logger.error(f"Failed to find field state with ID {id}: {str(e)}")
            return None
    
    def find_by_window_id(self, window_id: str) -> List[Dict[str, Any]]:
        """
        Find field states by window ID.
        
        Args:
            window_id: The ID of the learning window.
            
        Returns:
            A list of field states associated with the window.
        """
        try:
            return self.repository.find_by_window_id(window_id)
        except Exception as e:
            logger.error(f"Failed to find field states for window {window_id}: {str(e)}")
            return []
    
    def update_coherence(self, field_id: str, coherence: float) -> bool:
        """
        Update the coherence of a field state.
        
        Args:
            field_id: The ID of the field state.
            coherence: The new coherence value.
            
        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            field_state = self.find_by_id(field_id)
            if not field_state:
                logger.warning(f"Field state with ID {field_id} not found")
                return False
            
            field_state["coherence"] = coherence
            self.save(field_state)
            return True
        except Exception as e:
            logger.error(f"Failed to update coherence for field {field_id}: {str(e)}")
            return False
    
    def update_stability(self, field_id: str, stability: float) -> bool:
        """
        Update the stability of a field state.
        
        Args:
            field_id: The ID of the field state.
            stability: The new stability value.
            
        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            field_state = self.find_by_id(field_id)
            if not field_state:
                logger.warning(f"Field state with ID {field_id} not found")
                return False
            
            field_state["stability"] = stability
            self.save(field_state)
            return True
        except Exception as e:
            logger.error(f"Failed to update stability for field {field_id}: {str(e)}")
            return False
    
    def update_density_centers(self, field_id: str, density_centers: List[List[float]]) -> bool:
        """
        Update the density centers of a field state.
        
        Args:
            field_id: The ID of the field state.
            density_centers: The new density centers.
            
        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            field_state = self.find_by_id(field_id)
            if not field_state:
                logger.warning(f"Field state with ID {field_id} not found")
                return False
            
            field_state["density_centers"] = density_centers
            self.save(field_state)
            return True
        except Exception as e:
            logger.error(f"Failed to update density centers for field {field_id}: {str(e)}")
            return False
    
    def update_eigenspace(self, field_id: str, eigenspace: Dict[str, Any]) -> bool:
        """
        Update the eigenspace of a field state.
        
        Args:
            field_id: The ID of the field state.
            eigenspace: The new eigenspace properties.
            
        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            field_state = self.find_by_id(field_id)
            if not field_state:
                logger.warning(f"Field state with ID {field_id} not found")
                return False
            
            # Update eigenspace properties
            for key, value in eigenspace.items():
                field_state[key] = value
            
            self.save(field_state)
            return True
        except Exception as e:
            logger.error(f"Failed to update eigenspace for field {field_id}: {str(e)}")
            return False
