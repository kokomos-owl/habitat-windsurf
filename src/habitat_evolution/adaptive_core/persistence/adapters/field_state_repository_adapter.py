"""
Field State Repository Adapter for Vector-Tonic Persistence Integration.

This module provides an adapter that implements the FieldStateRepositoryInterface
using the existing TonicHarmonicFieldStateRepository implementation.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from habitat_evolution.adaptive_core.persistence.interfaces.field_state_repository import FieldStateRepositoryInterface
from habitat_evolution.pattern_aware_rag.persistence.arangodb.field_state_repository import TonicHarmonicFieldStateRepository

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
    
    def find_by_timestamp(self, timestamp: datetime) -> List[Dict[str, Any]]:
        """
        Find field states by their timestamp.
        
        Args:
            timestamp: The timestamp to search for.
            
        Returns:
            A list of field states with the specified timestamp.
        """
        try:
            return self.repository.find_by_timestamp(timestamp)
        except Exception as e:
            logger.error(f"Failed to find field states for timestamp {timestamp}: {str(e)}")
            return []
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest field state.
        
        Returns:
            The latest field state if available, None otherwise.
        """
        try:
            return self.repository.get_latest()
        except Exception as e:
            logger.error(f"Failed to get latest field state: {str(e)}")
            return None
    
    def find_by_coherence_range(self, min_coherence: float, max_coherence: float) -> List[Dict[str, Any]]:
        """
        Find field states within a coherence range.
        
        Args:
            min_coherence: The minimum coherence value.
            max_coherence: The maximum coherence value.
            
        Returns:
            A list of field states within the specified coherence range.
        """
        try:
            return self.repository.find_by_coherence_range(min_coherence, max_coherence)
        except Exception as e:
            logger.error(f"Failed to find field states by coherence range: {str(e)}")
            return []
    
    def find_by_stability_range(self, min_stability: float, max_stability: float) -> List[Dict[str, Any]]:
        """
        Find field states within a stability range.
        
        Args:
            min_stability: The minimum stability value.
            max_stability: The maximum stability value.
            
        Returns:
            A list of field states within the specified stability range.
        """
        try:
            return self.repository.find_by_stability_range(min_stability, max_stability)
        except Exception as e:
            logger.error(f"Failed to find field states by stability range: {str(e)}")
            return []
    
    def find_by_time_range(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Find field states within a time range.
        
        Args:
            start_time: The start of the time range.
            end_time: The end of the time range.
            
        Returns:
            A list of field states within the specified time range.
        """
        try:
            return self.repository.find_by_time_range(start_time, end_time)
        except Exception as e:
            logger.error(f"Failed to find field states by time range: {str(e)}")
            return []
    
    def find_by_pattern_id(self, pattern_id: str) -> List[Dict[str, Any]]:
        """
        Find field states that contain a specific pattern.
        
        Args:
            pattern_id: The ID of the pattern to search for.
            
        Returns:
            A list of field states that contain the specified pattern.
        """
        try:
            return self.repository.find_by_pattern_id(pattern_id)
        except Exception as e:
            logger.error(f"Failed to find field states for pattern {pattern_id}: {str(e)}")
            return []
