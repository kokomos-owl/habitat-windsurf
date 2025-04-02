"""
Pattern Repository Adapter for Vector-Tonic Persistence Integration.

This module provides an adapter that implements the PatternRepositoryInterface
using the existing PatternRepository implementation.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from habitat_evolution.adaptive_core.persistence.interfaces.pattern_repository import PatternRepositoryInterface
from habitat_evolution.pattern_aware_rag.persistence.arangodb.pattern_repository import PatternRepository

logger = logging.getLogger(__name__)


class PatternRepositoryAdapter(PatternRepositoryInterface):
    """
    Adapter that implements PatternRepositoryInterface using PatternRepository.
    
    This adapter bridges the gap between the Vector-Tonic Window system and
    the ArangoDB persistence layer for patterns.
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
        self.repository = PatternRepository()
        if db_connection:
            self.repository.connection_manager.db = db_connection
    
    def save(self, pattern: Dict[str, Any]) -> str:
        """
        Save a pattern to the repository.
        
        Args:
            pattern: The pattern to save.
            
        Returns:
            The ID of the saved pattern.
        """
        try:
            # Ensure the pattern has an ID
            if "id" not in pattern:
                pattern["id"] = str(uuid.uuid4())
            
            # Save the pattern
            return self.repository.save(pattern)
        except Exception as e:
            logger.error(f"Failed to save pattern: {str(e)}")
            raise
    
    def find_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Find a pattern by its ID.
        
        Args:
            id: The ID of the pattern to find.
            
        Returns:
            The pattern if found, None otherwise.
        """
        try:
            return self.repository.find_by_id(id)
        except Exception as e:
            logger.error(f"Failed to find pattern with ID {id}: {str(e)}")
            return None
    
    def find_by_window_id(self, window_id: str) -> List[Dict[str, Any]]:
        """
        Find patterns by window ID.
        
        Args:
            window_id: The ID of the learning window.
            
        Returns:
            A list of patterns associated with the window.
        """
        try:
            return self.repository.find_by_window_id(window_id)
        except Exception as e:
            logger.error(f"Failed to find patterns for window {window_id}: {str(e)}")
            return []
    
    def find_by_vector_similarity(self, vector: List[float], threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Find patterns by vector similarity.
        
        Args:
            vector: The vector to compare with.
            threshold: The similarity threshold.
            
        Returns:
            A list of patterns similar to the given vector.
        """
        try:
            return self.repository.find_by_vector_similarity(vector, threshold)
        except Exception as e:
            logger.error(f"Failed to find patterns by vector similarity: {str(e)}")
            return []
    
    def update_quality(self, pattern_id: str, quality_data: Dict[str, Any]) -> bool:
        """
        Update the quality metrics of a pattern.
        
        Args:
            pattern_id: The ID of the pattern.
            quality_data: The new quality metrics.
            
        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            pattern = self.find_by_id(pattern_id)
            if not pattern:
                logger.warning(f"Pattern with ID {pattern_id} not found")
                return False
            
            # Update quality metrics
            for key, value in quality_data.items():
                pattern[key] = value
            
            self.save(pattern)
            return True
        except Exception as e:
            logger.error(f"Failed to update quality for pattern {pattern_id}: {str(e)}")
            return False
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all patterns.
        
        Returns:
            A list of all patterns.
        """
        try:
            return self.repository.get_all()
        except Exception as e:
            logger.error(f"Failed to get all patterns: {str(e)}")
            return []
    
    def get_latest_version(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest version of a pattern.
        
        Args:
            pattern_id: The ID of the pattern.
            
        Returns:
            The latest version of the pattern if found, None otherwise.
        """
        try:
            return self.repository.get_latest_version(pattern_id)
        except Exception as e:
            logger.error(f"Failed to get latest version of pattern {pattern_id}: {str(e)}")
            return None
    
    def get_version_history(self, pattern_id: str) -> List[Dict[str, Any]]:
        """
        Get the version history of a pattern.
        
        Args:
            pattern_id: The ID of the pattern.
            
        Returns:
            A list of pattern versions.
        """
        try:
            return self.repository.get_version_history(pattern_id)
        except Exception as e:
            logger.error(f"Failed to get version history of pattern {pattern_id}: {str(e)}")
            return []
