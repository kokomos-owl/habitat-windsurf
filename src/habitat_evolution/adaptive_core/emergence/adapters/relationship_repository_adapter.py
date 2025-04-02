"""
Relationship Repository Adapter for Vector-Tonic Persistence Integration.

This module provides an adapter that implements the RelationshipRepositoryInterface
using the existing predicate relationship repository implementation.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.habitat_evolution.adaptive_core.emergence.interfaces.relationship_repository import RelationshipRepositoryInterface
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.predicate_relationship_repository import PredicateRelationshipRepository

logger = logging.getLogger(__name__)


class RelationshipRepositoryAdapter(RelationshipRepositoryInterface):
    """
    Adapter that implements RelationshipRepositoryInterface using PredicateRelationshipRepository.
    
    This adapter bridges the gap between the Vector-Tonic Window system and
    the ArangoDB persistence layer for relationships between patterns.
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
        self.repository = PredicateRelationshipRepository()
        if db_connection:
            self.repository.connection_manager.db = db_connection
    
    def save(self, relationship: Dict[str, Any]) -> str:
        """
        Save a relationship to the repository.
        
        Args:
            relationship: The relationship to save.
            
        Returns:
            The ID of the saved relationship.
        """
        try:
            # Ensure the relationship has required fields
            if "source_id" not in relationship or "target_id" not in relationship:
                raise ValueError("Relationship must have source_id and target_id")
            
            # Ensure the relationship has an ID
            if "id" not in relationship:
                relationship["id"] = f"{relationship['source_id']}_{relationship['target_id']}"
            
            # Add predicate if not present
            if "predicate" not in relationship:
                relationship["predicate"] = relationship.get("type", "RELATED_TO")
            
            # Save the relationship
            return self.repository.save(relationship)
        except Exception as e:
            logger.error(f"Failed to save relationship: {str(e)}")
            raise
    
    def find_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Find a relationship by its ID.
        
        Args:
            id: The ID of the relationship to find.
            
        Returns:
            The relationship if found, None otherwise.
        """
        try:
            return self.repository.find_by_id(id)
        except Exception as e:
            logger.error(f"Failed to find relationship with ID {id}: {str(e)}")
            return None
    
    def find_by_source(self, source_id: str) -> List[Dict[str, Any]]:
        """
        Find relationships by source ID.
        
        Args:
            source_id: The ID of the source pattern.
            
        Returns:
            A list of relationships with the given source ID.
        """
        try:
            return self.repository.find_by_source(source_id)
        except Exception as e:
            logger.error(f"Failed to find relationships for source {source_id}: {str(e)}")
            return []
    
    def find_by_target(self, target_id: str) -> List[Dict[str, Any]]:
        """
        Find relationships by target ID.
        
        Args:
            target_id: The ID of the target pattern.
            
        Returns:
            A list of relationships with the given target ID.
        """
        try:
            return self.repository.find_by_target(target_id)
        except Exception as e:
            logger.error(f"Failed to find relationships for target {target_id}: {str(e)}")
            return []
    
    def find_by_predicate(self, predicate: str) -> List[Dict[str, Any]]:
        """
        Find relationships by predicate.
        
        Args:
            predicate: The predicate to search for.
            
        Returns:
            A list of relationships with the given predicate.
        """
        try:
            return self.repository.find_by_predicate(predicate)
        except Exception as e:
            logger.error(f"Failed to find relationships for predicate {predicate}: {str(e)}")
            return []
    
    def delete(self, id: str) -> bool:
        """
        Delete a relationship by its ID.
        
        Args:
            id: The ID of the relationship to delete.
            
        Returns:
            True if the deletion was successful, False otherwise.
        """
        try:
            return self.repository.delete(id)
        except Exception as e:
            logger.error(f"Failed to delete relationship {id}: {str(e)}")
            return False
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all relationships.
        
        Returns:
            A list of all relationships.
        """
        try:
            return self.repository.get_all()
        except Exception as e:
            logger.error(f"Failed to get all relationships: {str(e)}")
            return []
