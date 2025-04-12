"""
Mock implementation of PatternAwareRAGService for testing purposes.

This module provides a simplified mock implementation of the PatternAwareRAGService
that can be used for testing the dependency chain without requiring all the
actual dependencies to be available.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import uuid
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class MockPatternAwareRAGService:
    """
    Mock implementation of PatternAwareRAGService for testing purposes.
    
    This class provides a simplified implementation that mimics the behavior
    of the real PatternAwareRAGService without requiring all the actual
    dependencies to be available.
    """
    
    def __init__(
        self,
        db_connection=None,
        pattern_repository=None,
        vector_tonic_service=None,
        claude_adapter=None,
        event_service=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the MockPatternAwareRAGService.
        
        Args:
            db_connection: Optional database connection
            pattern_repository: Optional pattern repository
            vector_tonic_service: Optional vector tonic service
            claude_adapter: Optional Claude adapter
            event_service: Optional event service
            config: Optional configuration dictionary
        """
        self.db_connection = db_connection
        self.pattern_repository = pattern_repository
        self.vector_tonic_service = vector_tonic_service
        self.claude_adapter = claude_adapter
        self.event_service = event_service
        self.config = config or {}
        
        # Initialize fallback storage
        self.fallback_storage = {
            "patterns": {},
            "relationships": [],
            "metrics": {
                "fallback_count": 0,
                "successful_operations": 0,
                "failed_operations": 0
            }
        }
        
        self._initialized = False
        logger.info("MockPatternAwareRAGService created")
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the service with the given configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        if config:
            self.config.update(config)
        
        self._initialized = True
        logger.info("MockPatternAwareRAGService initialized")
    
    def create_pattern(self, pattern_data: Dict[str, Any]) -> str:
        """
        Create a new pattern in the mock storage.
        
        Args:
            pattern_data: Dictionary containing pattern data
            
        Returns:
            ID of the created pattern
        """
        pattern_id = str(uuid.uuid4())
        pattern_data["id"] = pattern_id
        pattern_data["created_at"] = datetime.now().isoformat()
        
        self.fallback_storage["patterns"][pattern_id] = pattern_data
        self.fallback_storage["metrics"]["successful_operations"] += 1
        
        logger.info(f"Created mock pattern with ID: {pattern_id}")
        return pattern_id
    
    def create_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        relationship_type: str, 
        relationship_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a relationship between two patterns in the mock storage.
        
        Args:
            source_id: ID of the source pattern
            target_id: ID of the target pattern
            relationship_type: Type of relationship
            relationship_data: Optional data for the relationship
            
        Returns:
            ID of the created relationship
        """
        relationship_id = str(uuid.uuid4())
        relationship = {
            "id": relationship_id,
            "source_id": source_id,
            "target_id": target_id,
            "type": relationship_type,
            "data": relationship_data or {},
            "created_at": datetime.now().isoformat()
        }
        
        self.fallback_storage["relationships"].append(relationship)
        self.fallback_storage["metrics"]["successful_operations"] += 1
        
        logger.info(f"Created mock relationship with ID: {relationship_id}")
        return relationship_id
    
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a pattern from the mock storage.
        
        Args:
            pattern_id: ID of the pattern to retrieve
            
        Returns:
            Pattern data or None if not found
        """
        pattern = self.fallback_storage["patterns"].get(pattern_id)
        
        if pattern:
            self.fallback_storage["metrics"]["successful_operations"] += 1
            return pattern
        else:
            self.fallback_storage["metrics"]["failed_operations"] += 1
            return None
    
    def get_related_patterns(
        self, 
        pattern_id: str, 
        relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get patterns related to the given pattern.
        
        Args:
            pattern_id: ID of the pattern to get related patterns for
            relationship_type: Optional type of relationship to filter by
            
        Returns:
            List of related patterns
        """
        related_patterns = []
        
        for relationship in self.fallback_storage["relationships"]:
            if relationship["source_id"] == pattern_id:
                if relationship_type is None or relationship["type"] == relationship_type:
                    target_id = relationship["target_id"]
                    target_pattern = self.fallback_storage["patterns"].get(target_id)
                    if target_pattern:
                        related_patterns.append(target_pattern)
        
        self.fallback_storage["metrics"]["successful_operations"] += 1
        return related_patterns
    
    def get_fallback_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about fallback operations.
        
        Returns:
            Dictionary containing fallback metrics
        """
        return self.fallback_storage["metrics"]
    
    def query(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query using the mock service.
        
        Args:
            query_text: The query text to process
            context: Optional context for the query
            
        Returns:
            Dictionary containing the query result
        """
        if self.claude_adapter:
            try:
                # Try to use the real Claude adapter if available
                result = self.claude_adapter.query(query_text, context)
                self.fallback_storage["metrics"]["successful_operations"] += 1
                return result
            except Exception as e:
                logger.warning(f"Error using Claude adapter: {e}, falling back to mock response")
                self.fallback_storage["metrics"]["fallback_count"] += 1
        
        # Generate a mock response
        mock_response = {
            "query": query_text,
            "response": f"Mock response for: {query_text}",
            "context_used": context or {},
            "generated_at": datetime.now().isoformat(),
            "is_mock": True
        }
        
        return mock_response
