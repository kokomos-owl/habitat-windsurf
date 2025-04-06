"""
Unified graph service interface for Habitat Evolution.

This module defines a unified interface for graph services in the Habitat Evolution system,
reconciling the different implementations found across the codebase.
"""

from typing import Protocol, Dict, List, Any, Optional, Union, Tuple
from abc import abstractmethod

from src.habitat_evolution.infrastructure.interfaces.service_interface import ServiceInterface
from src.habitat_evolution.infrastructure.interfaces.repositories.graph_repository_interface import GraphRepositoryInterface


class UnifiedGraphServiceInterface(ServiceInterface, Protocol):
    """
    Unified interface for graph services in Habitat Evolution.
    
    This interface reconciles the different GraphService implementations found
    across the codebase, providing a consistent approach to graph operations
    that supports the pattern evolution and co-evolution principles of Habitat.
    """
    
    @abstractmethod
    def create_concept(self, name: str, concept_type: str, 
                      attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a concept node in the graph.
        
        Args:
            name: The name of the concept
            concept_type: The type of the concept
            attributes: Optional attributes for the concept
            
        Returns:
            The ID of the created concept
        """
        ...
        
    @abstractmethod
    def create_relation(self, source_id: str, target_id: str, 
                       relation_type: str, 
                       attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a relation between two concepts.
        
        Args:
            source_id: The ID of the source concept
            target_id: The ID of the target concept
            relation_type: The type of relation
            attributes: Optional attributes for the relation
            
        Returns:
            The ID of the created relation
        """
        ...
        
    @abstractmethod
    def get_concept(self, concept_id: str) -> Dict[str, Any]:
        """
        Get a concept by its ID.
        
        Args:
            concept_id: The ID of the concept to get
            
        Returns:
            The concept
        """
        ...
        
    @abstractmethod
    def find_concepts_by_name(self, name: str, exact_match: bool = True) -> List[Dict[str, Any]]:
        """
        Find concepts by name.
        
        Args:
            name: The name to search for
            exact_match: Whether to require an exact match
            
        Returns:
            A list of matching concepts
        """
        ...
        
    @abstractmethod
    def find_concepts_by_type(self, concept_type: str, 
                             attributes: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Find concepts by type and optional attributes.
        
        Args:
            concept_type: The type of concepts to find
            attributes: Optional attributes to filter by
            
        Returns:
            A list of matching concepts
        """
        ...
        
    @abstractmethod
    def find_relations(self, source_id: Optional[str] = None, 
                      target_id: Optional[str] = None,
                      relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find relations by source, target, and/or type.
        
        Args:
            source_id: Optional ID of the source concept
            target_id: Optional ID of the target concept
            relation_type: Optional type of relations to find
            
        Returns:
            A list of matching relations
        """
        ...
        
    @abstractmethod
    def update_concept(self, concept_id: str, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a concept's attributes.
        
        Args:
            concept_id: The ID of the concept to update
            attributes: The new attributes of the concept
            
        Returns:
            The updated concept
        """
        ...
        
    @abstractmethod
    def update_relation(self, relation_id: str, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a relation's attributes.
        
        Args:
            relation_id: The ID of the relation to update
            attributes: The new attributes of the relation
            
        Returns:
            The updated relation
        """
        ...
        
    @abstractmethod
    def delete_concept(self, concept_id: str, cascade: bool = False) -> bool:
        """
        Delete a concept from the graph.
        
        Args:
            concept_id: The ID of the concept to delete
            cascade: Whether to also delete related relations
            
        Returns:
            True if the concept was deleted, False otherwise
        """
        ...
        
    @abstractmethod
    def delete_relation(self, relation_id: str) -> bool:
        """
        Delete a relation from the graph.
        
        Args:
            relation_id: The ID of the relation to delete
            
        Returns:
            True if the relation was deleted, False otherwise
        """
        ...
        
    @abstractmethod
    def create_pattern(self, name: str, concepts: List[str], 
                      attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a pattern from concepts.
        
        Args:
            name: The name of the pattern
            concepts: The IDs of the concepts in the pattern
            attributes: Optional attributes for the pattern
            
        Returns:
            The ID of the created pattern
        """
        ...
        
    @abstractmethod
    def get_pattern(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get a pattern by its ID.
        
        Args:
            pattern_id: The ID of the pattern to get
            
        Returns:
            The pattern
        """
        ...
        
    @abstractmethod
    def get_concept_neighborhood(self, concept_id: str, 
                                depth: int = 1,
                                relation_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get the neighborhood of a concept.
        
        Args:
            concept_id: The ID of the concept
            depth: The depth of the neighborhood
            relation_types: Optional types of relations to include
            
        Returns:
            The neighborhood graph
        """
        ...
        
    @abstractmethod
    def calculate_semantic_similarity(self, concept_id1: str, concept_id2: str) -> float:
        """
        Calculate the semantic similarity between two concepts.
        
        Args:
            concept_id1: The ID of the first concept
            concept_id2: The ID of the second concept
            
        Returns:
            The semantic similarity (0-1)
        """
        ...
        
    @abstractmethod
    def find_path(self, source_id: str, target_id: str, 
                 max_depth: int = 3) -> Optional[List[Dict[str, Any]]]:
        """
        Find a path between two concepts.
        
        Args:
            source_id: The ID of the source concept
            target_id: The ID of the target concept
            max_depth: The maximum path length
            
        Returns:
            The path if found, None otherwise
        """
        ...
        
    @abstractmethod
    def execute_graph_query(self, query: str, 
                           parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a graph query.
        
        Args:
            query: The query to execute
            parameters: Optional parameters for the query
            
        Returns:
            The query results
        """
        ...
        
    @abstractmethod
    def create_graph_snapshot(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a snapshot of the current graph state.
        
        Args:
            metadata: Optional metadata for the snapshot
            
        Returns:
            The ID of the created snapshot
        """
        ...
