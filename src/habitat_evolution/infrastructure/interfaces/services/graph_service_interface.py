"""
Graph service interface for Habitat Evolution.

This module defines the interface for the graph service, which is
responsible for managing the graph representation of patterns and their relationships.
"""

from typing import Protocol, Any, Dict, List, Optional, Tuple
from abc import abstractmethod

from src.habitat_evolution.infrastructure.interfaces.service_interface import ServiceInterface


class GraphServiceInterface(ServiceInterface, Protocol):
    """
    Interface for the graph service in Habitat Evolution.
    
    The graph service is responsible for managing the graph representation of
    patterns and their relationships, providing a topological view of the semantic
    field. It supports the pattern evolution and co-evolution principles of Habitat
    by enabling the observation and analysis of how patterns relate to each other
    and how these relationships evolve over time.
    """
    
    @abstractmethod
    def create_concept(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a concept node in the graph.
        
        Args:
            name: The name of the concept
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
    def find_concepts_by_name(self, name: str) -> List[Dict[str, Any]]:
        """
        Find concepts by name.
        
        Args:
            name: The name to search for
            
        Returns:
            A list of matching concepts
        """
        ...
        
    @abstractmethod
    def find_concepts_by_quality(self, quality_state: str) -> List[Dict[str, Any]]:
        """
        Find concepts by quality state.
        
        Args:
            quality_state: The quality state to filter by
            
        Returns:
            A list of matching concepts
        """
        ...
        
    @abstractmethod
    def find_relations_by_concept(self, concept_id: str) -> List[Dict[str, Any]]:
        """
        Find relations for a concept.
        
        Args:
            concept_id: The ID of the concept
            
        Returns:
            A list of relations
        """
        ...
        
    @abstractmethod
    def evolve_concept_quality(self, concept_id: str, new_quality: str, 
                              metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Evolve the quality state of a concept.
        
        Args:
            concept_id: The ID of the concept
            new_quality: The new quality state
            metrics: Optional metrics associated with the transition
            
        Returns:
            The updated concept
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
    def get_concept_neighborhood(self, concept_id: str, 
                                depth: int = 1) -> Dict[str, Any]:
        """
        Get the neighborhood of a concept.
        
        Args:
            concept_id: The ID of the concept
            depth: The depth of the neighborhood
            
        Returns:
            The neighborhood graph
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
