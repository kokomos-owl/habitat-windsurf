"""
Interface for graph state repositories.

This module defines the interface for repositories that handle graph state persistence,
including nodes, relations, patterns, and quality state transitions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.habitat_evolution.pattern_aware_rag.state.test_states import (
    GraphStateSnapshot, ConceptNode, ConceptRelation, PatternState
)


class GraphStateRepositoryInterface(ABC):
    """
    Interface for graph state repositories.
    
    This interface defines the contract for repositories that handle the persistence
    of graph states, including nodes, relations, patterns, and quality state transitions.
    """
    
    @abstractmethod
    def save_state(self, graph_state: GraphStateSnapshot) -> str:
        """
        Save a graph state snapshot.
        
        Args:
            graph_state: The graph state snapshot to save
            
        Returns:
            The ID of the saved graph state
        """
        pass
        
    @abstractmethod
    def find_by_id(self, id: str) -> Optional[GraphStateSnapshot]:
        """
        Find a graph state by ID.
        
        Args:
            id: The ID of the graph state to find
            
        Returns:
            The graph state snapshot if found, None otherwise
        """
        pass
        
    @abstractmethod
    def save_node(self, node: ConceptNode) -> str:
        """
        Save a concept node.
        
        Args:
            node: The concept node to save
            
        Returns:
            The ID of the saved node
        """
        pass
        
    @abstractmethod
    def find_node_by_id(self, id: str) -> Optional[ConceptNode]:
        """
        Find a node by ID.
        
        Args:
            id: The ID of the node to find
            
        Returns:
            The concept node if found, None otherwise
        """
        pass
        
    @abstractmethod
    def save_relation(self, relation: ConceptRelation) -> str:
        """
        Save a relationship between nodes.
        
        Args:
            relation: The concept relation to save
            
        Returns:
            The ID of the saved relation
        """
        pass
        
    @abstractmethod
    def find_relations_by_nodes(self, source_id: str, target_id: str) -> List[ConceptRelation]:
        """
        Find relations between two nodes.
        
        Args:
            source_id: The ID of the source node
            target_id: The ID of the target node
            
        Returns:
            A list of concept relations between the nodes
        """
        pass
        
    @abstractmethod
    def save_pattern(self, pattern: PatternState) -> str:
        """
        Save a pattern state.
        
        Args:
            pattern: The pattern state to save
            
        Returns:
            The ID of the saved pattern
        """
        pass
        
    @abstractmethod
    def find_pattern_by_id(self, id: str) -> Optional[PatternState]:
        """
        Find a pattern by ID.
        
        Args:
            id: The ID of the pattern to find
            
        Returns:
            The pattern state if found, None otherwise
        """
        pass
        
    @abstractmethod
    def find_nodes_by_quality(self, quality: str) -> List[ConceptNode]:
        """
        Find nodes by quality state.
        
        Args:
            quality: The quality state to filter by
            
        Returns:
            A list of concept nodes with the specified quality state
        """
        pass
        
    @abstractmethod
    def find_relations_by_quality(self, quality: str) -> List[ConceptRelation]:
        """
        Find relationships by quality state.
        
        Args:
            quality: The quality state to filter by
            
        Returns:
            A list of concept relations with the specified quality state
        """
        pass
        
    @abstractmethod
    def find_nodes_by_category(self, category: str) -> List[ConceptNode]:
        """
        Find nodes by category.
        
        Args:
            category: The category to filter by
            
        Returns:
            A list of concept nodes with the specified category
        """
        pass
        
    @abstractmethod
    def track_quality_transition(self, entity_id: str, from_quality: str, to_quality: str) -> None:
        """
        Track a quality transition for an entity.
        
        Args:
            entity_id: The ID of the entity
            from_quality: The previous quality state
            to_quality: The new quality state
        """
        pass
        
    @abstractmethod
    def get_quality_transitions(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        Get quality transitions for an entity.
        
        Args:
            entity_id: The ID of the entity
            
        Returns:
            A list of quality transitions for the entity
        """
        pass
