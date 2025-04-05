"""
Graph service for managing graph states in the PatternAwareRAG system.

This module provides a service layer for interacting with the graph state repository,
offering higher-level functionality for managing graph states, nodes, relations,
patterns, and quality transitions.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid
import logging
import asyncio

from src.habitat_evolution.pattern_aware_rag.state.test_states import (
    GraphStateSnapshot, ConceptNode, ConceptRelation, PatternState
)
from src.habitat_evolution.adaptive_core.persistence.interfaces.graph_state_repository import GraphStateRepositoryInterface
from src.habitat_evolution.adaptive_core.persistence.arangodb.graph_state_repository import ArangoDBGraphStateRepository
from src.habitat_evolution.adaptive_core.models.pattern import Pattern


class GraphService:
    """
    Service for managing graph states in the PatternAwareRAG system.
    
    This class provides methods for interacting with the graph state repository,
    offering higher-level functionality for managing graph states, nodes, relations,
    patterns, and quality transitions.
    """
    
    def __init__(self, repository: Optional[GraphStateRepositoryInterface] = None, logger=None):
        """
        Initialize the graph service.
        
        Args:
            repository: The graph state repository to use (defaults to ArangoDBGraphStateRepository)
            logger: Logger instance to use (defaults to standard logging)
        """
        self.repository = repository or ArangoDBGraphStateRepository()
        self.logger = logger or logging.getLogger(__name__)
    
    async def create_snapshot(self, nodes: List[ConceptNode], relations: List[ConceptRelation], 
                             patterns: List[PatternState], version: int = 1) -> str:
        """
        Create a snapshot of the current graph state.
        
        Args:
            nodes: List of concept nodes in the graph
            relations: List of concept relations in the graph
            patterns: List of pattern states in the graph
            version: Version number of the graph state
            
        Returns:
            The ID of the created graph state snapshot
        """
        snapshot = GraphStateSnapshot(
            id=str(uuid.uuid4()),
            nodes=nodes,
            relations=relations,
            patterns=patterns,
            timestamp=datetime.now(),
            version=version
        )
        
        # Validate the snapshot
        try:
            snapshot.validate_relations()
        except Exception as e:
            self.logger.error(f"Invalid graph state snapshot: {str(e)}")
            raise ValueError(f"Invalid graph state snapshot: {str(e)}")
        
        # Save the snapshot
        snapshot_id = self.repository.save_state(snapshot)
        self.logger.info(f"Created graph state snapshot with ID: {snapshot_id}")
        
        return snapshot_id
    
    async def get_snapshot(self, snapshot_id: str) -> Optional[GraphStateSnapshot]:
        """
        Get a graph state snapshot by ID.
        
        Args:
            snapshot_id: The ID of the snapshot to retrieve
            
        Returns:
            The graph state snapshot if found, None otherwise
        """
        return self.repository.find_by_id(snapshot_id)
    
    async def add_node(self, name: str, node_type: str, quality_state: str = "uncertain", 
                      attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a node to the graph.
        
        Args:
            name: The name of the node
            node_type: The type of the node
            quality_state: The quality state of the node (default: "uncertain")
            attributes: Additional attributes for the node
            
        Returns:
            The ID of the created node
        """
        node = ConceptNode(
            id=str(uuid.uuid4()),
            name=name,
            node_type=node_type,
            quality_state=quality_state,
            attributes=attributes or {}
        )
        
        node_id = self.repository.save_node(node)
        self.logger.info(f"Added node '{name}' with ID: {node_id}")
        
        return node_id
    
    async def add_relation(self, source_id: str, target_id: str, relation_type: str,
                          quality_state: str = "uncertain", weight: float = 1.0,
                          attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a relation between two nodes.
        
        Args:
            source_id: The ID of the source node
            target_id: The ID of the target node
            relation_type: The type of the relation
            quality_state: The quality state of the relation (default: "uncertain")
            weight: The weight of the relation (default: 1.0)
            attributes: Additional attributes for the relation
            
        Returns:
            The ID of the created relation
        """
        # Verify that source and target nodes exist
        source_node = self.repository.find_node_by_id(source_id)
        target_node = self.repository.find_node_by_id(target_id)
        
        if not source_node:
            raise ValueError(f"Source node with ID {source_id} not found")
        
        if not target_node:
            raise ValueError(f"Target node with ID {target_id} not found")
        
        relation = ConceptRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            quality_state=quality_state,
            weight=weight,
            attributes=attributes or {}
        )
        
        relation_id = self.repository.save_relation(relation)
        self.logger.info(f"Added relation '{relation_type}' from {source_id} to {target_id}")
        
        return relation_id
    
    async def add_pattern(self, content: str, confidence: float, 
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a pattern to the graph.
        
        Args:
            content: The content of the pattern
            confidence: The confidence score of the pattern
            metadata: Additional metadata for the pattern
            
        Returns:
            The ID of the created pattern
        """
        pattern = PatternState(
            id=str(uuid.uuid4()),
            content=content,
            metadata=metadata or {},
            timestamp=datetime.now(),
            confidence=confidence
        )
        
        pattern_id = self.repository.save_pattern(pattern)
        self.logger.info(f"Added pattern with content '{content}' and ID: {pattern_id}")
        
        return pattern_id
    
    async def store_pattern(self, pattern: Pattern) -> str:
        """
        Store a pattern in the graph.
        
        Args:
            pattern: The pattern to store
            
        Returns:
            The ID of the stored pattern
        """
        # Convert Pattern to PatternState
        pattern_state = PatternState(
            id=pattern.id if hasattr(pattern, 'id') and pattern.id else str(uuid.uuid4()),
            content=pattern.base_concept if hasattr(pattern, 'base_concept') else str(pattern),
            metadata={
                "creator_id": pattern.creator_id if hasattr(pattern, 'creator_id') else "system",
                "coherence": pattern.coherence if hasattr(pattern, 'coherence') else 0.0,
                "phase_stability": pattern.phase_stability if hasattr(pattern, 'phase_stability') else 0.0,
                "signal_strength": pattern.signal_strength if hasattr(pattern, 'signal_strength') else 0.0
            },
            timestamp=datetime.now(),
            confidence=pattern.confidence if hasattr(pattern, 'confidence') else 0.5
        )
        
        pattern_id = self.repository.save_pattern(pattern_state)
        self.logger.info(f"Stored pattern with ID: {pattern_id}")
        
        return pattern_id
    
    async def update_node_quality(self, node_id: str, new_quality: str) -> bool:
        """
        Update the quality state of a node.
        
        Args:
            node_id: The ID of the node
            new_quality: The new quality state
            
        Returns:
            True if the update was successful, False otherwise
        """
        # Find the node
        node = self.repository.find_node_by_id(node_id)
        if not node:
            self.logger.error(f"Node with ID {node_id} not found")
            return False
        
        # Track the quality transition
        old_quality = node.quality_state
        if old_quality != new_quality:
            self.repository.track_quality_transition(node_id, old_quality, new_quality)
            self.logger.info(f"Quality transition for node {node_id}: {old_quality} -> {new_quality}")
        
        # Update the node
        node.quality_state = new_quality
        self.repository.save_node(node)
        
        return True
    
    async def update_relation_quality(self, source_id: str, target_id: str, 
                                     relation_type: str, new_quality: str) -> bool:
        """
        Update the quality state of a relation.
        
        Args:
            source_id: The ID of the source node
            target_id: The ID of the target node
            relation_type: The type of the relation
            new_quality: The new quality state
            
        Returns:
            True if the update was successful, False otherwise
        """
        # Find the relation
        relations = self.repository.find_relations_by_nodes(source_id, target_id)
        target_relation = None
        
        for relation in relations:
            if relation.relation_type == relation_type:
                target_relation = relation
                break
        
        if not target_relation:
            self.logger.error(f"Relation of type {relation_type} between {source_id} and {target_id} not found")
            return False
        
        # Track the quality transition
        old_quality = target_relation.quality_state
        if old_quality != new_quality:
            # Create a unique ID for the relation for tracking purposes
            relation_id = f"{source_id}_{target_id}_{relation_type}"
            self.repository.track_quality_transition(relation_id, old_quality, new_quality)
            self.logger.info(f"Quality transition for relation {relation_id}: {old_quality} -> {new_quality}")
        
        # Update the relation
        target_relation.quality_state = new_quality
        self.repository.save_relation(target_relation)
        
        return True
    
    async def get_quality_distribution(self) -> Dict[str, Dict[str, int]]:
        """
        Get the distribution of quality states across nodes and relations.
        
        Returns:
            A dictionary with quality state distributions for nodes and relations
        """
        # Get all quality states
        quality_states = ["poor", "uncertain", "good"]
        
        # Initialize result
        result = {
            "nodes": {state: 0 for state in quality_states},
            "relations": {state: 0 for state in quality_states}
        }
        
        # Count nodes by quality
        for state in quality_states:
            nodes = self.repository.find_nodes_by_quality(state)
            result["nodes"][state] = len(nodes)
        
        # Count relations by quality
        for state in quality_states:
            relations = self.repository.find_relations_by_quality(state)
            result["relations"][state] = len(relations)
        
        return result
    
    async def get_nodes_by_category(self, category: str) -> List[ConceptNode]:
        """
        Get nodes by category.
        
        Args:
            category: The category to filter by
            
        Returns:
            A list of nodes with the specified category
        """
        return self.repository.find_nodes_by_category(category)
    
    async def map_density_centers(self, coherence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Map density centers in the graph based on coherence.
        
        Args:
            coherence_threshold: The minimum coherence threshold for density centers
            
        Returns:
            A list of density centers with their properties
        """
        # Find high-quality nodes
        good_nodes = self.repository.find_nodes_by_quality("good")
        
        # Filter by coherence threshold
        centers = []
        for node in good_nodes:
            coherence = node.attributes.get("coherence", 0.0)
            if isinstance(coherence, str):
                coherence = float(coherence)
                
            if coherence >= coherence_threshold:
                centers.append({
                    "id": node.id,
                    "name": node.name,
                    "coherence": coherence,
                    "category": node.attributes.get("category", "UNKNOWN")
                })
        
        # Sort by coherence (highest first)
        centers.sort(key=lambda x: x["coherence"], reverse=True)
        
        return centers
    
    async def get_quality_transitions_history(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        Get the quality transition history for an entity.
        
        Args:
            entity_id: The ID of the entity
            
        Returns:
            A list of quality transitions for the entity
        """
        return self.repository.get_quality_transitions(entity_id)
    
    async def get_quality_transition_summary(self) -> Dict[str, int]:
        """
        Get a summary of quality transitions in the system.
        
        Returns:
            A dictionary with counts of different types of transitions
        """
        # This would require a more complex query in ArangoDB
        # For now, we'll implement a simplified version
        
        # Define transition types
        transition_types = {
            "poor_to_uncertain": 0,
            "uncertain_to_good": 0,
            "good_to_uncertain": 0,
            "uncertain_to_poor": 0
        }
        
        # Get all nodes
        nodes = []
        for quality in ["poor", "uncertain", "good"]:
            nodes.extend(self.repository.find_nodes_by_quality(quality))
        
        # Get transitions for each node
        for node in nodes:
            transitions = self.repository.get_quality_transitions(node.id)
            
            for transition in transitions:
                from_quality = transition["from_quality"]
                to_quality = transition["to_quality"]
                
                transition_key = f"{from_quality}_to_{to_quality}"
                if transition_key in transition_types:
                    transition_types[transition_key] += 1
        
        return transition_types
