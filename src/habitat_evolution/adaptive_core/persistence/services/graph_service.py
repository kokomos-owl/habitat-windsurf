"""
GraphService: A higher-level service for graph operations.

This service provides domain-specific methods for working with the graph state repository,
implementing common operations and patterns for the Habitat Evolution system.
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

from src.habitat_evolution.adaptive_core.persistence.arangodb.graph_state_repository import ArangoDBGraphStateRepository
from src.tests.adaptive_core.persistence.arangodb.test_state_models import (
    ConceptNode, ConceptRelation, PatternState, GraphStateSnapshot
)


class GraphService:
    """
    A higher-level service for graph operations.
    
    This service uses the GraphStateRepository to provide domain-specific methods
    for working with graph data in the Habitat Evolution system.
    """
    
    def __init__(self, repository: ArangoDBGraphStateRepository):
        """
        Initialize the graph service.
        
        Args:
            repository: The graph state repository to use
        """
        self.repository = repository
        
    # ===== Node Operations =====
    
    def create_concept(self, name: str, attributes: Dict[str, str] = None, 
                       quality_state: str = "uncertain", confidence: float = 0.3) -> ConceptNode:
        """
        Create a new concept node.
        
        Args:
            name: The name of the concept
            attributes: Optional attributes for the concept
            quality_state: Initial quality state (default: uncertain)
            confidence: Initial confidence score (default: 0.3)
            
        Returns:
            The created concept node
        """
        # Ensure attributes exists
        attributes = attributes or {}
        
        # Add quality state to attributes
        attributes["quality_state"] = quality_state
        if confidence is not None:
            attributes["confidence"] = str(confidence)
            
        # Create node
        node = ConceptNode(
            id=str(uuid.uuid4()),
            name=name,
            attributes=attributes,
            created_at=datetime.now()
        )
        
        # Save to repository
        self.repository.save_node(node)
        return node
    
    def find_concepts_by_name(self, name: str) -> List[ConceptNode]:
        """
        Find concept nodes by name.
        
        Args:
            name: The name to search for
            
        Returns:
            List of matching concept nodes
        """
        # Use AQL to find nodes by name
        query = """
        FOR node IN @@collection
            FILTER node.name == @name
            RETURN node
        """
        
        bind_vars = {
            "@collection": self.repository.nodes_collection,
            "name": name
        }
        
        cursor = self.repository.db.aql.execute(query, bind_vars=bind_vars)
        
        # Convert to ConceptNode objects
        nodes = []
        for doc in cursor:
            node = ConceptNode(
                id=doc["_key"],
                name=doc["name"],
                attributes=doc.get("attributes", {}),
                created_at=doc.get("created_at")
            )
            nodes.append(node)
            
        return nodes
    
    def evolve_concept_quality(self, node_id: str, to_quality: str, 
                              confidence: float = None, context: Dict[str, Any] = None) -> ConceptNode:
        """
        Evolve a concept's quality state.
        
        Args:
            node_id: The ID of the concept node
            to_quality: The new quality state
            confidence: Optional confidence score
            context: Optional context information
            
        Returns:
            The updated concept node
        """
        # Get current node
        node = self.repository.find_node_by_id(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")
            
        # Get current quality state
        from_quality = node.attributes.get("quality_state", "uncertain")
        
        # Track quality transition
        self.repository.track_quality_transition(
            entity_id=node_id,
            from_quality=from_quality,
            to_quality=to_quality,
            confidence=confidence,
            context=context
        )
        
        # Return updated node
        return self.repository.find_node_by_id(node_id)
    
    # ===== Relation Operations =====
    
    def create_relation(self, source_id: str, target_id: str, relation_type: str, 
                       weight: float = 1.0, quality_state: str = "uncertain") -> ConceptRelation:
        """
        Create a relation between concepts.
        
        Args:
            source_id: The ID of the source concept
            target_id: The ID of the target concept
            relation_type: The type of relation
            weight: The weight of the relation (default: 1.0)
            quality_state: Initial quality state (default: uncertain)
            
        Returns:
            The created relation
        """
        # Create relation
        relation = ConceptRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight
        )
        
        # Save to repository with quality state
        self.repository.save_relation(relation, quality_state=quality_state)
        return relation
    
    def find_relations_by_concept(self, concept_id: str) -> List[ConceptRelation]:
        """
        Find all relations involving a concept.
        
        Args:
            concept_id: The ID of the concept
            
        Returns:
            List of relations involving the concept
        """
        # Use AQL to find relations by concept ID
        query = """
        FOR rel IN @@collection
            FILTER rel.source_id == @concept_id OR rel.target_id == @concept_id
            RETURN rel
        """
        
        bind_vars = {
            "@collection": self.repository.relations_collection,
            "concept_id": concept_id
        }
        
        cursor = self.repository.db.aql.execute(query, bind_vars=bind_vars)
        
        # Convert to ConceptRelation objects
        relations = []
        for doc in cursor:
            relation = ConceptRelation(
                source_id=doc["source_id"],
                target_id=doc["target_id"],
                relation_type=doc["relation_type"],
                weight=doc["weight"]
            )
            relations.append(relation)
            
        return relations
    
    # ===== Pattern Operations =====
    
    def create_pattern(self, content: str, metadata: Dict[str, str] = None,
                      confidence: float = 0.3) -> PatternState:
        """
        Create a new pattern.
        
        Args:
            content: The pattern content
            metadata: Optional metadata for the pattern
            confidence: Initial confidence score (default: 0.3)
            
        Returns:
            The created pattern
        """
        # Ensure metadata exists
        metadata = metadata or {}
        
        # Create pattern
        pattern = PatternState(
            id=str(uuid.uuid4()),
            content=content,
            metadata=metadata,
            timestamp=datetime.now(),
            confidence=confidence
        )
        
        # Save to repository
        self.repository.save_pattern(pattern)
        return pattern
    
    def evolve_pattern_confidence(self, pattern_id: str, new_confidence: float,
                                context: Dict[str, Any] = None) -> PatternState:
        """
        Evolve a pattern's confidence score.
        
        Args:
            pattern_id: The ID of the pattern
            new_confidence: The new confidence score
            context: Optional context information
            
        Returns:
            The updated pattern
        """
        # Get current pattern
        pattern = self.repository.find_pattern_by_id(pattern_id)
        if not pattern:
            raise ValueError(f"Pattern {pattern_id} not found")
            
        # Determine quality states based on confidence
        old_quality = self._confidence_to_quality(pattern.confidence)
        new_quality = self._confidence_to_quality(new_confidence)
        
        # Update pattern with new confidence
        pattern.confidence = new_confidence
        self.repository.save_pattern(pattern)
        
        # Track quality transition if quality changed
        if old_quality != new_quality:
            self.repository.track_quality_transition(
                entity_id=pattern_id,
                from_quality=old_quality,
                to_quality=new_quality,
                confidence=new_confidence,
                context=context
            )
        
        return pattern
    
    def _confidence_to_quality(self, confidence: float) -> str:
        """
        Convert a confidence score to a quality state.
        
        Args:
            confidence: The confidence score (0.0-1.0)
            
        Returns:
            The corresponding quality state
        """
        if confidence < 0.4:
            return "poor"
        elif confidence < 0.7:
            return "uncertain"
        else:
            return "good"
    
    # ===== Graph Operations =====
    
    def create_graph_snapshot(self, nodes: List[ConceptNode] = None, 
                             relations: List[ConceptRelation] = None,
                             patterns: List[PatternState] = None) -> str:
        """
        Create a snapshot of the current graph state.
        
        Args:
            nodes: Optional list of nodes to include (default: all)
            relations: Optional list of relations to include (default: all)
            patterns: Optional list of patterns to include (default: all)
            
        Returns:
            The ID of the created snapshot
        """
        # If no nodes specified, get all nodes
        if nodes is None:
            # This would need a method to get all nodes, which we'd implement in the repository
            # For now, we'll just use an empty list
            nodes = []
            
        # If no relations specified, get all relations
        if relations is None:
            # This would need a method to get all relations
            relations = []
            
        # If no patterns specified, get all patterns
        if patterns is None:
            # This would need a method to get all patterns
            patterns = []
            
        # Create snapshot
        snapshot = GraphStateSnapshot(
            id=str(uuid.uuid4()),
            nodes=nodes,
            relations=relations,
            patterns=patterns,
            timestamp=datetime.now()
        )
        
        # Save to repository
        self.repository.save_state(snapshot)
        return snapshot.id
    
    def get_concept_neighborhood(self, concept_id: str, depth: int = 1) -> Tuple[List[ConceptNode], List[ConceptRelation]]:
        """
        Get the neighborhood of a concept.
        
        Args:
            concept_id: The ID of the concept
            depth: The depth of the neighborhood (default: 1)
            
        Returns:
            A tuple of (nodes, relations) in the neighborhood
        """
        # This would require a more complex graph traversal query
        # For now, we'll just implement a simple version for depth=1
        
        # Get the concept
        concept = self.repository.find_node_by_id(concept_id)
        if not concept:
            raise ValueError(f"Concept {concept_id} not found")
            
        # Get relations involving the concept
        relations = self.find_relations_by_concept(concept_id)
        
        # Get related concepts
        related_ids = set()
        for relation in relations:
            if relation.source_id == concept_id:
                related_ids.add(relation.target_id)
            else:
                related_ids.add(relation.source_id)
                
        # Get related concepts
        related_concepts = [self.repository.find_node_by_id(node_id) for node_id in related_ids]
        related_concepts = [node for node in related_concepts if node]  # Filter out None
        
        # Return neighborhood
        return [concept] + related_concepts, relations
