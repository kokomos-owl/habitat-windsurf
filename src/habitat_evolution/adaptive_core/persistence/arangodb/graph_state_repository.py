"""
ArangoDB implementation of the GraphStateRepository interface.

This module provides an ArangoDB-based implementation of the GraphStateRepository
interface for persisting graph states, nodes, relations, patterns, and quality transitions.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid

from src.habitat_evolution.pattern_aware_rag.state.test_states import (
    GraphStateSnapshot, ConceptNode, ConceptRelation, PatternState
)
from src.habitat_evolution.adaptive_core.persistence.interfaces.graph_state_repository import GraphStateRepositoryInterface
from src.habitat_evolution.adaptive_core.persistence.arangodb.base_repository import ArangoDBBaseRepository
from src.habitat_evolution.adaptive_core.persistence.arangodb.connection import ArangoDBConnectionManager


class ArangoDBGraphStateRepository(ArangoDBBaseRepository, GraphStateRepositoryInterface):
    """
    ArangoDB implementation of the GraphStateRepository interface.
    
    This class provides methods for persisting and retrieving graph states,
    including nodes, relations, patterns, and quality transitions in ArangoDB.
    """
    
    def __init__(self):
        """Initialize the repository with the required collections."""
        super().__init__()
        # Override collection name from base class
        self.collection_name = "GraphStates"
        
        # Define additional collections
        self.nodes_collection = "ConceptNodes"
        self.relations_collection = "ConceptRelations"
        self.patterns_collection = "PatternStates"
        self.quality_transitions_collection = "QualityTransitions"
        
        # Get database connection
        self.db = self.connection_manager.get_db()
        
        # Initialize collections
        self._initialize_collections()
        
    def _initialize_collections(self):
        """Initialize collections needed for graph state persistence."""
        collections = [
            self.collection_name,
            self.nodes_collection,
            self.patterns_collection,
            self.quality_transitions_collection
        ]
        
        for collection in collections:
            if not self.db.has_collection(collection):
                self.db.create_collection(collection)
                
        # Initialize edge collection for relations
        if not self.db.has_collection(self.relations_collection):
            self.db.create_collection(self.relations_collection, edge=True)
    
    def save_state(self, graph_state: GraphStateSnapshot) -> str:
        """
        Save a graph state snapshot.
        
        Args:
            graph_state: The graph state snapshot to save
            
        Returns:
            The ID of the saved graph state
        """
        # Validate the graph state
        graph_state.validate_relations()
        
        # Convert to dictionary
        state_dict = {
            "_key": graph_state.id,
            "timestamp": graph_state.timestamp.isoformat(),
            "version": graph_state.version,
            "node_ids": [node.id for node in graph_state.nodes],
            "relation_ids": [f"{rel.source_id}_{rel.target_id}_{rel.relation_type}" for rel in graph_state.relations],
            "pattern_ids": [pattern.id for pattern in graph_state.patterns]
        }
        
        # Save to ArangoDB
        collection = self.db.collection(self.collection_name)
        result = collection.insert(state_dict, return_new=True)
        
        # Save nodes, relations, and patterns
        for node in graph_state.nodes:
            self.save_node(node)
            
        for relation in graph_state.relations:
            self.save_relation(relation)
            
        for pattern in graph_state.patterns:
            self.save_pattern(pattern)
            
        return result["_key"]
    
    def find_by_id(self, id: str) -> Optional[GraphStateSnapshot]:
        """
        Find a graph state by ID.
        
        Args:
            id: The ID of the graph state to find
            
        Returns:
            The graph state snapshot if found, None otherwise
        """
        collection = self.db.collection(self.collection_name)
        
        try:
            # Get document by key
            doc = collection.get(id)
            if not doc:
                return None
                
            # Load nodes
            nodes = []
            for node_id in doc["node_ids"]:
                node = self.find_node_by_id(node_id)
                if node:
                    nodes.append(node)
            
            # Load relations
            relations = []
            for relation_id in doc["relation_ids"]:
                # Parse relation ID to get source, target, and type
                parts = relation_id.split("_")
                if len(parts) >= 3:
                    source_id = parts[0]
                    target_id = parts[1]
                    relation_type = "_".join(parts[2:])  # Handle relation types with underscores
                    
                    # Find relations between these nodes
                    found_relations = self.find_relations_by_nodes(source_id, target_id)
                    for rel in found_relations:
                        if rel.relation_type == relation_type:
                            relations.append(rel)
                            break
            
            # Load patterns
            patterns = []
            for pattern_id in doc["pattern_ids"]:
                pattern = self.find_pattern_by_id(pattern_id)
                if pattern:
                    patterns.append(pattern)
            
            # Create and return graph state
            timestamp = datetime.fromisoformat(doc["timestamp"])
            return GraphStateSnapshot(
                id=doc["_key"],
                nodes=nodes,
                relations=relations,
                patterns=patterns,
                timestamp=timestamp,
                version=doc["version"]
            )
            
        except Exception as e:
            print(f"Error finding graph state: {str(e)}")
            return None
    
    def save_node(self, node: ConceptNode) -> str:
        """
        Save a concept node.
        
        Args:
            node: The concept node to save
            
        Returns:
            The ID of the saved node
        """
        collection = self.db.collection(self.nodes_collection)
        
        # Get quality state from attributes if available, default to 'poor'
        quality_state = node.attributes.get("quality_state", "poor")
        
        # Convert to dictionary
        node_dict = {
            "_key": node.id,
            "name": node.name,
            "quality_state": quality_state,
            "attributes": node.attributes
        }
        
        # Add created_at timestamp if available
        if node.created_at:
            node_dict["created_at"] = node.created_at.isoformat()
        
        # Ensure category is stored as an assigned property
        if "category" in node.attributes:
            category = node.attributes["category"]
            node_dict["attributes"]["category"] = category
        
        # Check if node already exists
        existing = collection.get(node.id)
        if existing:
            # Check if we need to track a quality state transition
            if "quality_state" in existing and existing["quality_state"] != quality_state:
                self.track_quality_transition(
                    entity_id=node.id,
                    from_quality=existing["quality_state"],
                    to_quality=quality_state,
                    confidence=float(node.attributes.get("confidence", 0.5)) if node.attributes.get("confidence") else None
                )
            
            # Update existing node
            collection.update(node.id, node_dict)
        else:
            # Insert new node
            collection.insert(node_dict)
            
        return node.id
    
    def find_node_by_id(self, id: str) -> Optional[ConceptNode]:
        """
        Find a node by ID.
        
        Args:
            id: The ID of the node to find
            
        Returns:
            The concept node if found, None otherwise
        """
        collection = self.db.collection(self.nodes_collection)
        
        try:
            # Get document by key
            doc = collection.get(id)
            if not doc:
                return None
            
            # Get created_at if available
            created_at = None
            if "created_at" in doc:
                try:
                    created_at = datetime.fromisoformat(doc["created_at"])
                except (ValueError, TypeError):
                    pass
                
            # Convert to ConceptNode
            return ConceptNode(
                id=doc["_key"],
                name=doc["name"],
                attributes=doc["attributes"],
                created_at=created_at
            )
            
        except Exception as e:
    
    cursor = self.db.aql.execute(query, bind_vars=bind_vars)
    
    # Return first match
    doc = next(cursor, None)
    if doc:
        return ConceptRelation(
            source_id=doc["source_id"],
            target_id=doc["target_id"],
            relation_type=doc["relation_type"],
            weight=doc["weight"]
        )
    
    return None

def find_relations_by_nodes(self, source_id: str, target_id: str) -> List[ConceptRelation]:
    """
    Find relations between two nodes.
    
    Args:
        source_id: The source node ID
        target_id: The target node ID
        
    Returns:
        A list of concept relations between the nodes
    """
    query = """
    FOR e IN @@collection
        FILTER e._from == @from_id AND e._to == @to_id
        LET source_id = SUBSTRING(e._from, LENGTH(@nodes_prefix))
        LET target_id = SUBSTRING(e._to, LENGTH(@nodes_prefix))
        RETURN {
            source_id: source_id,
            target_id: target_id,
            relation_type: e.relation_type,
            weight: e.weight
        }
    """
    
    bind_vars = {
        "@collection": self.relations_collection,
        "from_id": f"{self.nodes_collection}/{source_id}",
        "to_id": f"{self.nodes_collection}/{target_id}",
        "nodes_prefix": f"{self.nodes_collection}/"
    }
    
    cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        Args:
            category: The category to filter by
            
        Returns:
            A list of concept nodes with the specified category
        """
        query = """
        FOR doc IN @@collection
            FILTER doc.attributes.category == @category
            RETURN doc
        """
        
        bind_vars = {
            "@collection": self.nodes_collection,
            "category": category
        }
        
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        
        nodes = []
        for doc in cursor:
            # Get created_at if available
            created_at = None
            if "created_at" in doc:
                try:
                    created_at = datetime.fromisoformat(doc["created_at"])
                except (ValueError, TypeError):
                    pass
                    
            node = ConceptNode(
                id=doc["_key"],
                name=doc["name"],
                attributes=doc["attributes"],
                created_at=created_at
            )
            nodes.append(node)
            
        return nodes
    
    def track_quality_transition(self, entity_id: str, from_quality: str, to_quality: str, confidence: float = None, context: Dict[str, Any] = None) -> None:
        """
        Track a quality transition for an entity.
        
        Args:
            entity_id: The ID of the entity
            from_quality: The previous quality state
            to_quality: The new quality state
            confidence: Optional confidence score for the transition (0.0-1.0)
            context: Optional contextual information about the transition
        """
        collection = self.db.collection(self.quality_transitions_collection)
        
        # Create transition document
        transition_dict = {
            "_key": str(uuid.uuid4()),
            "entity_id": entity_id,
            "from_quality": from_quality,
            "to_quality": to_quality,
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "context": context or {}
        }
        
        # Insert transition
        collection.insert(transition_dict)
        
        # Update the node's quality state
        self._update_node_quality_state(entity_id, to_quality, confidence)
    
    def get_quality_transitions(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        Get quality transitions for an entity.
        
        Args:
            entity_id: The ID of the entity
            
        Returns:
            A list of quality transitions for the entity
        """
        query = """
        FOR doc IN @@collection
            FILTER doc.entity_id == @entity_id
            SORT doc.timestamp DESC
            RETURN {
                from_quality: doc.from_quality,
                to_quality: doc.to_quality,
                timestamp: doc.timestamp
            }
        """
        
        bind_vars = {
            "@collection": self.quality_transitions_collection,
            "entity_id": entity_id
        }
        
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        
        return list(cursor)
    
    def _update_node_quality_state(self, node_id: str, quality_state: str, confidence: float = None) -> None:
        """
        Update a node's quality state.
        
        Args:
            node_id: The ID of the node
            quality_state: The new quality state
            confidence: Optional confidence score (0.0-1.0)
        """
        try:
            # Get the node collection
            collection = self.db.collection(self.nodes_collection)
            
            # Get the current node
            node_doc = collection.get(node_id)
            if not node_doc:
                return
                
            # Update quality state
            node_doc["quality_state"] = quality_state
            
            # Update confidence if provided
            if confidence is not None:
                if "attributes" not in node_doc:
                    node_doc["attributes"] = {}
                node_doc["attributes"]["confidence"] = str(confidence)
                
            # Update the node
            collection.update(node_id, node_doc)
        except Exception as e:
            print(f"Error updating node quality state: {str(e)}")
    
    def assign_category(self, node_id: str, category: str, confidence: float = None) -> None:
        """
        Assign a category to a node.
        
        Args:
            node_id: The ID of the node
            category: The category to assign
            confidence: Optional confidence score (0.0-1.0)
        """
        try:
            # Get the node collection
            collection = self.db.collection(self.nodes_collection)
            
            # Get the current node
            node_doc = collection.get(node_id)
            if not node_doc:
                return
                
            # Update attributes
            if "attributes" not in node_doc:
                node_doc["attributes"] = {}
                
            node_doc["attributes"]["category"] = category
            
            # Update confidence if provided
            if confidence is not None:
                node_doc["attributes"]["confidence"] = str(confidence)
                
            # Update the node
            collection.update(node_id, node_doc)
        except Exception as e:
            print(f"Error assigning category: {str(e)}")
    
    def get_quality_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of quality states across all nodes.
        
        Returns:
            A dictionary mapping quality states to counts
        """
        query = """
        RETURN {
            "poor": COUNT(FOR doc IN @@collection FILTER doc.quality_state == "poor" RETURN 1),
            "uncertain": COUNT(FOR doc IN @@collection FILTER doc.quality_state == "uncertain" RETURN 1),
            "good": COUNT(FOR doc IN @@collection FILTER doc.quality_state == "good" RETURN 1)
        }
        """
        
        bind_vars = {
            "@collection": self.nodes_collection
        }
        
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        result = next(cursor, {"poor": 0, "uncertain": 0, "good": 0})
        
        return result
    
    def _dict_to_entity(self, entity_dict: Dict[str, Any]) -> Any:
        """
        Convert a dictionary to an entity.
        
        This method is required by the base repository but not used directly
        in this implementation since we have specific conversion methods for
        each entity type.
        
        Args:
            entity_dict: The dictionary to convert
            
        Returns:
            The converted entity
        """
        # This method is not used directly in this implementation
        # We have specific conversion methods for each entity type
        return entity_dict
