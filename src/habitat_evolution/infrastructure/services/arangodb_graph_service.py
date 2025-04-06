"""
ArangoDB implementation of the UnifiedGraphServiceInterface for Habitat Evolution.

This module provides a concrete implementation of the UnifiedGraphServiceInterface
using ArangoDB as the persistence layer, supporting the pattern evolution and
co-evolution principles of Habitat Evolution.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid

from src.habitat_evolution.infrastructure.interfaces.services.unified_graph_service_interface import UnifiedGraphServiceInterface
from src.habitat_evolution.infrastructure.interfaces.persistence.arangodb_connection_interface import ArangoDBConnectionInterface
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface

logger = logging.getLogger(__name__)


class ArangoDBGraphService(UnifiedGraphServiceInterface):
    """
    ArangoDB implementation of the UnifiedGraphServiceInterface.
    
    This service provides a consistent approach to graph operations using ArangoDB
    as the persistence layer, supporting the pattern evolution and co-evolution
    principles of Habitat Evolution.
    """
    
    def __init__(self, 
                 db_connection: ArangoDBConnectionInterface,
                 event_service: EventServiceInterface):
        """
        Initialize a new ArangoDB graph service.
        
        Args:
            db_connection: The ArangoDB connection to use
            event_service: The event service for publishing events
        """
        self._db_connection = db_connection
        self._event_service = event_service
        self._initialized = False
        logger.debug("ArangoDBGraphService created")
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the graph service with the specified configuration.
        
        Args:
            config: Optional configuration for the graph service
        """
        if self._initialized:
            logger.warning("ArangoDBGraphService already initialized")
            return
            
        logger.info("Initializing ArangoDBGraphService")
        
        # Ensure required collections exist
        self._db_connection.ensure_collection("concepts")
        self._db_connection.ensure_collection("patterns")
        self._db_connection.ensure_edge_collection("relations")
        self._db_connection.ensure_edge_collection("pattern_relations")
        
        # Create graph if it doesn't exist
        self._db_connection.ensure_graph(
            "habitat_graph",
            edge_definitions=[
                {
                    "collection": "relations",
                    "from": ["concepts"],
                    "to": ["concepts"]
                },
                {
                    "collection": "pattern_relations",
                    "from": ["patterns"],
                    "to": ["patterns"]
                }
            ]
        )
        
        self._initialized = True
        logger.info("ArangoDBGraphService initialized")
    
    def shutdown(self) -> None:
        """
        Release resources when shutting down the graph service.
        """
        if not self._initialized:
            logger.warning("ArangoDBGraphService not initialized")
            return
            
        logger.info("Shutting down ArangoDBGraphService")
        self._initialized = False
        logger.info("ArangoDBGraphService shut down")
    
    def create_concept(self, name: str, concept_type: str, 
                      attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a concept in the graph.
        
        Args:
            name: The name of the concept
            concept_type: The type of the concept
            attributes: Optional attributes for the concept
            
        Returns:
            The created concept
        """
        if not self._initialized:
            self.initialize()
            
        concept_id = f"concept/{str(uuid.uuid4())}"
        concept = {
            "_key": concept_id.split("/")[1],
            "name": name,
            "type": concept_type,
            "attributes": attributes or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = self._db_connection.insert("concepts", concept)
        
        # Publish event
        self._event_service.publish("concept.created", {
            "concept_id": result["_id"],
            "name": name,
            "type": concept_type
        })
        
        return result
    
    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a concept by ID.
        
        Args:
            concept_id: The ID of the concept to get
            
        Returns:
            The concept, or None if not found
        """
        if not self._initialized:
            self.initialize()
            
        try:
            return self._db_connection.get_document("concepts", concept_id)
        except Exception as e:
            logger.error(f"Error getting concept {concept_id}: {str(e)}")
            return None
    
    def find_concepts_by_name(self, name: str, exact_match: bool = False) -> List[Dict[str, Any]]:
        """
        Find concepts by name.
        
        Args:
            name: The name to search for
            exact_match: Whether to require an exact match
            
        Returns:
            A list of matching concepts
        """
        if not self._initialized:
            self.initialize()
            
        query = "FOR c IN concepts"
        
        if exact_match:
            query += f" FILTER c.name == @name"
        else:
            query += f" FILTER LIKE(c.name, @name, true)"
            name = f"%{name}%"
            
        query += " RETURN c"
        
        return self._db_connection.execute_query(query, {"name": name})
    
    def find_concepts_by_type(self, concept_type: str) -> List[Dict[str, Any]]:
        """
        Find concepts by type.
        
        Args:
            concept_type: The type to search for
            
        Returns:
            A list of matching concepts
        """
        if not self._initialized:
            self.initialize()
            
        query = "FOR c IN concepts FILTER c.type == @type RETURN c"
        return self._db_connection.execute_query(query, {"type": concept_type})
    
    def update_concept(self, concept_id: str, 
                      updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update a concept.
        
        Args:
            concept_id: The ID of the concept to update
            updates: The updates to apply to the concept
            
        Returns:
            The updated concept, or None if not found
        """
        if not self._initialized:
            self.initialize()
            
        # Get the current concept
        concept = self.get_concept(concept_id)
        if not concept:
            return None
            
        # Apply updates
        for key, value in updates.items():
            if key == "attributes":
                # Merge attributes
                concept["attributes"].update(value)
            else:
                concept[key] = value
                
        concept["updated_at"] = datetime.utcnow().isoformat()
        
        # Update the concept
        result = self._db_connection.update_document("concepts", concept_id, concept)
        
        # Publish event
        self._event_service.publish("concept.updated", {
            "concept_id": concept_id,
            "updates": updates
        })
        
        return result
    
    def delete_concept(self, concept_id: str) -> bool:
        """
        Delete a concept.
        
        Args:
            concept_id: The ID of the concept to delete
            
        Returns:
            True if the concept was deleted, False otherwise
        """
        if not self._initialized:
            self.initialize()
            
        try:
            # Delete all relations involving this concept
            query = """
            FOR r IN relations
            FILTER r._from == @concept_id OR r._to == @concept_id
            REMOVE r IN relations
            """
            self._db_connection.execute_query(query, {"concept_id": concept_id})
            
            # Delete the concept
            self._db_connection.delete_document("concepts", concept_id)
            
            # Publish event
            self._event_service.publish("concept.deleted", {
                "concept_id": concept_id
            })
            
            return True
        except Exception as e:
            logger.error(f"Error deleting concept {concept_id}: {str(e)}")
            return False
    
    def create_relation(self, source_id: str, target_id: str, relation_type: str,
                       attributes: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Create a relation between two concepts.
        
        Args:
            source_id: The ID of the source concept
            target_id: The ID of the target concept
            relation_type: The type of the relation
            attributes: Optional attributes for the relation
            
        Returns:
            The created relation, or None if the concepts don't exist
        """
        if not self._initialized:
            self.initialize()
            
        # Verify that both concepts exist
        source = self.get_concept(source_id)
        target = self.get_concept(target_id)
        
        if not source or not target:
            logger.error(f"Cannot create relation: source or target not found")
            return None
            
        relation = {
            "_from": source_id,
            "_to": target_id,
            "type": relation_type,
            "attributes": attributes or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = self._db_connection.insert("relations", relation)
        
        # Publish event
        self._event_service.publish("relation.created", {
            "relation_id": result["_id"],
            "source_id": source_id,
            "target_id": target_id,
            "type": relation_type
        })
        
        return result
    
    def find_relations(self, source_id: Optional[str] = None, 
                      target_id: Optional[str] = None,
                      relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find relations matching the specified criteria.
        
        Args:
            source_id: Optional ID of the source concept
            target_id: Optional ID of the target concept
            relation_type: Optional type of the relation
            
        Returns:
            A list of matching relations
        """
        if not self._initialized:
            self.initialize()
            
        query = "FOR r IN relations"
        filters = []
        params = {}
        
        if source_id:
            filters.append("r._from == @source_id")
            params["source_id"] = source_id
            
        if target_id:
            filters.append("r._to == @target_id")
            params["target_id"] = target_id
            
        if relation_type:
            filters.append("r.type == @relation_type")
            params["relation_type"] = relation_type
            
        if filters:
            query += " FILTER " + " AND ".join(filters)
            
        query += " RETURN r"
        
        return self._db_connection.execute_query(query, params)
    
    def update_relation(self, relation_id: str,
                       updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update a relation.
        
        Args:
            relation_id: The ID of the relation to update
            updates: The updates to apply to the relation
            
        Returns:
            The updated relation, or None if not found
        """
        if not self._initialized:
            self.initialize()
            
        try:
            # Get the current relation
            relation = self._db_connection.get_document("relations", relation_id)
            
            # Apply updates
            for key, value in updates.items():
                if key == "attributes":
                    # Merge attributes
                    relation["attributes"].update(value)
                else:
                    relation[key] = value
                    
            relation["updated_at"] = datetime.utcnow().isoformat()
            
            # Update the relation
            result = self._db_connection.update_document("relations", relation_id, relation)
            
            # Publish event
            self._event_service.publish("relation.updated", {
                "relation_id": relation_id,
                "updates": updates
            })
            
            return result
        except Exception as e:
            logger.error(f"Error updating relation {relation_id}: {str(e)}")
            return None
    
    def delete_relation(self, relation_id: str) -> bool:
        """
        Delete a relation.
        
        Args:
            relation_id: The ID of the relation to delete
            
        Returns:
            True if the relation was deleted, False otherwise
        """
        if not self._initialized:
            self.initialize()
            
        try:
            # Delete the relation
            self._db_connection.delete_document("relations", relation_id)
            
            # Publish event
            self._event_service.publish("relation.deleted", {
                "relation_id": relation_id
            })
            
            return True
        except Exception as e:
            logger.error(f"Error deleting relation {relation_id}: {str(e)}")
            return False
    
    def create_pattern(self, name: str, concepts: List[str],
                      attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a pattern from a set of concepts.
        
        Args:
            name: The name of the pattern
            concepts: The IDs of the concepts in the pattern
            attributes: Optional attributes for the pattern
            
        Returns:
            The created pattern
        """
        if not self._initialized:
            self.initialize()
            
        pattern_id = f"pattern/{str(uuid.uuid4())}"
        pattern = {
            "_key": pattern_id.split("/")[1],
            "name": name,
            "concepts": concepts,
            "attributes": attributes or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = self._db_connection.insert("patterns", pattern)
        
        # Publish event
        self._event_service.publish("pattern.created", {
            "pattern_id": result["_id"],
            "name": name,
            "concepts": concepts
        })
        
        return result
    
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a pattern by ID.
        
        Args:
            pattern_id: The ID of the pattern to get
            
        Returns:
            The pattern, or None if not found
        """
        if not self._initialized:
            self.initialize()
            
        try:
            return self._db_connection.get_document("patterns", pattern_id)
        except Exception as e:
            logger.error(f"Error getting pattern {pattern_id}: {str(e)}")
            return None
    
    def get_concept_neighborhood(self, concept_id: str, 
                                depth: int = 1) -> Dict[str, Any]:
        """
        Get the neighborhood of a concept.
        
        Args:
            concept_id: The ID of the concept
            depth: The depth of the neighborhood
            
        Returns:
            The neighborhood as a subgraph
        """
        if not self._initialized:
            self.initialize()
            
        query = """
        LET concept = DOCUMENT(@concept_id)
        LET neighbors = (
            FOR v, e, p IN 1..@depth ANY @concept_id GRAPH 'habitat_graph'
            RETURN {
                "vertex": v,
                "edge": e,
                "path": p
            }
        )
        RETURN {
            "center": concept,
            "neighbors": neighbors
        }
        """
        
        results = self._db_connection.execute_query(query, {
            "concept_id": concept_id,
            "depth": depth
        })
        
        return results[0] if results else {"center": None, "neighbors": []}
    
    def calculate_semantic_similarity(self, concept_id1: str, 
                                    concept_id2: str) -> float:
        """
        Calculate the semantic similarity between two concepts.
        
        Args:
            concept_id1: The ID of the first concept
            concept_id2: The ID of the second concept
            
        Returns:
            The semantic similarity (0-1)
        """
        if not self._initialized:
            self.initialize()
            
        # This is a placeholder implementation
        # In a real implementation, this would use vector embeddings
        # or other semantic similarity measures
        
        # For now, we'll check if the concepts are directly connected
        relations = self.find_relations(source_id=concept_id1, target_id=concept_id2)
        if relations:
            return 0.8
            
        relations = self.find_relations(source_id=concept_id2, target_id=concept_id1)
        if relations:
            return 0.8
            
        # Check if they share common neighbors
        query = """
        LET neighbors1 = (
            FOR v, e IN 1..1 ANY @concept_id1 GRAPH 'habitat_graph'
            RETURN v._id
        )
        LET neighbors2 = (
            FOR v, e IN 1..1 ANY @concept_id2 GRAPH 'habitat_graph'
            RETURN v._id
        )
        LET common = LENGTH(INTERSECTION(neighbors1, neighbors2))
        LET total = LENGTH(UNION(neighbors1, neighbors2))
        RETURN total > 0 ? common / total : 0
        """
        
        results = self._db_connection.execute_query(query, {
            "concept_id1": concept_id1,
            "concept_id2": concept_id2
        })
        
        return results[0] if results else 0.0
    
    def find_path(self, source_id: str, target_id: str, 
                 max_depth: int = 5) -> List[Dict[str, Any]]:
        """
        Find a path between two concepts.
        
        Args:
            source_id: The ID of the source concept
            target_id: The ID of the target concept
            max_depth: The maximum path depth to search
            
        Returns:
            A list of edges forming the path, or an empty list if no path exists
        """
        if not self._initialized:
            self.initialize()
            
        query = """
        FOR p IN 1..@max_depth ANY SHORTEST_PATH @source_id TO @target_id GRAPH 'habitat_graph'
        RETURN p
        """
        
        results = self._db_connection.execute_query(query, {
            "source_id": source_id,
            "target_id": target_id,
            "max_depth": max_depth
        })
        
        return results[0] if results else []
    
    def execute_graph_query(self, query: str, 
                          params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom graph query.
        
        Args:
            query: The AQL query to execute
            params: Optional parameters for the query
            
        Returns:
            The query results
        """
        if not self._initialized:
            self.initialize()
            
        return self._db_connection.execute_query(query, params or {})
    
    def create_graph_snapshot(self, name: str, 
                            metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a snapshot of the current graph state.
        
        Args:
            name: The name of the snapshot
            metadata: Optional metadata for the snapshot
            
        Returns:
            The created snapshot
        """
        if not self._initialized:
            self.initialize()
            
        # Ensure snapshot collection exists
        self._db_connection.ensure_collection("graph_snapshots")
        
        # Create snapshot
        snapshot_id = f"snapshot/{str(uuid.uuid4())}"
        snapshot = {
            "_key": snapshot_id.split("/")[1],
            "name": name,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Get all concepts and relations
        concepts = self._db_connection.execute_query("FOR c IN concepts RETURN c")
        relations = self._db_connection.execute_query("FOR r IN relations RETURN r")
        patterns = self._db_connection.execute_query("FOR p IN patterns RETURN p")
        
        snapshot["data"] = {
            "concepts": concepts,
            "relations": relations,
            "patterns": patterns
        }
        
        result = self._db_connection.insert("graph_snapshots", snapshot)
        
        # Publish event
        self._event_service.publish("graph.snapshot_created", {
            "snapshot_id": result["_id"],
            "name": name
        })
        
        return result
