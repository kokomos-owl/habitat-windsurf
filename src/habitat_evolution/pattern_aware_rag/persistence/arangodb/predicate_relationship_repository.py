"""
Predicate Relationship Repository for ArangoDB.

This module provides a repository for persisting and retrieving predicate relationships
between actants in the Habitat Evolution system. It supports both specific predicate
edge collections and the generic PredicateRelationship collection.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.connection_manager import ArangoDBConnectionManager

logger = logging.getLogger(__name__)

class PredicateRelationshipRepository:
    """
    Repository for persisting and retrieving predicate relationships between actants.
    
    This repository handles both specific predicate edge collections (Preserves, Protects, etc.)
    and the generic PredicateRelationship collection for dynamic predicate handling.
    """
    
    def __init__(self):
        """Initialize the repository with a database connection."""
        self.connection_manager = ArangoDBConnectionManager()
        self.db = self.connection_manager.get_db()
        
        # Define the specific predicate collections
        self.specific_predicates = [
            "Preserves", "Protects", "Maintains", "Enables", 
            "Enhances", "ReducesDependenceOn", "EvolvesInto", "CoEvolvesWith"
        ]
        
        # Generic predicate collection
        self.generic_predicate = "PredicateRelationship"
    
    def save_relationship(self, 
                         source_id: str, 
                         predicate: str, 
                         target_id: str, 
                         properties: Dict[str, Any]) -> str:
        """
        Save a predicate relationship between two actants.
        
        Args:
            source_id: ID of the source actant
            predicate: The predicate describing the relationship
            target_id: ID of the target actant
            properties: Additional properties of the relationship
        
        Returns:
            The ID of the created edge
        """
        # Normalize predicate to match collection naming convention
        normalized_predicate = self._normalize_predicate(predicate)
        
        # Check if we have a specific collection for this predicate
        if normalized_predicate in self.specific_predicates:
            collection_name = normalized_predicate
            # No need to store predicate_type in specific collections
            edge_properties = properties.copy()
        else:
            # Use generic predicate relationship
            collection_name = self.generic_predicate
            # Include predicate_type in generic collection
            edge_properties = properties.copy()
            edge_properties["predicate_type"] = predicate
        
        # Get the collection
        collection = self.db.collection(collection_name)
        
        # Prepare edge document
        edge = {
            "_from": source_id,
            "_to": target_id,
            "timestamp": datetime.now().isoformat(),
            "first_observed": properties.get("first_observed", datetime.now().isoformat()),
            "last_observed": properties.get("last_observed", datetime.now().isoformat()),
            "observation_count": properties.get("observation_count", 1),
            "confidence": properties.get("confidence", 1.0)
        }
        
        # Add harmonic properties if present
        if "harmonic_properties" in properties:
            edge["harmonic_properties"] = json.dumps(properties["harmonic_properties"])
        
        # Add vector properties if present
        if "vector_properties" in properties:
            edge["vector_properties"] = json.dumps(properties["vector_properties"])
        
        # Add any other properties
        for key, value in edge_properties.items():
            if key not in edge and key not in ["harmonic_properties", "vector_properties"]:
                edge[key] = value
        
        # Insert the edge
        result = collection.insert(edge)
        logger.info(f"Created {collection_name} relationship: {source_id} -> {target_id}")
        
        return result["_id"]
    
    def update_relationship(self, 
                           edge_id: str, 
                           properties: Dict[str, Any]) -> bool:
        """
        Update an existing predicate relationship.
        
        Args:
            edge_id: ID of the edge to update
            properties: New properties to set
        
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Parse the edge ID to get collection name and key
            collection_name, key = edge_id.split('/')
            
            # Get the collection
            collection = self.db.collection(collection_name)
            
            # Prepare update document
            update_doc = {}
            
            # Update last_observed and observation_count
            if "last_observed" in properties:
                update_doc["last_observed"] = properties["last_observed"]
            else:
                update_doc["last_observed"] = datetime.now().isoformat()
            
            if "observation_count" in properties:
                update_doc["observation_count"] = properties["observation_count"]
            else:
                # Increment observation count
                edge = collection.get(key)
                update_doc["observation_count"] = edge.get("observation_count", 1) + 1
            
            # Update confidence if present
            if "confidence" in properties:
                update_doc["confidence"] = properties["confidence"]
            
            # Update harmonic properties if present
            if "harmonic_properties" in properties:
                update_doc["harmonic_properties"] = json.dumps(properties["harmonic_properties"])
            
            # Update vector properties if present
            if "vector_properties" in properties:
                update_doc["vector_properties"] = json.dumps(properties["vector_properties"])
            
            # Add any other properties
            for key, value in properties.items():
                if key not in update_doc and key not in ["harmonic_properties", "vector_properties"]:
                    update_doc[key] = value
            
            # Update the edge
            collection.update(key, update_doc)
            logger.info(f"Updated relationship: {edge_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error updating relationship {edge_id}: {str(e)}")
            return False
    
    def find_by_source_and_target(self, 
                                 source_id: str, 
                                 target_id: str, 
                                 predicate: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find relationships between a source and target actant.
        
        Args:
            source_id: ID of the source actant
            target_id: ID of the target actant
            predicate: Optional specific predicate to search for
        
        Returns:
            List of relationship edges
        """
        results = []
        
        if predicate:
            # Normalize predicate
            normalized_predicate = self._normalize_predicate(predicate)
            
            # Check if we have a specific collection for this predicate
            if normalized_predicate in self.specific_predicates:
                # Query the specific collection
                collection = self.db.collection(normalized_predicate)
                query = f"""
                FOR edge IN {normalized_predicate}
                    FILTER edge._from == @source_id AND edge._to == @target_id
                    RETURN edge
                """
                cursor = self.db.aql.execute(
                    query, 
                    bind_vars={"source_id": source_id, "target_id": target_id}
                )
                results.extend([doc for doc in cursor])
            else:
                # Query the generic collection
                collection = self.db.collection(self.generic_predicate)
                query = f"""
                FOR edge IN {self.generic_predicate}
                    FILTER edge._from == @source_id AND edge._to == @target_id
                    AND edge.predicate_type == @predicate
                    RETURN edge
                """
                cursor = self.db.aql.execute(
                    query, 
                    bind_vars={
                        "source_id": source_id, 
                        "target_id": target_id,
                        "predicate": predicate
                    }
                )
                results.extend([doc for doc in cursor])
        else:
            # Query all collections
            for collection_name in self.specific_predicates + [self.generic_predicate]:
                query = f"""
                FOR edge IN {collection_name}
                    FILTER edge._from == @source_id AND edge._to == @target_id
                    RETURN edge
                """
                cursor = self.db.aql.execute(
                    query, 
                    bind_vars={"source_id": source_id, "target_id": target_id}
                )
                results.extend([doc for doc in cursor])
        
        return results
    
    def find_by_predicate(self, 
                         predicate: str, 
                         confidence_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Find relationships by predicate type.
        
        Args:
            predicate: The predicate to search for
            confidence_threshold: Minimum confidence threshold
        
        Returns:
            List of relationship edges
        """
        results = []
        
        # Normalize predicate
        normalized_predicate = self._normalize_predicate(predicate)
        
        # Check if we have a specific collection for this predicate
        if normalized_predicate in self.specific_predicates:
            # Query the specific collection
            collection = self.db.collection(normalized_predicate)
            query = f"""
            FOR edge IN {normalized_predicate}
                FILTER edge.confidence >= @confidence
                RETURN edge
            """
            cursor = self.db.aql.execute(
                query, 
                bind_vars={"confidence": confidence_threshold}
            )
            results.extend([doc for doc in cursor])
        else:
            # Query the generic collection
            collection = self.db.collection(self.generic_predicate)
            query = f"""
            FOR edge IN {self.generic_predicate}
                FILTER edge.predicate_type == @predicate
                AND edge.confidence >= @confidence
                RETURN edge
            """
            cursor = self.db.aql.execute(
                query, 
                bind_vars={
                    "predicate": predicate,
                    "confidence": confidence_threshold
                }
            )
            results.extend([doc for doc in cursor])
        
        return results
    
    def find_by_source(self, 
                      source_id: str, 
                      predicate: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find relationships by source actant.
        
        Args:
            source_id: ID of the source actant
            predicate: Optional specific predicate to search for
        
        Returns:
            List of relationship edges
        """
        results = []
        
        if predicate:
            # Normalize predicate
            normalized_predicate = self._normalize_predicate(predicate)
            
            # Check if we have a specific collection for this predicate
            if normalized_predicate in self.specific_predicates:
                # Query the specific collection
                query = f"""
                FOR edge IN {normalized_predicate}
                    FILTER edge._from == @source_id
                    RETURN edge
                """
                cursor = self.db.aql.execute(
                    query, 
                    bind_vars={"source_id": source_id}
                )
                results.extend([doc for doc in cursor])
            else:
                # Query the generic collection
                query = f"""
                FOR edge IN {self.generic_predicate}
                    FILTER edge._from == @source_id
                    AND edge.predicate_type == @predicate
                    RETURN edge
                """
                cursor = self.db.aql.execute(
                    query, 
                    bind_vars={
                        "source_id": source_id,
                        "predicate": predicate
                    }
                )
                results.extend([doc for doc in cursor])
        else:
            # Query all collections
            for collection_name in self.specific_predicates + [self.generic_predicate]:
                query = f"""
                FOR edge IN {collection_name}
                    FILTER edge._from == @source_id
                    RETURN edge
                """
                cursor = self.db.aql.execute(
                    query, 
                    bind_vars={"source_id": source_id}
                )
                results.extend([doc for doc in cursor])
        
        return results
    
    def find_by_target(self, 
                      target_id: str, 
                      predicate: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find relationships by target actant.
        
        Args:
            target_id: ID of the target actant
            predicate: Optional specific predicate to search for
        
        Returns:
            List of relationship edges
        """
        results = []
        
        if predicate:
            # Normalize predicate
            normalized_predicate = self._normalize_predicate(predicate)
            
            # Check if we have a specific collection for this predicate
            if normalized_predicate in self.specific_predicates:
                # Query the specific collection
                query = f"""
                FOR edge IN {normalized_predicate}
                    FILTER edge._to == @target_id
                    RETURN edge
                """
                cursor = self.db.aql.execute(
                    query, 
                    bind_vars={"target_id": target_id}
                )
                results.extend([doc for doc in cursor])
            else:
                # Query the generic collection
                query = f"""
                FOR edge IN {self.generic_predicate}
                    FILTER edge._to == @target_id
                    AND edge.predicate_type == @predicate
                    RETURN edge
                """
                cursor = self.db.aql.execute(
                    query, 
                    bind_vars={
                        "target_id": target_id,
                        "predicate": predicate
                    }
                )
                results.extend([doc for doc in cursor])
        else:
            # Query all collections
            for collection_name in self.specific_predicates + [self.generic_predicate]:
                query = f"""
                FOR edge IN {collection_name}
                    FILTER edge._to == @target_id
                    RETURN edge
                """
                cursor = self.db.aql.execute(
                    query, 
                    bind_vars={"target_id": target_id}
                )
                results.extend([doc for doc in cursor])
        
        return results
    
    def find_by_harmonic_properties(self, 
                                   frequency_range: Tuple[float, float] = None,
                                   amplitude_range: Tuple[float, float] = None,
                                   phase_range: Tuple[float, float] = None) -> List[Dict[str, Any]]:
        """
        Find relationships by harmonic properties.
        
        Args:
            frequency_range: Optional tuple of (min, max) frequency
            amplitude_range: Optional tuple of (min, max) amplitude
            phase_range: Optional tuple of (min, max) phase
        
        Returns:
            List of relationship edges
        """
        results = []
        
        # Build filter conditions
        filter_conditions = []
        bind_vars = {}
        
        if frequency_range:
            filter_conditions.append(
                "LOWER(edge.harmonic_properties) LIKE @freq_pattern AND " +
                "JSON_EXTRACT(edge.harmonic_properties, 'frequency') >= @freq_min AND " +
                "JSON_EXTRACT(edge.harmonic_properties, 'frequency') <= @freq_max"
            )
            bind_vars.update({
                "freq_pattern": "%frequency%",
                "freq_min": frequency_range[0],
                "freq_max": frequency_range[1]
            })
        
        if amplitude_range:
            filter_conditions.append(
                "LOWER(edge.harmonic_properties) LIKE @amp_pattern AND " +
                "JSON_EXTRACT(edge.harmonic_properties, 'amplitude') >= @amp_min AND " +
                "JSON_EXTRACT(edge.harmonic_properties, 'amplitude') <= @amp_max"
            )
            bind_vars.update({
                "amp_pattern": "%amplitude%",
                "amp_min": amplitude_range[0],
                "amp_max": amplitude_range[1]
            })
        
        if phase_range:
            filter_conditions.append(
                "LOWER(edge.harmonic_properties) LIKE @phase_pattern AND " +
                "JSON_EXTRACT(edge.harmonic_properties, 'phase') >= @phase_min AND " +
                "JSON_EXTRACT(edge.harmonic_properties, 'phase') <= @phase_max"
            )
            bind_vars.update({
                "phase_pattern": "%phase%",
                "phase_min": phase_range[0],
                "phase_max": phase_range[1]
            })
        
        if not filter_conditions:
            # No filters specified, return empty list
            return []
        
        # Combine filter conditions
        filter_clause = " AND ".join(filter_conditions)
        
        # Query all collections
        for collection_name in self.specific_predicates + [self.generic_predicate]:
            query = f"""
            FOR edge IN {collection_name}
                FILTER {filter_clause}
                RETURN edge
            """
            cursor = self.db.aql.execute(query, bind_vars=bind_vars)
            results.extend([doc for doc in cursor])
        
        return results
    
    def find_by_observation_count(self, 
                                 min_count: int) -> List[Dict[str, Any]]:
        """
        Find relationships by minimum observation count.
        
        Args:
            min_count: Minimum number of observations
        
        Returns:
            List of relationship edges
        """
        results = []
        
        # Query all collections
        for collection_name in self.specific_predicates + [self.generic_predicate]:
            query = f"""
            FOR edge IN {collection_name}
                FILTER edge.observation_count >= @min_count
                RETURN edge
            """
            cursor = self.db.aql.execute(
                query, 
                bind_vars={"min_count": min_count}
            )
            results.extend([doc for doc in cursor])
        
        return results
    
    def delete_relationship(self, edge_id: str) -> bool:
        """
        Delete a relationship edge.
        
        Args:
            edge_id: ID of the edge to delete
        
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Parse the edge ID to get collection name and key
            collection_name, key = edge_id.split('/')
            
            # Get the collection
            collection = self.db.collection(collection_name)
            
            # Delete the edge
            collection.delete(key)
            logger.info(f"Deleted relationship: {edge_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error deleting relationship {edge_id}: {str(e)}")
            return False
    
    def _normalize_predicate(self, predicate: str) -> str:
        """
        Normalize a predicate string to match collection naming convention.
        
        Args:
            predicate: The predicate string
        
        Returns:
            Normalized predicate string
        """
        # Convert to camel case
        words = predicate.lower().split('_')
        normalized = words[0].capitalize() + ''.join(word.capitalize() for word in words[1:])
        
        return normalized
