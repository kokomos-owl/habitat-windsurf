"""
Predicate Relationship Repository for ArangoDB.

This module provides a repository for persisting and retrieving predicate relationships
between actants in the Habitat Evolution system. It supports both specific predicate
edge collections and the generic PredicateRelationship collection.
"""

import json
import logging
import traceback
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
        
        # Flatten harmonic properties if present
        if "harmonic_properties" in properties:
            # Keep the original JSON for backward compatibility
            edge["harmonic_properties"] = json.dumps(properties["harmonic_properties"])
            
            # Flatten the properties for direct querying
            if isinstance(properties["harmonic_properties"], dict):
                harmonic_props = properties["harmonic_properties"]
                if "frequency" in harmonic_props:
                    edge["harmonic_frequency"] = harmonic_props["frequency"]
                if "amplitude" in harmonic_props:
                    edge["harmonic_amplitude"] = harmonic_props["amplitude"]
                if "phase" in harmonic_props:
                    edge["harmonic_phase"] = harmonic_props["phase"]
        
        # Add vector properties if present
        if "vector_properties" in properties:
            edge["vector_properties"] = json.dumps(properties["vector_properties"])
            
        # Add bidirectionality flag if applicable
        if "is_bidirectional" in properties:
            edge["is_bidirectional"] = properties["is_bidirectional"]
        elif normalized_predicate == "CoEvolvesWith":
            # CoEvolvesWith is inherently bidirectional
            edge["is_bidirectional"] = True
        else:
            edge["is_bidirectional"] = False
            
        # Add impact strength and direction if present
        if "impact_strength" in properties:
            edge["impact_strength"] = properties["impact_strength"]
        else:
            # Default impact strength based on confidence
            edge["impact_strength"] = properties.get("confidence", 1.0)
            
        if "impact_direction" in properties:
            edge["impact_direction"] = properties["impact_direction"]
        else:
            # Default impact direction based on predicate type
            if normalized_predicate in ["Enhances", "Enables", "Preserves", "Protects", "Maintains"]:
                edge["impact_direction"] = 1.0  # Positive impact
            elif normalized_predicate in ["ReducesDependenceOn"]:
                edge["impact_direction"] = -1.0  # Negative impact
            else:
                edge["impact_direction"] = 0.0  # Neutral impact
        
        # Add any other properties
        for key, value in edge_properties.items():
            if key not in edge and key not in ["harmonic_properties", "vector_properties"]:
                edge[key] = value
        
        # Insert the edge
        result = collection.insert(edge)
        logger.info(f"Created {collection_name} relationship: {source_id} -> {target_id}")
        
        # Log the edge ID for debugging
        edge_id = result["_id"]
        logger.info(f"Edge ID created: {edge_id}")
        
        return edge_id
    
    def update_relationship(self, 
                           edge_id: str, 
                           properties: Dict[str, Any]) -> bool:
        """
        Update an existing predicate relationship.
        
        This method updates a relationship with new properties, including harmonic properties
        that capture the resonance characteristics of predicates. It also tracks the evolution
        of properties over time to support the detection of emergent patterns.
        
        Args:
            edge_id: ID of the edge to update (format: collection_name/key)
            properties: New properties to set
        
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Log the incoming edge ID and properties
            logger.info(f"Updating relationship with edge_id: {edge_id}")
            logger.info(f"Update properties: {properties}")
            
            # Parse the edge ID to get collection name and key
            parts = edge_id.split('/')
            logger.info(f"Edge ID parts: {parts}")
            
            if len(parts) != 2:
                logger.error(f"Invalid edge ID format: {edge_id}")
                return False
                
            collection_name, key = parts
            logger.info(f"Collection name: {collection_name}, key: {key}")
            
            # Get the collection
            try:
                collection = self.db.collection(collection_name)
                logger.info(f"Successfully accessed collection: {collection_name}")
            except Exception as e:
                logger.error(f"Could not access collection {collection_name}: {str(e)}")
                return False
            
            # Check if the edge exists
            has_key = collection.has(key)
            logger.info(f"Collection has key {key}: {has_key}")
            
            if not has_key:
                logger.error(f"Edge not found: {edge_id}")
                return False
            
            # Get the current edge document
            try:
                edge = collection.get(key)
                logger.info(f"Retrieved edge document: {edge}")
            except Exception as e:
                logger.error(f"Error retrieving edge document: {str(e)}")
                return False
            
            # Create update document with required fields
            update_doc = {
                "_key": key,
                "_from": edge["_from"],
                "_to": edge["_to"],
                "timestamp": datetime.now().isoformat(),
                "first_observed": edge.get("first_observed", datetime.now().isoformat()),
                "last_observed": properties.get("last_observed", datetime.now().isoformat())
            }
            logger.info(f"Created update document with basic fields")
            
            # Update confidence and observation count
            update_doc["confidence"] = properties.get("confidence", edge.get("confidence", 1.0))
            update_doc["observation_count"] = properties.get("observation_count", edge.get("observation_count", 1) + 1)
            
            # Process harmonic properties - this is key for predicate resonance characteristics
            if "harmonic_properties" in properties and isinstance(properties["harmonic_properties"], dict):
                harmonic_props = properties["harmonic_properties"]
                
                # Store as JSON string for compatibility
                update_doc["harmonic_properties"] = json.dumps(harmonic_props)
                
                # Store flattened properties for direct querying
                if "frequency" in harmonic_props:
                    update_doc["harmonic_frequency"] = harmonic_props["frequency"]
                if "amplitude" in harmonic_props:
                    update_doc["harmonic_amplitude"] = harmonic_props["amplitude"]
                if "phase" in harmonic_props:
                    update_doc["harmonic_phase"] = harmonic_props["phase"]
            elif "harmonic_properties" in edge:
                # Preserve existing harmonic properties
                update_doc["harmonic_properties"] = edge["harmonic_properties"]
                
                # Preserve flattened properties
                if "harmonic_frequency" in edge:
                    update_doc["harmonic_frequency"] = edge["harmonic_frequency"]
                if "harmonic_amplitude" in edge:
                    update_doc["harmonic_amplitude"] = edge["harmonic_amplitude"]
                if "harmonic_phase" in edge:
                    update_doc["harmonic_phase"] = edge["harmonic_phase"]
            
            # Update bidirectionality flag if present
            if "is_bidirectional" in properties:
                update_doc["is_bidirectional"] = properties["is_bidirectional"]
            elif "is_bidirectional" in edge:
                update_doc["is_bidirectional"] = edge["is_bidirectional"]
            
            # Update impact metrics
            if "impact_strength" in properties:
                update_doc["impact_strength"] = properties["impact_strength"]
            elif "impact_strength" in edge:
                update_doc["impact_strength"] = edge["impact_strength"]
                
            if "impact_direction" in properties:
                update_doc["impact_direction"] = properties["impact_direction"]
            elif "impact_direction" in edge:
                update_doc["impact_direction"] = edge["impact_direction"]
            
            # Track property evolution
            evolution_entry = {
                "timestamp": datetime.now().isoformat(),
                "confidence": update_doc["confidence"],
                "observation_count": update_doc["observation_count"]
            }
            
            # Add harmonic properties to evolution tracking
            if "harmonic_frequency" in update_doc:
                evolution_entry["harmonic_frequency"] = update_doc["harmonic_frequency"]
            if "harmonic_amplitude" in update_doc:
                evolution_entry["harmonic_amplitude"] = update_doc["harmonic_amplitude"]
            if "harmonic_phase" in update_doc:
                evolution_entry["harmonic_phase"] = update_doc["harmonic_phase"]
            if "impact_strength" in update_doc:
                evolution_entry["impact_strength"] = update_doc["impact_strength"]
            if "impact_direction" in update_doc:
                evolution_entry["impact_direction"] = update_doc["impact_direction"]
            
            # Add evolution entry to document
            if "evolution" in edge and isinstance(edge["evolution"], list):
                update_doc["evolution"] = edge["evolution"] + [evolution_entry]
            else:
                update_doc["evolution"] = [evolution_entry]
            
            # Preserve predicate_type for generic relationships
            if "predicate_type" in edge:
                update_doc["predicate_type"] = edge["predicate_type"]
            
            # Preserve any other properties from the original edge
            for k, value in edge.items():
                if k not in update_doc and not k.startswith("_"):
                    update_doc[k] = value
            
            # Add any other new properties
            for k, value in properties.items():
                if k not in update_doc and k != "harmonic_properties":
                    update_doc[k] = value
            
            # Update the document
            try:
                logger.info(f"Replacing document with key: {key}")
                collection.replace(key, update_doc)
                logger.info(f"Successfully updated document in collection {collection_name}")
                return True
            except Exception as e:
                logger.error(f"Error replacing document: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return False
        except Exception as e:
            logger.error(f"Error updating relationship {edge_id}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
        # Build filter conditions for AQL query using the flattened properties
        filter_conditions = []
        bind_vars = {}
        
        if frequency_range:
            filter_conditions.append(
                "edge.harmonic_frequency >= @freq_min AND " +
                "edge.harmonic_frequency <= @freq_max"
            )
            bind_vars.update({
                "freq_min": frequency_range[0],
                "freq_max": frequency_range[1]
            })
        
        if amplitude_range:
            filter_conditions.append(
                "edge.harmonic_amplitude >= @amp_min AND " +
                "edge.harmonic_amplitude <= @amp_max"
            )
            bind_vars.update({
                "amp_min": amplitude_range[0],
                "amp_max": amplitude_range[1]
            })
        
        if phase_range:
            filter_conditions.append(
                "edge.harmonic_phase >= @phase_min AND " +
                "edge.harmonic_phase <= @phase_max"
            )
            bind_vars.update({
                "phase_min": phase_range[0],
                "phase_max": phase_range[1]
            })
        
        # If no filters specified, return empty list
        if not filter_conditions:
            return []
        
        # Combine filter conditions
        filter_clause = " AND ".join(filter_conditions)
        
        # Query all collections with the filter
        results = []
        for collection_name in self.specific_predicates + [self.generic_predicate]:
            query = f"""
            FOR edge IN {collection_name}
                FILTER {filter_clause}
                RETURN edge
            """
            try:
                cursor = self.db.aql.execute(query, bind_vars=bind_vars)
                results.extend([doc for doc in cursor])
            except Exception as e:
                logger.error(f"Error querying collection {collection_name}: {str(e)}")
                
                # Fallback to Python-side filtering if AQL query fails
                try:
                    collection = self.db.collection(collection_name)
                    cursor = self.db.aql.execute(
                        "FOR edge IN @@collection RETURN edge",
                        bind_vars={"@collection": collection_name}
                    )
                    
                    for edge in cursor:
                        meets_criteria = True
                        
                        if frequency_range and "harmonic_frequency" in edge:
                            freq = edge["harmonic_frequency"]
                            if not (frequency_range[0] <= freq <= frequency_range[1]):
                                meets_criteria = False
                        
                        if amplitude_range and "harmonic_amplitude" in edge:
                            amp = edge["harmonic_amplitude"]
                            if not (amplitude_range[0] <= amp <= amplitude_range[1]):
                                meets_criteria = False
                        
                        if phase_range and "harmonic_phase" in edge:
                            phase = edge["harmonic_phase"]
                            if not (phase_range[0] <= phase <= phase_range[1]):
                                meets_criteria = False
                        
                        if meets_criteria:
                            results.append(edge)
                except Exception as fallback_error:
                    logger.error(f"Fallback filtering failed: {str(fallback_error)}")
        
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
            cursor = self.db.aql.execute(query, bind_vars={"min_count": min_count})
            results.extend([doc for doc in cursor])
        
        return results
        
    def find_by_actant(self, 
                      actant_id: str, 
                      predicate: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find relationships where actant is either source or target.
        
        This method supports bidirectional relationships, returning edges where
        the actant is either the source or target, if the relationship is bidirectional.
        
        Args:
            actant_id: ID of the actant to search for
            predicate: Optional specific predicate to search for
        
        Returns:
            List of relationship edges
        """
        results = []
        
        # Normalize predicate if provided
        if predicate:
            normalized_predicate = self._normalize_predicate(predicate)
            collections_to_search = [normalized_predicate] if normalized_predicate in self.specific_predicates else [self.generic_predicate]
        else:
            collections_to_search = self.specific_predicates + [self.generic_predicate]
        
        # Query all relevant collections
        for collection_name in collections_to_search:
            # Query for relationships where actant is source or target (if bidirectional)
            query = f"""
            FOR edge IN {collection_name}
                FILTER edge._from == @actant_id OR 
                       (edge._to == @actant_id AND edge.is_bidirectional == true)
                {f"FILTER edge.predicate_type == @predicate" if predicate and collection_name == self.generic_predicate else ""}
                RETURN edge
            """
            
            bind_vars = {"actant_id": actant_id}
            if predicate and collection_name == self.generic_predicate:
                bind_vars["predicate"] = predicate
                
            cursor = self.db.aql.execute(query, bind_vars=bind_vars)
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
