"""
Vector Tonic Service implementation for Habitat Evolution.

This module provides a concrete implementation of the VectorTonicServiceInterface,
supporting the pattern evolution and co-evolution principles of Habitat Evolution.
"""

import logging
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from src.habitat_evolution.infrastructure.interfaces.services.vector_tonic_service_interface import VectorTonicServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_pattern_repository import ArangoDBPatternRepository
from src.habitat_evolution.infrastructure.interfaces.persistence.arangodb_connection_interface import ArangoDBConnectionInterface

logger = logging.getLogger(__name__)


class VectorTonicService(VectorTonicServiceInterface):
    """
    Implementation of the VectorTonicServiceInterface.
    
    This service provides vector operations and tonic-harmonic pattern detection
    functionality for the Habitat Evolution system.
    """
    
    def __init__(self, 
                 db_connection: ArangoDBConnectionInterface,
                 event_service: EventServiceInterface,
                 pattern_repository: ArangoDBPatternRepository):
        """
        Initialize a new vector tonic service.
        
        Args:
            db_connection: The ArangoDB connection to use
            event_service: The event service for publishing events
            pattern_repository: The pattern repository
        """
        self._db_connection = db_connection
        self._event_service = event_service
        self._pattern_repository = pattern_repository
        self._vector_spaces = {}
        self._initialized = False
        logger.debug("VectorTonicService created")
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the vector tonic service with the specified configuration.
        
        Args:
            config: Optional configuration for the service
        """
        if self._initialized:
            logger.warning("VectorTonicService already initialized")
            return
            
        logger.info("Initializing VectorTonicService")
        
        # Ensure collections exist
        self._db_connection.ensure_collection("vector_spaces")
        self._db_connection.ensure_collection("vectors")
        self._db_connection.ensure_collection("patterns")
        self._db_connection.ensure_edge_collection("pattern_vectors")
        
        # Ensure collections exist first
        self._db_connection.ensure_collection("vector_spaces")
        self._db_connection.ensure_collection("vectors")
        self._db_connection.ensure_collection("patterns")
        self._db_connection.ensure_edge_collection("pattern_vectors")
        
        # Check if graph exists
        if not self._db_connection.graph_exists("vector_tonic_graph"):
            try:
                # Create the graph using the ensure_graph method which handles different edge definition formats
                edge_definitions = [
                    {
                        "collection": "pattern_vectors",  # Use 'collection' instead of 'edge_collection'
                        "from": ["patterns"],           # Use 'from' instead of 'from_collections'
                        "to": ["vectors"]              # Use 'to' instead of 'to_collections'
                    }
                ]
                
                # Use the ensure_graph method which normalizes edge definitions
                self._db_connection.ensure_graph("vector_tonic_graph", edge_definitions)
                
                logger.info("Created vector_tonic_graph successfully")
            except Exception as e:
                logger.error(f"Error creating graph: {e}")
                # Log more detailed information about the error
                import traceback
                logger.error(f"Detailed error: {traceback.format_exc()}")
                raise
        
        # Load existing vector spaces
        query = "FOR vs IN vector_spaces RETURN vs"
        vector_spaces = self._db_connection.execute_query(query)
        
        for vs in vector_spaces:
            self._vector_spaces[vs["_id"]] = {
                "name": vs["name"],
                "dimensions": vs["dimensions"],
                "metadata": vs.get("metadata", {})
            }
        
        self._initialized = True
        logger.info("VectorTonicService initialized")
        
        # Publish initialization event
        self._event_service.publish("vector_tonic.initialized", {
            "vector_spaces_count": len(self._vector_spaces)
        })
    
    def register_vector_space(self, name: str, dimensions: int, 
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a new vector space.
        
        Args:
            name: The name of the vector space
            dimensions: The number of dimensions in the vector space
            metadata: Optional metadata for the vector space
            
        Returns:
            The ID of the registered vector space
        """
        if not self._initialized:
            self.initialize()
            
        # Check if vector space with this name already exists
        query = "FOR vs IN vector_spaces FILTER vs.name == @name RETURN vs"
        existing = self._db_connection.execute_query(query, {"name": name})
        
        if existing:
            logger.info(f"Vector space '{name}' already exists, returning existing ID")
            return existing[0]["_id"]
            
        # Create new vector space
        vector_space = {
            "name": name,
            "dimensions": dimensions,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = self._db_connection.insert("vector_spaces", vector_space)
        vector_space_id = result["_id"]
        
        # Cache vector space
        self._vector_spaces[vector_space_id] = {
            "name": name,
            "dimensions": dimensions,
            "metadata": metadata or {}
        }
        
        # Publish event
        self._event_service.publish("vector_tonic.vector_space_registered", {
            "vector_space_id": vector_space_id,
            "name": name,
            "dimensions": dimensions
        })
        
        return vector_space_id
    
    def store_vector(self, vector_space_id: str, vector: List[float], 
                    entity_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a vector in the specified vector space.
        
        Args:
            vector_space_id: The ID of the vector space
            vector: The vector to store
            entity_id: The ID of the entity associated with the vector
            metadata: Optional metadata for the vector
            
        Returns:
            The ID of the stored vector
        """
        if not self._initialized:
            self.initialize()
            
        # Validate vector space
        if vector_space_id not in self._vector_spaces:
            raise ValueError(f"Vector space {vector_space_id} not found")
            
        # Validate vector dimensions
        expected_dims = self._vector_spaces[vector_space_id]["dimensions"]
        if len(vector) != expected_dims:
            raise ValueError(f"Vector has {len(vector)} dimensions, expected {expected_dims}")
            
        # Create vector document
        vector_doc = {
            "vector": vector,
            "vector_space_id": vector_space_id,
            "entity_id": entity_id,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = self._db_connection.insert("vectors", vector_doc)
        vector_id = result["_id"]
        
        # Publish event
        self._event_service.publish("vector_tonic.vector_stored", {
            "vector_id": vector_id,
            "vector_space_id": vector_space_id,
            "entity_id": entity_id
        })
        
        return vector_id
    
    def find_similar_vectors(self, vector_space_id: str, 
                            query_vector: List[float],
                            limit: int = 10,
                            threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Find vectors similar to the query vector.
        
        Args:
            vector_space_id: The ID of the vector space
            query_vector: The query vector
            limit: The maximum number of results
            threshold: The minimum similarity threshold
            
        Returns:
            A list of similar vectors with their metadata and similarity scores
        """
        if not self._initialized:
            self.initialize()
            
        # Validate vector space
        if vector_space_id not in self._vector_spaces:
            raise ValueError(f"Vector space {vector_space_id} not found")
            
        # Get all vectors in the vector space
        query = """
        FOR v IN vectors
        FILTER v.vector_space_id == @vector_space_id
        RETURN v
        """
        
        vectors = self._db_connection.execute_query(query, {"vector_space_id": vector_space_id})
        
        # Calculate cosine similarity for each vector
        query_vector_np = np.array(query_vector)
        results = []
        
        for vec_doc in vectors:
            vec = np.array(vec_doc["vector"])
            
            # Calculate cosine similarity
            similarity = np.dot(query_vector_np, vec) / (np.linalg.norm(query_vector_np) * np.linalg.norm(vec))
            
            if similarity >= threshold:
                results.append({
                    "vector_id": vec_doc["_id"],
                    "entity_id": vec_doc["entity_id"],
                    "similarity": float(similarity),
                    "vector": vec_doc["vector"],
                    "metadata": vec_doc.get("metadata", {})
                })
        
        # Sort by similarity (descending) and limit results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]
    
    def detect_tonic_patterns(self, vector_space_id: str, 
                             vectors: List[List[float]],
                             threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Detect tonic patterns in the specified vectors.
        
        Args:
            vector_space_id: The ID of the vector space
            vectors: The vectors to analyze
            threshold: The pattern detection threshold
            
        Returns:
            A list of detected patterns with their metadata
        """
        if not self._initialized:
            self.initialize()
            
        # Validate vector space
        if vector_space_id not in self._vector_spaces:
            raise ValueError(f"Vector space {vector_space_id} not found")
            
        # Convert vectors to numpy arrays
        np_vectors = [np.array(v) for v in vectors]
        
        # Simple clustering based on cosine similarity
        patterns = []
        assigned = [False] * len(np_vectors)
        
        for i in range(len(np_vectors)):
            if assigned[i]:
                continue
                
            # Start a new pattern
            pattern_vectors = [i]
            assigned[i] = True
            
            # Find similar vectors
            for j in range(i + 1, len(np_vectors)):
                if assigned[j]:
                    continue
                    
                # Calculate cosine similarity
                similarity = np.dot(np_vectors[i], np_vectors[j]) / (np.linalg.norm(np_vectors[i]) * np.linalg.norm(np_vectors[j]))
                
                if similarity >= threshold:
                    pattern_vectors.append(j)
                    assigned[j] = True
            
            # Only create patterns with at least 2 vectors
            if len(pattern_vectors) > 1:
                # Calculate centroid
                centroid = np.mean([np_vectors[idx] for idx in pattern_vectors], axis=0)
                
                # Calculate coherence (average similarity to centroid)
                coherence = np.mean([
                    np.dot(np_vectors[idx], centroid) / (np.linalg.norm(np_vectors[idx]) * np.linalg.norm(centroid))
                    for idx in pattern_vectors
                ])
                
                # Create pattern
                pattern = {
                    "id": str(uuid.uuid4()),
                    "vector_space_id": vector_space_id,
                    "centroid": centroid.tolist(),
                    "vector_indices": pattern_vectors,
                    "vectors": [vectors[idx] for idx in pattern_vectors],
                    "coherence": float(coherence),
                    "size": len(pattern_vectors)
                }
                
                patterns.append(pattern)
        
        # Publish event
        self._event_service.publish("vector_tonic.patterns_detected", {
            "vector_space_id": vector_space_id,
            "pattern_count": len(patterns)
        })
        
        return patterns
    
    def validate_harmonic_coherence(self, pattern_id: str, 
                                   new_vector: List[float]) -> Dict[str, Any]:
        """
        Validate the harmonic coherence of a new vector with an existing pattern.
        
        Args:
            pattern_id: The ID of the pattern
            new_vector: The new vector to validate
            
        Returns:
            Validation results including coherence score and recommendations
        """
        if not self._initialized:
            self.initialize()
            
        # Get pattern
        pattern = self._pattern_repository.find_by_id(pattern_id)
        if not pattern:
            raise ValueError(f"Pattern {pattern_id} not found")
            
        # Get pattern vectors
        pattern_vectors = self.get_pattern_vectors(pattern_id)
        if not pattern_vectors:
            raise ValueError(f"No vectors found for pattern {pattern_id}")
            
        # Get pattern centroid
        centroid = self.get_pattern_centroid(pattern_id)
        
        # Calculate similarity to centroid
        new_vector_np = np.array(new_vector)
        centroid_np = np.array(centroid)
        
        similarity = np.dot(new_vector_np, centroid_np) / (np.linalg.norm(new_vector_np) * np.linalg.norm(centroid_np))
        
        # Calculate similarities to all pattern vectors
        vector_similarities = []
        for pv in pattern_vectors:
            pv_vector = np.array(pv["vector"])
            pv_similarity = np.dot(new_vector_np, pv_vector) / (np.linalg.norm(new_vector_np) * np.linalg.norm(pv_vector))
            vector_similarities.append({
                "vector_id": pv["vector_id"],
                "similarity": float(pv_similarity)
            })
        
        # Sort by similarity
        vector_similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Calculate coherence impact
        current_coherence = pattern.metadata.get("coherence", 0.5)
        
        # Simulate adding the new vector to the pattern
        all_similarities = [vs["similarity"] for vs in vector_similarities]
        all_similarities.append(float(similarity))
        new_coherence = np.mean(all_similarities)
        
        coherence_impact = new_coherence - current_coherence
        
        # Determine recommendation
        recommendation = "accept"
        if similarity < 0.5:
            recommendation = "reject"
        elif similarity < 0.7:
            recommendation = "review"
        
        result = {
            "pattern_id": pattern_id,
            "similarity_to_centroid": float(similarity),
            "current_coherence": float(current_coherence),
            "projected_coherence": float(new_coherence),
            "coherence_impact": float(coherence_impact),
            "most_similar_vectors": vector_similarities[:3],
            "recommendation": recommendation
        }
        
        # Publish event
        self._event_service.publish("vector_tonic.coherence_validated", {
            "pattern_id": pattern_id,
            "similarity": float(similarity),
            "recommendation": recommendation
        })
        
        return result
    
    def update_pattern_with_vector(self, pattern_id: str, 
                                 vector: List[float],
                                 weight: float = 1.0) -> Dict[str, Any]:
        """
        Update a pattern with a new vector.
        
        Args:
            pattern_id: The ID of the pattern
            vector: The vector to add to the pattern
            weight: The weight of the vector in the pattern update
            
        Returns:
            The updated pattern with metadata
        """
        if not self._initialized:
            self.initialize()
            
        # Get pattern
        pattern = self._pattern_repository.find_by_id(pattern_id)
        if not pattern:
            raise ValueError(f"Pattern {pattern_id} not found")
            
        # Get pattern metadata
        metadata = pattern.metadata.copy() if pattern.metadata else {}
        
        # Get vector space ID
        vector_space_id = metadata.get("vector_space_id")
        if not vector_space_id:
            # Try to find it from existing pattern vectors
            pattern_vectors = self.get_pattern_vectors(pattern_id)
            if pattern_vectors:
                # Get vector document to find vector space
                vector_doc = self._db_connection.get_document("vectors", pattern_vectors[0]["vector_id"])
                vector_space_id = vector_doc.get("vector_space_id")
                
                # Update pattern metadata
                metadata["vector_space_id"] = vector_space_id
                self._pattern_repository.update_node(pattern_id, {"metadata": metadata})
        
        if not vector_space_id:
            raise ValueError(f"Vector space ID not found for pattern {pattern_id}")
            
        # Store the vector
        entity_id = f"pattern:{pattern_id}"
        vector_id = self.store_vector(vector_space_id, vector, entity_id)
        
        # Create edge between pattern and vector
        edge_data = {
            "weight": weight,
            "created_at": datetime.utcnow().isoformat()
        }
        
        self._db_connection.insert_edge(
            "pattern_vectors",
            pattern_id,
            vector_id,
            edge_data
        )
        
        # Update pattern centroid
        centroid = self.get_pattern_centroid(pattern_id)
        metadata["centroid"] = centroid
        
        # Update pattern coherence
        pattern_vectors = self.get_pattern_vectors(pattern_id)
        centroid_np = np.array(centroid)
        
        similarities = []
        for pv in pattern_vectors:
            pv_vector = np.array(pv["vector"])
            similarity = np.dot(pv_vector, centroid_np) / (np.linalg.norm(pv_vector) * np.linalg.norm(centroid_np))
            similarities.append(float(similarity))
        
        coherence = np.mean(similarities)
        metadata["coherence"] = float(coherence)
        metadata["vector_count"] = len(pattern_vectors)
        metadata["updated_at"] = datetime.utcnow().isoformat()
        
        # Update pattern
        self._pattern_repository.update_node(pattern_id, {"metadata": metadata})
        
        # Publish event
        self._event_service.publish("vector_tonic.pattern_updated", {
            "pattern_id": pattern_id,
            "vector_id": vector_id,
            "coherence": float(coherence)
        })
        
        # Return updated pattern
        return {
            "pattern_id": pattern_id,
            "centroid": centroid,
            "coherence": float(coherence),
            "vector_count": len(pattern_vectors),
            "metadata": metadata
        }
    
    def get_pattern_centroid(self, pattern_id: str) -> List[float]:
        """
        Get the centroid vector of a pattern.
        
        Args:
            pattern_id: The ID of the pattern
            
        Returns:
            The centroid vector of the pattern
        """
        if not self._initialized:
            self.initialize()
            
        # Get pattern vectors
        pattern_vectors = self.get_pattern_vectors(pattern_id)
        
        if not pattern_vectors:
            raise ValueError(f"No vectors found for pattern {pattern_id}")
            
        # Calculate centroid
        vectors = [np.array(pv["vector"]) for pv in pattern_vectors]
        centroid = np.mean(vectors, axis=0)
        
        return centroid.tolist()
    
    def get_pattern_vectors(self, pattern_id: str) -> List[Dict[str, Any]]:
        """
        Get all vectors associated with a pattern.
        
        Args:
            pattern_id: The ID of the pattern
            
        Returns:
            A list of vectors with their metadata
        """
        if not self._initialized:
            self.initialize()
            
        # Query for vectors connected to the pattern
        query = """
        FOR v, e IN OUTBOUND @pattern_id pattern_vectors
        RETURN {
            "vector_id": v._id,
            "vector": v.vector,
            "entity_id": v.entity_id,
            "metadata": v.metadata,
            "weight": e.weight,
            "created_at": e.created_at
        }
        """
        
        return self._db_connection.execute_query(query, {"pattern_id": pattern_id})
    
    def calculate_vector_gradient(self, vector_space_id: str,
                                 start_vector: List[float],
                                 end_vector: List[float],
                                 steps: int = 10) -> List[List[float]]:
        """
        Calculate a gradient between two vectors.
        
        Args:
            vector_space_id: The ID of the vector space
            start_vector: The starting vector
            end_vector: The ending vector
            steps: The number of steps in the gradient
            
        Returns:
            A list of vectors forming the gradient
        """
        if not self._initialized:
            self.initialize()
            
        # Validate vector space
        if vector_space_id not in self._vector_spaces:
            raise ValueError(f"Vector space {vector_space_id} not found")
            
        # Convert to numpy arrays
        start_np = np.array(start_vector)
        end_np = np.array(end_vector)
        
        # Calculate step size
        step_size = 1.0 / (steps - 1) if steps > 1 else 1.0
        
        # Generate gradient
        gradient = []
        for i in range(steps):
            t = i * step_size
            interpolated = (1 - t) * start_np + t * end_np
            
            # Normalize
            norm = np.linalg.norm(interpolated)
            if norm > 0:
                interpolated = interpolated / norm
                
            gradient.append(interpolated.tolist())
        
        # Publish event
        self._event_service.publish("vector_tonic.gradient_calculated", {
            "vector_space_id": vector_space_id,
            "steps": steps
        })
        
        return gradient
