"""
Vector Tonic Service Interface for Habitat Evolution.

This module defines the interface for vector tonic services in Habitat Evolution,
supporting the pattern evolution and co-evolution principles.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union


class VectorTonicServiceInterface(ABC):
    """
    Interface for vector tonic services.
    
    This interface defines the operations for vector tonic integration,
    which is responsible for managing vector operations and tonic-harmonic
    pattern detection in the Habitat Evolution system.
    """
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the vector tonic service with the specified configuration.
        
        Args:
            config: Optional configuration for the service
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_pattern_centroid(self, pattern_id: str) -> List[float]:
        """
        Get the centroid vector of a pattern.
        
        Args:
            pattern_id: The ID of the pattern
            
        Returns:
            The centroid vector of the pattern
        """
        pass
    
    @abstractmethod
    def get_pattern_vectors(self, pattern_id: str) -> List[Dict[str, Any]]:
        """
        Get all vectors associated with a pattern.
        
        Args:
            pattern_id: The ID of the pattern
            
        Returns:
            A list of vectors with their metadata
        """
        pass
    
    @abstractmethod
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
        pass
