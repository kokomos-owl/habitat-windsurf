"""
Pattern-aware RAG service implementation for Habitat Evolution.

This module provides a concrete implementation of the PatternAwareRAGInterface,
supporting the pattern evolution and co-evolution principles of Habitat Evolution.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import uuid
from datetime import datetime

from src.habitat_evolution.infrastructure.interfaces.services.pattern_aware_rag_interface import PatternAwareRAGInterface
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.vector_tonic_service_interface import VectorTonicServiceInterface
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_pattern_repository import ArangoDBPatternRepository, Pattern
from src.habitat_evolution.infrastructure.interfaces.persistence.arangodb_connection_interface import ArangoDBConnectionInterface
from src.habitat_evolution.infrastructure.adapters.pattern_bridge import PatternBridge
from src.habitat_evolution.adaptive_core.models.pattern import Pattern as AdaptiveCorePattern

logger = logging.getLogger(__name__)


class PatternAwareRAGService(PatternAwareRAGInterface):
    """
    Implementation of the PatternAwareRAGInterface.
    
    This service provides pattern-aware retrieval augmented generation functionality
    for the Habitat Evolution system, supporting the pattern evolution and
    co-evolution principles.
    """
    
    def __init__(self, 
                 db_connection: ArangoDBConnectionInterface,
                 event_service: EventServiceInterface,
                 vector_tonic_service: VectorTonicServiceInterface,
                 pattern_repository: ArangoDBPatternRepository,
                 pattern_bridge: PatternBridge[AdaptiveCorePattern]):
        """
        Initialize a new pattern-aware RAG service.
        
        Args:
            db_connection: The ArangoDB connection to use
            event_service: The event service for publishing events
            vector_tonic_service: The vector tonic service
            pattern_repository: The pattern repository
        """
        self._db_connection = db_connection
        self._event_service = event_service
        self._vector_tonic_service = vector_tonic_service
        self._pattern_repository = pattern_repository
        self._pattern_bridge = pattern_bridge
        self._initialized = False
        self._vector_space_id = None
        self._coherence_threshold = 0.5
        self._quality_threshold = 0.6
        self._quality_weight = 0.7
        logger.debug("PatternAwareRAGService created")
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the pattern-aware RAG service with the specified configuration.
        
        Args:
            config: Optional configuration for the service
        """
        if self._initialized:
            logger.warning("PatternAwareRAGService already initialized")
            return
            
        logger.info("Initializing PatternAwareRAGService")
        
        # Initialize pattern repository
        if not self._pattern_repository._initialized:
            self._pattern_repository.initialize()
            
        # Initialize vector tonic service
        if not self._vector_tonic_service._initialized:
            self._vector_tonic_service.initialize()
        
        # Create vector space for RAG
        self._vector_space_id = self._vector_tonic_service.register_vector_space(
            name="rag_vector_space",
            dimensions=768,  # Standard embedding dimension
            metadata={"purpose": "pattern_aware_rag"}
        )
        
        # Set configuration
        if config:
            self._coherence_threshold = config.get("coherence_threshold", self._coherence_threshold)
            self._quality_threshold = config.get("quality_threshold", self._quality_threshold)
            self._quality_weight = config.get("quality_weight", self._quality_weight)
        
        self._initialized = True
        logger.info(f"PatternAwareRAGService initialized with vector space {self._vector_space_id}")
        
        # Publish initialization event
        self._event_service.publish("pattern_aware_rag.initialized", {
            "vector_space_id": self._vector_space_id,
            "coherence_threshold": self._coherence_threshold,
            "quality_threshold": self._quality_threshold
        })
    
    def process_document(self, document: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document through the pattern-aware RAG system.
        
        Args:
            document: The document to process
            metadata: Optional metadata for the document
            
        Returns:
            Processing results including extracted patterns and metrics
        """
        if not self._initialized:
            self.initialize()
            
        logger.info(f"Processing document through pattern-aware RAG: {metadata.get('title', 'Untitled') if metadata else 'Untitled'}")
        
        # Extract entities (simplified implementation)
        entities = self._extract_entities(document)
        
        # Create patterns for entities
        patterns = []
        for entity in entities:
            pattern = self._create_entity_pattern(entity, document)
            patterns.append(pattern)
            
        # Update context with patterns
        context = self._update_context_with_patterns(patterns)
        
        # Calculate metrics
        metrics = self._calculate_metrics(patterns)
        
        # Publish document processed event
        self._event_service.publish("pattern_aware_rag.document_processed", {
            "document_id": metadata.get("id", str(uuid.uuid4())) if metadata else str(uuid.uuid4()),
            "pattern_count": len(patterns),
            "entity_count": len(entities)
        })
        
        return {
            "patterns": patterns,
            "metrics": metrics,
            "context": context
        }
    
    def query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the pattern-aware RAG system.
        
        Args:
            query: The query to process
            context: Optional context for the query
            
        Returns:
            Query results including relevant patterns and generated response
        """
        if not self._initialized:
            self.initialize()
            
        logger.info(f"Processing query through pattern-aware RAG: {query[:50]}...")
        
        # Get relevant patterns based on query
        relevant_patterns = self._retrieve_relevant_patterns(query, context)
        
        # Generate response
        response = self._generate_response(query, relevant_patterns, context)
        
        # Publish query processed event
        self._event_service.publish("pattern_aware_rag.query_processed", {
            "query_id": str(uuid.uuid4()),
            "relevant_pattern_count": len(relevant_patterns)
        })
        
        return {
            "query": query,
            "relevant_patterns": relevant_patterns,
            "response": response
        }
    
    def update_pattern_metadata(self, pattern_id: str, metadata: Dict[str, Any]) -> Optional[Pattern]:
        """
        Update the metadata of a pattern.
        
        Args:
            pattern_id: The ID of the pattern
            metadata: The metadata to update
            
        Returns:
            The updated pattern, or None if not found
        """
        if not self._initialized:
            self.initialize()
            
        # Get current pattern
        pattern = self._pattern_repository.find_by_id(pattern_id)
        if not pattern:
            logger.warning(f"Pattern {pattern_id} not found")
            return None
            
        # Update pattern
        return self._pattern_repository.update_node(pattern_id, {"metadata": metadata})
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[Pattern]:
        """
        Get a pattern by ID.
        
        Args:
            pattern_id: The ID of the pattern
            
        Returns:
            The pattern, or None if not found
        """
        if not self._initialized:
            self.initialize()
            
        return self._pattern_repository.find_by_id(pattern_id)
    
    def get_patterns_by_quality(self, min_quality: float = 0.7) -> List[Pattern]:
        """
        Get patterns by quality.
        
        Args:
            min_quality: The minimum quality threshold
            
        Returns:
            A list of patterns with quality >= min_quality
        """
        if not self._initialized:
            self.initialize()
            
        return self._pattern_repository.find_by_quality(min_quality=min_quality)
    
    def _extract_entities(self, document: str) -> List[str]:
        """
        Extract entities from a document.
        
        Args:
            document: The document to process
            
        Returns:
            A list of extracted entities
        """
        # Simplified entity extraction (in a real implementation, this would use NER)
        words = document.split()
        entities = []
        
        # Extract simple entities (2-3 word phrases)
        for i in range(len(words) - 2):
            if len(words[i]) > 3 and words[i][0].isupper():
                entity = " ".join(words[i:i+2])
                entities.append(entity)
                
                if i < len(words) - 3:
                    entity = " ".join(words[i:i+3])
                    entities.append(entity)
        
        return list(set(entities))
    
    def _create_entity_pattern(self, entity: str, document: str) -> Pattern:
        """
        Create a pattern for an entity.
        
        Args:
            entity: The entity to create a pattern for
            document: The document containing the entity
            
        Returns:
            The created pattern
        """
        # Create pattern
        pattern = self._pattern_repository.create_pattern(
            name=entity,
            pattern_type="entity",
            description=f"Entity pattern for '{entity}'",
            metadata={
                "type": "entity",
                "text": entity,
                "quality": 0.1,  # Start with low quality
                "coherence": 0.1,  # Start with low coherence
                "stability": 0.0,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        return pattern
    
    def _update_context_with_patterns(self, patterns: List[Pattern]) -> Dict[str, Any]:
        """
        Update context with patterns.
        
        Args:
            patterns: The patterns to add to context
            
        Returns:
            The updated context
        """
        # Count patterns by quality
        quality_counts = {"good": 0, "uncertain": 0, "poor": 0}
        
        for pattern in patterns:
            quality = pattern.metadata.get("quality", 0)
            
            if quality >= 0.7:
                quality_counts["good"] += 1
            elif quality >= 0.4:
                quality_counts["uncertain"] += 1
            else:
                quality_counts["poor"] += 1
        
        logger.info(f"Updated context from quality assessment: {quality_counts['good']} good, "
                   f"{quality_counts['uncertain']} uncertain, {quality_counts['poor']} poor entities")
        
        return {
            "pattern_count": len(patterns),
            "quality_counts": quality_counts
        }
    
    def _calculate_metrics(self, patterns: List[Pattern]) -> Dict[str, Any]:
        """
        Calculate metrics for patterns.
        
        Args:
            patterns: The patterns to calculate metrics for
            
        Returns:
            The calculated metrics
        """
        # Calculate average quality and coherence
        if not patterns:
            return {
                "avg_quality": 0.0,
                "avg_coherence": 0.0,
                "pattern_count": 0
            }
            
        avg_quality = sum(p.metadata.get("quality", 0) for p in patterns) / len(patterns)
        avg_coherence = sum(p.metadata.get("coherence", 0) for p in patterns) / len(patterns)
        
        return {
            "avg_quality": avg_quality,
            "avg_coherence": avg_coherence,
            "pattern_count": len(patterns)
        }
    
    def _retrieve_relevant_patterns(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[Pattern]:
        """
        Retrieve patterns relevant to a query.
        
        Args:
            query: The query to retrieve patterns for
            context: Optional context for the query
            
        Returns:
            A list of relevant patterns
        """
        # Get all patterns
        all_patterns = self._pattern_repository.find_all()
        
        # Enhance patterns with metadata using the bridge
        enhanced_patterns = self._pattern_bridge.enhance_patterns(all_patterns)
        
        # Filter by coherence threshold
        coherent_patterns = [
            p for p in enhanced_patterns 
            if p.metadata.get("coherence", 0) >= self._coherence_threshold
        ]
        
        # Simple relevance scoring (in a real implementation, this would use semantic similarity)
        scored_patterns = []
        for pattern in coherent_patterns:
            # Simple text matching (in a real implementation, this would use embeddings)
            relevance = 0.0
            if pattern.name and query:
                if pattern.name.lower() in query.lower():
                    relevance = 0.8
                elif any(word.lower() in pattern.name.lower() for word in query.lower().split()):
                    relevance = 0.5
            
            # Combine relevance and quality
            quality = pattern.metadata.get("quality", 0)
            combined_score = (relevance * (1 - self._quality_weight)) + (quality * self._quality_weight)
            
            if combined_score >= self._quality_threshold:
                scored_patterns.append((pattern, combined_score))
        
        # Sort by combined score
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Return patterns
        return [p[0] for p in scored_patterns]
    
    def _generate_response(self, query: str, relevant_patterns: List[Pattern], 
                         context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response to a query.
        
        Args:
            query: The query to respond to
            relevant_patterns: The relevant patterns for the query
            context: Optional context for the query
            
        Returns:
            The generated response
        """
        # Simple response generation (in a real implementation, this would use an LLM)
        if not relevant_patterns:
            return "No relevant information found."
            
        # Generate response based on relevant patterns
        response_parts = [f"Found {len(relevant_patterns)} relevant patterns:"]
        
        for i, pattern in enumerate(relevant_patterns[:5]):  # Limit to top 5
            quality = pattern.metadata.get("quality", 0)
            coherence = pattern.metadata.get("coherence", 0)
            response_parts.append(f"{i+1}. {pattern.name} (quality: {quality:.2f}, coherence: {coherence:.2f})")
        
        return "\n".join(response_parts)
