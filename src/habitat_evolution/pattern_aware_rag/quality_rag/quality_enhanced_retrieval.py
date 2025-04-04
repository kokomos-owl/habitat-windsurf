"""
Quality-enhanced retrieval for context-aware RAG.

This module provides the QualityEnhancedRetrieval class which enhances
retrieval capabilities based on entity quality assessments.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from dataclasses import dataclass, field
import networkx as nx
from datetime import datetime

from src.habitat_evolution.pattern_aware_rag.context.quality_aware_context import QualityAwarePatternContext
from src.habitat_evolution.adaptive_core.models import Pattern, Relationship
from src.habitat_evolution.core.pattern import PatternState
from src.habitat_evolution.adaptive_core.quality.predicate_quality_tracker import PredicateQualityTracker
from src.habitat_evolution.adaptive_core.persistence.semantic_memory_persistence import SemanticMemoryPersistence

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Result of quality-enhanced retrieval."""
    patterns: List[Pattern]
    quality_distribution: Dict[str, int]
    confidence: float
    retrieval_explanation: str
    quality_context: Dict[str, Any]
    predicate_quality: Dict[str, str] = field(default_factory=dict)
    entity_relationships: List[Dict[str, Any]] = field(default_factory=list)
    persistence_info: Dict[str, Any] = field(default_factory=dict)

class QualityEnhancedRetrieval:
    """Enhance retrieval based on quality assessments.
    
    This class implements quality-aware retrieval strategies that prioritize
    high-quality patterns and leverage quality assessment paths to improve
    retrieval results. It integrates with the elastic semantic memory system
    to create a complete RAG↔Evolution↔Persistence loop.
    """
    
    def __init__(self, quality_weight: float = 0.7, coherence_threshold: float = 0.6,
                 predicate_quality_tracker: Optional[PredicateQualityTracker] = None,
                 persistence_layer: Optional[SemanticMemoryPersistence] = None,
                 event_bus = None):
        """Initialize quality-enhanced retrieval.
        
        Args:
            quality_weight: Weight to give to quality in ranking
            coherence_threshold: Threshold for coherence filtering
            predicate_quality_tracker: Optional predicate quality tracker
            persistence_layer: Optional semantic memory persistence layer
            event_bus: Optional event bus for publishing events
        """
        self.quality_weight = quality_weight
        self.coherence_threshold = coherence_threshold
        self.event_bus = event_bus
        
        # Initialize predicate quality tracker if not provided
        self.predicate_quality_tracker = predicate_quality_tracker or PredicateQualityTracker(event_bus, logger)
        
        # Initialize persistence layer if not provided
        self.persistence_layer = persistence_layer or SemanticMemoryPersistence(logger=logger)
        
        # Entity network for tracking relationships
        self.entity_network = nx.DiGraph()
        
        # Cache for entity and predicate quality states
        self.entity_quality_cache = {}
        self.entity_confidence_cache = {}
        self.entity_transition_history = {}
        
        # Field metrics
        self.field_metrics = {
            'local_density': 0.0,
            'global_density': 0.0,
            'stability': 0.5,
            'coherence': 0.5
        }
        
        logger.info(f"Initialized QualityEnhancedRetrieval with quality_weight={quality_weight} and elastic semantic memory integration")
    
    def retrieve_with_quality(self, query: str, context: QualityAwarePatternContext, 
                             max_results: int = 10, use_persistence: bool = True) -> RetrievalResult:
        """Retrieve patterns with quality awareness.
        
        Args:
            query: Query string
            context: Quality-aware pattern context
            max_results: Maximum number of results to return
            
        Returns:
            RetrievalResult with retrieved patterns
        """
        # Get prioritized patterns from context
        prioritized_patterns = context.prioritize_patterns_by_quality()
        
        # Filter by coherence threshold
        coherent_patterns = [
            p for p in prioritized_patterns 
            if p.metadata.get("coherence", 0) >= self.coherence_threshold
        ]
        
        # Simple relevance scoring (in a real implementation, this would use semantic similarity)
        scored_patterns = []
        for pattern in coherent_patterns:
            # Basic text matching for demonstration
            relevance_score = self._calculate_relevance(query, pattern.text)
            
            # Quality score from metadata
            quality_score = self._calculate_quality_score(pattern)
            
            # Combined score
            combined_score = (relevance_score * (1 - self.quality_weight) + 
                             quality_score * self.quality_weight)
            
            scored_patterns.append((pattern, combined_score))
        
        # Sort by combined score
        sorted_patterns = sorted(scored_patterns, key=lambda x: x[1], reverse=True)
        
        # Take top results
        top_patterns = [p[0] for p in sorted_patterns[:max_results]]
        
        # Calculate quality distribution
        quality_distribution = {
            "good": sum(1 for p in top_patterns if p.metadata.get("quality_state") == "good"),
            "uncertain": sum(1 for p in top_patterns if p.metadata.get("quality_state") == "uncertain"),
            "poor": sum(1 for p in top_patterns if p.metadata.get("quality_state") == "poor")
        }
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(top_patterns)
        
        # Generate retrieval explanation
        explanation = self._generate_explanation(query, top_patterns, confidence)
        
        # Prepare quality context
        quality_context = {
            "high_quality_ratio": quality_distribution["good"] / max(1, len(top_patterns)),
            "coherence_level": context.coherence_level,
            "pattern_state_distribution": {k.name: v for k, v in context.pattern_state_distribution.items()},
            "quality_transitions": context.quality_transitions.get_transition_summary() if hasattr(context, "quality_transitions") else {}
        }
        
        # Extract entity relationships from patterns
        entity_relationships = self._extract_entity_relationships(top_patterns)
        
        # Update entity network with relationships
        self._update_entity_network(entity_relationships)
        
        # Get predicate quality information
        predicate_quality = {pred: self.predicate_quality_tracker.get_predicate_quality(pred) 
                           for pred in self._get_predicates_from_relationships(entity_relationships)}
        
        # Persistence information
        persistence_info = {}
        if use_persistence:
            # Save current state to persistence layer
            persistence_info = self._save_to_persistence(top_patterns, context)
        
        logger.info(f"Retrieved {len(top_patterns)} patterns with quality enhancement, confidence: {confidence:.2f}")
        
        return RetrievalResult(
            patterns=top_patterns,
            quality_distribution=quality_distribution,
            confidence=confidence,
            retrieval_explanation=explanation,
            quality_context=quality_context,
            predicate_quality=predicate_quality,
            entity_relationships=entity_relationships,
            persistence_info=persistence_info
        )
    
    def _calculate_relevance(self, query: str, pattern_text: str) -> float:
        """Calculate relevance score between query and pattern.
        
        Args:
            query: Query string
            pattern_text: Pattern text
            
        Returns:
            Relevance score between 0 and 1
        """
        # Simple word overlap for demonstration
        # In a real implementation, this would use semantic similarity
        query_words = set(query.lower().split())
        pattern_words = set(pattern_text.lower().split())
        
        if not query_words or not pattern_words:
            return 0.0
        
        overlap = len(query_words.intersection(pattern_words))
        return min(1.0, overlap / max(len(query_words), 1))
    
    def _calculate_quality_score(self, pattern: Pattern) -> float:
        """Calculate quality score for a pattern.
        
        Args:
            pattern: Pattern to calculate score for
            
        Returns:
            Quality score between 0 and 1
        """
        metadata = pattern.metadata
        quality_state = metadata.get("quality_state", "uncertain")
        
        if quality_state == "good":
            # For good patterns, use coherence and stability
            coherence = metadata.get("coherence", 0.5)
            stability = metadata.get("stability", 0.5)
            return (coherence * 0.6) + (stability * 0.4)
        elif quality_state == "uncertain":
            # For uncertain patterns, use coherence and emergence_rate
            coherence = metadata.get("coherence", 0.3)
            emergence_rate = metadata.get("emergence_rate", 0.3)
            return (coherence * 0.4) + (emergence_rate * 0.3) + 0.3  # Base score
        else:
            # Poor patterns get a low score
            return 0.2
    
    def _calculate_confidence(self, patterns: List[Pattern]) -> float:
        """Calculate overall confidence in retrieval results.
        
        Args:
            patterns: List of retrieved patterns
            
        Returns:
            Confidence score between 0 and 1
        """
        if not patterns:
            return 0.0
        
        # Calculate based on quality distribution
        good_count = sum(1 for p in patterns if p.metadata.get("quality_state") == "good")
        uncertain_count = sum(1 for p in patterns if p.metadata.get("quality_state") == "uncertain")
        
        # Weight good patterns more heavily
        weighted_count = (good_count * 1.0) + (uncertain_count * 0.5)
        
        return min(1.0, weighted_count / len(patterns))
    
    def _generate_explanation(self, query: str, patterns: List[Pattern], confidence: float) -> str:
        """Generate explanation for retrieval results.
        
        Args:
            query: Query string
            patterns: List of retrieved patterns
            confidence: Overall confidence
            
        Returns:
            Explanation string
        """
        good_patterns = [p for p in patterns if p.metadata.get("quality_state") == "good"]
        uncertain_patterns = [p for p in patterns if p.metadata.get("quality_state") == "uncertain"]
        
        explanation = f"Retrieved {len(patterns)} patterns for query '{query}' with confidence {confidence:.2f}.\n"
        
        if good_patterns:
            explanation += f"Found {len(good_patterns)} high-quality patterns that match your query.\n"
        if uncertain_patterns:
            explanation += f"Found {len(uncertain_patterns)} patterns of uncertain quality that may be relevant.\n"
        
        # Add predicate quality information to explanation
        predicates = set()
        for pattern in patterns:
            for rel in pattern.metadata.get('relationships', []):
                if isinstance(rel, dict) and 'predicate' in rel:
                    predicates.add(rel['predicate'])
        "good": sum(1 for p in top_patterns if p.metadata.get("quality_state") == "good"),
        "uncertain": sum(1 for p in top_patterns if p.metadata.get("quality_state") == "uncertain"),
        "poor": sum(1 for p in top_patterns if p.metadata.get("quality_state") == "poor")
    }
        
    # Calculate overall confidence
    confidence = self._calculate_confidence(top_patterns)
        
    # Generate retrieval explanation
    explanation = self._generate_explanation(query, top_patterns, confidence)
        
    # Prepare quality context
    quality_context = {
        "high_quality_ratio": quality_distribution["good"] / max(1, len(top_patterns)),
        "coherence_level": context.coherence_level,
        "pattern_state_distribution": {k.name: v for k, v in context.pattern_state_distribution.items()},
        "quality_transitions": context.quality_transitions.get_transition_summary() if hasattr(context, "quality_transitions") else {}
    }
        
    # Extract entity relationships from patterns
    entity_relationships = self._extract_entity_relationships(top_patterns)
        
    # Update entity network with relationships
    self._update_entity_network(entity_relationships)
        
    # Get predicate quality information
    predicate_quality = {pred: self.predicate_quality_tracker.get_predicate_quality(pred) 
                       for pred in self._get_predicates_from_relationships(entity_relationships)}
        
    # Persistence information
    persistence_info = {}
    if use_persistence:
        # Save current state to persistence layer
        persistence_info = self._save_to_persistence(top_patterns, context)
        
    logger.info(f"Retrieved {len(top_patterns)} patterns with quality enhancement, confidence: {confidence:.2f}")
        
    return RetrievalResult(
        patterns=top_patterns,
        quality_distribution=quality_distribution,
        confidence=confidence,
        retrieval_explanation=explanation,
        quality_context=quality_context,
        predicate_quality=predicate_quality,
        entity_relationships=entity_relationships,
        persistence_info=persistence_info
    )

def _calculate_relevance(self, query: str, pattern_text: str) -> float:
    """Calculate relevance score between query and pattern.
        return explanation
    
    def get_related_patterns(self, pattern: Pattern, context: QualityAwarePatternContext, 
                           max_results: int = 5) -> List[Tuple[Pattern, str, float]]:
        """Get patterns related to a given pattern.
        
        Args:
            pattern: Pattern to find related patterns for
            context: Quality-aware pattern context
            max_results: Maximum number of results to return
            
        Returns:
            List of (related_pattern, relationship_type, confidence) tuples
        """
        related = []
        
        # Get related entities from context
        entity_relations = context.get_related_entities(pattern.text)
        
        # Convert to pattern relations
        for entity, relation_type, confidence in entity_relations:
            # Find corresponding pattern
            related_pattern = next(
                (p for p in context.retrieval_patterns if p.text == entity),
                None
            )
            
            if related_pattern:
                related.append((related_pattern, relation_type, confidence))
        
        # Sort by confidence and take top results
        return sorted(related, key=lambda x: x[2], reverse=True)[:max_results]
