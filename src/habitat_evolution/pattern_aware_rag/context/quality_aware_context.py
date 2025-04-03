"""
Quality-aware pattern context for RAG operations.

This module provides the QualityAwarePatternContext class which extends
the RAGPatternContext with quality assessment paths and context-aware
pattern extraction capabilities.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from habitat_evolution.pattern_aware_rag.pattern_aware_rag import RAGPatternContext
from habitat_evolution.adaptive_core.models import Pattern, Relationship
from habitat_evolution.core.pattern import PatternState
from habitat_evolution.adaptive_core.emergence.context_aware_extraction.quality_assessment import QualityAssessment

from .quality_transitions import QualityTransitionTracker

logger = logging.getLogger(__name__)

@dataclass
class QualityAssessmentInfo:
    """Information about quality assessment for an entity."""
    entity: str
    quality_state: str  # "good", "uncertain", or "poor"
    pattern_state: PatternState
    metrics: Dict[str, float]
    contexts: List[Dict[str, str]] = field(default_factory=list)
    transitions: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class QualityAwarePatternContext(RAGPatternContext):
    """Extended context with quality assessment paths.
    
    This class extends the RAGPatternContext with quality assessment paths
    and context-aware pattern extraction capabilities, enabling the RAG system
    to prioritize high-quality patterns and track pattern evolution.
    """
    
    # Inherit from RAGPatternContext
    query_patterns: List[Pattern] = field(default_factory=list)
    retrieval_patterns: List[Pattern] = field(default_factory=list)
    augmentation_patterns: List[Pattern] = field(default_factory=list)
    coherence_level: float = 0.0
    
    # Quality-specific fields
    quality_assessments: Dict[str, QualityAssessmentInfo] = field(default_factory=dict)
    quality_transitions: QualityTransitionTracker = field(default_factory=QualityTransitionTracker)
    context_aware_extraction_results: Dict[str, Any] = field(default_factory=dict)
    
    # Pattern state distributions
    pattern_state_distribution: Dict[PatternState, int] = field(default_factory=lambda: {
        PatternState.EMERGENT: 0,
        PatternState.COHERENT: 0,
        PatternState.STABLE: 0,
        PatternState.DEGRADING: 0
    })
    
    # Quality state distributions
    quality_state_distribution: Dict[str, int] = field(default_factory=lambda: {
        "good": 0,
        "uncertain": 0,
        "poor": 0
    })
    
    def add_quality_assessment(self, entity: str, quality_info: QualityAssessmentInfo) -> None:
        """Add quality assessment information for an entity.
        
        Args:
            entity: The entity to add assessment for
            quality_info: Quality assessment information
        """
        self.quality_assessments[entity] = quality_info
        
        # Update distributions
        self.pattern_state_distribution[quality_info.pattern_state] += 1
        self.quality_state_distribution[quality_info.quality_state] += 1
        
        logger.info(f"Added quality assessment for '{entity}': {quality_info.quality_state}")
    
    def update_from_quality_assessment(self, quality_assessment: QualityAssessment) -> None:
        """Update context from quality assessment results.
        
        Args:
            quality_assessment: QualityAssessment instance with results
        """
        quality_states = quality_assessment.get_quality_states()
        transitions = quality_assessment.get_transitions()
        
        # Process good entities
        for entity, data in quality_states["good"].items():
            pattern_state = PatternState[data["pattern_state"]]
            
            quality_info = QualityAssessmentInfo(
                entity=entity,
                quality_state="good",
                pattern_state=pattern_state,
                metrics=data["metrics"]
            )
            
            self.add_quality_assessment(entity, quality_info)
            
            # Convert to Pattern for RAG
            pattern = Pattern(
                text=entity,
                pattern_type="entity",
                metadata={
                    "quality_state": "good",
                    "pattern_state": pattern_state.name,
                    "coherence": data["metrics"]["coherence"],
                    "stability": data["metrics"]["stability"]
                }
            )
            
            # Add to retrieval patterns with high priority
            self.retrieval_patterns.append(pattern)
        
        # Process uncertain entities
        for entity, data in quality_states["uncertain"].items():
            pattern_state = PatternState[data["pattern_state"]]
            
            quality_info = QualityAssessmentInfo(
                entity=entity,
                quality_state="uncertain",
                pattern_state=pattern_state,
                metrics=data["metrics"]
            )
            
            self.add_quality_assessment(entity, quality_info)
            
            # Convert to Pattern for RAG with lower priority
            pattern = Pattern(
                text=entity,
                pattern_type="entity",
                metadata={
                    "quality_state": "uncertain",
                    "pattern_state": pattern_state.name,
                    "coherence": data["metrics"]["coherence"],
                    "emergence_rate": data["metrics"]["emergence_rate"]
                }
            )
            
            # Add to retrieval patterns with lower priority
            if data["metrics"]["emergence_rate"] > 0.5:
                self.retrieval_patterns.append(pattern)
        
        # Process poor entities (typically not used for retrieval)
        for entity, data in quality_states["poor"].items():
            pattern_state = PatternState[data["pattern_state"]]
            
            quality_info = QualityAssessmentInfo(
                entity=entity,
                quality_state="poor",
                pattern_state=pattern_state,
                metrics=data["metrics"]
            )
            
            self.add_quality_assessment(entity, quality_info)
        
        # Update transitions
        for entity, entity_transitions in transitions.items():
            for transition in entity_transitions:
                if entity in self.quality_assessments:
                    self.quality_assessments[entity].transitions.append(transition)
        
        # Update coherence level based on good entities
        if quality_states["good"]:
            avg_coherence = sum(data["metrics"]["coherence"] for data in quality_states["good"].values()) / len(quality_states["good"])
            self.coherence_level = avg_coherence
        
        logger.info(f"Updated context from quality assessment: {len(quality_states['good'])} good, {len(quality_states['uncertain'])} uncertain, {len(quality_states['poor'])} poor entities")
    
    def get_high_quality_patterns(self) -> List[Pattern]:
        """Get high-quality patterns for retrieval.
        
        Returns:
            List of high-quality patterns
        """
        return [p for p in self.retrieval_patterns 
                if p.metadata.get("quality_state") == "good"]
    
    def get_emerging_patterns(self) -> List[Pattern]:
        """Get emerging patterns that show potential.
        
        Returns:
            List of emerging patterns
        """
        return [p for p in self.retrieval_patterns 
                if p.metadata.get("pattern_state") == PatternState.EMERGENT.name
                and p.metadata.get("emergence_rate", 0) > 0.6]
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get a summary of quality assessment.
        
        Returns:
            Dictionary with quality summary
        """
        return {
            "quality_state_distribution": self.quality_state_distribution,
            "pattern_state_distribution": {k.name: v for k, v in self.pattern_state_distribution.items()},
            "coherence_level": self.coherence_level,
            "high_quality_patterns_count": len(self.get_high_quality_patterns()),
            "emerging_patterns_count": len(self.get_emerging_patterns()),
            "total_patterns": len(self.retrieval_patterns)
        }
    
    def prioritize_patterns_by_quality(self) -> List[Pattern]:
        """Prioritize patterns based on quality for retrieval.
        
        Returns:
            List of patterns prioritized by quality
        """
        # Sort patterns by quality metrics
        return sorted(
            self.retrieval_patterns,
            key=lambda p: (
                1 if p.metadata.get("quality_state") == "good" else 0,
                p.metadata.get("coherence", 0),
                p.metadata.get("stability", 0) if p.metadata.get("quality_state") == "good" else p.metadata.get("emergence_rate", 0)
            ),
            reverse=True
        )
    
    def get_entity_quality_info(self, entity: str) -> Optional[QualityAssessmentInfo]:
        """Get quality information for an entity.
        
        Args:
            entity: The entity to get information for
            
        Returns:
            QualityAssessmentInfo or None if not found
        """
        return self.quality_assessments.get(entity)
    
    def get_related_entities(self, entity: str) -> List[Tuple[str, str, float]]:
        """Get entities related to the given entity.
        
        Args:
            entity: The entity to get related entities for
            
        Returns:
            List of (related_entity, relationship_type, confidence) tuples
        """
        related = []
        
        # Check for relationships in context-aware extraction results
        if "relationships" in self.context_aware_extraction_results:
            for rel in self.context_aware_extraction_results["relationships"]:
                if rel["source"] == entity:
                    related.append((rel["target"], rel["predicate"], rel["confidence"]))
                elif rel["target"] == entity:
                    related.append((rel["source"], f"inverse_{rel['predicate']}", rel["confidence"]))
        
        return related
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary representation.
        
        Returns:
            Dictionary representation of context
        """
        return {
            "query_patterns": [p.to_dict() for p in self.query_patterns],
            "retrieval_patterns": [p.to_dict() for p in self.retrieval_patterns],
            "augmentation_patterns": [p.to_dict() for p in self.augmentation_patterns],
            "coherence_level": self.coherence_level,
            "quality_state_distribution": self.quality_state_distribution,
            "pattern_state_distribution": {k.name: v for k, v in self.pattern_state_distribution.items()},
            "quality_summary": self.get_quality_summary(),
            "context_aware_extraction_summary": {
                "entities_count": len(self.context_aware_extraction_results.get("entities", [])),
                "relationships_count": len(self.context_aware_extraction_results.get("relationships", [])),
                "quality_summary": self.context_aware_extraction_results.get("quality_summary", {})
            }
        }
