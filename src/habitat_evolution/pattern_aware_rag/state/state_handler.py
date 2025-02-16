"""State handling for Pattern-Aware RAG.

This module manages the transformation and validation of graph states,
ensuring concept-relationship coherence throughout the RAG process.
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime

from .test_states import GraphStateSnapshot, ConceptNode, ConceptRelation, PatternState

@dataclass
class StateCoherenceMetrics:
    """Metrics for state coherence validation."""
    concept_confidence: float
    relationship_strength: float
    pattern_stability: float
    overall_coherence: float
    temporal_stability: float

class GraphStateHandler:
    """Handles graph state transformations and validation."""
    
    def __init__(self, min_coherence: float = 0.6):
        self.min_coherence = min_coherence
    
    def validate_state_coherence(
        self,
        state: GraphStateSnapshot
    ) -> StateCoherenceMetrics:
        """Validate coherence of graph state."""
        # Calculate concept confidence
        concept_confidences = [
            concept.confidence
            for concept in state.concepts.values()
        ]
        avg_concept_confidence = sum(concept_confidences) / len(concept_confidences)
        
        # Calculate relationship strength
        relationship_strengths = [
            rel.strength
            for rel in state.relations
        ]
        avg_relationship_strength = sum(relationship_strengths) / len(relationship_strengths)
        
        # Calculate pattern stability
        pattern_stabilities = [
            pattern.stability
            for pattern in state.patterns.values()
        ]
        avg_pattern_stability = sum(pattern_stabilities) / len(pattern_stabilities)
        
        # Get existing metrics
        overall_coherence = state.metrics.get('coherence', 0.0)
        temporal_stability = float(state.temporal_context.get('stability', 0.0))
        
        return StateCoherenceMetrics(
            concept_confidence=avg_concept_confidence,
            relationship_strength=avg_relationship_strength,
            pattern_stability=avg_pattern_stability,
            overall_coherence=overall_coherence,
            temporal_stability=temporal_stability
        )
    
    def prepare_prompt_context(
        self,
        state: GraphStateSnapshot,
        max_concepts: int = 15,
        max_relations: int = 25
    ) -> Dict[str, str]:
        """Prepare graph state context for prompt template."""
        # Format graph state overview
        state_overview = (
            f"ID: {state.id}\n"
            f"Timestamp: {state.timestamp}\n"
            f"Total Concepts: {len(state.concepts)}\n"
            f"Total Relations: {len(state.relations)}\n"
            f"Active Patterns: {len(state.patterns)}"
        )
        
        # Format concept details (most confident first)
        sorted_concepts = sorted(
            state.concepts.values(),
            key=lambda c: c.confidence,
            reverse=True
        )[:max_concepts]
        
        concept_str = "\n".join(
            f"- {c.content} ({c.type}): confidence={c.confidence:.2f}"
            for c in sorted_concepts
        )
        
        # Format relationships (strongest first)
        sorted_relations = sorted(
            state.relations,
            key=lambda r: r.strength,
            reverse=True
        )[:max_relations]
        
        relation_str = "\n".join(
            f"- {r.source_id} --[{r.type}: {r.strength:.2f}]--> {r.target_id}"
            for r in sorted_relations
        )
        
        # Format pattern context
        pattern_str = "\n".join(
            f"Pattern {pid}:\n"
            f"  - Coherence: {p.coherence:.2f}\n"
            f"  - Stability: {p.stability:.2f}\n"
            f"  - Stage: {p.emergence_stage}"
            for pid, p in state.patterns.items()
        )
        
        # Format temporal context
        temporal = (
            f"Evolution Stage: {state.temporal_context['stage']}\n"
            f"Stability Index: {state.temporal_context['stability']:.2f}\n"
            f"Recent Changes: {', '.join(state.temporal_context['recent_changes'])}"
        )
        
        return {
            "graph_state": state_overview,
            "concept_details": concept_str,
            "concept_relations": relation_str,
            "pattern_context": pattern_str,
            "temporal_context": temporal,
            "coherence_metrics": str(state.metrics)
        }
    
    def validate_state_transition(
        self,
        from_state: GraphStateSnapshot,
        to_state: GraphStateSnapshot
    ) -> bool:
        """Validate that state transition maintains coherence."""
        from_metrics = self.validate_state_coherence(from_state)
        to_metrics = self.validate_state_coherence(to_state)
        
        # Core coherence must not decrease significantly
        coherence_maintained = (
            to_metrics.overall_coherence >= from_metrics.overall_coherence * 0.95
        )
        
        # Pattern stability should improve or maintain
        stability_maintained = (
            to_metrics.pattern_stability >= from_metrics.pattern_stability
        )
        
        # Relationship strength should not weaken
        relationships_maintained = (
            to_metrics.relationship_strength >= from_metrics.relationship_strength * 0.95
        )
        
        return all([
            coherence_maintained,
            stability_maintained,
            relationships_maintained
        ])
