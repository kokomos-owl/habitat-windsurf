"""
Quality assessment for context-aware pattern extraction.

This module provides the QualityAssessment class which assesses and tracks
quality states of entities based on contextual evidence.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import math

from .entity_context_manager import EntityContextManager
from src.habitat_evolution.adaptive_core.resonance.tonic_harmonic_metrics import TonicHarmonicMetrics
from src.habitat_evolution.core.pattern import PatternState, SignalMetrics, FlowMetrics

logger = logging.getLogger(__name__)

class QualityAssessment:
    """Assess and track quality states of entities.
    
    This class implements a Habitat-aligned quality state machine that tracks entity quality
    transitions through states based on contextual evidence, harmonic coherence,
    and pattern evolution metrics.
    """
    
    def __init__(self, threshold: float = 0.7):
        """Initialize the quality assessment.
        
        Args:
            threshold: Confidence threshold for "good" quality state
        """
        self.threshold = threshold
        
        # Quality states aligned with Habitat pattern states
        self.quality_states = {
            "good": {},       # entity -> quality metrics
            "uncertain": {},  # entity -> quality metrics
            "poor": {}        # entity -> quality metrics
        }
        
        # Track transitions between states
        self.quality_transitions = {}  # entity -> list of transitions
        
        # Harmonic metrics calculator
        self.harmonic_metrics = TonicHarmonicMetrics()
        
        # Domain-specific context patterns for climate risk
        self.domain_patterns = [
            "climate", "risk", "impact", "assessment", "system", 
            "coastal", "sea level", "erosion", "flooding", "storm",
            "temperature", "precipitation", "drought", "heat", "cold",
            "infrastructure", "adaptation", "mitigation", "vulnerability"
        ]
        
        # Pattern state mapping
        self.pattern_state_mapping = {
            PatternState.EMERGING: "uncertain",
            PatternState.STABLE: "good",
            PatternState.DECLINING: "poor",
            PatternState.TRANSFORMING: "uncertain",
            PatternState.NOISE: "poor",
            PatternState.MERGED: "good"
        }
        
        # Initialize metrics for quality assessment
        self.signal_metrics = SignalMetrics(
            strength=0.0,
            noise_ratio=0.0,
            persistence=0.0,
            reproducibility=0.0
        )
        self.flow_metrics = FlowMetrics(
            viscosity=0.0,
            back_pressure=0.0,
            volume=0.0,
            current=0.0
        )
    
    def assess_entities(self, entities: List[str], context_manager: EntityContextManager) -> None:
        """Assess the quality of entities based on their contexts.
        
        Args:
            entities: List of entities to assess
            context_manager: EntityContextManager with context information
        """
        for entity in entities:
            contexts = context_manager.get_contexts(entity)
            
            # Calculate basic quality score
            basic_score = self._calculate_basic_quality_score(entity, contexts)
            
            # Calculate Habitat-specific quality metrics
            quality_metrics = self._calculate_habitat_quality_metrics(entity, contexts)
            
            # Determine pattern state based on metrics
            pattern_state, state_reason = self._determine_pattern_state(quality_metrics)
            quality_state = self.pattern_state_mapping[pattern_state]
            
            # Store quality metrics with the entity
            quality_data = {
                "basic_score": basic_score,
                "metrics": quality_metrics,
                "pattern_state": pattern_state.name,
                "state_reason": state_reason
            }
            
            # Handle state transitions
            current_state = self._get_current_state(entity)
            if current_state and current_state != quality_state:
                self._record_transition(
                    entity, 
                    current_state, 
                    quality_state, 
                    quality_metrics
                )
                logger.info(f"Entity '{entity}' transitioned from {current_state} to {quality_state} ({state_reason})")
                
                # Remove from previous state
                if entity in self.quality_states[current_state]:
                    del self.quality_states[current_state][entity]
            
            # Add to appropriate state
            self.quality_states[quality_state][entity] = quality_data
            
            if quality_state == "good":
                logger.info(f"Entity '{entity}' assessed as good: coherence={quality_metrics['coherence']:.2f}, stability={quality_metrics['stability']:.2f}")
            elif quality_state == "uncertain":
                logger.debug(f"Entity '{entity}' remains uncertain: coherence={quality_metrics['coherence']:.2f}, emergence={quality_metrics['emergence_rate']:.2f}")
            elif quality_state == "poor":
                logger.warning(f"Entity '{entity}' assessed as poor: stability={quality_metrics['stability']:.2f}, energy={quality_metrics['energy_state']:.2f}")
    
    def _calculate_basic_quality_score(self, entity: str, contexts: List[Dict[str, str]]) -> float:
        """Calculate a basic quality score for an entity based on its contexts.
        
        Args:
            entity: The entity to calculate score for
            contexts: List of contexts for the entity
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        
        # Higher score for entities that appear in multiple contexts
        score += min(1.0, len(contexts) / 5.0) * 0.3
        
        # Check for domain-specific context patterns
        domain_pattern_matches = 0
        for context in contexts:
            for pattern in self.domain_patterns:
                if pattern in context["before"].lower() or pattern in context["after"].lower():
                    domain_pattern_matches += 1
        
        # Add score based on domain pattern matches
        score += min(0.5, domain_pattern_matches * 0.05)
        
        # Higher score for entities that start with capital letters
        if entity and entity[0].isupper():
            score += 0.1
        
        # Higher score for multi-word entities (more specific)
        if " " in entity:
            score += 0.1
        
        # Cap score at 1.0
        return min(1.0, score)
        
    def _calculate_habitat_quality_metrics(self, entity: str, contexts: List[Dict[str, str]]) -> Dict[str, float]:
        """Calculate Habitat-specific quality metrics for an entity.
        
        Args:
            entity: The entity to calculate metrics for
            contexts: List of contexts for the entity
            
        Returns:
            Dictionary of quality metrics
        """
        # Calculate basic metrics
        context_count = len(contexts)
        context_diversity = len(set(ctx["full_text"] for ctx in contexts))
        domain_relevance = self._calculate_domain_relevance(contexts)
        
        # Calculate harmonic metrics
        tonic_value = self._calculate_tonic_value(entity, contexts)
        harmonic_coherence = self._calculate_harmonic_coherence(contexts)
        phase_alignment = self._calculate_phase_alignment(entity, contexts)
        
        # Calculate pattern evolution metrics
        emergence_rate = self._calculate_emergence_rate(entity, contexts)
        stability = self._calculate_stability(contexts)
        energy_state = self._calculate_energy_state(entity, contexts)
        adaptation_rate = self._calculate_adaptation_rate(entity)
        cross_pattern_flow = self._calculate_cross_pattern_flow(entity, contexts)
        
        return {
            "coherence": harmonic_coherence,
            "emergence_rate": emergence_rate,
            "cross_pattern_flow": cross_pattern_flow,
            "energy_state": energy_state,
            "adaptation_rate": adaptation_rate,
            "stability": stability,
            "tonic_value": tonic_value,
            "phase_alignment": phase_alignment,
            "context_count": context_count,
            "context_diversity": context_diversity,
            "domain_relevance": domain_relevance
        }
    
    def _calculate_domain_relevance(self, contexts: List[Dict[str, str]]) -> float:
        """Calculate domain relevance based on context patterns.
        
        Args:
            contexts: List of contexts
            
        Returns:
            Domain relevance score
        """
        if not contexts:
            return 0.0
            
        pattern_matches = 0
        for context in contexts:
            for pattern in self.domain_patterns:
                if pattern in context["before"].lower() or pattern in context["after"].lower():
                    pattern_matches += 1
        
        return min(1.0, pattern_matches / (len(contexts) * 2))
    
    def _calculate_tonic_value(self, entity: str, contexts: List[Dict[str, str]]) -> float:
        """Calculate tonic value for an entity based on its contexts.
        
        Args:
            entity: The entity
            contexts: List of contexts
            
        Returns:
            Tonic value
        """
        if not contexts:
            return 0.0
            
        # Use frequency and position as proxy for tonic value
        context_count = len(contexts)
        word_count = len(entity.split())
        
        # More specific (multi-word) entities have higher tonic value
        specificity = min(1.0, word_count / 3)
        
        # More frequent entities have higher tonic value
        frequency = min(1.0, context_count / 10)
        
        return (specificity * 0.6) + (frequency * 0.4)
    
    def _calculate_harmonic_coherence(self, contexts: List[Dict[str, str]]) -> float:
        """Calculate harmonic coherence based on context similarity.
        
        Args:
            contexts: List of contexts
            
        Returns:
            Harmonic coherence score
        """
        if not contexts or len(contexts) < 2:
            return 0.5  # Neutral coherence for single context
            
        # Use context similarity as proxy for harmonic coherence
        unique_contexts = set(ctx["full_text"] for ctx in contexts)
        similarity_ratio = 1 - (len(unique_contexts) / len(contexts))
        
        # Higher similarity means higher coherence
        return 0.3 + (similarity_ratio * 0.7)  # Scale to 0.3-1.0 range
    
    def _calculate_phase_alignment(self, entity: str, contexts: List[Dict[str, str]]) -> float:
        """Calculate phase alignment based on context positions.
        
        Args:
            entity: The entity
            contexts: List of contexts
            
        Returns:
            Phase alignment score
        """
        if not contexts:
            return 0.0
            
        # Use consistent position in contexts as proxy for phase alignment
        before_positions = []
        after_positions = []
        
        for ctx in contexts:
            before_words = ctx["before"].split()
            after_words = ctx["after"].split()
            before_positions.append(len(before_words))
            after_positions.append(len(after_words))
        
        # Calculate variance in positions
        if before_positions:
            before_variance = self._calculate_variance(before_positions)
            before_alignment = 1.0 / (1.0 + before_variance)
        else:
            before_alignment = 0.0
            
        if after_positions:
            after_variance = self._calculate_variance(after_positions)
            after_alignment = 1.0 / (1.0 + after_variance)
        else:
            after_alignment = 0.0
        
        # Average alignment score
        return (before_alignment + after_alignment) / 2.0
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values.
        
        Args:
            values: List of values
            
        Returns:
            Variance
        """
        if not values:
            return 0.0
            
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def _calculate_emergence_rate(self, entity: str, contexts: List[Dict[str, str]]) -> float:
        """Calculate emergence rate based on context growth.
        
        Args:
            entity: The entity
            contexts: List of contexts
            
        Returns:
            Emergence rate score
        """
        # For now, use context count as proxy for emergence rate
        # In a real implementation, this would track context growth over time
        return min(1.0, len(contexts) / 10)
    
    def _calculate_stability(self, contexts: List[Dict[str, str]]) -> float:
        """Calculate stability based on context consistency.
        
        Args:
            contexts: List of contexts
            
        Returns:
            Stability score
        """
        if not contexts:
            return 0.0
            
        # Use context similarity as proxy for stability
        unique_contexts = set(ctx["full_text"] for ctx in contexts)
        similarity_ratio = 1 - (len(unique_contexts) / len(contexts))
        
        # More similar contexts indicate higher stability
        return similarity_ratio
    
    def _calculate_energy_state(self, entity: str, contexts: List[Dict[str, str]]) -> float:
        """Calculate energy state based on context activity.
        
        Args:
            entity: The entity
            contexts: List of contexts
            
        Returns:
            Energy state score
        """
        if not contexts:
            return 0.0
            
        # Use context diversity and domain relevance as proxy for energy
        unique_contexts = len(set(ctx["full_text"] for ctx in contexts))
        domain_relevance = self._calculate_domain_relevance(contexts)
        
        # Higher diversity and relevance indicate higher energy
        return (min(1.0, unique_contexts / 5) * 0.6) + (domain_relevance * 0.4)
    
    def _calculate_adaptation_rate(self, entity: str) -> float:
        """Calculate adaptation rate based on transition history.
        
        Args:
            entity: The entity
            
        Returns:
            Adaptation rate score
        """
        # Use transition history as proxy for adaptation rate
        transitions = self.quality_transitions.get(entity, [])
        return min(1.0, len(transitions) / 3)
    
    def _calculate_cross_pattern_flow(self, entity: str, contexts: List[Dict[str, str]]) -> float:
        """Calculate cross-pattern flow based on context connections.
        
        Args:
            entity: The entity
            contexts: List of contexts
            
        Returns:
            Cross-pattern flow score
        """
        if not contexts:
            return 0.0
            
        # Use presence of other entities in contexts as proxy for cross-pattern flow
        other_entities = set()
        for ctx in contexts:
            # Simple heuristic: look for capitalized words in context
            words = ctx["before"].split() + ctx["after"].split()
            for word in words:
                if word and word[0].isupper() and word != entity:
                    other_entities.add(word)
        
        # More connected entities indicate higher cross-pattern flow
        return min(1.0, len(other_entities) / 10)
    
    def _record_transition(self, entity: str, from_state: str, to_state: str, metrics: Dict[str, float]) -> None:
        """Record a quality state transition for an entity.
        
        Args:
            entity: The entity transitioning
            from_state: Starting quality state
            to_state: Ending quality state
            metrics: Quality metrics at transition time
        """
        if entity not in self.quality_transitions:
            self.quality_transitions[entity] = []
        
        self.quality_transitions[entity].append({
            "from": from_state,
            "to": to_state,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_quality_states(self) -> Dict[str, Dict[str, float]]:
        """Get the current quality states.
        
        Returns:
            Dictionary of quality states
        """
        return self.quality_states
    
    def get_transitions(self, entity: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get quality transitions for an entity or all entities.
        
        Args:
            entity: Optional entity to get transitions for
            
        Returns:
            Dictionary of transitions
        """
        if entity:
            return {entity: self.quality_transitions.get(entity, [])}
        return self.quality_transitions
    
    def _determine_pattern_state(self, metrics: Dict[str, float]) -> Tuple[PatternState, str]:
        """Determine pattern state based on quality metrics.
        
        Args:
            metrics: Quality metrics
            
        Returns:
            Tuple of (PatternState, reason)
        """
        coherence = metrics["coherence"]
        stability = metrics["stability"]
        emergence_rate = metrics["emergence_rate"]
        energy_state = metrics["energy_state"]
        
        # Determine state based on metrics
        if coherence > 0.7 and stability > 0.6:
            return PatternState.STABLE, "high coherence and stability"
        elif coherence > 0.7 and energy_state > 0.5:
            return PatternState.STABLE, "high coherence with good energy"
        elif emergence_rate > 0.6 and energy_state > 0.4:
            return PatternState.EMERGING, "high emergence rate"
        elif stability < 0.3 or energy_state < 0.2:
            return PatternState.DECLINING, "low stability or energy"
        else:
            return PatternState.EMERGING, "default emergent state"
    
    def _get_current_state(self, entity: str) -> Optional[str]:
        """Get the current quality state for an entity.
        
        Args:
            entity: The entity
            
        Returns:
            Current quality state or None if not found
        """
        for state, entities in self.quality_states.items():
            if entity in entities:
                return state
        return None
        
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get a summary of quality assessment.
        
        Returns:
            Dictionary with quality summary
        """
        # Calculate average metrics for each state
        good_metrics = {}
        uncertain_metrics = {}
        poor_metrics = {}
        
        if self.quality_states["good"]:
            for metric in next(iter(self.quality_states["good"].values()))["metrics"]:
                good_metrics[metric] = sum(e["metrics"][metric] for e in self.quality_states["good"].values()) / len(self.quality_states["good"])
                
        if self.quality_states["uncertain"]:
            for metric in next(iter(self.quality_states["uncertain"].values()))["metrics"]:
                uncertain_metrics[metric] = sum(e["metrics"][metric] for e in self.quality_states["uncertain"].values()) / len(self.quality_states["uncertain"])
                
        if self.quality_states["poor"]:
            for metric in next(iter(self.quality_states["poor"].values()))["metrics"]:
                poor_metrics[metric] = sum(e["metrics"][metric] for e in self.quality_states["poor"].values()) / len(self.quality_states["poor"])
        
        return {
            "good_entities_count": len(self.quality_states["good"]),
            "uncertain_entities_count": len(self.quality_states["uncertain"]),
            "poor_entities_count": len(self.quality_states["poor"]),
            "transition_count": sum(len(transitions) for transitions in self.quality_transitions.values()),
            "good_metrics": good_metrics,
            "uncertain_metrics": uncertain_metrics,
            "poor_metrics": poor_metrics,
            "pattern_states": {
                PatternState.STABLE.name: sum(1 for e in self.quality_states["good"].values() if e["pattern_state"] == PatternState.STABLE.name),
                PatternState.COHERENT.name: sum(1 for e in self.quality_states["good"].values() if e["pattern_state"] == PatternState.COHERENT.name),
                PatternState.EMERGENT.name: sum(1 for e in self.quality_states["uncertain"].values() if e["pattern_state"] == PatternState.EMERGENT.name),
                PatternState.DEGRADING.name: sum(1 for e in self.quality_states["poor"].values() if e["pattern_state"] == PatternState.DEGRADING.name)
            }
        }
