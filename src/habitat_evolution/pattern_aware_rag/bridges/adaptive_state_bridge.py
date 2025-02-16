"""Bridge between Pattern-Aware RAG state evolution and AdaptiveID system.

This module ensures that state transitions and pattern evolution are properly
reflected in the AdaptiveID system, maintaining coherence and provenance.
"""

from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

from ..adaptive_core.id.adaptive_id import AdaptiveID, Version
from .state_evolution import (
    StateTransaction,
    PatternEvolutionEvent,
    StateEvolutionTracker
)
from .test_states import (
    GraphStateSnapshot,
    PatternState,
    ConceptNode
)

class AdaptiveStateManager:
    """Manages the bridge between graph state evolution and adaptive IDs."""
    
    def __init__(self):
        self._evolution_tracker = StateEvolutionTracker()
        self._pattern_to_adaptive: Dict[str, AdaptiveID] = {}
        self._concept_to_adaptive: Dict[str, AdaptiveID] = {}
        
        # Initialize learning control
        from .learning_control import EventCoordinator
        self._event_coordinator = EventCoordinator()
        
        # Start initial learning window
        self._event_coordinator.create_learning_window()
    
    def process_state_transaction(
        self,
        transaction: StateTransaction
    ) -> Tuple[List[AdaptiveID], List[str]]:
        """Process a state transaction and update adaptive IDs.
        
        Returns:
            Tuple of (updated_ids, removed_ids)
        """
        # Calculate overall stability score
        stability_score = self._calculate_stability(transaction)
        
        # Queue transaction event and get recommended delay
        delay = self._event_coordinator.queue_event(
            event_type='state_transaction',
            entity_id=transaction.transaction_id,
            data={
                'from_state_id': transaction.from_state.id,
                'to_state_id': transaction.to_state.id,
                'changes': transaction.changes
            },
            stability_score=stability_score
        )
        
        # Apply back pressure delay if needed
        if delay > 0:
            import time
            time.sleep(delay)
        
        updated_ids = []
        removed_ids = []
        
        # Process concept changes
        self._process_concept_changes(
            transaction.from_state,
            transaction.to_state,
            updated_ids,
            removed_ids
        )
        
        # Process pattern changes
        self._process_pattern_changes(
            transaction.from_state,
            transaction.to_state,
            updated_ids,
            removed_ids
        )
        
        # Update version history
        self._update_version_history(transaction)
        
        return updated_ids, removed_ids
    
    def _process_concept_changes(
        self,
        from_state: GraphStateSnapshot,
        to_state: GraphStateSnapshot,
        updated_ids: List[AdaptiveID],
        removed_ids: List[str]
    ):
        """Process concept-level changes."""
        # Handle new or updated concepts
        for concept_id, concept in to_state.concepts.items():
            if concept_id not in self._concept_to_adaptive:
                # New concept - create adaptive ID
                adaptive_id = AdaptiveID(
                    base_concept=concept.content,
                    creator_id="pattern_aware_rag",
                    weight=1.0,
                    confidence=concept.confidence,
                    uncertainty=1.0 - concept.confidence
                )
                self._concept_to_adaptive[concept_id] = adaptive_id
                updated_ids.append(adaptive_id)
            else:
                # Update existing concept
                adaptive_id = self._concept_to_adaptive[concept_id]
                if concept_id in from_state.concepts:
                    old_concept = from_state.concepts[concept_id]
                    if (old_concept.confidence != concept.confidence or
                        old_concept.content != concept.content):
                        self._update_concept_adaptive(
                            adaptive_id,
                            old_concept,
                            concept
                        )
                        updated_ids.append(adaptive_id)
        
        # Handle removed concepts
        for concept_id in from_state.concepts:
            if concept_id not in to_state.concepts:
                if concept_id in self._concept_to_adaptive:
                    removed_ids.append(concept_id)
                    del self._concept_to_adaptive[concept_id]
    
    def _process_pattern_changes(
        self,
        from_state: GraphStateSnapshot,
        to_state: GraphStateSnapshot,
        updated_ids: List[AdaptiveID],
        removed_ids: List[str]
    ):
        """Process pattern-level changes."""
        # Handle new or updated patterns
        for pattern_id, pattern in to_state.patterns.items():
            if pattern_id not in self._pattern_to_adaptive:
                # New pattern - create adaptive ID
                adaptive_id = AdaptiveID(
                    base_concept=f"pattern_{pattern_id}",
                    creator_id="pattern_aware_rag",
                    weight=pattern.stability,
                    confidence=pattern.coherence,
                    uncertainty=1.0 - pattern.stability
                )
                self._pattern_to_adaptive[pattern_id] = adaptive_id
                updated_ids.append(adaptive_id)
            else:
                # Update existing pattern
                adaptive_id = self._pattern_to_adaptive[pattern_id]
                if pattern_id in from_state.patterns:
                    old_pattern = from_state.patterns[pattern_id]
                    if (old_pattern.coherence != pattern.coherence or
                        old_pattern.stability != pattern.stability or
                        old_pattern.concepts != pattern.concepts):
                        self._update_pattern_adaptive(
                            adaptive_id,
                            old_pattern,
                            pattern
                        )
                        updated_ids.append(adaptive_id)
        
        # Handle removed patterns
        for pattern_id in from_state.patterns:
            if pattern_id not in to_state.patterns:
                if pattern_id in self._pattern_to_adaptive:
                    removed_ids.append(pattern_id)
                    del self._pattern_to_adaptive[pattern_id]
    
    def _update_concept_adaptive(
        self,
        adaptive_id: AdaptiveID,
        old_concept: ConceptNode,
        new_concept: ConceptNode
    ):
        """Update an adaptive ID for a concept change."""
        version_data = {
            "content": new_concept.content,
            "confidence": new_concept.confidence,
            "type": new_concept.type,
            "properties": new_concept.properties
        }
        
        adaptive_id.add_version(
            data=version_data,
            origin="pattern_aware_rag"
        )
        
        # Update core properties
        adaptive_id.confidence = new_concept.confidence
        adaptive_id.uncertainty = 1.0 - new_concept.confidence
    
    def _update_pattern_adaptive(
        self,
        adaptive_id: AdaptiveID,
        old_pattern: PatternState,
        new_pattern: PatternState
    ):
        """Update an adaptive ID for a pattern change."""
        version_data = {
            "coherence": new_pattern.coherence,
            "stability": new_pattern.stability,
            "concepts": list(new_pattern.concepts),
            "emergence_stage": new_pattern.emergence_stage
        }
        
        adaptive_id.add_version(
            data=version_data,
            origin="pattern_aware_rag"
        )
        
        # Update core properties
        adaptive_id.weight = new_pattern.stability
        adaptive_id.confidence = new_pattern.coherence
        adaptive_id.uncertainty = 1.0 - new_pattern.stability
    
    def _calculate_stability(self, transaction: StateTransaction) -> float:
        """Calculate overall stability score for a transaction."""
        # Get window stats
        stats = self._event_coordinator.get_window_stats()
        
        # Base stability on multiple factors
        stability_factors = [
            transaction.coherence_delta.pattern_stability,
            transaction.coherence_delta.concept_confidence,
            0.5 if stats.get('is_saturated', False) else 1.0,  # Penalize saturated windows
            max(0.0, 1.0 - (stats.get('change_count', 0) / 100))  # Penalize high change rates
        ]
        
        return sum(stability_factors) / len(stability_factors)
    
    def _update_version_history(self, transaction: StateTransaction):
        """Update version history for all affected adaptive IDs."""
        timestamp = transaction.timestamp.isoformat()
        
        # Update concept versions
        for concept_id, concept in transaction.to_state.concepts.items():
            if concept_id in self._concept_to_adaptive:
                adaptive_id = self._concept_to_adaptive[concept_id]
                adaptive_id.temporal_context[timestamp] = {
                    "transaction_id": transaction.transaction_id,
                    "coherence_delta": transaction.coherence_delta.concept_confidence,
                    "emergence_type": transaction.emergence_type,
                    "learning_window": self._event_coordinator.get_window_stats()
                }
        
        # Update pattern versions
        for pattern_id, pattern in transaction.to_state.patterns.items():
            if pattern_id in self._pattern_to_adaptive:
                adaptive_id = self._pattern_to_adaptive[pattern_id]
                adaptive_id.temporal_context[timestamp] = {
                    "transaction_id": transaction.transaction_id,
                    "coherence_delta": transaction.coherence_delta.pattern_stability,
                    "emergence_type": transaction.emergence_type
                }
    
    def get_adaptive_id(self, entity_id: str) -> Optional[AdaptiveID]:
        """Get the adaptive ID for a concept or pattern."""
        if entity_id in self._concept_to_adaptive:
            return self._concept_to_adaptive[entity_id]
        return self._pattern_to_adaptive.get(entity_id)
