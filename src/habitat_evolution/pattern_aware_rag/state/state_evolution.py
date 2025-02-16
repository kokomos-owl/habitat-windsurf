"""Pattern-Aware RAG state evolution tracking.

This module manages the transaction-based evolution of graph states,
ensuring proper tracking of pattern changes and maintaining state history.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set
from uuid import uuid4

from .test_states import GraphStateSnapshot, PatternState
from .state_handler import GraphStateHandler, StateCoherenceMetrics

@dataclass
class StateTransaction:
    """Represents a state transition transaction."""
    transaction_id: str
    timestamp: datetime
    from_state: GraphStateSnapshot
    to_state: GraphStateSnapshot
    changes: Dict[str, any]
    coherence_delta: StateCoherenceMetrics
    emergence_type: str  # NATURAL, GUIDED, or POTENTIAL
    
    @property
    def is_valid(self) -> bool:
        """Check if the state transition is valid."""
        handler = GraphStateHandler()
        return handler.validate_state_transition(
            self.from_state,
            self.to_state
        )

@dataclass
class PatternEvolutionEvent:
    """Represents a pattern evolution event."""
    pattern_id: str
    event_type: str  # EMERGE, MERGE, TRANSFORM, MAINTAIN
    timestamp: datetime
    from_state: Optional[PatternState]
    to_state: PatternState
    context: Dict[str, any]
    confidence: float

class StateEvolutionTracker:
    """Tracks and manages state evolution through transactions."""
    
    def __init__(self):
        self._transactions: List[StateTransaction] = []
        self._pattern_events: Dict[str, List[PatternEvolutionEvent]] = {}
        self._state_handler = GraphStateHandler()
        
        # Initialize adaptive state management
        from .adaptive_state_bridge import AdaptiveStateManager
        self._adaptive_manager = AdaptiveStateManager()
    
    def create_transaction(
        self,
        from_state: GraphStateSnapshot,
        to_state: GraphStateSnapshot,
        changes: Dict[str, any],
        emergence_type: str
    ) -> StateTransaction:
        """Create a new state transition transaction."""
        # Calculate coherence metrics delta
        from_metrics = self._state_handler.validate_state_coherence(from_state)
        to_metrics = self._state_handler.validate_state_coherence(to_state)
        
        transaction = StateTransaction(
            transaction_id=str(uuid4()),
            timestamp=datetime.now(),
            from_state=from_state,
            to_state=to_state,
            changes=changes,
            coherence_delta=StateCoherenceMetrics(
                concept_confidence=to_metrics.concept_confidence - from_metrics.concept_confidence,
                relationship_strength=to_metrics.relationship_strength - from_metrics.relationship_strength,
                pattern_stability=to_metrics.pattern_stability - from_metrics.pattern_stability,
                overall_coherence=to_metrics.overall_coherence - from_metrics.overall_coherence,
                temporal_stability=to_metrics.temporal_stability - from_metrics.temporal_stability
            ),
            emergence_type=emergence_type
        )
        
        if transaction.is_valid:
            self._transactions.append(transaction)
            self._track_pattern_events(transaction)
            
            # Update adaptive IDs
            updated_ids, removed_ids = self._adaptive_manager.process_state_transaction(transaction)
            
            # Add adaptive context to transaction metadata
            transaction.changes['adaptive_updates'] = {
                'updated_ids': [aid.id for aid in updated_ids],
                'removed_ids': removed_ids
            }
            
            return transaction
        else:
            raise ValueError("Invalid state transition - coherence requirements not met")
    
    def _track_pattern_events(self, transaction: StateTransaction):
        """Track pattern evolution events from a transaction."""
        # Track pattern additions
        for pattern_id, new_pattern in transaction.to_state.patterns.items():
            if pattern_id not in transaction.from_state.patterns:
                self._add_pattern_event(
                    pattern_id=pattern_id,
                    event_type="EMERGE",
                    from_state=None,
                    to_state=new_pattern,
                    context=transaction.changes,
                    confidence=new_pattern.coherence
                )
        
        # Track pattern updates
        for pattern_id, old_pattern in transaction.from_state.patterns.items():
            if pattern_id in transaction.to_state.patterns:
                new_pattern = transaction.to_state.patterns[pattern_id]
                event_type = self._determine_evolution_type(old_pattern, new_pattern)
                self._add_pattern_event(
                    pattern_id=pattern_id,
                    event_type=event_type,
                    from_state=old_pattern,
                    to_state=new_pattern,
                    context=transaction.changes,
                    confidence=new_pattern.coherence
                )
    
    def _add_pattern_event(
        self,
        pattern_id: str,
        event_type: str,
        from_state: Optional[PatternState],
        to_state: PatternState,
        context: Dict[str, any],
        confidence: float
    ):
        """Add a pattern evolution event."""
        if pattern_id not in self._pattern_events:
            self._pattern_events[pattern_id] = []
            
        event = PatternEvolutionEvent(
            pattern_id=pattern_id,
            event_type=event_type,
            timestamp=datetime.now(),
            from_state=from_state,
            to_state=to_state,
            context=context,
            confidence=confidence
        )
        self._pattern_events[pattern_id].append(event)
    
    def _determine_evolution_type(
        self,
        old_pattern: PatternState,
        new_pattern: PatternState
    ) -> str:
        """Determine the type of pattern evolution."""
        if old_pattern.concepts == new_pattern.concepts:
            return "MAINTAIN"
        elif len(new_pattern.concepts) > len(old_pattern.concepts):
            return "MERGE"
        elif old_pattern.coherence < new_pattern.coherence:
            return "TRANSFORM"
        else:
            return "MAINTAIN"
    
    def get_pattern_history(
        self,
        pattern_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[PatternEvolutionEvent]:
        """Get evolution history for a specific pattern."""
        events = self._pattern_events.get(pattern_id, [])
        if not (start_time or end_time):
            return events
            
        return [
            event for event in events
            if (not start_time or event.timestamp >= start_time) and
               (not end_time or event.timestamp <= end_time)
        ]
    
    def get_state_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[StateTransaction]:
        """Get state transition history."""
        if not (start_time or end_time):
            return self._transactions
            
        return [
            tx for tx in self._transactions
            if (not start_time or tx.timestamp >= start_time) and
               (not end_time or tx.timestamp <= end_time)
        ]
