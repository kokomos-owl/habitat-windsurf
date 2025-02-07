"""Document evolution tracking system.

This module tracks how patterns emerge from document interfaces,
maintaining the relationship between source documents, adaptive IDs,
and evolved patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import logging
from pathlib import Path
import json
from collections import defaultdict

from ..processor import RiskMetric, ProcessingResult
from .state_space import StateSpace, StateTransition
from .pattern_feedback import PatternFeedback, LearningWindow

logger = logging.getLogger(__name__)

@dataclass
class DocumentInterface:
    """Represents a document's interface for pattern emergence."""
    doc_id: str  # MongoDB _id
    adaptive_id: str
    content_hash: str
    patterns: Set[str] = field(default_factory=set)
    interface_metrics: Dict[str, float] = field(default_factory=dict)
    last_processed: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'doc_id': self.doc_id,
            'adaptive_id': self.adaptive_id,
            'content_hash': self.content_hash,
            'patterns': list(self.patterns),
            'interface_metrics': self.interface_metrics,
            'last_processed': self.last_processed.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentInterface':
        """Create from dictionary."""
        return cls(
            doc_id=data['doc_id'],
            adaptive_id=data['adaptive_id'],
            content_hash=data['content_hash'],
            patterns=set(data['patterns']),
            interface_metrics=data['interface_metrics'],
            last_processed=datetime.fromisoformat(data['last_processed'])
        )

@dataclass
class DocumentEvolutionTracker:
    """Tracks pattern evolution from document interfaces."""
    
    state_space: StateSpace = field(default_factory=StateSpace)
    pattern_feedback: PatternFeedback = field(default_factory=lambda: PatternFeedback(
        LearningWindow(window_size=timedelta(days=30))
    ))
    interfaces: Dict[str, DocumentInterface] = field(default_factory=dict)
    
    def process_document(self, 
                        doc_id: str,
                        adaptive_id: str,
                        content: str,
                        processor_result: ProcessingResult) -> Dict[str, Any]:
        """Process document and track pattern evolution."""
        try:
            # Create or update interface
            interface = self._get_or_create_interface(doc_id, adaptive_id, content)
            
            # Track patterns from processing
            new_patterns = set()
            for metric in processor_result.metrics:
                pattern_id = f"{metric.risk_type}_{metric.timeframe}"
                new_patterns.add(pattern_id)
                
                # Record state transition
                self._record_pattern_transition(
                    pattern_id=pattern_id,
                    interface=interface,
                    metric=metric
                )
            
            # Update interface patterns
            removed_patterns = interface.patterns - new_patterns
            added_patterns = new_patterns - interface.patterns
            interface.patterns = new_patterns
            
            # Get feedback for pattern changes
            feedback = self._get_pattern_feedback(
                interface=interface,
                added_patterns=added_patterns,
                removed_patterns=removed_patterns
            )
            
            # Update interface metrics
            interface.interface_metrics.update({
                'pattern_count': len(interface.patterns),
                'pattern_stability': feedback.get('stability', 0.0),
                'emergence_rate': feedback.get('emergence_rate', 0.0)
            })
            
            interface.last_processed = datetime.utcnow()
            
            return {
                'interface_id': interface.adaptive_id,
                'patterns': list(interface.patterns),
                'metrics': interface.interface_metrics,
                'feedback': feedback
            }
            
        except Exception as e:
            logger.error(f"Error processing document evolution: {e}")
            return {}
    
    def _get_or_create_interface(self,
                               doc_id: str,
                               adaptive_id: str,
                               content: str) -> DocumentInterface:
        """Get or create document interface."""
        content_hash = hash(content)
        
        if adaptive_id in self.interfaces:
            interface = self.interfaces[adaptive_id]
            if interface.content_hash != content_hash:
                # Content changed, update hash
                interface.content_hash = content_hash
        else:
            interface = DocumentInterface(
                doc_id=doc_id,
                adaptive_id=adaptive_id,
                content_hash=content_hash
            )
            self.interfaces[adaptive_id] = interface
            
        return interface
    
    def _record_pattern_transition(self,
                                 pattern_id: str,
                                 interface: DocumentInterface,
                                 metric: RiskMetric) -> None:
        """Record pattern state transition from document."""
        current_state = {
            'confidence': metric.confidence,
            'semantic_weight': metric.semantic_weight,
            'value': metric.value,
            'timeframe': metric.timeframe
        }
        
        # Get previous state if exists
        previous_state = None
        if pattern_id in interface.patterns:
            transitions = self.state_space.transitions.get(pattern_id, [])
            if transitions:
                previous_state = transitions[-1].to_state
        
        # Record transition
        self.state_space.record_transition(
            adaptive_id=interface.adaptive_id,
            from_state=previous_state or {},
            to_state=current_state,
            interface_context={
                'interface_ids': [interface.adaptive_id],
                'doc_id': interface.doc_id,
                'pattern_type': 'risk_metric'
            },
            energy_delta=metric.semantic_weight * metric.confidence
        )
    
    def _get_pattern_feedback(self,
                            interface: DocumentInterface,
                            added_patterns: Set[str],
                            removed_patterns: Set[str]) -> Dict[str, Any]:
        """Get feedback about pattern changes."""
        feedback = {
            'added_patterns': [],
            'removed_patterns': [],
            'stability': 0.0,
            'emergence_rate': 0.0
        }
        
        # Process added patterns
        for pattern in added_patterns:
            state = self._get_pattern_state(pattern)
            if state:
                pattern_feedback = self.pattern_feedback.process_pattern_state(
                    pattern_id=pattern,
                    interface_id=interface.adaptive_id,
                    current_state=state
                )
                feedback['added_patterns'].append(pattern_feedback)
        
        # Process removed patterns
        for pattern in removed_patterns:
            state = self._get_pattern_state(pattern)
            if state:
                pattern_feedback = self.pattern_feedback.process_pattern_state(
                    pattern_id=pattern,
                    interface_id=interface.adaptive_id,
                    current_state=state,
                    previous_state=self._get_previous_state(pattern)
                )
                feedback['removed_patterns'].append(pattern_feedback)
        
        # Calculate overall metrics
        if feedback['added_patterns'] or feedback['removed_patterns']:
            metrics = []
            for f in feedback['added_patterns'] + feedback['removed_patterns']:
                if 'metrics' in f:
                    metrics.append(f['metrics'])
            
            if metrics:
                feedback['stability'] = sum(m.get('stability', 0) for m in metrics) / len(metrics)
                feedback['emergence_rate'] = sum(m.get('emergence_rate', 0) for m in metrics) / len(metrics)
        
        return feedback
    
    def _get_pattern_state(self, pattern: str) -> Optional[Dict[str, Any]]:
        """Get current state of a pattern."""
        transitions = self.state_space.transitions.get(pattern, [])
        if transitions:
            return transitions[-1].to_state
        return None
    
    def _get_previous_state(self, pattern: str) -> Optional[Dict[str, Any]]:
        """Get previous state of a pattern."""
        transitions = self.state_space.transitions.get(pattern, [])
        if len(transitions) > 1:
            return transitions[-2].to_state
        return None
    
    def save_state(self, filepath: str) -> None:
        """Save evolution state to file."""
        state = {
            'interfaces': {
                id_: interface.to_dict()
                for id_, interface in self.interfaces.items()
            }
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str) -> None:
        """Load evolution state from file."""
        with open(filepath) as f:
            state = json.load(f)
            self.interfaces = {
                id_: DocumentInterface.from_dict(data)
                for id_, data in state['interfaces'].items()
            }
