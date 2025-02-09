"""Enhanced flow-based metric extraction system with pattern evolution."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import re
from enum import Enum
import logging
from .pattern_learner import PatternLearner
from .pattern_recognition import ClimatePatternRecognizer, PatternMatch

logger = logging.getLogger(__name__)

class FlowState(Enum):
    """States of a metric flow."""
    ACTIVE = "active"
    LEARNING = "learning"
    STABLE = "stable"
    DEPRECATED = "deprecated"

@dataclass
class MetricFlow:
    """Represents a flow of metrics through the system."""
    flow_id: str
    source_pattern: str
    confidence: float = 1.0
    viscosity: float = 0.35  # Base viscosity
    density: float = 1.0
    temporal_stability: float = 1.0
    cross_validation_score: float = 1.0
    state: FlowState = FlowState.ACTIVE
    
    # Track metric evolution
    history: List[Dict[str, Any]] = field(default_factory=list)
    pattern_variants: List[Tuple[str, float]] = field(default_factory=list)
    
    def calculate_flow_confidence(self) -> float:
        """Calculate overall confidence based on flow metrics."""
        weights = {
            'base_confidence': 0.25,
            'viscosity': 0.15,
            'density': 0.15,
            'temporal_stability': 0.25,
            'cross_validation': 0.20
        }
        
        scores = {
            'base_confidence': self.confidence,
            'viscosity': 1.0 - (self.viscosity / 2),  # Lower viscosity is better
            'density': self.density,
            'temporal_stability': self.temporal_stability,
            'cross_validation': self.cross_validation_score
        }
        
        return sum(scores[k] * weights[k] for k in weights)

    def add_pattern_variant(self, pattern: str, success_rate: float) -> None:
        """Add a new pattern variant with its success rate."""
        self.pattern_variants.append((pattern, success_rate))
        self.pattern_variants.sort(key=lambda x: x[1], reverse=True)

class MetricFlowManager:
    """Manages metric flows through the system with pattern evolution."""
    
    def __init__(self):
        self.active_flows: Dict[str, MetricFlow] = {}
        self.pattern_flows: Dict[str, List[str]] = {}
        self.pattern_recognizer = ClimatePatternRecognizer()
        
        # Adaptive confidence thresholds
        self.base_confidence = 0.6
        self.min_confidence = 0.4
        self.learning_threshold = 0.7
        self.stability_threshold = 0.85
        
        # Flow evolution settings
        self.max_pattern_variants = 5
        self.variant_retention_period = 30  # days
        
    def extract_metrics(self, content: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract metrics using evolved patterns and flow management."""
        metrics = []
        pattern_matches = self.pattern_recognizer.find_patterns(content)
        
        for match in pattern_matches:
            flow_id = self._get_or_create_flow(match)
            flow = self.active_flows[flow_id]
            
            # Update flow metrics
            self._update_flow_metrics(flow, match, context)
            
            # Calculate confidence and check threshold
            confidence = flow.calculate_flow_confidence()
            if confidence >= self.min_confidence:
                metrics.append(self._create_metric_dict(match, flow, confidence))
            
            # Handle pattern evolution
            self._evolve_pattern(flow, match, confidence)
        
        return metrics
    
    def _get_or_create_flow(self, match: PatternMatch) -> str:
        """Get existing flow or create new one for the pattern."""
        pattern_type = match.pattern_type
        if pattern_type not in self.pattern_flows:
            self.pattern_flows[pattern_type] = []
            
        # Try to find matching flow
        for flow_id in self.pattern_flows[pattern_type]:
            flow = self.active_flows[flow_id]
            if self._pattern_matches_flow(match, flow):
                return flow_id
                
        # Create new flow
        flow_id = f"flow_{pattern_type}_{len(self.pattern_flows[pattern_type])}"
        self.active_flows[flow_id] = MetricFlow(
            flow_id=flow_id,
            source_pattern=match.pattern_type
        )
        self.pattern_flows[pattern_type].append(flow_id)
        return flow_id
    
    def _pattern_matches_flow(self, match: PatternMatch, flow: MetricFlow) -> bool:
        """Check if pattern match corresponds to existing flow."""
        # Check if pattern type matches
        if match.pattern_type != flow.source_pattern:
            return False
            
        # Check temporal context overlap
        if 'temporal_indicators' in match.context:
            flow_temporal = self._get_flow_temporal_context(flow)
            match_temporal = set(ind['value'] for ind in match.context['temporal_indicators'])
            return bool(flow_temporal & match_temporal)
            
        return False
    
    def _get_flow_temporal_context(self, flow: MetricFlow) -> set:
        """Get temporal context from flow history."""
        temporal_context = set()
        for entry in flow.history:
            if 'temporal_indicators' in entry:
                temporal_context.update(ind['value'] for ind in entry['temporal_indicators'])
        return temporal_context
    
    def _update_flow_metrics(self, flow: MetricFlow, match: PatternMatch, context: Dict[str, Any]) -> None:
        """Update flow metrics based on new match."""
        # Update confidence based on pattern match
        flow.confidence = match.confidence
        
        # Update temporal stability if available
        if 'temporal_distance' in context:
            flow.temporal_stability = max(0.0, 1.0 - (context['temporal_distance'] / 365))
        
        # Update cross validation score
        if 'cross_validation_score' in context:
            flow.cross_validation_score = context['cross_validation_score']
        
        # Update history
        flow.history.append({
            'timestamp': datetime.now(),
            'pattern_type': match.pattern_type,
            'value': match.value,
            'confidence': match.confidence,
            'temporal_indicators': match.context.get('temporal_indicators', [])
        })
    
    def _evolve_pattern(self, flow: MetricFlow, match: PatternMatch, confidence: float) -> None:
        """Handle pattern evolution based on match results."""
        if confidence < self.learning_threshold:
            # Pattern needs improvement
            flow.state = FlowState.LEARNING
            self.pattern_recognizer.learner.record_result(
                match.pattern_type,
                match.context['surrounding_text'],
                False
            )
        elif confidence > self.stability_threshold:
            # Pattern is performing well
            flow.state = FlowState.STABLE
            self.pattern_recognizer.learner.record_result(
                match.pattern_type,
                match.context['surrounding_text'],
                True
            )
        
        # Get evolved patterns
        stats = self.pattern_recognizer.learner.get_pattern_stats(match.pattern_type)
        if stats and 'best_pattern' in stats:
            flow.add_pattern_variant(stats['best_pattern'], stats['success_rate'])
    
    def _create_metric_dict(self, match: PatternMatch, flow: MetricFlow, confidence: float) -> Dict[str, Any]:
        """Create metric dictionary from match and flow data."""
        return {
            'type': match.pattern_type,
            'value': match.value,
            'unit': match.unit,
            'confidence': confidence,
            'flow_id': flow.flow_id,
            'flow_state': flow.state.value,
            'temporal_context': match.context.get('temporal_indicators', []),
            'pattern_evolution': {
                'current_pattern': flow.source_pattern,
                'variants': flow.pattern_variants[:self.max_pattern_variants],
                'stability': flow.temporal_stability
            }
        }
    
    def get_flow_stats(self) -> Dict[str, Any]:
        """Get statistics about all metric flows."""
        return {
            'total_flows': len(self.active_flows),
            'pattern_types': list(self.pattern_flows.keys()),
            'flow_states': {
                state.value: len([f for f in self.active_flows.values() if f.state == state])
                for state in FlowState
            },
            'pattern_evolution': {
                pattern_type: self.pattern_recognizer.learner.get_pattern_stats(pattern_type)
                for pattern_type in self.pattern_flows.keys()
            }
        }
