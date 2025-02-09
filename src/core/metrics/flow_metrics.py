"""Flow-based metric extraction system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import re
from enum import Enum

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
    
    # Track metric evolution
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def calculate_flow_confidence(self) -> float:
        """Calculate overall confidence based on flow metrics."""
        weights = {
            'base_confidence': 0.3,
            'viscosity': 0.2,
            'density': 0.2,
            'temporal_stability': 0.15,
            'cross_validation': 0.15
        }
        
        scores = {
            'base_confidence': self.confidence,
            'viscosity': 1.0 - (self.viscosity / 2),  # Lower viscosity is better
            'density': self.density,
            'temporal_stability': self.temporal_stability,
            'cross_validation': self.cross_validation_score
        }
        
        return sum(scores[k] * weights[k] for k in weights)

class MetricFlowManager:
    """Manages metric flows through the system."""
    
    def __init__(self):
        self.active_flows: Dict[str, MetricFlow] = {}
        self.pattern_flows: Dict[str, List[str]] = {}
        
        # Confidence thresholds
        self.min_confidence = 0.6
        self.preferred_confidence = 0.8
        
        # Pattern recognition settings
        self.pattern_matchers = {
            'number': r'[-+]?\d*\.?\d+',
            'percentage': r'[-+]?\d*\.?\d+\s*%',
            'range': r'[-+]?\d*\.?\d+\s*(?:to|-)\s*[-+]?\d*\.?\d+',
            'trend': r'(?:increase|decrease|rise|fall|grew|dropped)\s+(?:by|to|from)?\s*[-+]?\d*\.?\d+\s*%?'
        }
        
    def create_flow(self, pattern: str) -> MetricFlow:
        """Create a new metric flow from a pattern."""
        flow_id = f"flow_{len(self.active_flows) + 1}"
        flow = MetricFlow(flow_id=flow_id, source_pattern=pattern)
        
        self.active_flows[flow_id] = flow
        if pattern not in self.pattern_flows:
            self.pattern_flows[pattern] = []
        self.pattern_flows[pattern].append(flow_id)
        
        return flow
        
    def extract_metrics(self, text: str, context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Extract metrics using flow-based pattern matching."""
        metrics = []
        
        for pattern_type, regex in self.pattern_matchers.items():
            matches = re.finditer(regex, text, re.IGNORECASE)
            
            for match in matches:
                value = match.group(0)
                flow = self.create_flow(value)
                
                # Calculate initial confidence
                confidence = self._calculate_initial_confidence(value, pattern_type)
                
                # Adjust for context if provided
                if context:
                    confidence = self._adjust_confidence_for_context(confidence, context)
                
                flow.confidence = confidence
                
                metrics.append({
                    'value': value,
                    'type': pattern_type,
                    'confidence': confidence,
                    'flow_id': flow.flow_id
                })
        
        return metrics
    
    def _calculate_initial_confidence(self, value: str, pattern_type: str) -> float:
        """Calculate initial confidence for a metric."""
        base_confidence = 0.7  # Start with moderate confidence
        
        # Adjust based on pattern type
        type_modifiers = {
            'number': 0.1,
            'percentage': 0.15,
            'range': 0.05,
            'trend': 0.2
        }
        
        # Add type modifier
        confidence = base_confidence + type_modifiers.get(pattern_type, 0)
        
        # Penalize unusual values
        try:
            if pattern_type in ['number', 'percentage']:
                num = float(re.findall(r'[-+]?\d*\.?\d+', value)[0])
                if abs(num) > 1000 or abs(num) < 0.0001:
                    confidence *= 0.8
        except (ValueError, IndexError):
            confidence *= 0.7
            
        return min(1.0, max(0.0, confidence))
    
    def _adjust_confidence_for_context(self, base_confidence: float, context: Dict) -> float:
        """Adjust confidence based on contextual information."""
        confidence = base_confidence
        
        # Adjust for temporal context
        if 'temporal_distance' in context:
            # Reduce confidence for predictions far in the future
            temporal_factor = 1.0 - (min(context['temporal_distance'], 100) / 200)
            confidence *= temporal_factor
            
        # Adjust for source reliability
        if 'source_reliability' in context:
            confidence *= context['source_reliability']
            
        # Adjust for cross-validation
        if 'cross_validation_score' in context:
            confidence = (confidence + context['cross_validation_score']) / 2
            
        return min(1.0, max(0.0, confidence))
    
    def update_flow_metrics(self, flow_id: str, metrics: Dict[str, float]) -> None:
        """Update metrics for a specific flow."""
        if flow_id not in self.active_flows:
            return
            
        flow = self.active_flows[flow_id]
        
        # Update flow metrics
        for metric, value in metrics.items():
            if hasattr(flow, metric):
                setattr(flow, metric, value)
                
        # Record history
        flow.history.append({
            'timestamp': datetime.utcnow(),
            'metrics': metrics.copy(),
            'confidence': flow.calculate_flow_confidence()
        })
    
    def get_flow_confidence(self, flow_id: str) -> float:
        """Get current confidence for a flow."""
        if flow_id not in self.active_flows:
            return 0.0
            
        return self.active_flows[flow_id].calculate_flow_confidence()
    
    def get_pattern_confidence(self, pattern: str) -> float:
        """Get aggregated confidence for all flows of a pattern."""
        if pattern not in self.pattern_flows:
            return 0.0
            
        # Get confidences and weights
        confidences_and_weights = [
            (self.get_flow_confidence(flow_id), self.active_flows[flow_id].density)
            for flow_id in self.pattern_flows[pattern]
            if flow_id in self.active_flows
        ]
        
        if not confidences_and_weights:
            return 0.0
            
        # Calculate weighted average
        total_weight = sum(weight for _, weight in confidences_and_weights)
        weighted_sum = sum(conf * weight for conf, weight in confidences_and_weights)
        
        # Normalize and apply density penalty for too many flows
        base_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
        flow_count_penalty = max(0.0, 1.0 - (len(confidences_and_weights) - 1) * 0.1)
        
        return min(0.9, base_confidence * flow_count_penalty)
