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
        """Calculate overall confidence based on flow metrics and field topology."""
        weights = {
            'base_confidence': 0.25,
            'viscosity': 0.15,
            'density': 0.15,
            'temporal_stability': 0.15,
            'cross_validation': 0.15,
            'field_stability': 0.15  # New weight for field topology
        }
        
        # Get field stability from topology
        field_state = self._analyze_vector_field()
        field_stability = 1.0 - min(1.0, field_state.divergence / self.collapse_threshold)
        
        scores = {
            'base_confidence': self.confidence,
            'viscosity': 1.0 - (self.viscosity / 2),  # Lower viscosity is better
            'density': self.density,
            'temporal_stability': self.temporal_stability,
            'cross_validation': self.cross_validation_score,
            'field_stability': field_stability
        }
        
        return sum(scores[k] * weights[k] for k in weights)

@dataclass
class VectorFieldState:
    """Represents the state of the vector field at a point in time."""
    magnitude: float
    direction: float
    divergence: float
    curl: float
    critical_points: List[Dict[str, Any]] = field(default_factory=list)

class MetricFlowManager:
    """Manages metric flows through the system."""
    
    def __init__(self):
        self.active_flows: Dict[str, MetricFlow] = {}
        self.pattern_flows: Dict[str, List[str]] = {}
        self.field_history: List[VectorFieldState] = []
        
        # Topology parameters
        self.field_resolution = 0.1
        self.attractor_radius = 0.2
        self.collapse_threshold = 0.3
        
        # Confidence thresholds
        self.min_confidence = 0.6
        
    def create_flow(self, flow_id: str, source_pattern: str) -> MetricFlow:
        """Create a new metric flow."""
        if flow_id in self.active_flows:
            return self.active_flows[flow_id]
            
        flow = MetricFlow(flow_id, source_pattern)
        self.active_flows[flow_id] = flow
        
        if source_pattern not in self.pattern_flows:
            self.pattern_flows[source_pattern] = []
        self.pattern_flows[source_pattern].append(flow_id)
        
        return flow
        
    def _analyze_vector_field(self) -> VectorFieldState:
        """Analyze vector field topology to detect pattern collapse."""
        if not self.history:
            return VectorFieldState(0.0, 0.0, 0.0, 0.0)
            
        # Calculate field characteristics
        current = self.history[-1]
        previous = self.history[-2] if len(self.history) > 1 else current
        
        # Calculate vector components
        dx = current['confidence'] - previous['confidence']
        dy = current['temporal_stability'] - previous['temporal_stability']
        
        # Calculate field properties
        magnitude = (dx**2 + dy**2)**0.5
        direction = math.atan2(dy, dx)
        
        # Calculate field derivatives
        if len(self.history) > 2:
            ddx = dx - (previous['confidence'] - self.history[-3]['confidence'])
            ddy = dy - (previous['temporal_stability'] - self.history[-3]['temporal_stability'])
            divergence = ddx + ddy
            curl = ddy - ddx
        else:
            divergence = 0.0
            curl = 0.0
            
        # Identify critical points
        critical_points = []
        if magnitude < self.field_resolution:
            critical_type = 'attractor' if divergence < 0 else 'source'
            critical_points.append({
                'type': critical_type,
                'position': (current['confidence'], current['temporal_stability']),
                'strength': abs(divergence)
            })
            
        return VectorFieldState(
            magnitude=magnitude,
            direction=direction,
            divergence=divergence,
            curl=curl,
            critical_points=critical_points
        )
        
    def detect_pattern_collapse(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """Detect if a pattern is collapsing using vector field topology."""
        flow = self.active_flows.get(flow_id)
        if not flow or len(flow.history) < 3:
            return None
            
        # Analyze field topology
        field_state = self._analyze_vector_field()
        
        # Check for collapse conditions
        is_collapsing = (
            field_state.divergence > self.collapse_threshold or
            (field_state.magnitude > 0.5 and field_state.curl > 0.7) or
            any(p['type'] == 'source' and p['strength'] > 0.8 
                for p in field_state.critical_points)
        )
        
        if is_collapsing:
            return {
                'flow_id': flow_id,
                'collapse_type': 'topology_based',
                'severity': min(1.0, field_state.divergence / self.collapse_threshold),
                'field_state': field_state,
                'recovery_chance': 1.0 - (field_state.magnitude / 2)
            }
            
        return None
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
