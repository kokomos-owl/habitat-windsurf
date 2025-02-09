"""Coherence and emergence metrics calculation and visualization."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from enum import Enum

class InsightType(Enum):
    """Types of system insights for coherence and emergence."""
    PATTERN_EMERGENCE = "pattern_emergence"
    COHERENCE_SHIFT = "coherence_shift"
    STABILITY_THREAT = "stability_threat"
    EVOLUTION_OPPORTUNITY = "evolution_opportunity"

@dataclass
class SystemInsight:
    """Represents a system-generated insight about coherence or emergence."""
    insight_type: InsightType
    confidence: float
    timestamp: datetime
    context: Dict[str, Any]
    affected_patterns: List[str]
    recommendation: Optional[str] = None

class CoherenceMetrics:
    """Calculates and tracks system coherence and emergence metrics."""
    
    def __init__(self):
        self.coherence_window = timedelta(days=7)
        self.emergence_threshold = 0.15
        self.coherence_threshold = 0.75
        self.insight_history: List[SystemInsight] = []
        
    def calculate_metrics(self, 
                         flow_states: Dict[str, Any],
                         pattern_stats: Dict[str, Any],
                         temporal_context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate current coherence and emergence metrics."""
        
        # Calculate pattern coherence
        pattern_coherence = self._calculate_pattern_coherence(pattern_stats)
        
        # Calculate temporal coherence
        temporal_coherence = self._calculate_temporal_coherence(temporal_context)
        
        # Calculate emergence indicators
        emergence_indicators = self._calculate_emergence_indicators(flow_states)
        
        # Generate system insights
        insights = self._generate_insights(
            pattern_coherence,
            temporal_coherence,
            emergence_indicators
        )
        
        # Calculate overall system health
        system_health = self._calculate_system_health(
            pattern_coherence,
            temporal_coherence,
            emergence_indicators
        )
        
        return {
            'system_health': system_health,
            'pattern_coherence': pattern_coherence,
            'temporal_coherence': temporal_coherence,
            'emergence_indicators': emergence_indicators,
            'insights': insights,
            'visualization_data': self._prepare_visualization_data(
                pattern_coherence,
                temporal_coherence,
                emergence_indicators
            )
        }
    
    def _calculate_pattern_coherence(self, pattern_stats: Dict[str, Any]) -> Dict[str, float]:
        """Calculate coherence metrics for patterns."""
        coherence_metrics = {}
        
        for pattern_type, stats in pattern_stats.items():
            # Base coherence on pattern stability and success rate
            stability = stats.get('stability', 0.0)
            success_rate = stats.get('success_rate', 0.0)
            variant_count = len(stats.get('variants', []))
            
            # More variants might indicate lower coherence
            variant_factor = 1.0 / (1.0 + (variant_count / 5.0))
            
            coherence = (stability * 0.4 + success_rate * 0.4 + variant_factor * 0.2)
            coherence_metrics[pattern_type] = coherence
            
        return coherence_metrics
    
    def _calculate_temporal_coherence(self, temporal_context: Dict[str, Any]) -> float:
        """Calculate coherence in temporal relationships."""
        if not temporal_context:
            return 1.0
            
        # Check temporal consistency
        temporal_markers = temporal_context.get('markers', [])
        if not temporal_markers:
            return 1.0
            
        # Calculate temporal distance variance
        distances = []
        for i in range(1, len(temporal_markers)):
            try:
                current = datetime.strptime(temporal_markers[i]['time'], '%Y-%m-%d')
                previous = datetime.strptime(temporal_markers[i-1]['time'], '%Y-%m-%d')
                distances.append((current - previous).days)
            except (ValueError, KeyError):
                continue
                
        if not distances:
            return 1.0
            
        # Lower variance indicates higher coherence
        variance = np.var(distances) if len(distances) > 1 else 0
        return 1.0 / (1.0 + (variance / 365.0))  # Normalize by year
    
    def _calculate_emergence_indicators(self, flow_states: Dict[str, Any]) -> Dict[str, float]:
        """Calculate emergence indicators from flow states."""
        indicators = {
            'pattern_emergence': 0.0,
            'state_flux': 0.0,
            'evolution_pressure': 0.0
        }
        
        # Count patterns in each state
        state_counts = {}
        for flow_id, state in flow_states.items():
            state_counts[state] = state_counts.get(state, 0) + 1
            
        total_flows = sum(state_counts.values())
        if total_flows == 0:
            return indicators
            
        # Calculate emergence indicators
        indicators['pattern_emergence'] = state_counts.get('emerging', 0) / total_flows
        indicators['state_flux'] = len([s for s in flow_states.values() if s in ['learning', 'emerging']]) / total_flows
        indicators['evolution_pressure'] = 1.0 - (state_counts.get('stable', 0) / total_flows)
        
        return indicators
    
    def _generate_insights(self,
                         pattern_coherence: Dict[str, float],
                         temporal_coherence: float,
                         emergence_indicators: Dict[str, float]) -> List[SystemInsight]:
        """Generate system insights based on metrics."""
        insights = []
        timestamp = datetime.now()
        
        # Check for emerging patterns
        if emergence_indicators['pattern_emergence'] > self.emergence_threshold:
            insights.append(SystemInsight(
                insight_type=InsightType.PATTERN_EMERGENCE,
                confidence=emergence_indicators['pattern_emergence'],
                timestamp=timestamp,
                context={'emergence_level': emergence_indicators['pattern_emergence']},
                affected_patterns=[p for p, c in pattern_coherence.items() if c < self.coherence_threshold],
                recommendation="Monitor emerging patterns for potential evolution"
            ))
        
        # Check for coherence shifts
        low_coherence_patterns = [p for p, c in pattern_coherence.items() if c < self.coherence_threshold]
        if low_coherence_patterns:
            insights.append(SystemInsight(
                insight_type=InsightType.COHERENCE_SHIFT,
                confidence=1.0 - min(pattern_coherence.values()),
                timestamp=timestamp,
                context={'low_coherence_patterns': low_coherence_patterns},
                affected_patterns=low_coherence_patterns,
                recommendation="Review and potentially adjust affected patterns"
            ))
        
        return insights
    
    def _calculate_system_health(self,
                               pattern_coherence: Dict[str, float],
                               temporal_coherence: float,
                               emergence_indicators: Dict[str, float]) -> float:
        """Calculate overall system health score."""
        if not pattern_coherence:
            return 1.0
            
        # Weights for different components
        weights = {
            'pattern_coherence': 0.4,
            'temporal_coherence': 0.3,
            'emergence_balance': 0.3
        }
        
        # Calculate average pattern coherence
        avg_pattern_coherence = sum(pattern_coherence.values()) / len(pattern_coherence)
        
        # Calculate emergence balance (some emergence is good, too much isn't)
        emergence_balance = 1.0 - abs(0.2 - emergence_indicators['pattern_emergence'])
        
        return (
            avg_pattern_coherence * weights['pattern_coherence'] +
            temporal_coherence * weights['temporal_coherence'] +
            emergence_balance * weights['emergence_balance']
        )
    
    def _prepare_visualization_data(self,
                                  pattern_coherence: Dict[str, float],
                                  temporal_coherence: float,
                                  emergence_indicators: Dict[str, float]) -> Dict[str, Any]:
        """Prepare data for both system and user visualizations."""
        return {
            'system_view': {
                'coherence_matrix': [
                    {
                        'pattern': pattern,
                        'coherence': coherence,
                        'emergence_potential': emergence_indicators['pattern_emergence'],
                        'stability_index': coherence * (1.0 - emergence_indicators['state_flux'])
                    }
                    for pattern, coherence in pattern_coherence.items()
                ],
                'temporal_stability': temporal_coherence,
                'emergence_vectors': emergence_indicators
            },
            'user_view': {
                'health_indicators': {
                    'system_stability': self._calculate_system_health(
                        pattern_coherence,
                        temporal_coherence,
                        emergence_indicators
                    ),
                    'pattern_health': pattern_coherence,
                    'temporal_alignment': temporal_coherence
                },
                'emerging_patterns': [
                    {
                        'pattern': pattern,
                        'confidence': coherence,
                        'status': 'emerging' if coherence < self.coherence_threshold else 'stable'
                    }
                    for pattern, coherence in pattern_coherence.items()
                ],
                'system_insights': [
                    {
                        'type': insight.insight_type.value,
                        'message': insight.recommendation,
                        'confidence': insight.confidence
                    }
                    for insight in self._generate_insights(
                        pattern_coherence,
                        temporal_coherence,
                        emergence_indicators
                    )
                ]
            }
        }
