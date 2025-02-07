"""
HabitatFlow: Core flow management for pattern evolution and coherence.

This module manages the natural flow of patterns and their evolution,
ensuring coherence is maintained throughout the system. It integrates
with existing pattern evolution and processing components.

Key Components:
    - FlowState: Represents the current state of pattern flow
    - FlowType: Defines types of flows in the system
    - HabitatFlow: Main flow management class
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime
import logging

from ..pattern_evolution import PatternEvolutionTracker

logger = logging.getLogger(__name__)

@dataclass
class FlowState:
    """Represents the current state of a pattern flow."""
    strength: float = 0.0
    coherence: float = 0.0
    emergence_potential: float = 0.0
    temporal_context: Optional[Dict[str, Any]] = None
    last_updated: datetime = datetime.now()

    def is_valid(self) -> bool:
        """Validate flow state meets minimum thresholds."""
        return all([
            self.strength >= 0.3,
            self.coherence >= 0.3,
            self.emergence_potential >= 0.3
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            'strength': self.strength,
            'coherence': self.coherence,
            'emergence_potential': self.emergence_potential,
            'temporal_context': self.temporal_context,
            'last_updated': self.last_updated.isoformat()
        }

class FlowType(Enum):
    """Defines types of flows in the system."""
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    EMERGENT = "emergent"

class HabitatFlow:
    """Core flow management for pattern evolution."""
    
    def __init__(self):
        """Initialize flow management system."""
        self.pattern_evolution = PatternEvolutionTracker()
        self.active_flows: Dict[str, FlowState] = {}
        self.flow_history: List[Dict[str, Any]] = []
        
        # Flow configuration
        self.flow_thresholds = {
            'emergence': 0.3,
            'coherence': 0.3,
            'stability': 0.3
        }
        
        logger.info("Initialized HabitatFlow manager")

    async def process_flow(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process content through the flow system.
        
        Args:
            content: Input content to process
            
        Returns:
            Dict containing flow results and states
        """
        try:
            # Initialize flow state
            flow_id = self._generate_flow_id()
            flow_state = FlowState()
            
            # Extract pattern from content
            pattern = content.get('text', '')
            temporal_context = content.get('metadata', {})
            
            # Process through pattern evolution
            self.pattern_evolution.observe_pattern(
                pattern=pattern,
                confidence=0.8,  # Initial confidence, will be refined
                temporal_context=temporal_context
            )
            
            # Get evolved patterns to assess flow
            evolved_patterns = self.pattern_evolution.get_evolved_patterns(
                min_confidence=0.5,
                min_flow=0.3
            )
            
            # Update flow state based on pattern evolution
            if evolved_patterns:
                pattern_state = evolved_patterns[0]  # Most significant evolution
                flow_state.strength = pattern_state.confidence
                flow_state.coherence = self._calculate_coherence(pattern_state)
                flow_state.emergence_potential = self._calculate_emergence({
                    'flow_velocity': pattern_state.flow_velocity,
                    'flow_direction': pattern_state.flow_direction,
                    'confidence': pattern_state.confidence
                })
            
            # Track flow state
            self.active_flows[flow_id] = flow_state
            self._record_flow_history(flow_id, flow_state)
            
            return {
                'flow_id': flow_id,
                'state': flow_state.to_dict(),
                'pattern_state': evolved_patterns[0] if evolved_patterns else None,
                'is_valid': flow_state.is_valid()
            }
            
        except Exception as e:
            logger.error(f"Error processing flow: {str(e)}")
            raise

    def _calculate_coherence(self, pattern_state: 'PatternState') -> float:
        """Calculate coherence from pattern state."""
        # Get relationship strengths
        relationships = self.pattern_evolution.get_pattern_relationships(
            pattern_state.pattern,
            min_strength=0.3
        )
        
        if not relationships:
            return 0.0
            
        # Calculate average relationship strength
        avg_strength = sum(strength for _, strength in relationships) / len(relationships)
        
        # Weight relationship strength with confidence
        return (0.7 * avg_strength + 0.3 * pattern_state.confidence)

    def _calculate_emergence(self, pattern_result: Dict[str, Any]) -> float:
        """Calculate emergence potential from pattern results."""
        velocity = pattern_result.get('flow_velocity', 0.0)
        direction = pattern_result.get('flow_direction', 0.0)
        confidence = pattern_result.get('confidence', 0.0)
        
        # Normalize direction to [0,1] range
        norm_direction = (direction % (2 * 3.14159)) / (2 * 3.14159)
        
        # Weight the factors
        return (0.4 * velocity + 
                0.3 * norm_direction + 
                0.3 * confidence)

    def _generate_flow_id(self) -> str:
        """Generate unique flow identifier."""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"flow_{timestamp}"

    def _record_flow_history(self, flow_id: str, state: FlowState):
        """Record flow state in history."""
        self.flow_history.append({
            'flow_id': flow_id,
            'timestamp': datetime.now().isoformat(),
            'state': state.to_dict()
        })
        
        # Maintain history size
        if len(self.flow_history) > 1000:
            self.flow_history = self.flow_history[-1000:]
