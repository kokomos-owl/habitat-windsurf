"""
HabitatFlow: Core flow management for pattern evolution and coherence.

This module facilitates the natural emergence and evolution of patterns,
fostering coherence while remaining open to new forms and behaviors.
It builds upon evolutionary insights from previous iterations while
allowing patterns to form and stabilize naturally.

Key Components:
    - FlowState: Represents the current state of pattern flow
    - FlowType: Defines types of flows in the system
    - HabitatFlow: Main flow management class
    
Evolutionary Principles:
    - Patterns emerge and evolve naturally rather than being enforced
    - System sensitivity adapts to emerging patterns
    - Coherence maintained through flexible, dynamic relationships
    - Previous insights inform but don't constrain new evolution
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from datetime import datetime
import logging
import json

from ..pattern_evolution import PatternEvolutionTracker
from .flow_types import FlowDynamics, FlowState, FlowType, FlowMetrics, ProcessingContext

logger = logging.getLogger(__name__)

@dataclass
class FlowResult:
    """Result of flow processing with evolution context."""
    flow_id: str
    state: FlowState
    metrics: FlowMetrics
    pattern_state: Optional[Dict[str, Any]] = None
    evolution_context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'flow_id': self.flow_id,
            'state': self.state.to_dict(),
            'metrics': self.metrics.to_dict(),
            'pattern_state': self.pattern_state,
            'evolution_context': self.evolution_context,
            'is_valid': self.state.is_valid()
        }

@dataclass
class FlowConfiguration:
    """Configuration for flow processing."""
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        'emergence': 0.3,
        'coherence': 0.3,
        'stability': 0.3
    })
    max_history: int = 1000
    enable_feedback: bool = True  # For future pattern feedback

class HabitatFlow:
    """Core flow management for pattern evolution and emergence.
    
    This class facilitates the natural evolution of patterns and their flows,
    building on insights from previous iterations while remaining open to
    new emergent behaviors. It provides a flexible foundation that allows
    patterns to form, evolve, and stabilize naturally.
    
    Key aspects:
    - Pattern Evolution: Tracks and facilitates pattern growth
    - Flow Dynamics: Allows natural movement and emergence
    - Coherence: Maintains system stability while allowing change
    - Emergence: Recognizes and nurtures new pattern formation
    """
    
    def __init__(self, config: Optional[FlowConfiguration] = None):
        """Initialize the evolutionary flow system.
        
        Creates an environment where patterns can naturally emerge and evolve,
        while maintaining enough structure to ensure coherence.
        
        Args:
            config: Optional flow configuration for tuning emergence sensitivity
        """
        self.config = config or FlowConfiguration()
        self.pattern_evolution = PatternEvolutionTracker()
        
        # Core evolutionary state
        self.active_flows: Dict[str, FlowState] = {}
        self.flow_history: List[Dict[str, Any]] = []
        self.processing_contexts: Dict[str, ProcessingContext] = {}
        
        # Dynamic pattern tracking
        self.dynamics_tracker = FlowDynamics()
        
        # Initialize emergence sensitivity
        self._init_emergence_sensitivity()
        
        logger.info("Initialized evolutionary flow system with sensitivity: %s", 
                   json.dumps(self.config.thresholds))
                   
    def _init_emergence_sensitivity(self):
        """Initialize system sensitivity to emerging patterns.
        
        This affects how readily the system recognizes and adapts to
        new pattern formation.
        """
        # Start with moderate sensitivity that can self-adjust
        self.emergence_sensitivity = 0.5  # Mid-point between stability and change

    async def process_flow(self, content: Dict[str, Any]) -> FlowResult:
        """Process content through the evolutionary flow system.
        
        This method allows patterns to naturally emerge and evolve while
        maintaining coherence. It doesn't force specific flows but rather
        facilitates natural pattern formation and evolution.
        
        Args:
            content: Input content to process
            
        Returns:
            FlowResult containing evolutionary state and emergence metrics
        """
        try:
            # Initialize flow tracking
            flow_id = self._generate_flow_id()
            flow_state = FlowState()
            flow_metrics = FlowMetrics()
            
            # Create processing context
            context = ProcessingContext(
                flow_id=flow_id,
                flow_type=FlowType.STRUCTURAL,  # Default to structural
                metadata=content.get('metadata', {})
            )
            self.processing_contexts[flow_id] = context
            
            # Extract and process pattern
            pattern = content.get('text', '')
            if pattern:
                # Observe pattern evolution
                evolution_result = self.pattern_evolution.observe_pattern(
                    pattern=pattern,
                    confidence=0.8,  # Initial confidence
                    temporal_context=context.metadata
                )
                
                # Update flow dynamics
                self._update_flow_dynamics(evolution_result)
                
                # Get evolved patterns
                evolved_patterns = self.pattern_evolution.get_evolved_patterns(
                    min_confidence=self.config.thresholds['coherence'],
                    min_flow=self.config.thresholds['emergence']
                )
                
                if evolved_patterns:
                    # Update state and metrics
                    pattern_state = evolved_patterns[0]
                    flow_state.dynamics = self.dynamics_tracker
                    flow_metrics = self._calculate_flow_metrics(pattern_state)
            
            # Create and track result
            result = FlowResult(
                flow_id=flow_id,
                state=flow_state,
                metrics=flow_metrics,
                pattern_state=evolved_patterns[0] if evolved_patterns else None,
                evolution_context=context.metadata
            )
            
            # Track flow state
            self.active_flows[flow_id] = flow_state
            self._record_flow_history(flow_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing flow: {str(e)}")
            raise

    def _update_flow_dynamics(self, evolution_result: Dict[str, Any]) -> None:
        """Update flow dynamics based on pattern evolution.
        
        Rather than forcing specific dynamics, this method observes and
        facilitates natural pattern movement and evolution.
        """
        # Extract evolutionary indicators
        velocity = evolution_result.get('velocity', 0.0)
        direction = evolution_result.get('direction', 0.0)
        energy = evolution_result.get('energy', 0.0)
        confidence = evolution_result.get('confidence', 0.0)
        
        # Adjust emergence sensitivity based on pattern stability
        self._adjust_emergence_sensitivity(energy, confidence)
        
        # Update dynamics tracker with evolutionary state
        self.dynamics_tracker.velocity = velocity
        self.dynamics_tracker.direction = direction
        self.dynamics_tracker.energy = energy
        self.dynamics_tracker.propensity = confidence
        
    def _adjust_emergence_sensitivity(self, energy: float, confidence: float):
        """Dynamically adjust sensitivity to emerging patterns.
        
        Higher energy with lower confidence suggests new patterns forming,
        which may require increased sensitivity.
        """
        emergence_indicator = energy * (1 - confidence)
        if emergence_indicator > 0.7:
            # High energy, low confidence suggests new patterns
            self.emergence_sensitivity = min(0.8, self.emergence_sensitivity + 0.1)
        elif confidence > 0.8:
            # High confidence suggests stable patterns
            self.emergence_sensitivity = max(0.3, self.emergence_sensitivity - 0.05)
        
    def _calculate_flow_metrics(self, pattern_state: Dict[str, Any]) -> FlowMetrics:
        """Calculate flow metrics from pattern state."""
        # Get relationship context
        relationships = self.pattern_evolution.get_pattern_relationships(
            pattern_state['pattern'],
            min_strength=self.config.thresholds['coherence']
        )
        
        # Calculate metrics
        metrics = FlowMetrics()
        
        if relationships:
            # Coherence from relationship strength
            avg_strength = sum(strength for _, strength in relationships) / len(relationships)
            metrics.coherence = 0.7 * avg_strength + 0.3 * pattern_state['confidence']
            
            # Stability from confidence consistency
            metrics.stability = min(1.0, pattern_state['confidence'] * 0.8 + 0.2)
            
            # Emergence rate from dynamics
            metrics.emergence_rate = self.dynamics_tracker.emergence_readiness
            
            # Cross-flow from relationship network
            metrics.cross_flow = avg_strength * self.dynamics_tracker.energy
            
        return metrics

    def get_flow_state(self, flow_id: str) -> Optional[FlowState]:
        """Get current state of a flow."""
        return self.active_flows.get(flow_id)
        
    def get_flow_metrics(self, flow_id: str) -> Optional[FlowMetrics]:
        """Get current metrics for a flow."""
        context = self.processing_contexts.get(flow_id)
        return context.metrics if context else None
        
    def get_flow_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent flow history."""
        return self.flow_history[-limit:] if self.flow_history else []

    def _generate_flow_id(self) -> str:
        """Generate unique flow identifier."""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"flow_{timestamp}"

    def _record_flow_history(self, flow_id: str, result: FlowResult) -> None:
        """Record flow result in history."""
        self.flow_history.append({
            'flow_id': flow_id,
            'timestamp': datetime.now().isoformat(),
            'result': result.to_dict()
        })
        
        # Maintain history size
        if len(self.flow_history) > self.config.max_history:
            self.flow_history = self.flow_history[-self.config.max_history:]
