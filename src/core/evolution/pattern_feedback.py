"""Pattern feedback mechanisms for evolutionary learning.

This module implements feedback mechanisms that are cognizant of how
patterns evolve out of interfaces. It provides both immediate feedback
loops and learning windows for pattern evolution analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import deque

@dataclass
class LearningWindow:
    """Sliding window for pattern evolution learning."""
    window_size: timedelta
    observations: deque = field(default_factory=lambda: deque(maxlen=1000))
    interface_patterns: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    
    def add_observation(self, 
                       pattern_id: str,
                       interface_id: str,
                       state: Dict[str, Any],
                       metrics: Dict[str, float],
                       timestamp: datetime) -> None:
        """Add an observation to the learning window."""
        observation = {
            'pattern_id': pattern_id,
            'interface_id': interface_id,
            'state': state,
            'metrics': metrics,
            'timestamp': timestamp
        }
        self.observations.append(observation)
        
        # Track by interface
        if interface_id not in self.interface_patterns:
            self.interface_patterns[interface_id] = []
        self.interface_patterns[interface_id].append(observation)
        
        # Cleanup old observations
        cutoff = datetime.utcnow() - self.window_size
        while (self.observations and 
               self.observations[0]['timestamp'] < cutoff):
            self.observations.popleft()
    
    def get_interface_metrics(self, interface_id: str) -> Dict[str, float]:
        """Calculate metrics for patterns evolved from an interface."""
        patterns = self.interface_patterns.get(interface_id, [])
        if not patterns:
            return {}
            
        metrics = defaultdict(list)
        for p in patterns:
            for k, v in p['metrics'].items():
                metrics[k].append(v)
                
        return {
            k: np.mean(v) for k, v in metrics.items()
        }

@dataclass
class PatternFeedback:
    """Feedback mechanism for pattern evolution."""
    learning_window: LearningWindow
    feedback_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'energy_variance': 0.3,
        'stability_threshold': 0.7,
        'emergence_sensitivity': 0.5
    })
    
    def process_pattern_state(self,
                            pattern_id: str,
                            interface_id: str,
                            current_state: Dict[str, Any],
                            previous_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process pattern state and generate feedback."""
        timestamp = datetime.utcnow()
        
        # Calculate state transition metrics
        metrics = self._calculate_transition_metrics(
            current_state, previous_state
        )
        
        # Add to learning window
        self.learning_window.add_observation(
            pattern_id, interface_id, current_state, metrics, timestamp
        )
        
        # Generate feedback based on learning window
        feedback = self._generate_feedback(
            pattern_id, interface_id, current_state, metrics
        )
        
        return feedback
    
    def _calculate_transition_metrics(self,
                                   current: Dict[str, Any],
                                   previous: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Calculate metrics for state transition."""
        if not previous:
            return {
                'energy': current.get('energy', 0.0),
                'stability': current.get('stability', 0.0),
                'emergence_phase': current.get('emergence_phase', 0.0)
            }
            
        return {
            'energy_delta': current.get('energy', 0.0) - previous.get('energy', 0.0),
            'stability_delta': current.get('stability', 0.0) - previous.get('stability', 0.0),
            'emergence_delta': current.get('emergence_phase', 0.0) - previous.get('emergence_phase', 0.0)
        }
    
    def _generate_feedback(self,
                         pattern_id: str,
                         interface_id: str,
                         state: Dict[str, Any],
                         metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate feedback based on learning window analysis."""
        interface_metrics = self.learning_window.get_interface_metrics(interface_id)
        
        feedback = {
            'pattern_id': pattern_id,
            'interface_id': interface_id,
            'timestamp': datetime.utcnow().isoformat(),
            'adjustments': {}
        }
        
        # Check energy variance
        if metrics.get('energy_delta', 0) > self.feedback_thresholds['energy_variance']:
            feedback['adjustments']['energy'] = {
                'action': 'stabilize',
                'reason': 'High energy variance detected',
                'suggested_delta': -0.1
            }
        
        # Check stability
        if state.get('stability', 0) < self.feedback_thresholds['stability_threshold']:
            feedback['adjustments']['emergence_sensitivity'] = {
                'action': 'increase',
                'reason': 'Low stability detected',
                'suggested_delta': 0.1
            }
        
        # Compare with interface patterns
        if interface_metrics:
            feedback['interface_context'] = {
                'avg_metrics': interface_metrics,
                'pattern_count': len(self.learning_window.interface_patterns.get(interface_id, []))
            }
        
        return feedback
