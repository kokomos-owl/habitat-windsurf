"""Connector between flow metrics and topology visualization."""

from typing import Dict, List, Optional
import json
from dataclasses import asdict
from src.core.metrics.flow_metrics import MetricFlowManager, VectorFieldState

class TopologyConnector:
    """Connects flow metrics to topology visualization."""
    
    def __init__(self, flow_manager: MetricFlowManager):
        self.flow_manager = flow_manager
        
    def get_visualization_state(self) -> Dict:
        """Get current state for visualization."""
        field_state = self.flow_manager._analyze_vector_field()
        
        return {
            'vector_field': self._prepare_field_data(field_state),
            'critical_points': self._prepare_critical_points(field_state),
            'collapse_warning': self._check_collapse_warning(field_state)
        }
    
    def _prepare_field_data(self, state: VectorFieldState) -> Dict:
        """Prepare vector field data for visualization."""
        return {
            'magnitude': state.magnitude,
            'direction': state.direction,
            'divergence': state.divergence,
            'curl': state.curl
        }
    
    def _prepare_critical_points(self, state: VectorFieldState) -> List[Dict]:
        """Prepare critical points data for visualization."""
        return [{
            'type': point['type'],
            'position': point['position'],
            'strength': point['strength'],
            'field_state': asdict(state)
        } for point in state.critical_points]
    
    def _check_collapse_warning(self, state: VectorFieldState) -> Optional[Dict]:
        """Check if collapse warning should be shown."""
        if state.divergence > self.flow_manager.collapse_threshold:
            return {
                'severity': min(1.0, state.divergence / self.flow_manager.collapse_threshold),
                'type': 'topology_based',
                'recovery_chance': 1.0 - (state.magnitude / 2),
                'field_state': asdict(state)
            }
        return None
    
    def to_json(self) -> str:
        """Convert current state to JSON for frontend."""
        return json.dumps(self.get_visualization_state())
