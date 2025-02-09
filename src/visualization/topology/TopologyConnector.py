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
        
        # Get active flows and patterns
        flows = self.flow_manager.active_flows
        patterns = self.flow_manager.pattern_flows
        
        # Create nodes for each pattern and flow
        nodes = []
        links = []
        
        # Add pattern nodes
        for pattern, flow_ids in patterns.items():
            stage = 'stage1'
            if 'precipitation' in pattern.lower():
                stage = 'stage1'
            elif 'drought' in pattern.lower():
                stage = 'stage2'
            elif 'wildfire' in pattern.lower():
                stage = 'stage3'
                
            nodes.append({
                'id': f'pattern_{pattern}',
                'label': pattern,
                'type': 'pattern',
                'stage': stage
            })
            
            # Add flow nodes and links
            for flow_id in flow_ids:
                flow = flows[flow_id]
                nodes.append({
                    'id': flow_id,
                    'label': f'Flow {flow_id}',
                    'type': 'flow',
                    'stage': stage,
                    'metrics': {
                        'stability': getattr(flow, 'stability', 0.5),
                        'coherence': getattr(flow, 'coherence', 0.5),
                        'energy_state': getattr(flow, 'energy_state', 0.5)
                    }
                })
                
                links.append({
                    'source': f'pattern_{pattern}',
                    'target': flow_id,
                    'weight': getattr(flow, 'confidence', 0.5),
                    'stage': stage
                })
        
        # Calculate average metrics across all flows
        total_flows = len(flows)
        if total_flows > 0:
            avg_stability = sum(getattr(flow, 'stability', 0.5) for flow in flows.values()) / total_flows
            avg_coherence = sum(getattr(flow, 'coherence', 0.5) for flow in flows.values()) / total_flows
            avg_energy = sum(getattr(flow, 'energy_state', 0.5) for flow in flows.values()) / total_flows
        else:
            avg_stability = avg_coherence = avg_energy = 0.5
            
        return {
            'nodes': nodes,
            'links': links,
            'metadata': {
                'stability': avg_stability,
                'coherence': avg_coherence,
                'energy_state': avg_energy,
                'total_patterns': len(patterns),
                'total_flows': total_flows
            }
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
