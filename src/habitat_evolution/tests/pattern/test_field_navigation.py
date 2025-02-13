"""Test field navigation through climate concept space."""

import pytest
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from habitat_evolution.core.pattern.attention import (
    AttentionSet, create_climate_attention_set, NeighborContext
)

from habitat_evolution.core.services.field.interfaces import (
    FieldStateService, GradientService, FlowDynamicsService,
    FieldState, GradientVector
)
from habitat_evolution.core.pattern.quality import PatternQualityAnalyzer
from habitat_evolution.core.pattern.observation import PatternObserver

logger = logging.getLogger(__name__)

class FieldNavigationObserver:
    """Observes and logs field navigation through multiple modalities."""
    
    def __init__(self, field_service: FieldStateService, 
                 gradient_service: GradientService,
                 flow_service: FlowDynamicsService,
                 attention_set: Optional[AttentionSet] = None):
        self.field_service = field_service
        self.gradient_service = gradient_service
        self.flow_service = flow_service
        self.pattern_observer = PatternObserver()
        self.quality_analyzer = PatternQualityAnalyzer()
        self.attention_set = attention_set or create_climate_attention_set()
        
    async def _get_neighbor_context(self, field_id: str, position: Dict[str, float]) -> NeighborContext:
        """Get observations from neighboring positions."""
        neighbors = {}
        distances = {}
        gradients = {}
        
        # Check 8 surrounding positions
        deltas = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dx, dy in deltas:
            neighbor_pos = {
                'x': position['x'] + dx,
                'y': position['y'] + dy
            }
            # Get base observations without neighbor context to avoid recursion
            neighbors[f"{dx},{dy}"] = await self._observe_base(field_id, neighbor_pos)
            distances[f"{dx},{dy}"] = np.sqrt(dx*dx + dy*dy)
            gradients[f"{dx},{dy}"] = await self.gradient_service.calculate_gradient(
                field_id, neighbor_pos
            )
        
        return NeighborContext(
            position=position,
            neighbors=neighbors,
            distances=distances,
            gradients=gradients
        )
    
    async def _observe_base(self, field_id: str, position: Dict[str, float]) -> Dict:
        """Base observation without neighbor context."""
        observations = {}
        
        # Wave mechanics observation
        field_state = await self.field_service.get_field_state(field_id)
        phase = self.pattern_observer.calculate_phase_relationship(
            field_state.potential, position
        )
        observations['wave'] = {
            'phase': phase,
            'coherence': field_state.stability,
            'potential': field_state.potential
        }
        
        # Field theory observation
        gradient = await self.gradient_service.calculate_gradient(field_id, position)
        observations['field'] = {
            'gradient_magnitude': gradient.magnitude,
            'gradient_direction': gradient.direction,
            'stability': gradient.stability
        }
        
        # Flow dynamics observation
        viscosity = await self.flow_service.calculate_viscosity(field_id, position)
        turbulence = await self.flow_service.calculate_turbulence(field_id, position)
        observations['flow'] = {
            'viscosity': viscosity,
            'turbulence': turbulence
        }
        
        return observations
    
    async def observe_position(self, field_id: str, position: Dict[str, float]) -> Dict:
        """Observe a position through all modalities."""
        # Get base observations
        observations = await self._observe_base(field_id, position)
        
        # Get neighbor context
        neighbor_context = await self._get_neighbor_context(field_id, position)
        
        # Pattern emergence detection with neighbor awareness
        pattern_potential = self._calculate_pattern_potential(
            observations['wave']['potential'],
            observations['field']['gradient_magnitude'],
            observations['field']['stability'],
            observations['flow']['viscosity'],
            observations['flow']['turbulence']
        )
        
        emergence_type = self._detect_emergence(pattern_potential, observations)
        observations['pattern'] = {
            'potential': pattern_potential,
            'emergence_type': emergence_type
        }
        
        # Evaluate attention filters with neighbor context
        attention_results = self.attention_set.evaluate(observations, neighbor_context)
        observations['attention'] = attention_results
        
        # Log any strong attention signals as facts
        for filter_name, score in attention_results.items():
            if score >= 0.8:  # Strong attention signal
                logger.info(f"[ATTENTION FACT] High {filter_name} detected (score: {score:.2f})")
                
        # Log neighbor-based observations
        coherent_neighbors = sum(1 for n in neighbor_context.neighbors.values() 
                               if n['wave']['coherence'] >= 0.7)
        if coherent_neighbors >= 4:  # More than half neighbors are coherent
            logger.info(f"[NEIGHBOR FACT] Strong local coherence detected with {coherent_neighbors} coherent neighbors")
            
        # Check for gradient alignment
        aligned_gradients = sum(1 for g in neighbor_context.gradients.values()
                              if g.magnitude >= 0.7)
        if aligned_gradients >= 4:
            logger.info(f"[NEIGHBOR FACT] Strong gradient alignment detected with {aligned_gradients} aligned neighbors")
        
        return observations
        
    def _calculate_pattern_potential(self, field_potential: float,
                                    gradient_magnitude: float,
                                    stability: float,
                                    viscosity: float,
                                    turbulence: float) -> float:
        """Calculate pattern potential from field conditions."""
        # Higher potential when:
        # - Field potential is high
        # - Strong gradients exist
        # - System is stable
        # - Viscosity allows flow
        # - Turbulence is moderate (some mixing but not chaotic)
        base_potential = field_potential * gradient_magnitude * stability
        flow_factor = viscosity * (1.0 - (turbulence * 0.7))  # Reduce effect of high turbulence
        return base_potential * flow_factor
    
    def _detect_emergence(self, pattern_potential: float, observations: Dict) -> str:
        """Detect what type of pattern is emerging based on field conditions."""
        # Thresholds for emergence detection
        NODE_THRESHOLD = 0.7
        EDGE_THRESHOLD = 0.6
        
        coherence = observations['wave']['coherence']
        stability = observations['field']['stability']
        
        if pattern_potential >= NODE_THRESHOLD and coherence >= 0.8:
            return "NODE_EMERGENCE"
        elif pattern_potential >= EDGE_THRESHOLD and stability >= 0.6:
            return "EDGE_EMERGENCE"
        else:
            return "NO_EMERGENCE"

    def log_observations(self, position_name: str, observations: Dict):
        """Log observations in a structured way."""
        logger.info(f"\n=== Observations at {position_name} ===")
        
        logger.info("Wave Mechanics:")
        logger.info(f"  Phase: {observations['wave']['phase']:.2f}")
        logger.info(f"  Coherence: {observations['wave']['coherence']:.2f}")
        
        logger.info("Field Theory:")
        logger.info(f"  Gradient Magnitude: {observations['field']['gradient_magnitude']:.2f}")
        logger.info(f"  Gradient Direction: {observations['field']['gradient_direction']}")
        logger.info(f"  Field Stability: {observations['field']['stability']:.2f}")
        
        logger.info("Flow Dynamics:")
        logger.info(f"  Viscosity: {observations['flow']['viscosity']:.2f}")
        logger.info(f"  Turbulence: {observations['flow']['turbulence']:.2f}")

@pytest.mark.asyncio
async def test_climate_field_navigation():
    """Test navigation through climate concept fields."""
    
    # Setup test data from Martha's Vineyard climate assessment
    climate_hazards = {
        'extreme_precipitation': {
            'position': {'x': 0, 'y': 0},
            'strength': 7.34  # Historical 100-year rainfall amount
        },
        'extreme_drought': {
            'position': {'x': 10, 'y': 0},
            'strength': 0.085  # Historical annual likelihood
        },
        'wildfire': {
            'position': {'x': 0, 'y': 10},
            'strength': 1.0  # Baseline for increase calculation
        }
    }
    
    # Create field state for testing
    field_state = FieldState(
        field_id="climate_mv",
        timestamp=datetime.now(),
        potential=1.0,
        gradient={'x': 0.0, 'y': 0.0},
        stability=0.8,
        metadata={"region": "Martha's Vineyard"}
    )
    
    # Initialize services (mock implementations for test)
    field_service = MockFieldStateService(field_state)
    gradient_service = MockGradientService()
    flow_service = MockFlowDynamicsService()
    
    # Create observer
    observer = FieldNavigationObserver(field_service, gradient_service, flow_service)
    
    # Test navigation path: precipitation -> flood risk -> return period
    path_positions = [
        ('start', climate_hazards['extreme_precipitation']['position']),
        ('impact', {'x': 2, 'y': 1}),  # Position where hazard impacts vulnerability
        ('risk', {'x': 4, 'y': 2})     # Position where risk level is assessed
    ]
    
    # Walk the path and observe
    for position_name, position in path_positions:
        observations = await observer.observe_position("climate_mv", position)
        observer.log_observations(position_name, observations)
        
        # Assert some basic expectations about the observations
        assert 0 <= observations['wave']['coherence'] <= 1, "Coherence should be between 0 and 1"
        assert observations['field']['gradient_magnitude'] >= 0, "Gradient magnitude should be non-negative"
        assert observations['flow']['turbulence'] >= 0, "Turbulence should be non-negative"

# Mock service implementations for testing
class MockFieldStateService(FieldStateService):
    def __init__(self, initial_state: FieldState):
        self.state = initial_state
        # Martha's Vineyard climate data points
        self.positions = {
            # Extreme precipitation area (100-year rainfall: 7.34 inches)
            (0,0): {'potential': 0.9, 'stability': 0.3},  # High energy, unstable
            # Drought area (26% annual likelihood by late century)
            (10,0): {'potential': 0.7, 'stability': 0.8},  # Moderate energy, stable
            # Wildfire area (94% increase in danger days)
            (0,10): {'potential': 0.8, 'stability': 0.4}   # High energy, moderate stability
        }
    
    async def get_field_state(self, field_id: str):
        return self.state
        
    async def update_field_state(self, field_id: str, state: FieldState):
        self.state = state
        
    async def calculate_field_stability(self, field_id: str):
        x, y = int(self.state.gradient['x']), int(self.state.gradient['y'])
        return self.positions.get((x,y), {'stability': 0.5})['stability']

class MockGradientService(GradientService):
    async def calculate_gradient(self, field_id: str, position: Dict[str, float]):
        return GradientVector(
            direction={'x': 0.5, 'y': 0.3},
            magnitude=0.6,
            stability=0.7
        )
        
    async def get_flow_direction(self, field_id: str, position: Dict[str, float]):
        return {'x': 0.5, 'y': 0.3}
        
    async def calculate_potential_difference(self, field_id: str, 
                                          position1: Dict[str, float],
                                          position2: Dict[str, float]):
        return 0.5

class MockFlowDynamicsService(FlowDynamicsService):
    def __init__(self):
        # Define flow characteristics for climate regions
        self.flow_fields = {
            (0,0): {'viscosity': 0.3, 'turbulence': 0.8},    # High turbulence in precipitation area
            (10,0): {'viscosity': 0.7, 'turbulence': 0.2},   # Low turbulence in drought area
            (0,10): {'viscosity': 0.4, 'turbulence': 0.6}    # Moderate turbulence in wildfire area
        }
    
    async def calculate_viscosity(self, field_id: str, position: Dict[str, float]):
        x, y = int(position['x']), int(position['y'])
        return self.flow_fields.get((x,y), {'viscosity': 0.5})['viscosity']
        
    async def calculate_turbulence(self, field_id: str, position: Dict[str, float]):
        x, y = int(position['x']), int(position['y'])
        return self.flow_fields.get((x,y), {'turbulence': 0.3})['turbulence']
        
    async def calculate_flow_rate(self, field_id: str,
                                start_position: Dict[str, float],
                                end_position: Dict[str, float]):
        # Calculate flow rate based on viscosity and turbulence differences
        x1, y1 = int(start_position['x']), int(start_position['y'])
        x2, y2 = int(end_position['x']), int(end_position['y'])
        
        start_flow = self.flow_fields.get((x1,y1), {'viscosity': 0.5, 'turbulence': 0.3})
        end_flow = self.flow_fields.get((x2,y2), {'viscosity': 0.5, 'turbulence': 0.3})
        
        # Flow rate increases with viscosity difference and decreases with turbulence
        visc_diff = abs(start_flow['viscosity'] - end_flow['viscosity'])
        turb_factor = 1.0 - (max(start_flow['turbulence'], end_flow['turbulence']) * 0.7)
        
        return visc_diff * turb_factor
