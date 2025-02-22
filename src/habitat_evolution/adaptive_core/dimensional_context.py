from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

class DimensionType(Enum):
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    SYSTEMIC = "systemic"
    REFERENCE = "reference"

class WindowState(Enum):
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"

class DimensionalWindow:
    """Represents a window of observation for pattern emergence in a dimension."""
    def __init__(self, dimension_type: DimensionType):
        self.type = dimension_type
        self.state = WindowState.CLOSED
        self.observations = []
        self.attraction_points = {}  # Track concept attraction like SemanticPotential
        self.semantic_gradients = []  # Track relationships between concepts
        self.boundary_tension = 0.0
        self.transcendence_score = 0.0  # Track how much pattern transcends dimension
        self.last_update = datetime.now()
    
    def observe(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Observe patterns without enforcing relationships."""
        suggestions = []
        
        # Extract concepts from observation
        concepts = set()
        for key, value in observation.items():
            if isinstance(value, (str, float)):
                concepts.add(key)
                if value and isinstance(value, str):
                    concepts.add(value)
        
        # Look for dimensional alignments
        for concept in concepts:
            # Record observation frequency
            if concept not in self.attraction_points:
                self.attraction_points[concept] = 0.0
            self.attraction_points[concept] += 0.1
            
            # Suggest potential relationships based on context proximity
            related = [c for c in concepts if c != concept]
            if related:
                suggestion = {
                    'concept': concept,
                    'potential_alignments': related,
                    'context_strength': self.attraction_points[concept],
                    'dimension': self.type.value,
                    'timestamp': datetime.now()
                }
                suggestions.append(suggestion)
                
                # Update boundary tension based on relationship potential
                self.boundary_tension = max(0.0, 
                    self.boundary_tension + (0.05 * len(related)))
        
        self.observations.append({
            'raw': observation,
            'processed': suggestions,
            'timestamp': datetime.now()
        })
        
        self._update_state()
        return {
            'suggestions': suggestions,
            'boundary_tension': self.boundary_tension,
            'window_state': self.state
        }
    
    def record_gradient(self, from_concept: str, to_concept: str, strength: float):
        """Record semantic gradient between concepts."""
        self.semantic_gradients.append({
            "from": from_concept,
            "to": to_concept,
            "strength": strength,
            "timestamp": datetime.now()
        })
    
    def _update_state(self) -> None:
        """Update window state based on observations and boundary tension."""
        if self.state == WindowState.CLOSED:
            if len(self.observations) >= 2:
                self.state = WindowState.OPENING
        
        elif self.state == WindowState.OPENING:
            if self.boundary_tension > 0.3:  # Threshold for significant relationships
                self.state = WindowState.OPEN
            elif len(self.observations) > 5:  # If no significant relationships after 5 observations
                self.state = WindowState.CLOSING
        
        elif self.state == WindowState.OPEN:
            if self.boundary_tension < 0.2:  # Relationships weakening
                self.state = WindowState.CLOSING
        
        elif self.state == WindowState.CLOSING:
            if len(self.observations) == 0:
                self.state = WindowState.CLOSED

class DimensionalContext:
    """Manages pattern observation and evolution across multiple dimensions."""
    def __init__(self):
        self.dimensions = {dim: DimensionalWindow(dim) for dim in DimensionType}
        self.evolution_history = []
        self.dimension_weights = defaultdict(float)
    
    def observe_pattern(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Allow patterns to emerge naturally across dimensions."""
        results = {}
        
        # Let each dimension observe and suggest patterns
        for dim_type, window in self.dimensions.items():
            result = window.observe(observation)
            
            # Only include dimensions that found meaningful patterns
            if result['suggestions']:
                results[dim_type.value] = result
                self._update_dimension_weights(dim_type)
        
        # Record evolution if we see cross-dimensional patterns
        if len(results) > 1:
            self._record_evolution(observation, results)
        
        return results
    
    def _update_dimension_weights(self, dimension: DimensionType) -> None:
        """Update dimension weights based on observation frequency and boundary tension."""
        window = self.dimensions[dimension]
        self.dimension_weights[dimension] = (
            len(window.observations) * window.boundary_tension
        )
    
    def _record_evolution(self, observation: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Record natural pattern evolution across dimensions."""
        self.evolution_history.append({
            'timestamp': datetime.now(),
            'observation': observation,
            'dimensional_patterns': results,
            'active_dimensions': [dim for dim, res in results.items() 
                                if res['window_state'] in (WindowState.OPENING.value, WindowState.OPEN.value)],
            'dimension_weights': {dim.value: weight 
                                for dim, weight in self.dimension_weights.items()}
        })
    
    def get_active_dimensions(self) -> List[DimensionType]:
        """Get dimensions currently showing pattern activity."""
        return [dim_type for dim_type, window in self.dimensions.items()
                if window.state in (WindowState.OPENING, WindowState.OPEN)]
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of pattern evolution across dimensions."""
        return {
            'active_dimensions': self.get_active_dimensions(),
            'boundary_tensions': {
                dim_type.value: window.boundary_tension
                for dim_type, window in self.dimensions.items()
            },
            'dimension_weights': {
                dim_type.value: self.dimension_weights[dim_type]
                for dim_type in DimensionType
            },
            'total_observations': sum(len(window.observations)
                                    for window in self.dimensions.values())
        }

class DimensionalContext:
    def __init__(self):
        self.dimensions = {dim: DimensionalWindow(dim) for dim in DimensionType}
        self.evolution_history = []
        self.reference_frame = None
        self.dimension_weights = defaultdict(float)
        
    def observe_pattern(self, observation: Dict[str, Any]) -> None:
        """Process new pattern observation and track dimensional transcendence."""
        potential_dimensions = self._detect_dimensions(observation)
        
        results = {}
        for dim in potential_dimensions:
            window = self.dimensions[dim]
            result = window.observe(observation)
            
            if result['suggestions']:
                results[dim.value] = result
                self._update_dimension_weights(dim)
            
            if self._detect_transcendence(dim, observation):
                self._record_evolution(dim, observation)
                self._update_dimension_weights(dim)
        
        return results

    def _detect_dimensions(self, observation: Dict[str, Any]) -> List[DimensionType]:
        """Detect which dimensions are relevant to this observation."""
        dimensions = []
        
        # Spatial dimension - location-based patterns
        if any(key in observation for key in ['location', 'position', 'area', 'region']):
            dimensions.append(DimensionType.SPATIAL)
        
        # Temporal dimension - time-based patterns
        if any(key in observation for key in ['time', 'date', 'period', 'duration']):
            dimensions.append(DimensionType.TEMPORAL)
        
        # Systemic dimension - system-wide patterns
        if any(key in observation for key in ['system', 'network', 'process', 'flow']):
            dimensions.append(DimensionType.SYSTEMIC)
        
        # Reference dimension - external reference points
        if any(key in observation for key in ['reference', 'baseline', 'benchmark']):
            dimensions.append(DimensionType.REFERENCE)
        
        # If no specific dimensions detected, observe in all dimensions
        return dimensions if dimensions else list(DimensionType)
        
        # Systemic indicators
        systemic_indicators = {"ecosystem", "interaction", "relationship", "coupling", "feedback"}
        if any(ind in observation for ind in systemic_indicators):
            dimensions.append(DimensionType.SYSTEMIC)
        
        # Reference indicators
        if "reference" in observation or self.reference_frame in observation:
            dimensions.append(DimensionType.REFERENCE)
            
        return dimensions

    def _detect_transcendence(self, dimension: DimensionType, observation: Dict[str, Any]) -> bool:
        """Detect if a pattern is transcending its original dimension."""
        window = self.dimensions[dimension]
        
        # Check if pattern shows significant boundary tension
        if window.boundary_tension > 0.3:
            # Look for cross-dimensional relationships
            for other_dim, other_window in self.dimensions.items():
                if other_dim != dimension and other_window.boundary_tension > 0:
                    return True
        
        return False
        
    def _record_evolution(self, dimension: DimensionType, observation: Dict[str, Any]) -> None:
        """Record pattern evolution event."""
        self.evolution_history.append({
            'timestamp': datetime.now(),
            'dimension': dimension.value,
            'observation': observation,
            'boundary_tension': self.dimensions[dimension].boundary_tension,
            'window_state': self.dimensions[dimension].state.value
        })
        
    def _update_dimension_weights(self, dimension: DimensionType) -> None:
        """Update dimension weights based on observed pattern evolution."""
        window = self.dimensions[dimension]
        self.dimension_weights[dimension] = (
            len(window.observations) * window.boundary_tension
        )
        
    def get_active_dimensions(self) -> List[DimensionType]:
        """Get dimensions currently showing pattern activity."""
        return [dim_type for dim_type, window in self.dimensions.items()
                if window.state in (WindowState.OPENING, WindowState.OPEN)]
    def _record_evolution(self, dimension: DimensionType, observation: Dict[str, Any]) -> None:
        """Record pattern evolution event."""
        self.evolution_history.append({
            "dimension": dimension,
            "timestamp": datetime.now(),
            "observation": observation,
            "transcendence_score": self.dimensions[dimension].transcendence_score,
            "window_state": self.dimensions[dimension].state.value
        })
    
    def _update_dimension_weights(self, dimension: DimensionType) -> None:
        """Update dimension weights based on observed pattern evolution."""
        # Increase weight for dimensions showing consistent evolution
        self.dimension_weights[dimension] += 0.1
        
        # Decay weights for other dimensions
        for dim in DimensionType:
            if dim != dimension:
                self.dimension_weights[dim] *= 0.95
    
    def get_active_dimensions(self) -> List[DimensionType]:
        """Get list of currently active dimensions."""
        return [dim for dim, window in self.dimensions.items() 
                if window.state in (WindowState.OPENING, WindowState.OPEN)]
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of pattern evolution across dimensions."""
        return {
            "active_dimensions": self.get_active_dimensions(),
            "dimension_weights": dict(self.dimension_weights),
            "transcendence_scores": {
                dim.value: window.transcendence_score 
                for dim, window in self.dimensions.items()
            },
            "total_observations": sum(len(window.observations) 
                                   for window in self.dimensions.values())
        }
