"""Tests for semantic pattern discovery and evolution in visualization."""

import pytest
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# Use visualization test classes
from ...visualization.test_visualization import (
    TestVisualizationConfig,
    TestPatternVisualizer
)
from ...visualization.pattern_id import PatternAdaptiveID
from ...visualization.semantic_validation import SemanticValidator, ValidationStatus

from enum import Enum
from typing import Optional, List, Dict, Set

class WindowState(Enum):
    """Window states for pattern evolution."""
    CLOSED = "CLOSED"      # Initial state, no pattern flow
    OPENING = "OPENING"    # Beginning pattern absorption
    OPEN = "OPEN"         # Actively processing patterns
    CLOSING = "CLOSING"    # Reducing pattern flow

class SemanticPotential:
    """Represents potential meaning structures that could emerge."""
    def __init__(self):
        self.attraction_points: Dict[str, float] = {}
        self.semantic_gradients: List[Dict] = []
        self.boundary_tension: float = 0.0
        self.concept_resonance: Dict[str, Dict] = {}
        self.pattern_correlations: Dict[tuple, List[float]] = {}
        
    def observe_semantic_potential(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Observe semantic potential without enforcing relationships."""
        suggestions = []
        
        # Extract concepts from observation
        concepts = set()
        for key, value in observation.items():
            if isinstance(value, (str, float)):
                concepts.add(key)
                if value and isinstance(value, str):
                    concepts.add(value)
        
        # Look for semantic alignments
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
                    'timestamp': datetime.now()
                }
                suggestions.append(suggestion)
                
                # Update boundary tension based on relationship potential
                self.boundary_tension = max(0.0, 
                    self.boundary_tension + (0.05 * len(related)))
        
        return suggestions
    
    def suggest_pattern(self, pattern_type: str, confidence: float):
        """Suggest a potential pattern without forcing it."""
        self.attraction_points[pattern_type] = confidence
        
    def record_gradient(self, from_pattern: str, to_pattern: str, strength: float):
        """Record semantic gradient between patterns."""
        self.semantic_gradients.append({
            "from": from_pattern,
            "to": to_pattern,
            "strength": strength,
            "timestamp": datetime.now()
        })

class WindowTransition:
    """Represents a semantic window state transition."""
    def __init__(self, from_state: WindowState, to_state: WindowState, pressure_threshold: float):
        self.from_state = from_state
        self.to_state = to_state
        self.pressure_threshold = pressure_threshold
        self.transition_count = 0
        self.last_transition = None
        self.semantic_potential = SemanticPotential()

from dataclasses import dataclass
from typing import List, Optional
from hashlib import blake2b

@dataclass
class PatternLineage:
    """Compact representation of pattern evolution history."""
    base_id: str
    state: str
    evolution_step: int
    observer_hash: str  # Condensed hash of observer sequence
    parent_hash: Optional[str] = None  # Condensed hash of parent pattern

@dataclass
class PatternIntuition:
    """Represents evolving intuitive resonance with a pattern."""
    pattern_id: str
    resonance_strength: float   # How strongly pattern resonates
    encounter_count: int       # Number of pattern encounters
    last_resonance: datetime
    intuition_state: str       # glimpse, resonance, attunement
    hints: List[Dict]         # Intuitive hints about pattern connections

class UserContext:
    """User context with pattern intuition tracking."""
    def __init__(self, user_id: str):
        self._hasher = blake2b(digest_size=8)  # 8-byte hash for users
        self._user_id = user_id
        self._pattern_intuitions: Dict[str, PatternIntuition] = {}
        self._resonance_networks: Dict[str, List[str]] = {}
        self._intuition_flow: List[Dict] = []
    
    @property
    def hash(self) -> str:
        """Generate consistent 8-byte hash for user identity."""
        self._hasher.update(self._user_id.encode())
        return self._hasher.hexdigest()[:16]  # 16 chars = 8 bytes
    
    def resonate_with_pattern(self, pattern_id: str, hint: Dict = None) -> PatternIntuition:
        """Create or deepen intuitive resonance with a pattern."""
        if pattern_id not in self._pattern_intuitions:
            # Initial glimpse
            intuition = PatternIntuition(
                pattern_id=pattern_id,
                resonance_strength=0.1,
                encounter_count=1,
                last_resonance=datetime.now(),
                intuition_state="glimpse",
                hints=[hint] if hint else []
            )
            self._pattern_intuitions[pattern_id] = intuition
        else:
            # Deepen existing resonance
            intuition = self._pattern_intuitions[pattern_id]
            intuition.encounter_count += 1
            intuition.last_resonance = datetime.now()
            
            # Evolve resonance naturally
            if intuition.encounter_count > 10:
                intuition.intuition_state = "attunement"
                intuition.resonance_strength = min(1.0, intuition.resonance_strength + 0.1)
            elif intuition.encounter_count > 5:
                intuition.intuition_state = "resonance"
                intuition.resonance_strength = min(0.8, intuition.resonance_strength + 0.15)
            
            if hint:
                intuition.hints.append(hint)
        
        # Record intuition flow
        self._intuition_flow.append({
            "timestamp": datetime.now(),
            "pattern_id": pattern_id,
            "intuition": intuition,
            "hint": hint
        })
        
        return intuition
    
    def find_resonant_patterns(self, pattern_id: str) -> List[str]:
        """Find patterns with similar resonance signatures."""
        resonant = []
        if pattern_id in self._pattern_intuitions:
            # Find patterns with similar intuition states
            current = self._pattern_intuitions[pattern_id]
            for pid, intuit in self._pattern_intuitions.items():
                if pid != pattern_id:
                    # Check for resonance alignment
                    state_match = intuit.intuition_state == current.intuition_state
                    strength_diff = abs(intuit.resonance_strength - current.resonance_strength)
                    if state_match and strength_diff < 0.3:  # Allow for natural variation
                        resonant.append(pid)
        return resonant
    
    def get_intuition_frame(self, pattern_id: str) -> Dict:
        """Get current intuitive resonance frame for a pattern."""
        if pattern_id in self._pattern_intuitions:
            intuit = self._pattern_intuitions[pattern_id]
            return {
                "pattern_id": pattern_id,
                "resonance_state": intuit.intuition_state,
                "resonance": intuit.resonance_strength,
                "attunement": intuit.encounter_count / 10.0,  # Natural progression
                "recent_hints": intuit.hints[-3:],  # Recent intuitive hints
                "resonant_patterns": self.find_resonant_patterns(pattern_id)
            }
        return None

class AdaptiveId:
    """Base class for adaptive pattern identification with window state management.
    
    This class implements a consistent pattern for handling contextual information:
    - temporal_context: Stored as a JSON string to ensure consistent serialization/deserialization
      across the system. This pattern should be followed for any future context types
      (e.g., spatial_context) that need to store structured data.
      
    The context pattern follows these rules:
    1. Initialize as a JSON string containing an empty dict
    2. Always serialize when setting
    3. Always deserialize when reading
    4. Use json.loads/dumps for consistency
    
    This approach ensures:
    - Consistent data representation across the system
    - Safe serialization for database storage
    - Clear contract for context handling in derived classes
    """
    def __init__(self, base_id: str, pattern_type: str = None, hazard_type: str = None,
                 initial_state: WindowState = WindowState.CLOSED, 
                 observer_id: str = None, parent_pattern_id: str = None):
        from hashlib import blake2b
        self._hasher = blake2b(digest_size=4)  # Small, efficient hash
        self._base_id = base_id
        self._window_state = initial_state
        self._evolution_history = []
        self._stability_score = 1.0
        self._pressure_level = 0.0
        self._observer_id = observer_id  # Track who's observing/modifying
        self._parent_pattern_id = parent_pattern_id  # Track pattern lineage
        self._child_patterns: Set[str] = set()  # Track derived patterns
        # Initialize contexts following the standard pattern
        # temporal_context: JSON string for structured temporal data
        self.temporal_context = json.dumps({})  # Initialize as empty JSON dict
        self.temporal_horizon = None  # Current temporal scope
        self.probability = None       # Event probability
        # spatial_context: Currently a simple string, but could be extended to use
        # the same JSON pattern as temporal_context if needed in the future
        self.spatial_context = None
        self.pattern_type = pattern_type
        self.hazard_type = hazard_type
        
        # Define valid transitions and thresholds
        self._transitions = {
            WindowState.CLOSED: [
                WindowTransition(WindowState.CLOSED, WindowState.OPENING, 0.3)
            ],
            WindowState.OPENING: [
                WindowTransition(WindowState.OPENING, WindowState.OPEN, 0.5),  # Lower threshold
                WindowTransition(WindowState.OPENING, WindowState.CLOSED, 0.1)
            ],
            WindowState.OPEN: [
                WindowTransition(WindowState.OPEN, WindowState.CLOSING, 0.4),
                WindowTransition(WindowState.OPEN, WindowState.OPEN, 0.6)  # Lower stability requirement
            ],
            WindowState.CLOSING: [
                WindowTransition(WindowState.CLOSING, WindowState.CLOSED, 0.2)
            ]
        }
        
        # Track connected patterns for Neo4j relationships
        self._connected_patterns: Set[str] = set()
        self._relationship_strengths: Dict[str, float] = {}
    
    def _generate_hash(self, value: str) -> str:
        """Generate short, consistent hash for observers/patterns."""
        self._hasher.update(value.encode())
        return self._hasher.hexdigest()[:8]
    
    @property
    def current_id(self) -> str:
        state_abbrev = self._window_state.value[:3].lower()
        obs_hash = self._generate_hash(self._observer_id) if self._observer_id else ""
        parent_hash = self._generate_hash(self._parent_pattern_id) if self._parent_pattern_id else ""
        
        # Compact format: base_state_step#obs#parent
        return f"{self._base_id}_{state_abbrev}_{len(self._evolution_history)}#{obs_hash}#{parent_hash}"
    
    @staticmethod
    def parse_pattern_id(pattern_id: str) -> PatternLineage:
        """Deconstruct pattern ID into its components."""
        base_state_step, obs_hash, parent_hash = pattern_id.split('#')
        base, state, step = base_state_step.rsplit('_', 2)
        
        return PatternLineage(
            base_id=base,
            state=state,
            evolution_step=int(step),
            observer_hash=obs_hash,
            parent_hash=parent_hash if parent_hash else None
        )
    
    def connect_pattern(self, pattern_id: str, relationship_strength: float, pattern_type: str = None):
        """Establish semantic relationship with another pattern."""
        self._connected_patterns.add(pattern_id)
        self._relationship_strengths[pattern_id] = relationship_strength
        
        # Always record gradients but with varying strengths based on window state
        gradient_strength = relationship_strength
        if self._window_state == WindowState.CLOSED:
            gradient_strength *= 0.3  # Weak initial connections
        elif self._window_state == WindowState.OPENING:
            gradient_strength *= 0.7  # Growing connections
        elif self._window_state == WindowState.CLOSING:
            gradient_strength *= 0.5  # Fading connections
        
        # Record gradient once per window state
        current_transitions = self._transitions.get(self._window_state, [])
        if current_transitions:
            # Use the first transition for gradient recording
            transition = current_transitions[0]
            if pattern_type:
                transition.semantic_potential.suggest_pattern(pattern_type, gradient_strength)
            
            transition.semantic_potential.record_gradient(
                from_pattern=self.current_id,
                to_pattern=pattern_id,
                strength=gradient_strength
            )
            
            # Share pattern suggestions with other transitions
            for other_transition in current_transitions[1:]:
                if pattern_type:
                    other_transition.semantic_potential.suggest_pattern(pattern_type, gradient_strength)
        
        # Generate Neo4j relationship properties with potential structure hints
        return {
            "source_id": self.current_id,
            "target_id": pattern_id,
            "strength": relationship_strength,
            "window_state": self._window_state.value,
            "stability": self._stability_score,
            "potential_patterns": [p for p, c in self._get_potential_patterns().items() if c > 0.7],
            "semantic_gradients": self._get_active_gradients()
        }
    
    def _get_potential_patterns(self) -> Dict[str, float]:
        """Get potential patterns from all transitions."""
        potentials = {}
        for transitions in self._transitions.values():
            for transition in transitions:
                potentials.update(transition.semantic_potential.attraction_points)
        return potentials
    
    def _get_active_gradients(self) -> List[Dict]:
        """Get active semantic gradients from current state."""
        active_gradients = []
        for transition in self._transitions.get(self._window_state, []):
            active_gradients.extend([
                g for g in transition.semantic_potential.semantic_gradients
                if (datetime.now() - g["timestamp"]).seconds < 3600  # Active in last hour
            ])
        return active_gradients
    
    def _check_transitions(self) -> Optional[WindowState]:
        """Check if any transitions should occur based on pressure."""
        current_transitions = self._transitions.get(self._window_state, [])
        
        for transition in current_transitions:
            if self._pressure_level >= transition.pressure_threshold:
                # Record semantic state before transition
                transition.semantic_potential.boundary_tension = self._pressure_level
                transition.transition_count += 1
                transition.last_transition = datetime.now()
                return transition.to_state
        return None
    
    def evolve(self, pressure: float, stability: float, observer_insight: Dict = None):
        """Record evolution step with pressure and stability metrics.
        
        Args:
            pressure: Current pressure level
            stability: Current stability score
            observer_insight: Optional insights from observer, should be concise
        """
        evolution_step = {
            "timestamp": datetime.now(),
            "pressure": pressure,
            "stability": stability,
            "window_state": self._window_state.value,
            "connected_patterns": list(self._connected_patterns),
            "observer_id": self._observer_id,
            "parent_pattern": self._parent_pattern_id,
            "child_patterns": list(self._child_patterns)
        }
        
        if observer_insight:
            evolution_step["observer_insight"] = observer_insight
            
        self._evolution_history.append(evolution_step)
        
        self._stability_score = stability
        self._pressure_level = pressure
        
        # Check for state transitions
        new_state = self._check_transitions()
        if new_state:
            self._window_state = new_state
            
        # Return Neo4j node properties
        return {
            "id": self.current_id,
            "base_id": self._base_id,
            "window_state": self._window_state.value,
            "stability": stability,
            "pressure": pressure,
            "evolution_step": len(self._evolution_history)
        }

def test_pattern_and_user_intuition():
    """Test pattern ID evolution and user-pattern intuitive resonance."""
    # Create user context
    user = UserContext("researcher_1")
    print(f"\nUser Hash: {user.hash}")
    
    # Create initial pattern with user as observer
    drought = AdaptiveId("drought_risk", observer_id=user.hash)
    initial_id = drought.current_id
    
    # Initial glimpse of pattern
    intuition = user.resonate_with_pattern(
        initial_id,
        hint={"feeling": "Seems connected to seasonal cycles"}
    )
    print(f"\nInitial Pattern Glimpse:")
    print(f"Pattern: {initial_id}")
    print(f"State: {intuition.intuition_state}")
    print(f"Resonance: {intuition.resonance_strength:.2f}")
    
    # Parse and verify components
    lineage = AdaptiveId.parse_pattern_id(initial_id)
    print("\nPattern Lineage:")
    print(f"Base: {lineage.base_id}")
    print(f"State: {lineage.state}")
    print(f"Step: {lineage.evolution_step}")
    print(f"Observer: {lineage.observer_hash}")
    
    # Deepen resonance through encounters
    for i in range(6):  # Progress through resonance states
        hint = {
            "feeling": f"Intuitive hint {i+1}",
            "certainty": 0.5 + i*0.1,
            "connections": ["weather", "climate"]
        }
        drought.evolve(
            pressure=0.4 + i*0.1,
            stability=0.8,
            observer_insight=hint
        )
        evolved_id = drought.current_id
        intuition = user.resonate_with_pattern(evolved_id, hint)
    
    print(f"\nDeepened Resonance:")
    frame = user.get_intuition_frame(evolved_id)
    print(f"Pattern: {evolved_id}")
    print(f"State: {frame['resonance_state']}")
    print(f"Attunement: {frame['attunement']:.2f}")
    print(f"Recent Hints: {len(frame['recent_hints'])}")
    
    # Create and explore resonant patterns
    rainfall = AdaptiveId("rainfall_pattern", observer_id=user.hash)
    temperature = AdaptiveId("temperature_pattern", observer_id=user.hash)
    
    # Build resonance network
    for pattern in [rainfall, temperature]:
        for i in range(4):
            pattern_id = pattern.current_id
            hint = {
                "feeling": f"Network resonance {i+1}",
                "connection_type": "harmony",
                "intensity": 0.6 + i*0.1
            }
            pattern.evolve(pressure=0.5, stability=0.7, observer_insight=hint)
            user.resonate_with_pattern(pattern_id, hint)
    
    print("\nResonance Network:")
    resonant = user.find_resonant_patterns(evolved_id)
    print(f"Patterns resonating with {evolved_id}:")
    for rel in resonant:
        rel_frame = user.get_intuition_frame(rel)
        print(f"  - {rel} ({rel_frame['resonance_state']})")
    
    # Demonstrate natural resonance progression
    print(f"\nResonance Journey:")
    for entry in user._intuition_flow[-3:]:
        print(f"  {entry['timestamp']}: {entry['hint']['feeling']} ({entry['intuition'].intuition_state})")

def test_semantic_potential_evolution():
    """Test the natural evolution of semantic potential in pattern discovery."""
    # Initialize test nodes with semantic context and observers
    drought = AdaptiveId("drought_risk", observer_id="agent_1")
    rainfall = AdaptiveId("rainfall_pattern", observer_id="agent_2")
    
    # Create a derived pattern from drought
    drought_severity = AdaptiveId(
        "drought_severity",
        observer_id="agent_3",
        parent_pattern_id=drought.current_id
    )
    wildfire = AdaptiveId("wildfire_risk")
    
    # Define initial concept relationships
    climate_concepts = {
        "drought": {
            "precipitation": 0.8,  # Strong direct relationship
            "soil_moisture": 0.7,
            "temperature": 0.6,
            "wildfire": 0.5   # Initial moderate relationship
        },
        "precipitation": {
            "drought": 0.8,
            "flood_risk": 0.6,
            "vegetation": 0.5
        },
        "wildfire": {
            "temperature": 0.7,
            "drought": 0.5,
            "wind_pattern": 0.4
        }
    }
    
    # Observe natural pattern emergence
    observations = []
    
    # Phase 1: Initial Pattern Suggestion (CLOSED state)
    # Observe initial context without enforcing relationships
    initial_observation = {
        "pattern_type": "climate_risk",
        "risk_factor": "drought",
        "related_measure": "rainfall",
        "confidence": 0.4
    }
    
    # Let semantic potential emerge from observation
    semantic_suggestions = drought._transitions[WindowState.CLOSED][0].semantic_potential.observe_semantic_potential(
        initial_observation
    )
    
    # Connect patterns based on observed potential
    drought_props = drought.connect_pattern(
        pattern_id=rainfall.current_id,
        relationship_strength=0.4,
        pattern_type="precipitation_impact"
    )
    
    observations.append({
        "phase": "suggestion",
        "pattern_id": drought.current_id,
        "window_state": drought._window_state.value,
        "semantic_suggestions": semantic_suggestions,
        "gradients": drought_props["semantic_gradients"],
        "boundary_tension": drought._transitions[drought._window_state][0].semantic_potential.boundary_tension
    })
    
    # Phase 2: Initial Evolution (OPENING state)
    drought.evolve(pressure=0.35, stability=0.85)
    props = drought.connect_pattern(
        pattern_id=rainfall.current_id,
        relationship_strength=0.6,
        pattern_type="precipitation_impact"
    )
    observations.append({
        "phase": "evolution",
        "window_state": drought._window_state.value,
        "pattern_ids": {
            "drought": drought.current_id,
            "rainfall": rainfall.current_id
        },
        "gradients": props["semantic_gradients"]
    })
    
    # Phase 2: Pattern Evolution
    for i in range(3):
        # Gradually increase pressure to force state transitions
        pressure = 0.3 + (i * 0.2)  # 0.3 -> 0.5 -> 0.7
        drought.evolve(pressure=pressure, stability=0.75)
        rainfall.evolve(pressure=pressure, stability=0.8)
        
        # Connect emerging patterns
        props = drought.connect_pattern(
            pattern_id=wildfire.current_id,
            relationship_strength=0.7,
            pattern_type="risk_cascade"
        )
        observations.append({
            "phase": "evolution",
            "window_state": props["window_state"],
            "potentials": props["potential_patterns"],
            "gradients": props["semantic_gradients"]
        })
    
    # Validate and display semantic potential
    assert len(observations) > 0, "Should record observations"
    
    print("\nSemantic Evolution Timeline:")
    print("-" * 50)
    
    for i, obs in enumerate(observations):
        print(f"\nStep {i + 1}: {obs['phase'].upper()}")
        if 'pattern_ids' in obs:
            print("Pattern IDs:")
            for name, pid in obs['pattern_ids'].items():
                print(f"  {name}: {pid}")
        elif 'pattern_id' in obs:
            print(f"Pattern ID: {obs['pattern_id']}")
            
        if 'window_state' in obs:
            print(f"Window State: {obs['window_state']}")
        
        if obs.get('semantic_suggestions'):
            print("\nSemantic Potential:")
            # Group by concept type
            risk_concepts = [s for s in obs['semantic_suggestions'] 
                           if 'risk' in s['concept']]
            measure_concepts = [s for s in obs['semantic_suggestions']
                              if 'measure' in s['concept'] or 'pattern' in s['concept']]
            
            if risk_concepts:
                print("  Risk Factors:")
                for suggestion in risk_concepts:
                    strength_bar = "█" * int(suggestion['context_strength'] * 10)
                    print(f"    {suggestion['concept']}: {suggestion['context_strength']:.2f} [{strength_bar:<10}]")
                    if suggestion['potential_alignments']:
                        measures = [a for a in suggestion['potential_alignments'] 
                                  if 'measure' in a or 'pattern' in a]
                        if measures:
                            print("      Related Measures:")
                            for measure in measures:
                                print(f"        • {measure}")
            
            if measure_concepts:
                print("  Measurements:")
                for suggestion in measure_concepts:
                    strength_bar = "█" * int(suggestion['context_strength'] * 10)
                    print(f"    {suggestion['concept']}: {suggestion['context_strength']:.2f} [{strength_bar:<10}]")
        
        if 'boundary_tension' in obs:
            tension_bar = "█" * int(obs['boundary_tension'] * 10)
            print(f"\nBoundary Tension: {obs['boundary_tension']:.2f} [{tension_bar:<10}]")
        
        print("\nSemantic Gradients:")
        for gradient in obs['gradients']:
            strength = gradient['strength']
            strength_bar = "█" * int(strength * 10)
            print(f"  {gradient['from']} → {gradient['to']}")
            print(f"    Strength: {strength:.2f} [{strength_bar:<10}]")
            print(f"    Timestamp: {gradient['timestamp'].strftime('%H:%M:%S')}")
            
        # Show relationship evolution
        if i > 0:
            prev_gradients = {(g['from'], g['to']): g['strength'] 
                            for g in observations[i-1]['gradients']}
            for gradient in obs['gradients']:
                key = (gradient['from'], gradient['to'])
                if key in prev_gradients:
                    prev_strength = prev_gradients[key]
                    delta = gradient['strength'] - prev_strength
                    if abs(delta) > 0.01:
                        print(f"    Δ: {delta:+.2f} {'↑' if delta > 0 else '↓'}")
    
    # Check for natural emergence
    evolution_phases = [obs["phase"] for obs in observations]
    assert "suggestion" in evolution_phases, "Should show initial pattern suggestion"
    assert "evolution" in evolution_phases, "Should show pattern evolution"
    
    # Verify semantic gradients formed
    final_obs = observations[-1]
    assert len(final_obs["gradients"]) > 0, "Should develop semantic gradients"
    
    # Confirm window state progression
    states = [obs.get("window_state") for obs in observations if "window_state" in obs]
    assert WindowState.OPEN.value in states, "Should reach OPEN state for pattern flow"

@dataclass
class SemanticNode:
    """Base class for semantic graph nodes with adaptive identification.
    
    Establishes the standard context handling pattern:
    - All context fields (temporal, spatial, etc.) are stored as JSON strings
    - Contexts are always serialized when setting
    - Contexts are always deserialized when accessing
    - Uses json.loads/dumps for consistency
    """
    id: str
    type: str
    created_at: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    
    def __post_init__(self):
        self.adaptive_id = AdaptiveId(self.id)
        self._coherence_score = 0.0
        self._stability_threshold = 0.8  # From success criteria
        # Initialize base contexts using the datetime from top-level import
        now = datetime.now()
        self.temporal_context = json.dumps({
            "created_at": self.created_at.isoformat(),
            "last_modified": now.isoformat()
        })
        self.spatial_context = json.dumps({})
    
    def update_coherence(self, score: float):
        """Update coherence score and evolve adaptive ID if needed."""
        self._coherence_score = score
        pressure = 1.0 - score if score < self._stability_threshold else 0.0
        self.adaptive_id.evolve(pressure=pressure, stability=score)
        
    def update_temporal_context(self, updates: Dict[str, Any]) -> None:
        """Safely update temporal context by deserializing, updating, and reserializing.
        
        Args:
            updates: Dictionary of updates to apply to temporal context
        """
        current = json.loads(self.temporal_context) if isinstance(self.temporal_context, str) else {}
        current.update(updates)
        current['last_modified'] = datetime.now().isoformat()
        self.temporal_context = json.dumps(current)
    
    def update_spatial_context(self, updates: Dict[str, Any]) -> None:
        """Safely update spatial context by deserializing, updating, and reserializing.
        
        Args:
            updates: Dictionary of updates to apply to spatial context
        """
        current = json.loads(self.spatial_context) if isinstance(self.spatial_context, str) else {}
        current.update(updates)
        current['last_modified'] = datetime.now().isoformat()
        self.spatial_context = json.dumps(current)
        
    def to_dict(self):
        """Convert node to dictionary for Neo4j export.
        
        Returns:
            dict: Node representation with properly deserialized contexts:
                - temporal_context: JSON string -> dict
                - spatial_context: JSON string -> dict
                - Other fields: Direct property access
        """
        return {
            'id': self.id,
            'type': self.type,
            'created_at': self.created_at.isoformat(),
            'confidence': self.confidence,
            'coherence': self._coherence_score,
            'adaptive_id': self.adaptive_id.current_id(),
            'node_type': getattr(self, 'node_type', None),
            'temporal_horizon': getattr(self, 'temporal_horizon', None),
            'temporal_context': json.loads(self.temporal_context) if isinstance(self.temporal_context, str) else {},
            'spatial_context': json.loads(self.spatial_context) if isinstance(self.spatial_context, str) else {},
            'probability': getattr(self, 'probability', None)
        }

class TemporalNode(SemanticNode):
    """Represents a temporal context.
    
    Follows the standard context handling pattern defined in AdaptiveId:
    - temporal_context is stored as a JSON string
    - Handles serialization/deserialization consistently
    - Maintains the contract for Neo4j export
    """
    def __init__(self, period: str, year: int, id: str):
        super().__init__(id=id, type="temporal")
        self.period = period
        self.year = year
        # Initialize temporal_context using helper method
        self.update_temporal_context({
            "period": period,
            "year": year
        })
        
    def to_dict(self):
        """Convert temporal node to dictionary with period and year.
        
        Returns:
            dict: Node representation with properly deserialized contexts:
                - temporal_context: Deserialized from JSON string
                - period, year: Direct property access
        """
        base_dict = super().to_dict()
        base_dict.update({
            'period': self.period,
            'year': self.year,
            'hazard_type': 'temporal_context',
            'temporal_horizon': self.period,
            'probability': 1.0,
            'spatial_context': 'MarthasVineyard'
        })
        return base_dict
    
class EventNode(SemanticNode):
    """Represents climate events.
    
    Follows the standard context handling pattern defined in AdaptiveId:
    - temporal_context is stored as a JSON string
    - Handles serialization/deserialization consistently
    - Maintains the contract for Neo4j export
    """
    def __init__(self, id: str, event_type: str, metrics: Dict[str, float]):
        super().__init__(id=id, type="event")
        self.event_type = event_type
        self.metrics = metrics
        
    def to_dict(self):
        """Convert event node to dictionary with event type and metrics.
        
        Returns:
            dict: Node representation with properly deserialized contexts:
                - temporal_context: Deserialized from JSON string
                - metrics: Raw metrics dictionary
                - Other fields: Direct property access
        """
        base_dict = super().to_dict()
        base_dict.update({
            'event_type': self.event_type,
            'metrics': self.metrics,
            'hazard_type': self.event_type,
            'temporal_horizon': getattr(self, 'temporal_horizon', 'current'),
            'probability': self.metrics.get(f'{getattr(self, "temporal_horizon", "current")}_probability', 1.0),
            'increase_percent': self.metrics.get(f'{getattr(self, "temporal_horizon", "current")}_increase', None),
            'spatial_context': 'MarthasVineyard',
            'temporal_context': json.loads(self.temporal_context) if isinstance(self.temporal_context, str) else {}
        })
        return base_dict

class SemanticRelation(SemanticNode):
    """Represents evolving relationships between semantic patterns.
    
    As part of the Unified Adaptive Pattern Evolution Framework, relationships are
    treated as first-class evolution points alongside nodes. This enables:
    
    1. Learning Windows:
       - Inherits window state management from SemanticNode
       - Enables gradual relationship pattern emergence
       - Transitions: CLOSED -> OPENING -> OPEN -> CLOSING
    
    2. Adaptive Identity:
       - Composite ID structure: source_relation_target
       - Evolves with connected patterns
       - Maintains semantic continuity
    
    3. Context Management:
       - temporal_context: JSON string tracking evolution history
       - spatial_context: JSON string for geographical relevance
       - Consistent serialization/deserialization
    
    4. Pattern Recognition:
       - Relationship patterns emerge independently
       - Contributes to graph-level meaning structures
       - Adapts with connected node evolution
    
    This approach ensures that meaning structures evolve cohesively at both
    node and relationship levels, enabling organic pattern emergence across
    the entire semantic graph.
    """
    
    def __init__(self, source_id: str, target_id: str, relation_type: str, strength: float = 1.0, evidence: List[str] = None):
        # Create adaptive ID from relationship components
        relation_id = f"{source_id}_{relation_type}_{target_id}"
        super().__init__(id=relation_id, type="relation")
        
        # Set relationship properties
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.strength = strength
        self.evidence = evidence or []
        
        # Initialize contexts
        self.update_temporal_context({"created_at": self.created_at.isoformat()})
        self.update_spatial_context({})  # Initialize empty but following pattern
        
    def to_dict(self):
        """Convert relationship to dictionary for Neo4j export.
        
        Returns:
            dict: Relationship representation with properly deserialized contexts:
                - temporal_context: JSON string -> dict
                - spatial_context: JSON string -> dict
                - Other fields: Direct property access
        """
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'type': self.relation_type,
            'strength': self.strength,
            'evidence': self.evidence,
            'temporal_context': json.loads(self.temporal_context) if isinstance(self.temporal_context, str) else {},
            'spatial_context': json.loads(self.spatial_context) if isinstance(self.spatial_context, str) else {}
        }

class SemanticPatternVisualizer(TestPatternVisualizer):
    """Enhanced visualizer with semantic pattern capabilities."""
    
    def __init__(self, config: Optional[TestVisualizationConfig] = None):
        super().__init__(config)
        self.validator = SemanticValidator()
        
    def extract_patterns_from_semantic_graph(
        self, 
        semantic_graph: Dict
    ) -> List[PatternAdaptiveID]:
        """Extract patterns from semantic graph with validation."""
        patterns = []
        
        # Process temporal nodes
        if "temporal_nodes" in semantic_graph:
            for node in semantic_graph["temporal_nodes"]:
                validation_result = self.validator.validate_node_structure(node)
                self.validator.log_validation(validation_result)
                if validation_result.status == ValidationStatus.RED:
                    raise ValueError(validation_result.message)
                pattern = PatternAdaptiveID(
                    pattern_type="temporal",
                    hazard_type="temporal_context",
                    creator_id=node.id,
                    confidence=1.0
                )
                pattern.temporal_context = json.dumps({
                    node.period: node.year,
                    "created_at": str(node.created_at),
                    "last_modified": str(node.created_at)
                })
                pattern.temporal_horizon = node.period
                pattern.probability = 1.0
                pattern.spatial_context = json.dumps({"location": "MarthasVineyard"})
                pattern.update_metrics(
                    position=(0, node.year),  # Use year as y-coordinate
                    field_state=1.0,  # Base field state
                    coherence=1.0,
                    energy_state=1.0  # Base energy state
                )
                patterns.append(pattern)
        
        # Process event nodes
        if "event_nodes" in semantic_graph:
            for event in semantic_graph["event_nodes"]:
                validation_result = self.validator.validate_node_structure(event)
                self.validator.log_validation(validation_result)
                if validation_result.status == ValidationStatus.RED:
                    raise ValueError(validation_result.message)
                pattern = PatternAdaptiveID(
                    pattern_type="event",
                    hazard_type=event.event_type,
                    creator_id=event.id,
                    confidence=1.0
                )
                pattern.temporal_context = json.dumps({
                    "created_at": str(event.created_at),
                    "last_modified": str(event.created_at)
                })
                pattern.temporal_horizon = getattr(event, 'temporal_horizon', 'current')
                pattern.probability = event.metrics.get(f'{pattern.temporal_horizon}_probability', 1.0)
                pattern.increase_percent = event.metrics.get(f'{pattern.temporal_horizon}_increase')
                pattern.spatial_context = json.dumps({"location": "Martha's Vineyard"})
                pattern.update_metrics(
                    position=(0, 0),  # Default position
                    field_state=1.0,  # Base field state
                    coherence=1.0,
                    energy_state=1.0  # Base energy state
                )
                patterns.append(pattern)
        
        # Process relations
        if "relations" in semantic_graph:
            for rel in semantic_graph["relations"]:
                source_pattern = next((p for p in patterns if p.creator_id == rel.source_id), None)
                target_pattern = next((p for p in patterns if p.creator_id == rel.target_id), None)
                
                if source_pattern and target_pattern:
                    source_pattern.add_relationship(
                        target_id=target_pattern.id,
                        relationship_type=rel.relation_type,
                        metrics={
                            "strength": rel.strength,
                            "spatial_distance": 1.0,
                            "coherence_similarity": 1.0,
                            "combined_strength": rel.strength
                        }
                    )
        
        return patterns
    
    def discover_pattern_relationships(
        self,
        patterns: List[PatternAdaptiveID]
    ) -> List[Dict]:
        """Discover and validate relationships between patterns."""
        relationships = []
        
        for p1 in patterns:
            for p2 in patterns:
                if p1 == p2:
                    continue
                    
                # Calculate relationship metrics
                temporal_alignment = self._calculate_temporal_alignment(p1, p2)
                causal_strength = self._calculate_causal_strength(p1, p2)
                
                if temporal_alignment > 0.5 or causal_strength > 0.5:
                    relationship = {
                        "source": p1.pattern_type,
                        "target": p2.pattern_type,
                        "metrics": {
                            "temporal_alignment": temporal_alignment,
                            "causal_strength": causal_strength
                        }
                    }
                    
                    # Validate relationship
                    rel_result = self.validator.validate_relationship(
                        SemanticRelation(
                            source_id=p1.id,
                            target_id=p2.id,
                            relation_type="temporal" if temporal_alignment > causal_strength else "causal",
                            strength=max(temporal_alignment, causal_strength),
                            evidence=[]
                        )
                    )
                    self.validator.log_validation(rel_result)
                    
                    if rel_result.status != ValidationStatus.RED:
                        relationships.append(relationship)
        
        return relationships
    
    def track_temporal_evolution(
        self,
        patterns: List[Dict],
        temporal_nodes: List[TemporalNode]
    ) -> List[List[Dict]]:
        """Track pattern evolution across time periods."""
        evolution_chains = []
        
        # Group patterns by type
        pattern_groups = {}
        for pattern in patterns:
            if pattern.pattern_type not in pattern_groups:
                pattern_groups[pattern.pattern_type] = []
            pattern_groups[pattern.pattern_type].append(pattern)
        
        # Build evolution chains
        for pattern_type, group in pattern_groups.items():
            chain = []
            for temporal_node in sorted(temporal_nodes, key=lambda x: x.year):
                matching_patterns = [
                    p for p in group
                    if hasattr(p, 'temporal_context') and 
                    json.loads(p.temporal_context).get(temporal_node.period) == temporal_node.year
                ]
                
                if matching_patterns:
                    chain.append({
                        "year": temporal_node.year,
                        "period": temporal_node.period,
                        "pattern": matching_patterns[0]
                    })
            
            if len(chain) > 1:
                evolution_chains.append(chain)
        
        return evolution_chains
    
    def _calculate_temporal_alignment(self, p1: PatternAdaptiveID, p2: PatternAdaptiveID) -> float:
        """Calculate temporal alignment between patterns."""
        # For temporal patterns, check hazard_type alignment
        if p1.pattern_type == "temporal" and p2.pattern_type == "temporal":
            # Perfect alignment if same hazard_type
            return 1.0 if p1.hazard_type == p2.hazard_type else 0.0
        
        # For event patterns, check hazard_type alignment
        elif p1.pattern_type == "event" and p2.pattern_type == "event":
            return 1.0 if p1.hazard_type == p2.hazard_type else 0.0
        
        # For mixed patterns (temporal and event)
        else:
            # Use confidence as a measure of alignment
            return min(p1.confidence, p2.confidence)
    
    def _calculate_causal_strength(self, p1: PatternAdaptiveID, p2: PatternAdaptiveID) -> float:
        """Calculate causal relationship strength between patterns."""
        # Example causation rules (can be expanded)
        causation_rules = {
            ("drought", "wildfire"): 0.8,
            ("extreme_precipitation", "storm_surge"): 0.7,
            ("drought", "extratropical_storms"): 0.5,
            ("precipitation", "flood"): 0.7,
            ("temporal", "event"): 0.6,  # Temporal context influences events
            ("event", "temporal"): 0.4   # Events can shape temporal patterns
        }
        
        # Get the key types for lookup
        type1 = p1.pattern_type
        type2 = p2.pattern_type
        
        # Check direct causation
        if (type1, type2) in causation_rules:
            return causation_rules[(type1, type2)]
        
        # Check hazard type causation
        if p1.pattern_type == "event" and p2.pattern_type == "event":
            if (p1.hazard_type, p2.hazard_type) in causation_rules:
                return causation_rules[(p1.hazard_type, p2.hazard_type)]
        
        return 0.0
    
    def export_to_neo4j(self, patterns: List[PatternAdaptiveID]) -> Dict:
        """Export patterns and relationships to Neo4j format."""
        nodes = []
        relationships = []
        
        # Create nodes
        for pattern in patterns:
            node = {
                "id": pattern._base_id,
                "type": getattr(pattern, 'pattern_type', 'unknown'),
                "hazard_type": getattr(pattern, 'hazard_type', 'unknown'),
                "confidence": pattern._stability_score,
                "temporal_horizon": getattr(pattern, 'temporal_horizon', 'current'),
                "probability": getattr(pattern, 'probability', 1.0),
                "spatial_context": getattr(pattern, 'spatial_context', 'MarthasVineyard'),
                "temporal_context": json.loads(pattern.temporal_context) if isinstance(pattern.temporal_context, str) else pattern.temporal_context,
                "metrics": {
                    "stability": pattern._stability_score,
                    "pressure": pattern._pressure_level
                }
            }
            nodes.append(node)
            
            # Create relationships
            for connected_id in pattern._connected_patterns:
                strength = pattern._relationship_strengths.get(connected_id, 0.0)
                relationship = {
                    "source": pattern._base_id,
                    "target": connected_id,
                    "type": "RELATED_TO",
                    "properties": {
                        "strength": strength,
                        "created_at": datetime.now().isoformat()
                    },
                    "color": "#ff0000" if strength > 0.7 else "#0000ff",
                    "width": 2 if strength > 0.7 else 1,
                    "arrow": "triangle"
                }
                relationships.append(relationship)
        
        return {
            "nodes": nodes,
            "relationships": relationships
        }
    
    def visualize_test_structure(self, patterns: List[PatternAdaptiveID]) -> Dict:
        """Create Neo4j visualization of test structure."""
        # Export patterns to Neo4j format
        graph_data = self.export_to_neo4j(patterns)
        
        # Add visualization metadata
        for node in graph_data["nodes"]:
            if node["type"] == "test_group":
                node["color"] = "#4CAF50"  # Green for test groups
                node["size"] = 40
            else:
                node["color"] = "#2196F3"  # Blue for test cases
                node["size"] = 30
        
        for rel in graph_data["relationships"]:
            if rel["type"] == "CONTAINS":
                rel["color"] = "#4CAF50"  # Green for contains
                rel["width"] = 2
            elif rel["type"] == "PRECEDES":
                rel["color"] = "#FFC107"  # Yellow for precedes
                rel["width"] = 1
            rel["arrow"] = "triangle"
        
        return graph_data
    
    def generate_neo4j_query(self, graph_data: Dict) -> str:
        """Generate Neo4j Cypher query to create the graph."""
        query_parts = []
        
        # Create nodes
        for node in graph_data["nodes"]:
            props = {
                "id": node["id"],
                "type": node["type"],
                "hazard_type": node["hazard_type"],
                "confidence": node["confidence"],
                "temporal_context": node["temporal_context"],
                "metrics": node["metrics"]
            }
            props_str = ', '.join(f'{k}: {json.dumps(v)}' for k, v in props.items())
            query_parts.append(f"CREATE (n:Pattern {{ {props_str} }})")        
        
        # Create relationships
        for rel in graph_data["relationships"]:
            props = {
                "type": rel["type"],
                **rel["properties"]
            }
            props_str = ', '.join(f'{k}: {json.dumps(v)}' for k, v in props.items())
            query_parts.append(
                f"MATCH (s:Pattern {{id: '{rel['source']}'}}), (t:Pattern {{id: '{rel['target']}'}})"
                f" CREATE (s)-[r:{rel['type']} {{ {props_str} }}]->(t)")
        
        return "\n".join(query_parts)


def test_neo4j_visualization():
    """Test Neo4j visualization capabilities."""
    # Initialize visualizer
    visualizer = SemanticPatternVisualizer()
    
    # Create test patterns
    patterns = [
        AdaptiveId(
            "drought_2025",  # base_id
            initial_state=WindowState.OPEN,
            observer_id="test_observer"
        ),
        AdaptiveId(
            "wildfire_2025",  # base_id
            initial_state=WindowState.OPEN,
            observer_id="test_observer"
        ),
        AdaptiveId(
            "storm_surge_2025",  # base_id
            initial_state=WindowState.OPEN,
            observer_id="test_observer"
        )
    ]
    
    # Set pattern properties
    for pattern in patterns:
        pattern._pattern_type = "event"
        pattern._confidence = 0.8
        pattern.temporal_context = json.dumps({"current": 2025})
        pattern._field_state = 0.7
        pattern._coherence = 0.9
        pattern._energy_state = 0.6
    
    patterns[0]._hazard_type = "drought"
    patterns[1]._hazard_type = "wildfire"
    patterns[2]._hazard_type = "storm_surge"
    
    # Create relationships
    patterns[0].connect_pattern(patterns[1]._base_id, 0.8)  # drought -> wildfire
    patterns[1].connect_pattern(patterns[2]._base_id, 0.6)  # wildfire -> storm_surge
    
    # Export to Neo4j format
    graph_data = visualizer.export_to_neo4j(patterns)
    
    # Verify nodes
    assert len(graph_data["nodes"]) == 3
    for node in graph_data["nodes"]:
        assert "id" in node
        assert "type" in node
        assert "hazard_type" in node
        assert "confidence" in node
        assert "temporal_context" in node
        assert "metrics" in node
        assert node["confidence"] == 1.0  # Default confidence from PatternAdaptiveID
        assert node["temporal_context"] == {"current": 2025}
    
    # Verify relationships
    assert len(graph_data["relationships"]) == 2
    for rel in graph_data["relationships"]:
        assert "source" in rel
        assert "target" in rel
        assert "type" in rel
        assert "properties" in rel
        assert "strength" in rel["properties"]
        assert "created_at" in rel["properties"]
    
    # Test visualization metadata
    viz_data = visualizer.visualize_test_structure(patterns)
    for node in viz_data["nodes"]:
        assert "color" in node
        assert "size" in node
    
    for rel in viz_data["relationships"]:
        assert "color" in rel
        assert "width" in rel
        assert "arrow" in rel
    
    # Test Neo4j query generation
    query = visualizer.generate_neo4j_query(graph_data)
    assert isinstance(query, str)
    assert "CREATE" in query
    assert "MATCH" in query
    assert "Pattern" in query


@pytest.fixture
def semantic_graph_selection():
    """Create semantic graph from Martha's Vineyard text selection."""
    current_time = datetime.now()
    
    temporal_nodes = [
        TemporalNode(period="current", year=2025, id="current"),
        TemporalNode(period="mid_century", year=2050, id="mid_century"),
        TemporalNode(period="late_century", year=2075, id="late_century")
    ]
    
    event_nodes = [
        EventNode(
            id="rainfall_100yr",
            event_type="extreme_precipitation",
            metrics={
                "current_probability": 1.0,
                "mid_increase": 1.2,
                "late_increase": 1.5
            }
        ),
        EventNode(
            id="drought_severe",
            event_type="drought",
            metrics={
                "current_probability": 0.8,
                "mid_increase": 1.3,
                "late_increase": 1.8
            }
        ),
        EventNode(
            id="wildfire_high",
            event_type="wildfire",
            metrics={
                "current_probability": 0.6,
                "mid_increase": 1.4,
                "late_increase": 2.0
            }
        )
    ]
    
    relations = [
        SemanticRelation(
            source_id="drought_severe",
            target_id="wildfire_high",
            relation_type="increases_probability",
            strength=0.8,
            evidence=["historical correlation", "scientific studies"]
        ),
        SemanticRelation(
            source_id="rainfall_100yr",
            target_id="drought_severe",
            relation_type="decreases_probability",
            strength=0.6,
            evidence=["moisture availability", "soil conditions"]
        )
    ]
    
    return {
        "temporal_nodes": temporal_nodes,
        "event_nodes": event_nodes,
        "relations": relations,
        "created_at": current_time,
        "location": "Martha's Vineyard"
    }

def test_semantic_pattern_discovery(semantic_graph_selection):
    """Test pattern discovery with validation."""
    visualizer = SemanticPatternVisualizer()
    
    # Extract and validate patterns
    patterns = visualizer.extract_patterns_from_semantic_graph(
        semantic_graph_selection
    )
    assert len(patterns) > 0
    
    # Get UI status
    ui_status = visualizer.validator.get_ui_status()
    assert ui_status["status"] in ValidationStatus
    
    # Discover relationships
    relationships = visualizer.discover_pattern_relationships(patterns)
    assert len(relationships) > 0
    
    # Track evolution
    evolution_chains = visualizer.track_temporal_evolution(
        patterns,
        semantic_graph_selection["temporal_nodes"]
    )
    assert len(evolution_chains) > 0
    
    # Export to Neo4j
    visualizer.export_pattern_graph_to_neo4j(
        patterns=patterns,
        field=np.zeros((20, 20))  # Placeholder field
    )

def test_validation_status_tracking(semantic_graph_selection):
    """Test validation status tracking and UI updates."""
    visualizer = SemanticPatternVisualizer()
    
    # Initial status should be green
    initial_status = visualizer.validator.get_ui_status()
    assert initial_status["status"] == ValidationStatus.GREEN
    
    # Introduce an invalid node
    invalid_event = EventNode(
        id="invalid",
        event_type="unknown",
        metrics={}
    )
    invalid_event.confidence = 0.5  # Set confidence after creation
    semantic_graph_selection["event_nodes"].append(invalid_event)
    
    # Extract patterns (should trigger validation warnings)
    try:
        visualizer.extract_patterns_from_semantic_graph(semantic_graph_selection)
    except ValueError:
        # Expected error due to invalid node
        pass
    
    # Check updated status - should be RED due to invalid event type
    updated_status = visualizer.validator.get_ui_status()
    assert updated_status["status"] == ValidationStatus.RED
    assert "validation_history" in dir(visualizer.validator)

def test_dimensional_pattern_emergence():
    """Test natural pattern emergence through dimensional context."""
    from ...adaptive_core.dimensional_context import DimensionalContext, DimensionType, WindowState
    
    # Initialize dimensional context
    context = DimensionalContext()
    
    # Test pattern emergence across dimensions
    observation = {
        "location": "coastal_zone",
        "time": "2024-02",
        "system": "climate",
        "event": "storm_surge",
        "impact": "flooding",
        "severity": 0.8
    }
    
    # Observe pattern and check results
    results = context.observe_pattern(observation)
    
    # Verify dimensional activation
    assert len(results) > 0, "Should detect patterns in at least one dimension"
    
    # Check that spatial dimension recognized location-based patterns
    spatial_result = results.get(DimensionType.SPATIAL.value)
    assert spatial_result is not None, "Should detect spatial patterns"
    assert spatial_result['suggestions'][0]['concept'] == 'location'
    assert 'coastal_zone' in spatial_result['suggestions'][0]['potential_alignments']
    
    # Check boundary tension and window state
    assert spatial_result['boundary_tension'] > 0, "Should have non-zero boundary tension"
    assert spatial_result['window_state'] == WindowState.OPENING.value
    
    # Test pattern evolution over time
    for _ in range(3):
        new_observation = {
            "location": "coastal_zone",
            "time": "2024-03",
            "system": "climate",
            "event": "erosion",
            "impact": "infrastructure_damage",
            "severity": 0.6
        }
        # Use observe() instead of update()
        results = context.observe_pattern(new_observation)
    
    # Get evolution summary
    summary = context.get_evolution_summary()
    
    # Verify pattern evolution
    assert len(summary['active_dimensions']) > 0, "Should have active dimensions"
    assert summary['total_observations'] == 4, "Should have recorded all observations"
    assert any(tension > 0.3 for tension in summary['boundary_tensions'].values()), "At least one dimension should have significant boundary tension"

def test_abstract_visualization():
    """Test visualization of test structure as a graph."""
    # Create test structure nodes
    test_nodes = [
        SemanticNode(id="pattern_validation", type="test_group"),
        SemanticNode(id="evolution_tracking", type="test_group"),
        SemanticNode(id="neo4j_integration", type="test_group"),
        
        # Pattern validation tests
        SemanticNode(id="event_type_validation", type="test_case"),
        SemanticNode(id="temporal_context_validation", type="test_case"),
        SemanticNode(id="relationship_validation", type="test_case"),
        
        # Evolution tracking tests
        SemanticNode(id="pattern_transition", type="test_case"),
        SemanticNode(id="temporal_alignment", type="test_case"),
        SemanticNode(id="causal_strength", type="test_case"),
        
        # Neo4j integration tests
        SemanticNode(id="graph_structure", type="test_case"),
        SemanticNode(id="pattern_persistence", type="test_case"),
        SemanticNode(id="relationship_preservation", type="test_case")
    ]
    
    # Create relationships between tests
    test_relations = [
        # Pattern validation relationships
        SemanticRelation(
            source_id="pattern_validation",
            target_id="event_type_validation",
            relation_type="CONTAINS",
            strength=1.0,
            evidence=["Validates climate hazard event types"]
        ),
        SemanticRelation(
            source_id="pattern_validation",
            target_id="temporal_context_validation",
            relation_type="CONTAINS",
            strength=1.0,
            evidence=["Validates temporal context fields"]
        ),
        SemanticRelation(
            source_id="pattern_validation",
            target_id="relationship_validation",
            relation_type="CONTAINS",
            strength=1.0,
            evidence=["Validates pattern relationships"]
        ),
        
        # Evolution tracking relationships
        SemanticRelation(
            source_id="evolution_tracking",
            target_id="pattern_transition",
            relation_type="CONTAINS",
            strength=1.0,
            evidence=["Tracks pattern state transitions"]
        ),
        SemanticRelation(
            source_id="evolution_tracking",
            target_id="temporal_alignment",
            relation_type="CONTAINS",
            strength=1.0,
            evidence=["Tracks temporal alignment between patterns"]
        ),
        SemanticRelation(
            source_id="evolution_tracking",
            target_id="causal_strength",
            relation_type="CONTAINS",
            strength=1.0,
            evidence=["Monitors causal relationship strength"]
        ),
        
        # Neo4j integration relationships
        SemanticRelation(
            source_id="neo4j_integration",
            target_id="graph_structure",
            relation_type="CONTAINS",
            strength=1.0,
            evidence=["Validates Neo4j graph structure"]
        ),
        SemanticRelation(
            source_id="neo4j_integration",
            target_id="pattern_persistence",
            relation_type="CONTAINS",
            strength=1.0,
            evidence=["Tests pattern persistence in Neo4j"]
        ),
        SemanticRelation(
            source_id="neo4j_integration",
            target_id="relationship_preservation",
            relation_type="CONTAINS",
            strength=1.0,
            evidence=["Tests relationship preservation in Neo4j"]
        ),
        
        # Cross-group relationships
        SemanticRelation(
            source_id="pattern_validation",
            target_id="evolution_tracking",
            relation_type="PRECEDES",
            strength=0.8,
            evidence=["Validation must pass before evolution"]
        ),
        SemanticRelation(
            source_id="evolution_tracking",
            target_id="neo4j_integration",
            relation_type="PRECEDES",
            strength=0.8,
            evidence=["Evolution must be tracked before persistence"]
        )
    ]
    
    # Create temporal nodes
    temporal_nodes = [
        TemporalNode(period="current", year=2025, id="current"),
        TemporalNode(period="mid_century", year=2050, id="mid_century"),
        TemporalNode(period="late_century", year=2075, id="late_century")
    ]
    
    # Create event nodes
    event_nodes = [
        EventNode(
            id="drought_severe",
            event_type="drought",
            metrics={
                "current_probability": 0.8,
                "mid_increase": 1.3,
                "late_increase": 1.8
            }
        ),
        EventNode(
            id="wildfire_high",
            event_type="wildfire",
            metrics={
                "current_probability": 0.6,
                "mid_increase": 1.4,
                "late_increase": 2.0
            }
        )
    ]
    
    # Create semantic graph
    semantic_graph = {
        "relations": test_relations,
        "temporal_nodes": temporal_nodes,
        "event_nodes": event_nodes,
        "created_at": datetime.now(),
        "location": "Test Environment"
    }
    
    visualizer = SemanticPatternVisualizer()
    patterns = visualizer.extract_patterns_from_semantic_graph(semantic_graph)
    
    # Discover and validate relationships
    visualizer.discover_pattern_relationships(patterns)
    
    # Verify test structure
    assert len(patterns) == len(temporal_nodes) + len(event_nodes)
    
    # Verify relationships
    validation_result = visualizer.validator.get_ui_status()
    assert validation_result["status"] == ValidationStatus.GREEN
