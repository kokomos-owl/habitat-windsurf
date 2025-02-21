"""Tests for semantic pattern discovery and evolution in visualization."""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from habitat_evolution.visualization.semantic_validation import (
    SemanticValidator,
    ValidationStatus,
    ValidationResult
)

from habitat_evolution.visualization.test_visualization import (
    TestVisualizationConfig,
    TestPatternVisualizer
)

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

class AdaptiveId:
    """Base class for adaptive pattern identification with window state management."""
    def __init__(self, base_id: str, initial_state: WindowState = WindowState.CLOSED):
        self._base_id = base_id
        self._window_state = initial_state
        self._evolution_history = []
        self._stability_score = 1.0
        self._pressure_level = 0.0
        
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
    
    @property
    def current_id(self) -> str:
        return f"{self._base_id}_{len(self._evolution_history)}"
    
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
    
    def evolve(self, pressure: float, stability: float):
        """Record evolution step with pressure and stability metrics."""
        self._evolution_history.append({
            "timestamp": datetime.now(),
            "pressure": pressure,
            "stability": stability,
            "window_state": self._window_state.value,
            "connected_patterns": list(self._connected_patterns)
        })
        
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

def test_semantic_potential_evolution():
    """Test the natural evolution of semantic potential in pattern discovery."""
    # Initialize test nodes
    drought = AdaptiveId("drought_risk")
    rainfall = AdaptiveId("rainfall_pattern")
    wildfire = AdaptiveId("wildfire_risk")
    
    # Observe natural pattern emergence
    observations = []
    
    # Phase 1: Initial Pattern Suggestion (CLOSED state)
    drought_props = drought.connect_pattern(
        pattern_id=rainfall.current_id,
        relationship_strength=0.4,
        pattern_type="precipitation_impact"
    )
    observations.append({
        "phase": "suggestion",
        "window_state": drought._window_state.value,
        "potentials": drought_props["potential_patterns"],
        "gradients": drought_props["semantic_gradients"]
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
        "potentials": props["potential_patterns"],
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
        if 'window_state' in obs:
            print(f"Window State: {obs['window_state']}")
        
        print("\nPotential Patterns:")
        for pattern in obs['potentials']:
            print(f"  - {pattern}")
        
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
    """Base class for semantic graph nodes with adaptive identification."""
    id: str
    type: str
    created_at: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    
    def __post_init__(self):
        self.adaptive_id = AdaptiveId(self.id)
        self._coherence_score = 0.0
        self._stability_threshold = 0.8  # From success criteria
    
    def update_coherence(self, score: float):
        """Update coherence score and evolve adaptive ID if needed."""
        self._coherence_score = score
        pressure = 1.0 - score if score < self._stability_threshold else 0.0
        self.adaptive_id.evolve(pressure=pressure, stability=score)

class TemporalNode(SemanticNode):
    """Represents a temporal context."""
    def __init__(self, period: str, year: int, id: str):
        super().__init__(id=id, type="temporal")
        self.period = period
        self.year = year
    
class EventNode(SemanticNode):
    """Represents climate events."""
    def __init__(self, id: str, event_type: str, metrics: Dict[str, float]):
        super().__init__(id=id, type="event")
        self.event_type = event_type
        self.metrics = metrics

@dataclass
class SemanticRelation:
    """Represents relationships between nodes."""
    source_id: str
    target_id: str
    relation_type: str
    strength: float
    evidence: List[str]

class SemanticPatternVisualizer(TestPatternVisualizer):
    """Enhanced visualizer with semantic pattern capabilities."""
    
    def __init__(self, config: Optional[TestVisualizationConfig] = None):
        super().__init__(config)
        self.validator = SemanticValidator()
        
    def extract_patterns_from_semantic_graph(
        self, 
        semantic_graph: Dict
    ) -> List[Dict]:
        """Extract patterns from semantic graph with validation."""
        patterns = []
        
        # Validate temporal nodes
        temporal_result = self.validator.validate_temporal_sequence(
            semantic_graph["temporal_nodes"]
        )
        self.validator.log_validation(temporal_result)
        
        if temporal_result.status == ValidationStatus.RED:
            raise ValueError(f"Invalid temporal sequence: {temporal_result.message}")
        
        # Process event nodes
        for event in semantic_graph["event_nodes"]:
            # Validate node structure
            node_result = self.validator.validate_node_structure(event)
            self.validator.log_validation(node_result)
            
            if node_result.status == ValidationStatus.RED:
                continue
                
            # Extract pattern from event
            pattern = {
                "type": event.event_type,
                "metrics": {
                    "confidence": event.confidence,
                    **event.metrics
                },
                "temporal_context": {
                    period: next(
                        t for t in semantic_graph["temporal_nodes"]
                        if t.period == period
                    ).year
                    for period in ["current", "mid_century", "late_century"]
                }
            }
            patterns.append(pattern)
            
        return patterns
    
    def discover_pattern_relationships(
        self,
        patterns: List[Dict]
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
                        "source": p1["type"],
                        "target": p2["type"],
                        "metrics": {
                            "temporal_alignment": temporal_alignment,
                            "causal_strength": causal_strength
                        }
                    }
                    
                    # Validate relationship
                    rel_result = self.validator.validate_relationship(
                        SemanticRelation(
                            source_id=p1["type"],
                            target_id=p2["type"],
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
            if pattern["type"] not in pattern_groups:
                pattern_groups[pattern["type"]] = []
            pattern_groups[pattern["type"]].append(pattern)
        
        # Build evolution chains
        for pattern_type, group in pattern_groups.items():
            chain = []
            for temporal_node in sorted(temporal_nodes, key=lambda x: x.year):
                matching_patterns = [
                    p for p in group
                    if p["temporal_context"][temporal_node.period] == temporal_node.year
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
    
    def _calculate_temporal_alignment(self, p1: Dict, p2: Dict) -> float:
        """Calculate temporal alignment between patterns."""
        periods = ["current", "mid_century", "late_century"]
        alignments = []
        
        for period in periods:
            year = p1["temporal_context"][period]
            if year == p2["temporal_context"][period]:
                alignments.append(1.0)
            else:
                alignments.append(0.0)
                
        return sum(alignments) / len(alignments)
    
    def _calculate_causal_strength(self, p1: Dict, p2: Dict) -> float:
        """Calculate causal relationship strength between patterns."""
        # Example causation rules (can be expanded)
        causation_rules = {
            ("drought", "wildfire"): 0.8,
            ("precipitation", "flood"): 0.7
        }
        
        return causation_rules.get((p1["type"], p2["type"]), 0.0)

@pytest.fixture
def semantic_graph_selection():
    """Create semantic graph from Martha's Vineyard text selection."""
    current_time = datetime.now()
    
    temporal_nodes = [
        TemporalNode(
            id="current",
            type="temporal",
            period="current",
            year=2025,
            created_at=current_time
        ),
        TemporalNode(
            id="mid_century",
            type="temporal",
            period="mid_century",
            year=2050,
            created_at=current_time
        ),
        TemporalNode(
            id="late_century",
            type="temporal",
            period="late_century",
            year=2100,
            created_at=current_time
        )
    ]
    
    event_nodes = [
        EventNode(
            id="rainfall_100yr",
            type="event",
            event_type="precipitation",
            metrics={
                "current_probability": 1.0,
                "mid_increase": 1.2,
                "late_multiplier": 5.0
            },
            created_at=current_time
        ),
        EventNode(
            id="drought",
            type="event",
            event_type="drought",
            metrics={
                "current_likelihood": 0.085,
                "mid_likelihood": 0.13,
                "late_likelihood": 0.26
            },
            created_at=current_time
        ),
        EventNode(
            id="wildfire",
            type="event",
            event_type="wildfire",
            metrics={
                "current_baseline": 1.0,
                "mid_increase": 1.44,
                "late_increase": 1.94
            },
            created_at=current_time
        )
    ]
    
    relations = [
        SemanticRelation(
            source_id="drought",
            target_id="wildfire",
            relation_type="causation",
            strength=0.8,
            evidence=["Linked to this increase in drought stress"]
        ),
        SemanticRelation(
            source_id="rainfall_100yr",
            target_id="flood_risk",
            relation_type="correlation",
            strength=0.9,
            evidence=["useful indicator of flood risk"]
        )
    ]
    
    return {
        "temporal_nodes": temporal_nodes,
        "event_nodes": event_nodes,
        "relations": relations
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
        type="event",
        event_type="unknown",
        metrics={},
        created_at=datetime.now()
    )
    semantic_graph_selection["event_nodes"].append(invalid_event)
    
    # Extract patterns (should trigger validation warnings)
    visualizer.extract_patterns_from_semantic_graph(semantic_graph_selection)
    
    # Check updated status
    updated_status = visualizer.validator.get_ui_status()
    assert updated_status["status"] in [ValidationStatus.YELLOW, ValidationStatus.RED]
    assert "validation_history" in dir(visualizer.validator)
