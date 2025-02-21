"""Tests for Neo4j persistence of intuitive resonance patterns and pattern lifecycle."""

import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any

from habitat_evolution.pattern_aware_rag.services.neo4j_service import Neo4jStateStore
from visualization.core.db.neo4j_client import Neo4jConfig
from habitat_evolution.tests.visualization.test_semantic_pattern_visualization import (
    UserContext, AdaptiveId, PatternIntuition, WindowState, SemanticPotential
)

class PatternObservation:
    def __init__(self, text: str):
        self.text = text
        self.semantic_potential = SemanticPotential()
        self.pressure_threshold = 0.3
        self.stability_threshold = 0.7
        
    def _calculate_semantic_pressure(self, window: AdaptiveId) -> float:
        frame = self._current_semantic_frame(window)
        pressure = 0.0
        
        # Pressure from temporal markers
        if "mid-century" in self.text or "late-century" in self.text:
            pressure += 0.35
            
        # Pressure from quantitative changes
        if "increase" in self.text or "more likely" in self.text:
            pressure += 0.25
            
        # Pressure from historical context
        if "historical" in self.text:
            pressure += 0.2
            
        return min(pressure, 1.0)
        
    def _calculate_stability(self, window: AdaptiveId) -> float:
        # Base stability
        stability = 0.8
        
        # Reduce for uncertainty markers
        if "expected" in self.text or "about" in self.text:
            stability -= 0.1
            
        return max(0.0, min(stability, 1.0))
        
    def _current_semantic_frame(self, window: AdaptiveId) -> Dict:
        frame = {
            "temporal_markers": [],
            "relationships": [],
            "attention_points": []
        }
        
        # Find temporal markers
        if "mid-century" in self.text:
            frame["temporal_markers"].append("mid-century")
        if "late-century" in self.text:
            frame["temporal_markers"].append("late-century")
            
        # Find attention points
        if "100-year rainfall event" in self.text:
            frame["attention_points"].append("historical-baseline")
        if "five times more likely" in self.text:
            frame["attention_points"].append("significant-increase")
            
        # Find relationships
        if "increase" in self.text:
            frame["relationships"].append("temporal-progression")
            
        return frame
        
    def _extract_core_observable(self, window: AdaptiveId) -> Dict:
        return {
            "base_concept": window._base_id,
            "state": window._window_state.value,
            "connected_patterns": list(window._connected_patterns)
        }

import pytest
from datetime import datetime
from typing import Dict, List, Any

from habitat_evolution.pattern_aware_rag.services.neo4j_service import Neo4jStateStore
from visualization.core.db.neo4j_client import Neo4jConfig
from habitat_evolution.tests.visualization.test_semantic_pattern_visualization import (
    UserContext, AdaptiveId, PatternIntuition
)

class ResonanceNetwork:
    """Manages a network of resonating patterns."""
    
    def __init__(self):
        self.patterns: Dict[str, PatternIntuition] = {}
        self.resonance_edges: List[Dict[str, Any]] = []
        
    def add_pattern(self, pattern_id: str, intuition: PatternIntuition):
        """Add a pattern to the network."""
        self.patterns[pattern_id] = intuition
        
    def add_resonance(self, source_id: str, target_id: str, strength: float):
        """Add a resonance connection between patterns."""
        self.resonance_edges.append({
            "source": source_id,
            "target": target_id,
            "strength": strength,
            "timestamp": datetime.now()
        })
        
    def _extract_semantic_context(self, hints: List[Dict]) -> Dict[str, List[str]]:
        """Extract semantic context from pattern hints."""
        context = {
            'spatial': [],
            'temporal': [],
            'metrics': [],
            'phenomena': []
        }
        
        for hint in hints:
            if 'period' in hint or 'timeframe' in hint:
                context['temporal'].extend([v for v in hint.values() if isinstance(v, str)])
            if 'location' in hint or 'area' in hint:
                context['spatial'].extend([v for v in hint.values() if isinstance(v, str)])
            if 'metric' in hint or 'measure' in hint:
                context['metrics'].extend([v for v in hint.values() if isinstance(v, str)])
            if 'phenomenon' in hint or 'hazard' in hint:
                context['phenomena'].extend([v for v in hint.values() if isinstance(v, str)])
        
        return context
    
    def _infer_concept_type(self, base_concept: str) -> str:
        """Infer the type of concept based on its semantic pattern."""
        # Broad semantic patterns rather than specific domains
        change_patterns = {'intensification', 'reduction', 'transformation', 'cycle'}
        system_patterns = {'threshold', 'cascade', 'feedback', 'network'}
        space_patterns = {'local', 'regional', 'distributed', 'concentrated'}
        time_patterns = {'periodic', 'progressive', 'acute', 'chronic'}
        interaction_patterns = {'coupling', 'dependency', 'influence', 'response'}
        
        # Focus on pattern type rather than domain specifics
        if any(pattern in base_concept for pattern in change_patterns):
            return 'change_pattern'
        elif any(pattern in base_concept for pattern in system_patterns):
            return 'system_pattern'
        elif any(pattern in base_concept for pattern in space_patterns):
            return 'space_pattern'
        elif any(pattern in base_concept for pattern in time_patterns):
            return 'time_pattern'
        elif any(pattern in base_concept for pattern in interaction_patterns):
            return 'interaction_pattern'
        return 'emerging_pattern'  # Allow new pattern types to emerge
    
    def to_neo4j_structure(self) -> Dict[str, Any]:
        """Convert resonance network to Neo4j-compatible structure."""
        nodes = []
        relationships = []
        
        # Convert patterns to nodes
        for pattern_id, intuition in self.patterns.items():
            # Extract semantic dimensions from pattern_id and hints
            base_concept = pattern_id.split('_')[0]  # e.g. 'drought' from 'drought_risk'
            semantic_context = self._extract_semantic_context(intuition.hints)
            
            nodes.append({
                "id": pattern_id,
                "type": "Pattern",
                # Core pattern properties
                "base_concept": base_concept,
                "concept_type": self._infer_concept_type(base_concept),
                "pattern_class": pattern_id.split('_')[1] if len(pattern_id.split('_')) > 1 else None,
                
                # Semantic dimensions
                "spatial_context": semantic_context.get('spatial', []),  # e.g. ['Martha\'s Vineyard', 'coastal areas']
                "temporal_context": semantic_context.get('temporal', []),  # e.g. ['2050s', 'mid-century']
                "impact_metrics": semantic_context.get('metrics', []),  # e.g. ['rainfall probability', '5x more likely']
                "related_phenomena": semantic_context.get('phenomena', []),  # e.g. ['drought', 'wildfire']
                
                # Evolution tracking
                "resonance_state": intuition.intuition_state,
                "resonance_strength": intuition.resonance_strength,
                "encounter_count": intuition.encounter_count,
                "last_resonance": intuition.last_resonance.isoformat(),
                "hints": [hint["spatial_context"] for hint in intuition.hints],
                "observations": [hint["scale_observation"] if "scale_observation" in hint else hint["local_observation"] if "local_observation" in hint else hint["regional_observation"] for hint in intuition.hints],
                "place_connections": [hint["place_connection"] for hint in intuition.hints],
                "spatial_scales": [hint["spatial_scale"] for hint in intuition.hints],
                "stability": intuition.stability if hasattr(intuition, "stability") else None,
                "created_at": datetime.now().isoformat()
            })
            
        # Convert resonance edges to relationships
        for edge in self.resonance_edges:
            relationships.append({
                "source": edge["source"],
                "target": edge["target"],
                "type": "RESONATES_WITH",
                "strength": edge["strength"],
                "timestamp": edge["timestamp"].isoformat()
            })
            
        return {
            "nodes": nodes,
            "relationships": relationships
        }

@pytest.mark.asyncio
async def test_resonance_network_persistence():
    """Test persisting a resonance network to Neo4j."""
    # Create user context
    user = UserContext("researcher_1")
    
    # Create temporal context patterns
    mid_century = AdaptiveId("mid_century", observer_id=user.hash)
    late_century = AdaptiveId("late_century", observer_id=user.hash)
    
    # Create impact patterns
    rainfall_event = AdaptiveId("100year_rainfall", observer_id=user.hash)
    drought = AdaptiveId("drought_risk", observer_id=user.hash)
    wildfire = AdaptiveId("wildfire_danger", observer_id=user.hash)
    flood = AdaptiveId("flood_risk", observer_id=user.hash)
    storms = AdaptiveId("extratropical_storms", observer_id=user.hash)
    
    # Build resonance network
    network = ResonanceNetwork()
    
    # Add temporal contexts with strong resonance
    for pattern, hint in [
        (mid_century, {"period": "mid-century", "timeframe": "2050s"}),
        (late_century, {"period": "late-century", "timeframe": "2080s"})
    ]:
        pattern_id = pattern.current_id
        intuition = user.resonate_with_pattern(
            pattern_id,
            hint={"feeling": f"Clear temporal anchor: {hint['period']}"}
        )
        network.add_pattern(pattern_id, intuition)
    
    # Add impact patterns with initial glimpses
    for pattern, hint in [
        (rainfall_event, {"metric": "100-year rainfall probability", "mid_change": "slight increase", "late_change": "5x more likely"}),
        (drought, {"metric": "annual likelihood", "mid_change": "13%", "late_change": "26%"}),
        (wildfire, {"metric": "high-danger days", "mid_change": "44% increase", "late_change": "94% increase"}),
        (flood, {"metric": "risk indicator", "evidence": "linked to rainfall events"}),
        (storms, {"metric": "frequency", "evidence": "increase in extratropical storms"})
    ]:
        pattern_id = pattern.current_id
        intuition = user.resonate_with_pattern(
            pattern_id,
            hint=hint
        )
        network.add_pattern(pattern_id, intuition)
    
    # Create temporal impact connections
    for period in [mid_century, late_century]:
        for impact in [rainfall_event, drought, wildfire, flood, storms]:
            network.add_resonance(
                period.current_id,
                impact.current_id,
                strength=0.8  # Strong temporal relationships
            )
    
    # Create causal impact connections
    network.add_resonance(rainfall_event.current_id, flood.current_id, strength=0.9)  # Rainfall -> Flood
    network.add_resonance(drought.current_id, wildfire.current_id, strength=0.85)  # Drought -> Wildfire
    
    # Evolve patterns to show progression
    for i in range(2):
        for pattern in [rainfall_event, drought, wildfire, flood, storms]:
            hint = {
                "feeling": f"Deepening resonance {i+1}",
                "connection_type": "harmony",
                "intensity": 0.6 + i*0.1
            }
            pattern.evolve(
                pressure=0.5,
                stability=0.7,
                observer_insight=hint
            )
            pattern_id = pattern.current_id
            intuition = user.resonate_with_pattern(pattern_id, hint)
            network.add_pattern(pattern_id, intuition)
            
            # Create resonance edges with other patterns that have resonated
            resonant_patterns = [p for p in [rainfall_event, drought, wildfire, flood, storms] if p != pattern]
            for other in resonant_patterns:
                network.add_resonance(
                    pattern_id,
                    other.current_id,
                    strength=0.5 + i*0.1
                )
    
    # Convert to Neo4j structure
    graph_data = network.to_neo4j_structure()
    
    # Verify structure
    assert len(graph_data["nodes"]) > 0, "Should have pattern nodes"
    assert len(graph_data["relationships"]) > 0, "Should have resonance relationships"
    
    # Check node structure
    for node in graph_data["nodes"]:
        assert "resonance_state" in node, "Node should have resonance state"
        assert "resonance_strength" in node, "Node should have resonance strength"
        assert "hints" in node, "Node should have hints list"
        
    # Check relationship structure
    for rel in graph_data["relationships"]:
        assert rel["type"] == "RESONATES_WITH", "Should use RESONATES_WITH relationship"
        assert 0.0 <= rel["strength"] <= 1.0, "Resonance strength should be 0-1"
        
    # Store in Neo4j
    config = Neo4jConfig(
        username="neo4j",
        password="habitat123"
    )
    neo4j = Neo4jStateStore(config=config)
    state_id = await neo4j.store_graph_state(graph_data)
    assert state_id, "Should get valid state ID"
    
    # Retrieve and validate
    stored_state = await neo4j.get_graph_state(state_id)
    assert stored_state, "Should retrieve stored state"
    
    print("\nResonance Network Structure:")
    print(f"Nodes: {len(graph_data['nodes'])}")
    print(f"Relationships: {len(graph_data['relationships'])}")
    print("\nSample Pattern Node:")
    print(graph_data["nodes"][0])
    print("\nSample Resonance Edge:")
    print(graph_data["relationships"][0])

@pytest.mark.asyncio
async def test_pattern_lifecycle_with_feedback():
    """Test the fundamental pattern lifecycle with feedback at each stage."""
    # Initialize base patterns with rich semantic context
    network = ResonanceNetwork()
    
    # Initialize patterns from spatial-temporal context
    now = datetime.now()
    
    # Local intensification pattern (Martha's Vineyard context)
    local_pattern = PatternIntuition(
        pattern_id="local_intensification",
        intuition_state="glimpse",
        resonance_strength=0.3,
        encounter_count=1,
        last_resonance=now,
        hints=[{
            "spatial_context": "Martha's Vineyard",
            "local_observation": "flood risk increasing",
            "place_connection": "resident vulnerability",
            "spatial_scale": "island community",
            "local_impact": "community resilience needs"
        }]
    )
    
    # Regional pattern (Northeast context)
    regional_pattern = PatternIntuition(
        pattern_id="regional_dynamics",
        intuition_state="resonance",
        resonance_strength=0.6,
        encounter_count=3,
        last_resonance=now,
        hints=[{
            "spatial_context": "Northeast region",
            "regional_observation": "extratropical storms increasing",
            "place_connection": "coastal vulnerability",
            "spatial_scale": "regional impacts",
            "regional_impact": "storm surge patterns"
        }]
    )
    
    # Cross-scale pattern (Local-Regional interactions)
    cross_scale_pattern = PatternIntuition(
        pattern_id="cross_scale_interaction",
        intuition_state="attunement",
        resonance_strength=0.9,
        encounter_count=5,
        last_resonance=now,
        hints=[{
            "spatial_context": "Local-Regional interface",
            "scale_observation": "local risks amplified by regional changes",
            "place_connection": "nested vulnerabilities",
            "spatial_scale": "multi-scale dynamics",
            "cross_scale_impact": "compound risk emergence"
        }]
    )
    
    # Add spatially-aware patterns to network
    network.add_pattern("local_intensification", local_pattern)
    network.add_pattern("regional_dynamics", regional_pattern)
    network.add_pattern("cross_scale_interaction", cross_scale_pattern)
    
    # Create resonance connections showing spatial pattern evolution
    network.add_resonance("local_intensification", "regional_dynamics", 0.7)  # local patterns inform regional understanding
    network.add_resonance("regional_dynamics", "cross_scale_interaction", 0.8)  # regional context reveals cross-scale patterns
    network.add_resonance("cross_scale_interaction", "local_intensification", 0.6)  # cross-scale insights reshape local perception
    
    # Convert to Neo4j structure
    graph_data = network.to_neo4j_structure()
    
    # Store in Neo4j
    config = Neo4jConfig(
        username="neo4j",
        password="habitat123"
    )
    neo4j = Neo4jStateStore(config=config)
    state_id = await neo4j.store_graph_state(graph_data)
    
    # Verify structure and evolution potential
    stored_state = await neo4j.get_graph_state(state_id)
    assert stored_state, "Should retrieve stored state"
    
    # Validate semantic richness
    nodes = graph_data["nodes"]
    for node in nodes:
        assert "base_concept" in node, "Node should have base concept"
        assert "concept_type" in node, "Node should have concept type"
        assert "spatial_context" in node, "Node should have spatial context"
        assert "temporal_context" in node, "Node should have temporal context"
        assert "impact_metrics" in node, "Node should have impact metrics"
    
    # Validate evolution potential
    relationships = graph_data["relationships"]
    assert len(relationships) >= 3, "Should have sufficient connections for evolution"
    
    print("\nPattern Evolution Network:")
    print(f"Base Patterns: {len(nodes)}")
    print(f"Evolution Connections: {len(relationships)}")
    print("\nSample Pattern Node:")
    print(nodes[0])
    
    # Verify pattern evolution potential
    print("\nPattern Evolution Analysis:")
    for rel in graph_data["relationships"]:
        print(f"\nEvolution: {rel['source']} -> {rel['target']}")
        print(f"Strength: {rel['strength']}")
        print(f"Timestamp: {rel['timestamp']}")
    


