"""Tests for semantic pattern discovery and evolution in visualization."""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from habitat_evolution.visualization.semantic_validation import (
    SemanticValidator,
    ValidationStatus,
    ValidationResult
)

from habitat_evolution.visualization.test_visualization import (
    TestVisualizationConfig,
    TestPatternVisualizer
)

@dataclass
class SemanticNode:
    """Base class for semantic graph nodes."""
    id: str
    type: str
    created_at: datetime
    confidence: float = 1.0
    
@dataclass
class TemporalNode(SemanticNode):
    """Represents a temporal context."""
    period: str
    year: int
    
@dataclass
class EventNode(SemanticNode):
    """Represents climate events."""
    event_type: str
    metrics: Dict[str, float]
    
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
