"""Tests for observing semantic pattern discovery behavior."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from habitat_evolution.visualization.semantic_validation import (
    SemanticValidator,
    ValidationStatus,
    ValidationResult
)

@dataclass
class ObservationFrame:
    """Base class for observation frames."""
    start_time: datetime
    duration: timedelta
    observations: List[Dict] = field(default_factory=list)
    
    def record(self, observation: Dict):
        """Record an observation with timestamp."""
        self.observations.append({
            "timestamp": datetime.now(),
            **observation
        })
    
    def within_window(self) -> bool:
        """Check if current time is within observation window."""
        now = datetime.now()
        return self.start_time <= now <= (self.start_time + self.duration)

@dataclass
class PatternEmergenceFrame(ObservationFrame):
    """Observes pattern emergence from semantic structure."""
    
    def observe_emergence(self, pattern: Dict):
        """Record pattern emergence observation."""
        if not self.within_window():
            return
            
        self.record({
            "type": "emergence",
            "pattern_type": pattern["type"],
            "confidence": pattern.get("metrics", {}).get("confidence", 0.0),
            "context": pattern.get("temporal_context", {})
        })
    
    def get_emergence_timeline(self) -> List[Dict]:
        """Get timeline of pattern emergence."""
        return sorted(self.observations, key=lambda x: x["timestamp"])

@dataclass
class RelationshipDiscoveryFrame(ObservationFrame):
    """Observes relationship discovery between patterns."""
    
    def observe_relationship(self, relationship: Dict):
        """Record relationship discovery observation."""
        if not self.within_window():
            return
            
        self.record({
            "type": "relationship",
            "source": relationship["source"],
            "target": relationship["target"],
            "metrics": relationship["metrics"]
        })
    
    def get_relationship_strength_evolution(self) -> Dict[str, List[float]]:
        """Get evolution of relationship strengths over time."""
        strengths = {}
        for obs in self.observations:
            key = f"{obs['source']}->{obs['target']}"
            if key not in strengths:
                strengths[key] = []
            strengths[key].append(obs["metrics"]["causal_strength"])
        return strengths

@dataclass
class ValidationStateFrame(ObservationFrame):
    """Observes validation state changes."""
    
    def observe_validation(self, result: ValidationResult):
        """Record validation state observation."""
        if not self.within_window():
            return
            
        self.record({
            "type": "validation",
            "status": result.status.value,
            "message": result.message,
            "context": result.context
        })
    
    def get_validation_sequence(self) -> List[str]:
        """Get sequence of validation states."""
        return [obs["status"] for obs in self.observations]

class SemanticObserver:
    """Coordinates observation frames for semantic pattern discovery."""
    
    def __init__(self, observation_duration: timedelta = timedelta(hours=1)):
        self.start_time = datetime.now()
        self.duration = observation_duration
        
        # Initialize observation frames
        self.frames = {
            "emergence": PatternEmergenceFrame(self.start_time, self.duration),
            "relationship": RelationshipDiscoveryFrame(self.start_time, self.duration),
            "validation": ValidationStateFrame(self.start_time, self.duration)
        }
    
    def observe_pattern_emergence(self, pattern: Dict):
        """Record pattern emergence observation."""
        self.frames["emergence"].observe_emergence(pattern)
    
    def observe_relationship_discovery(self, relationship: Dict):
        """Record relationship discovery observation."""
        self.frames["relationship"].observe_relationship(relationship)
    
    def observe_validation_state(self, result: ValidationResult):
        """Record validation state observation."""
        self.frames["validation"].observe_validation(result)
    
    def get_observation_summary(self) -> Dict:
        """Get summary of all observations."""
        return {
            "start_time": self.start_time.isoformat(),
            "duration": str(self.duration),
            "pattern_emergence": self.frames["emergence"].get_emergence_timeline(),
            "relationships": self.frames["relationship"].get_relationship_strength_evolution(),
            "validation_sequence": self.frames["validation"].get_validation_sequence()
        }

@pytest.fixture
def semantic_observer():
    """Create semantic observer for testing."""
    return SemanticObserver(observation_duration=timedelta(minutes=5))

def test_pattern_emergence_observation(semantic_observer, semantic_graph_selection):
    """Test observation of pattern emergence process."""
    visualizer = SemanticPatternVisualizer()
    
    # Observe pattern extraction
    patterns = []
    for event in semantic_graph_selection["event_nodes"]:
        pattern = visualizer.extract_patterns_from_semantic_graph({
            "temporal_nodes": semantic_graph_selection["temporal_nodes"],
            "event_nodes": [event],
            "relations": []
        })[0]
        semantic_observer.observe_pattern_emergence(pattern)
        patterns.append(pattern)
    
    # Validate observations
    summary = semantic_observer.get_observation_summary()
    emergence_timeline = summary["pattern_emergence"]
    
    assert len(emergence_timeline) == len(semantic_graph_selection["event_nodes"])
    assert all(obs["type"] == "emergence" for obs in emergence_timeline)
    assert all(obs["confidence"] > 0 for obs in emergence_timeline)

def test_relationship_discovery_observation(semantic_observer, semantic_graph_selection):
    """Test observation of relationship discovery process."""
    visualizer = SemanticPatternVisualizer()
    
    # Extract patterns first
    patterns = visualizer.extract_patterns_from_semantic_graph(semantic_graph_selection)
    
    # Observe relationship discovery
    for relationship in visualizer.discover_pattern_relationships(patterns):
        semantic_observer.observe_relationship_discovery(relationship)
    
    # Validate observations
    summary = semantic_observer.get_observation_summary()
    relationship_evolution = summary["relationships"]
    
    assert len(relationship_evolution) > 0
    assert all(len(strengths) > 0 for strengths in relationship_evolution.values())
    assert all(0 <= strength <= 1 for strengths in relationship_evolution.values() 
              for strength in strengths)

def test_validation_state_observation(semantic_observer):
    """Test observation of validation state changes."""
    validator = SemanticValidator()
    
    # Create some test validation results
    results = [
        ValidationResult(
            status=ValidationStatus.GREEN,
            message="Initial validation",
            context={"phase": "start"}
        ),
        ValidationResult(
            status=ValidationStatus.YELLOW,
            message="Warning condition",
            context={"phase": "processing"}
        ),
        ValidationResult(
            status=ValidationStatus.GREEN,
            message="Resolution",
            context={"phase": "end"}
        )
    ]
    
    # Observe validation states
    for result in results:
        semantic_observer.observe_validation_state(result)
    
    # Validate observations
    summary = semantic_observer.get_observation_summary()
    validation_sequence = summary["validation_sequence"]
    
    assert len(validation_sequence) == len(results)
    assert validation_sequence == ["success", "warning", "success"]
