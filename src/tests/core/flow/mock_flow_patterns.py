"""
Mock patterns and states for flow testing.
Ensures consistent pattern evolution and structure-meaning relationships.
"""

from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass
from src.core.flow.habitat_flow import FlowState

@dataclass
class MockPattern:
    """Mock pattern for testing flow evolution."""
    content: str
    confidence: float
    temporal_context: Dict[str, Any]
    related_patterns: List[str]
    flow_velocity: float
    flow_direction: float
    structure_weight: float
    meaning_weight: float

class MockFlowPatterns:
    """Provides mock patterns with known evolution characteristics."""
    
    def __init__(self):
        self.patterns: Dict[str, MockPattern] = {}
        self._initialize_patterns()
        
    def _initialize_patterns(self):
        """Initialize mock patterns with known relationships."""
        self.patterns["temperature_rise"] = MockPattern(
            content="Temperature rise of 2.5°C",
            confidence=0.85,
            temporal_context={"year": 2050},
            related_patterns=["coastal_impact", "ecosystem_change"],
            flow_velocity=0.7,
            flow_direction=1.57,  # π/2 radians
            structure_weight=0.8,
            meaning_weight=0.9
        )
        
        self.patterns["coastal_impact"] = MockPattern(
            content="Coastal flooding increases",
            confidence=0.80,
            temporal_context={"year": 2050},
            related_patterns=["temperature_rise"],
            flow_velocity=0.6,
            flow_direction=2.09,  # 2π/3 radians
            structure_weight=0.7,
            meaning_weight=0.8
        )
        
        self.patterns["ecosystem_change"] = MockPattern(
            content="Ecosystem adaptation required",
            confidence=0.75,
            temporal_context={"year": 2050},
            related_patterns=["temperature_rise", "coastal_impact"],
            flow_velocity=0.5,
            flow_direction=1.05,  # π/3 radians
            structure_weight=0.6,
            meaning_weight=0.7
        )

    def get_mock_flow_state(self, pattern_id: str) -> FlowState:
        """Get a mock flow state for a pattern."""
        pattern = self.patterns.get(pattern_id)
        if not pattern:
            return FlowState()
            
        return FlowState(
            strength=pattern.confidence,
            coherence=self._calculate_mock_coherence(pattern),
            emergence_potential=self._calculate_mock_emergence(pattern),
            temporal_context=pattern.temporal_context,
            last_updated=datetime.now()
        )
    
    def _calculate_mock_coherence(self, pattern: MockPattern) -> float:
        """Calculate mock coherence based on structure-meaning weights."""
        return (pattern.structure_weight * 0.4 + 
                pattern.meaning_weight * 0.6)
    
    def _calculate_mock_emergence(self, pattern: MockPattern) -> float:
        """Calculate mock emergence based on flow characteristics."""
        return (pattern.flow_velocity * 0.4 + 
                (pattern.flow_direction % (2 * 3.14159)) / (2 * 3.14159) * 0.3 +
                pattern.confidence * 0.3)
