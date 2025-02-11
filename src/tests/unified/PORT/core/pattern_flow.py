"""Pattern flow types and analysis for structure-meaning evolution."""

from enum import Enum
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
import logging
import re
import uuid

from .graph_schema import RelationType as GraphRelationType

logger = logging.getLogger(__name__)

class FlowType(Enum):
    """Core flow types for pattern evolution aligned with graph schema."""
    
    # Primed patterns (from graph schema)
    MEASURED_BY = GraphRelationType.MEASURED_BY.value
    IMPACTS = GraphRelationType.IMPACTS.value
    EVOLVES_TO = GraphRelationType.EVOLVES_TO.value
    
    # Essential patterns (from graph schema)
    ADAPTS_TO = GraphRelationType.ADAPTS_TO.value
    EMERGES_FROM = GraphRelationType.EMERGES_FROM.value
    
    # Natural patterns (emergent flow)
    CONTAINS = GraphRelationType.CONTAINS.value
    RELATED_TO = GraphRelationType.RELATED_TO.value
    DERIVED_FROM = GraphRelationType.DERIVED_FROM.value
    
    # Interface recognition and adherence
    ADHERES_TO = GraphRelationType.ADHERES_TO.value
    RECOGNIZES = GraphRelationType.RECOGNIZES.value

@dataclass
class FlowState:
    """State of a pattern flow with climate-specific attributes."""
    strength: float = 0.0            # Current flow strength
    velocity: float = 0.0            # Rate of pattern evolution
    coherence: float = 0.0           # Flow consistency
    emergence_potential: float = 0.0  # Potential for new patterns
    temporal_scope: str = ""         # e.g., "mid-century", "late-century"
    confidence_interval: Optional[tuple] = None  # For measurement uncertainties

class ClimatePatternMatcher:
    """Matches climate-specific patterns in text."""
    
    def __init__(self):
        # Measurement patterns from climate assessment
        self.measurement_patterns = [
            (r"(\d+(?:\.\d+)?%?)(?:\s*[-–]\s*(\d+(?:\.\d+)?%?))?\s*(?:increase|decrease|change)\s+in\s+(.+?)(?=\s*[,.])", 0.9),
            (r"(\w+)\s+(?:is|are)\s+measured\s+(?:at|by)\s+(\d+(?:\.\d+)?%?)", 0.85),
            (r"between\s+(\d+(?:\.\d+)?%?)\s+and\s+(\d+(?:\.\d+)?%?)\s+of\s+(?:the|all)\s+(.+?)(?=\s*[,.])", 0.8)
        ]
        
        # Impact patterns from climate assessment
        self.impact_patterns = [
            (r"(putting|puts|put)\s+(.+?)\s+at\s+risk", 0.9),
            (r"(impacts?|affects?|influences?)\s+(.+?)\s+(?:through|by|via)\s+(.+?)(?=\s*[,.])", 0.85),
            (r"(leads?|leading)\s+to\s+(.+?)(?=\s*[,.])", 0.8)
        ]
        
        # Evolution patterns from climate assessment
        self.evolution_patterns = [
            (r"(?:increase|decrease)\s+(\d+(?:\.\d+)?%?)\s+by\s+(mid|late)-century", 0.9),
            (r"(?:from|between)\s+(.+?)\s+to\s+(.+?)\s+by\s+(mid|late)-century", 0.85),
            (r"expected\s+to\s+(?:increase|decrease)\s+(.+?)\s+by\s+(.+?)(?=\s*[,.])", 0.8)
        ]

class PatternFlow:
    """Manages pattern flows in structure-meaning space."""
    
    def __init__(self):
        self.active_flows: Dict[str, FlowState] = {}
        self.emerging_flows: Dict[str, FlowState] = {}
        self.pattern_matcher = ClimatePatternMatcher()
        self.interface_recognitions: Dict[str, Set[str]] = {}
        self.adherence_patterns: Dict[str, Dict[str, float]] = {}
        
    def register_flow(
        self,
        flow_type: FlowType,
        initial_state: Optional[FlowState] = None,
        interface_recognitions: Optional[Dict[str, float]] = None
    ) -> str:
        """Register a new pattern flow with interface recognition."""
        flow_id = str(uuid.uuid4())
        state = initial_state or FlowState()
        
        self.active_flows[flow_id] = state
        
        # Register interface recognitions if provided
        if interface_recognitions:
            for interface, strength in interface_recognitions.items():
                self.register_interface_recognition(
                    flow_id,
                    interface,
                    strength
                )
                
        return flow_id
        
    def register_interface_recognition(
        self,
        source_pattern: str,
        target_interface: str,
        recognition_strength: float = 0.0
    ):
        """Register interface recognition with adherence tracking."""
        if source_pattern not in self.interface_recognitions:
            self.interface_recognitions[source_pattern] = set()
            
        self.interface_recognitions[source_pattern].add(target_interface)
        
        # Track adherence pattern
        if source_pattern not in self.adherence_patterns:
            self.adherence_patterns[source_pattern] = {}
            
        self.adherence_patterns[source_pattern][target_interface] = recognition_strength
        
    def get_adherence_strength(
        self,
        pattern: str,
        interface: str
    ) -> float:
        """Get interface adherence strength."""
        return self.adherence_patterns.get(pattern, {}).get(interface, 0.0)
        
    def update_adherence(
        self,
        pattern: str,
        interface: str,
        new_strength: float
    ):
        """Update interface adherence strength."""
        if pattern not in self.adherence_patterns:
            self.adherence_patterns[pattern] = {}
        self.adherence_patterns[pattern][interface] = new_strength
        
    def get_recognized_interfaces(self, pattern: str) -> Set[str]:
        """Get all interfaces recognized by a pattern."""
        return self.interface_recognitions.get(pattern, set())
        
    def update_flow(
        self,
        flow_type: FlowType,
        state_updates: Dict[str, Any]
    ) -> None:
        """Update flow state with new measurements."""
        if flow_type.name not in self.active_flows:
            logger.warning(f"Flow {flow_type.name} not registered")
            return
            
        flow_state = self.active_flows[flow_type.name]
        for attr, value in state_updates.items():
            if hasattr(flow_state, attr):
                setattr(flow_state, attr, value)
                
    def register_emerging_flow(
        self,
        pattern_type: str,
        initial_state: Optional[FlowState] = None
    ) -> None:
        """Register a newly emerging pattern flow."""
        if pattern_type in self.emerging_flows:
            logger.warning(f"Emerging flow {pattern_type} already registered")
            return
            
        self.emerging_flows[pattern_type] = initial_state or FlowState()
        
    def extract_climate_patterns(self, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract climate-specific patterns from content."""
        patterns = {
            "measurements": [],
            "impacts": [],
            "evolution": []
        }
        
        # Extract measurements
        for pattern, confidence in self.pattern_matcher.measurement_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                patterns["measurements"].append({
                    "type": FlowType.MEASURED_BY.value,
                    "groups": match.groups(),
                    "confidence": confidence,
                    "text": match.group(0)
                })
        
        # Extract impacts
        for pattern, confidence in self.pattern_matcher.impact_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                patterns["impacts"].append({
                    "type": FlowType.IMPACTS.value,
                    "groups": match.groups(),
                    "confidence": confidence,
                    "text": match.group(0)
                })
        
        # Extract evolution
        for pattern, confidence in self.pattern_matcher.evolution_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                patterns["evolution"].append({
                    "type": FlowType.EVOLVES_TO.value,
                    "groups": match.groups(),
                    "confidence": confidence,
                    "text": match.group(0)
                })
        
        return patterns
        
    def get_flow_dynamics(self) -> Dict[str, Any]:
        """Get current flow dynamics."""
        return {
            "active_flows": [
                {
                    "type": flow_type,
                    "state": dataclasses.asdict(state)
                }
                for flow_type, state in self.active_flows.items()
            ],
            "emerging_flows": [
                {
                    "type": flow_type,
                    "state": dataclasses.asdict(state)
                }
                for flow_type, state in self.emerging_flows.items()
            ]
        }
        
    def calculate_flow_scores(self) -> Dict[str, Any]:
        """Calculate overall flow scores."""
        active_strengths = [
            state.strength for state in self.active_flows.values()
        ]
        emerging_strengths = [
            state.strength for state in self.emerging_flows.values()
        ]
        
        return {
            "coherence": sum(active_strengths) / len(active_strengths) if active_strengths else 0.0,
            "emergence": sum(emerging_strengths) / len(emerging_strengths) if emerging_strengths else 0.0,
            "dynamics": {
                "primed_flow": max(active_strengths) if active_strengths else 0.0,
                "natural_flow": max(emerging_strengths) if emerging_strengths else 0.0,
                "flow_integration": self._calculate_integration()
            }
        }
        
    def _calculate_integration(self) -> float:
        """Calculate how well primed and emerging flows are integrated."""
        if not self.active_flows or not self.emerging_flows:
            return 0.0
            
        active_coherence = sum(
            state.coherence for state in self.active_flows.values()
        ) / len(self.active_flows)
        
        emerging_coherence = sum(
            state.coherence for state in self.emerging_flows.values()
        ) / len(self.emerging_flows)
        
        return (active_coherence + emerging_coherence) / 2.0
