"""
Pattern flow types and analysis for structure-meaning evolution.

This module implements pattern flow analysis with a focus on natural emergence.
It provides mechanisms for analyzing how patterns naturally emerge, merge,
transform, and maintain over time.

Key Components:
    - FlowType: Core flow types aligned with natural emergence
    - FlowState: Natural state of pattern flow
    - PatternFlow: Main class managing natural pattern flows
    - PatternMatcher: Natural pattern recognition

The system focuses on:
    - Light observation of pattern flow
    - Natural interface recognition
    - Organic pattern matching
    - Unforced pattern evolution

Typical usage:
    1. Initialize PatternFlow
    2. Register natural flows
    3. Allow patterns to emerge
    4. Observe interface recognition
    5. Track natural adherence
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import re
import uuid

logger = logging.getLogger(__name__)

class FlowType(Enum):
    """Core flow types for natural pattern evolution."""
    
    # Natural emergence patterns
    EMERGES_FROM = "emerges_from"  # Natural pattern emergence
    ADAPTS_TO = "adapts_to"      # Natural adaptation
    EVOLVES_TO = "evolves_to"    # Natural evolution
    
    # Natural relationship patterns
    CONTAINS = "contains"        # Natural containment
    RELATED_TO = "related_to"    # Natural relationship
    DERIVED_FROM = "derived_from"  # Natural derivation
    
    # Natural interface patterns
    ADHERES_TO = "adheres_to"    # Natural adherence
    RECOGNIZES = "recognizes"    # Natural recognition

@dataclass
class FlowState:
    """Natural state of pattern flow."""
    strength: float = 0.0            # Natural flow strength
    velocity: float = 0.0            # Natural evolution rate
    coherence: float = 0.0           # Natural consistency
    emergence_potential: float = 0.0  # Natural emergence potential
    confidence: float = 0.0          # Natural confidence
    temporal_context: str = ""       # Natural temporal context

class PatternMatcher:
    """Matches natural patterns in content."""
    
    def __init__(self):
        # Natural emergence patterns
        self.emergence_patterns = [
            (r"naturally\s+emerges?\s+from\s+(.+?)(?=\s*[,.])", 0.9),
            (r"(?:forms?|appears?)\s+naturally\s+(?:in|within)\s+(.+?)(?=\s*[,.])", 0.85),
            (r"emerges?\s+organically\s+through\s+(.+?)(?=\s*[,.])", 0.8)
        ]
        
        # Natural adaptation patterns
        self.adaptation_patterns = [
            (r"naturally\s+adapts?\s+to\s+(.+?)(?=\s*[,.])", 0.9),
            (r"(?:changes?|evolves?)\s+with\s+(.+?)(?=\s*[,.])", 0.85),
            (r"responds?\s+to\s+(.+?)\s+by\s+(.+?)(?=\s*[,.])", 0.8)
        ]
        
        # Natural evolution patterns
        self.evolution_patterns = [
            (r"gradually\s+(?:becomes?|transforms?)\s+(.+?)(?=\s*[,.])", 0.9),
            (r"naturally\s+develops?\s+into\s+(.+?)(?=\s*[,.])", 0.85),
            (r"evolves?\s+through\s+(.+?)\s+into\s+(.+?)(?=\s*[,.])", 0.8)
        ]

class PatternFlow:
    """Manages natural pattern flows."""
    
    def __init__(self):
        self.active_flows: Dict[str, FlowState] = {}
        self.emerging_flows: Dict[str, FlowState] = {}
        self.pattern_matcher = PatternMatcher()
        self.interface_recognitions: Dict[str, Set[str]] = {}
        self.adherence_patterns: Dict[str, Dict[str, float]] = {}
        
    def register_flow(
        self,
        flow_type: FlowType,
        initial_state: Optional[FlowState] = None,
        interface_recognitions: Optional[Dict[str, float]] = None
    ) -> str:
        """Register a new natural pattern flow."""
        flow_id = str(uuid.uuid4())
        state = initial_state or FlowState()
        
        self.active_flows[flow_id] = state
        
        # Register natural interface recognitions
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
        """Register natural interface recognition."""
        if source_pattern not in self.interface_recognitions:
            self.interface_recognitions[source_pattern] = set()
            
        self.interface_recognitions[source_pattern].add(target_interface)
        
        # Track natural adherence
        if source_pattern not in self.adherence_patterns:
            self.adherence_patterns[source_pattern] = {}
            
        self.adherence_patterns[source_pattern][target_interface] = recognition_strength
        
    def get_adherence_strength(
        self,
        pattern: str,
        interface: str
    ) -> float:
        """Get natural interface adherence strength."""
        return self.adherence_patterns.get(pattern, {}).get(interface, 0.0)
        
    def update_adherence(
        self,
        pattern: str,
        interface: str,
        new_strength: float
    ):
        """Update natural interface adherence strength."""
        if pattern not in self.adherence_patterns:
            self.adherence_patterns[pattern] = {}
        self.adherence_patterns[pattern][interface] = new_strength
        
    def get_recognized_interfaces(self, pattern: str) -> Set[str]:
        """Get naturally recognized interfaces."""
        return self.interface_recognitions.get(pattern, set())
        
    def update_flow(
        self,
        flow_type: FlowType,
        state_updates: Dict[str, Any]
    ) -> None:
        """Update natural flow state."""
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
        """Register a naturally emerging pattern flow."""
        if pattern_type in self.emerging_flows:
            logger.warning(f"Emerging flow {pattern_type} already registered")
            return
            
        self.emerging_flows[pattern_type] = initial_state or FlowState()
        
    def extract_patterns(self, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract natural patterns from content."""
        patterns = {
            "emergence": [],
            "adaptation": [],
            "evolution": []
        }
        
        # Extract natural emergence
        for pattern, confidence in self.pattern_matcher.emergence_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                patterns["emergence"].append({
                    "type": FlowType.EMERGES_FROM.value,
                    "groups": match.groups(),
                    "confidence": confidence,
                    "text": match.group(0)
                })
        
        # Extract natural adaptation
        for pattern, confidence in self.pattern_matcher.adaptation_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                patterns["adaptation"].append({
                    "type": FlowType.ADAPTS_TO.value,
                    "groups": match.groups(),
                    "confidence": confidence,
                    "text": match.group(0)
                })
        
        # Extract natural evolution
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
