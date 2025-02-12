"""
Core pattern types and data structures.
"""
from typing import Dict, Any, TypedDict, List
from dataclasses import dataclass

class Pattern(TypedDict):
    """Pattern data structure."""
    id: str
    coherence: float
    energy: float
    state: str
    metrics: Dict[str, float]
    relationships: List[str]

@dataclass
class FieldGradients:
    """Field gradient measurements."""
    coherence: float
    energy: float
    density: float
    turbulence: float

@dataclass
class FieldState:
    """Current state of the field."""
    gradients: FieldGradients
    patterns: List[Pattern]
    timestamp: float
