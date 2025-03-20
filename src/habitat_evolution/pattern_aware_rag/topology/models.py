"""
Topology models for representing semantic landscape features.

These models define the core data structures for representing topological features
of the semantic landscape, including frequency domains, boundaries, resonance points,
and field dynamics metrics.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
import json
import uuid


def serialize_datetime(obj):
    """Helper function to serialize datetime objects to JSON."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


@dataclass
class FrequencyDomain:
    """Represents a region of the semantic landscape with consistent frequency characteristics."""
    
    id: str = field(default_factory=lambda: f"fd-{uuid.uuid4()}")
    dominant_frequency: float = 0.0  # Primary frequency in this domain
    bandwidth: float = 0.0  # Range of frequencies (standard deviation)
    phase_coherence: float = 0.0  # Degree of phase alignment (0-1)
    center_coordinates: Tuple[float, ...] = field(default_factory=tuple)  # N-dimensional coordinates
    radius: float = 0.0  # Approximate radius of influence
    pattern_ids: Set[str] = field(default_factory=set)  # Patterns in this domain
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional properties


@dataclass
class Boundary:
    """Represents a boundary between frequency domains in the semantic landscape."""
    
    id: str = field(default_factory=lambda: f"b-{uuid.uuid4()}")
    domain_ids: Tuple[str, str] = field(default_factory=tuple)  # IDs of domains this boundary separates
    sharpness: float = 0.0  # Rate of change across boundary (0-1)
    permeability: float = 0.0  # Ease of pattern movement across boundary (0-1)
    stability: float = 0.0  # Temporal persistence of boundary (0-1)
    dimensionality: int = 0  # Number of dimensions this boundary spans
    coordinates: List[Tuple[float, ...]] = field(default_factory=list)  # Points defining boundary
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional properties


@dataclass
class ResonancePoint:
    """Represents a point of constructive interference in the semantic landscape."""
    
    id: str = field(default_factory=lambda: f"r-{uuid.uuid4()}")
    coordinates: Tuple[float, ...] = field(default_factory=tuple)  # N-dimensional coordinates
    strength: float = 0.0  # Amplitude of harmonic peak (0-1)
    stability: float = 0.0  # Resistance to perturbation (0-1)
    attractor_radius: float = 0.0  # Radius of influence
    contributing_pattern_ids: Dict[str, float] = field(default_factory=dict)  # Pattern IDs and weights
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional properties


@dataclass
class FieldMetrics:
    """Metrics describing the overall state of the semantic field."""
    
    coherence: float = 0.0  # Global organization (0-1)
    energy_density: Dict[str, float] = field(default_factory=dict)  # Energy by region
    adaptation_rate: float = 0.0  # Speed of response to new patterns
    homeostasis_index: float = 0.0  # Ability to maintain stable state (0-1)
    entropy: float = 0.0  # Measure of disorder in the field
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional properties


@dataclass
class TopologyState:
    """Complete state of the semantic topology at a point in time."""
    
    id: str = field(default_factory=lambda: f"ts-{uuid.uuid4()}")
    frequency_domains: Dict[str, FrequencyDomain] = field(default_factory=dict)
    boundaries: Dict[str, Boundary] = field(default_factory=dict)
    resonance_points: Dict[str, ResonancePoint] = field(default_factory=dict)
    field_metrics: FieldMetrics = field(default_factory=FieldMetrics)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional properties
    
    def to_json(self) -> str:
        """Serialize the topology state to JSON."""
        return json.dumps(asdict(self), default=serialize_datetime)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TopologyState':
        """Deserialize a topology state from JSON."""
        data = json.loads(json_str)
        
        # Convert dictionaries back to proper objects
        if 'frequency_domains' in data:
            data['frequency_domains'] = {
                k: FrequencyDomain(**v) for k, v in data['frequency_domains'].items()
            }
        
        if 'boundaries' in data:
            data['boundaries'] = {
                k: Boundary(**v) for k, v in data['boundaries'].items()
            }
        
        if 'resonance_points' in data:
            data['resonance_points'] = {
                k: ResonancePoint(**v) for k, v in data['resonance_points'].items()
            }
        
        if 'field_metrics' in data:
            data['field_metrics'] = FieldMetrics(**data['field_metrics'])
        
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        return cls(**data)
    
    def diff(self, other: 'TopologyState') -> Dict[str, Any]:
        """Calculate difference between two topology states."""
        diff_result = {
            'added_domains': {},
            'removed_domains': {},
            'modified_domains': {},
            'added_boundaries': {},
            'removed_boundaries': {},
            'modified_boundaries': {},
            'added_resonance_points': {},
            'removed_resonance_points': {},
            'modified_resonance_points': {},
            'field_metrics_changes': {}
        }
        
        # Compare domains
        for domain_id, domain in self.frequency_domains.items():
            if domain_id not in other.frequency_domains:
                diff_result['added_domains'][domain_id] = domain
            elif domain != other.frequency_domains[domain_id]:
                diff_result['modified_domains'][domain_id] = {
                    'before': other.frequency_domains[domain_id],
                    'after': domain
                }
        
        for domain_id in other.frequency_domains:
            if domain_id not in self.frequency_domains:
                diff_result['removed_domains'][domain_id] = other.frequency_domains[domain_id]
        
        # Similar comparisons for boundaries and resonance points
        # ...
        
        # Compare field metrics
        for attr, value in asdict(self.field_metrics).items():
            if attr != 'timestamp' and attr != 'metadata':
                other_value = getattr(other.field_metrics, attr)
                if value != other_value:
                    diff_result['field_metrics_changes'][attr] = {
                        'before': other_value,
                        'after': value
                    }
        
        return diff_result


@dataclass
class TopologyDiff:
    """Represents changes between two topology states."""
    
    from_state_id: str
    to_state_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    added_domains: Dict[str, FrequencyDomain] = field(default_factory=dict)
    removed_domains: Dict[str, FrequencyDomain] = field(default_factory=dict)
    modified_domains: Dict[str, Dict[str, FrequencyDomain]] = field(default_factory=dict)
    added_boundaries: Dict[str, Boundary] = field(default_factory=dict)
    removed_boundaries: Dict[str, Boundary] = field(default_factory=dict)
    modified_boundaries: Dict[str, Dict[str, Boundary]] = field(default_factory=dict)
    added_resonance_points: Dict[str, ResonancePoint] = field(default_factory=dict)
    removed_resonance_points: Dict[str, ResonancePoint] = field(default_factory=dict)
    modified_resonance_points: Dict[str, Dict[str, ResonancePoint]] = field(default_factory=dict)
    field_metrics_changes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
