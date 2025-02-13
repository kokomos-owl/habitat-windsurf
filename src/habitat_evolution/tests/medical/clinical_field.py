"""Clinical field implementation for medical knowledge evolution."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

@dataclass
class VitalSign:
    """Vital sign measurement with metadata."""
    name: str
    value: float
    timestamp: datetime
    unit: str
    normal_range: tuple[float, float]

@dataclass
class LabResult:
    """Laboratory result with metadata."""
    name: str
    value: float
    timestamp: datetime
    unit: str
    normal_range: tuple[float, float]
    critical_range: Optional[tuple[float, float]] = None

@dataclass
class ClinicalEvent:
    """Clinical event (medication, procedure, etc)."""
    event_type: str
    name: str
    timestamp: datetime
    duration: Optional[timedelta] = None
    details: Optional[Dict[str, Any]] = None

class ClinicalField:
    """
    Multi-dimensional clinical field for pattern evolution.
    
    Dimensions:
    - Temporal: Time-based evolution
    - Vitals: Physiological measurements
    - Labs: Laboratory results
    - Events: Clinical interventions
    - Outcomes: Clinical states
    """
    
    def __init__(self, 
                 time_window: timedelta,
                 vital_types: List[str],
                 lab_types: List[str],
                 event_types: List[str],
                 outcome_types: List[str]):
        """Initialize clinical field dimensions."""
        # Time resolution: 5 minutes
        self.time_steps = int(time_window.total_seconds() / 300)
        
        # Initialize field dimensions
        self.dimensions = {
            "temporal": np.zeros((self.time_steps, len(vital_types) + len(lab_types))),
            "vitals": {vital: np.zeros(100) for vital in vital_types},
            "labs": {lab: np.zeros(100) for lab in lab_types},
            "events": {event: [] for event in event_types},
            "outcomes": {outcome: 0.0 for outcome in outcome_types}
        }
        
        # Track data ranges for normalization
        self.ranges = {
            "vitals": {},
            "labs": {}
        }
        
        # Field metadata
        self.metadata = {
            "created_at": datetime.now(),
            "last_updated": datetime.now(),
            "data_points": 0,
            "patterns_detected": 0
        }
    
    def add_vital_sign(self, vital: VitalSign) -> None:
        """Add vital sign measurement to field."""
        if vital.name not in self.dimensions["vitals"]:
            raise ValueError(f"Unknown vital sign type: {vital.name}")
            
        # Update range
        if vital.name not in self.ranges["vitals"]:
            self.ranges["vitals"][vital.name] = {
                "min": vital.value,
                "max": vital.value
            }
        else:
            self.ranges["vitals"][vital.name]["min"] = min(
                self.ranges["vitals"][vital.name]["min"], 
                vital.value
            )
            self.ranges["vitals"][vital.name]["max"] = max(
                self.ranges["vitals"][vital.name]["max"], 
                vital.value
            )
        
        # Normalize value to 0-1 range
        norm_value = (vital.value - vital.normal_range[0]) / (
            vital.normal_range[1] - vital.normal_range[0]
        )
        
        # Add to temporal dimension
        time_index = self._get_time_index(vital.timestamp)
        vital_index = list(self.dimensions["vitals"].keys()).index(vital.name)
        self.dimensions["temporal"][time_index, vital_index] = norm_value
        
        # Update metadata
        self.metadata["last_updated"] = datetime.now()
        self.metadata["data_points"] += 1
    
    def add_lab_result(self, lab: LabResult) -> None:
        """Add laboratory result to field."""
        if lab.name not in self.dimensions["labs"]:
            raise ValueError(f"Unknown lab type: {lab.name}")
            
        # Update range
        if lab.name not in self.ranges["labs"]:
            self.ranges["labs"][lab.name] = {
                "min": lab.value,
                "max": lab.value
            }
        else:
            self.ranges["labs"][lab.name]["min"] = min(
                self.ranges["labs"][lab.name]["min"], 
                lab.value
            )
            self.ranges["labs"][lab.name]["max"] = max(
                self.ranges["labs"][lab.name]["max"], 
                lab.value
            )
        
        # Normalize value to 0-1 range
        norm_value = (lab.value - lab.normal_range[0]) / (
            lab.normal_range[1] - lab.normal_range[0]
        )
        
        # Add to temporal dimension
        time_index = self._get_time_index(lab.timestamp)
        lab_index = len(self.dimensions["vitals"]) + list(self.dimensions["labs"].keys()).index(lab.name)
        self.dimensions["temporal"][time_index, lab_index] = norm_value
        
        # Update metadata
        self.metadata["last_updated"] = datetime.now()
        self.metadata["data_points"] += 1
    
    def add_clinical_event(self, event: ClinicalEvent) -> None:
        """Add clinical event to field."""
        if event.event_type not in self.dimensions["events"]:
            raise ValueError(f"Unknown event type: {event.event_type}")
            
        self.dimensions["events"][event.event_type].append(event)
        
        # Update metadata
        self.metadata["last_updated"] = datetime.now()
        self.metadata["data_points"] += 1
    
    def calculate_field_state(self, timestamp: datetime) -> Dict[str, float]:
        """Calculate field state at given timestamp."""
        time_index = self._get_time_index(timestamp)
        
        # Get temporal slice
        temporal_state = self.dimensions["temporal"][time_index, :]
        
        # Calculate vital signs state
        vital_state = np.mean([
            np.mean(self.dimensions["vitals"][vital])
            for vital in self.dimensions["vitals"]
        ])
        
        # Calculate lab results state
        lab_state = np.mean([
            np.mean(self.dimensions["labs"][lab])
            for lab in self.dimensions["labs"]
        ])
        
        # Calculate event density
        event_count = sum(
            1 for events in self.dimensions["events"].values()
            for event in events
            if abs((event.timestamp - timestamp).total_seconds()) <= 300
        )
        event_state = min(1.0, event_count / 10)  # Normalize to 0-1
        
        return {
            "temporal": float(np.mean(temporal_state)),
            "vitals": float(vital_state),
            "labs": float(lab_state),
            "events": float(event_state)
        }
    
    def _get_time_index(self, timestamp: datetime) -> int:
        """Convert timestamp to temporal index."""
        start_time = self.metadata["created_at"]
        seconds_elapsed = (timestamp - start_time).total_seconds()
        return min(self.time_steps - 1, max(0, int(seconds_elapsed / 300)))
