"""Test fixtures for sepsis cases based on MIMIC-III data."""

from datetime import datetime, timedelta
from typing import List, Dict, Any
from ..clinical_field import VitalSign, LabResult, ClinicalEvent

def create_sepsis_case(case_id: str, onset_time: datetime) -> Dict[str, Any]:
    """Create a sepsis case with progression based on Sepsis-3 criteria."""
    
    # Vital signs progression (6 hours before to 24 hours after onset)
    vitals: List[VitalSign] = []
    current_time = onset_time - timedelta(hours=6)
    
    while current_time <= onset_time + timedelta(hours=24):
        # Normal early vitals transitioning to sepsis
        if current_time < onset_time:
            # Early signs
            vitals.extend([
                VitalSign(
                    name="heart_rate",
                    value=95 + (current_time - onset_time).total_seconds() / 3600,
                    timestamp=current_time,
                    unit="bpm",
                    normal_range=(60, 100)
                ),
                VitalSign(
                    name="systolic_bp",
                    value=120 - (current_time - onset_time).total_seconds() / 7200,
                    timestamp=current_time,
                    unit="mmHg",
                    normal_range=(90, 140)
                ),
                VitalSign(
                    name="temperature",
                    value=37.5 + (current_time - onset_time).total_seconds() / 36000,
                    timestamp=current_time,
                    unit="Celsius",
                    normal_range=(36.5, 37.5)
                ),
                VitalSign(
                    name="respiratory_rate",
                    value=18 + (current_time - onset_time).total_seconds() / 7200,
                    timestamp=current_time,
                    unit="breaths/min",
                    normal_range=(12, 20)
                )
            ])
        else:
            # Sepsis criteria met
            vitals.extend([
                VitalSign(
                    name="heart_rate",
                    value=115 + (current_time - onset_time).total_seconds() / 7200,
                    timestamp=current_time,
                    unit="bpm",
                    normal_range=(60, 100)
                ),
                VitalSign(
                    name="systolic_bp",
                    value=85 - (current_time - onset_time).total_seconds() / 14400,
                    timestamp=current_time,
                    unit="mmHg",
                    normal_range=(90, 140)
                ),
                VitalSign(
                    name="temperature",
                    value=38.5 + (current_time - onset_time).total_seconds() / 72000,
                    timestamp=current_time,
                    unit="Celsius",
                    normal_range=(36.5, 37.5)
                ),
                VitalSign(
                    name="respiratory_rate",
                    value=24 + (current_time - onset_time).total_seconds() / 14400,
                    timestamp=current_time,
                    unit="breaths/min",
                    normal_range=(12, 20)
                )
            ])
        
        current_time += timedelta(minutes=30)
    
    # Lab results showing progression
    labs: List[LabResult] = [
        # Pre-sepsis labs
        LabResult(
            name="wbc",
            value=11.5,
            timestamp=onset_time - timedelta(hours=4),
            unit="K/uL",
            normal_range=(4.5, 11.0),
            critical_range=(20.0, 50.0)
        ),
        LabResult(
            name="lactate",
            value=2.1,
            timestamp=onset_time - timedelta(hours=4),
            unit="mmol/L",
            normal_range=(0.5, 2.0),
            critical_range=(4.0, 20.0)
        ),
        # Sepsis onset labs
        LabResult(
            name="wbc",
            value=16.5,
            timestamp=onset_time,
            unit="K/uL",
            normal_range=(4.5, 11.0),
            critical_range=(20.0, 50.0)
        ),
        LabResult(
            name="lactate",
            value=4.2,
            timestamp=onset_time,
            unit="mmol/L",
            normal_range=(0.5, 2.0),
            critical_range=(4.0, 20.0)
        ),
        LabResult(
            name="creatinine",
            value=1.8,
            timestamp=onset_time,
            unit="mg/dL",
            normal_range=(0.6, 1.2),
            critical_range=(3.0, 10.0)
        ),
        # Follow-up labs
        LabResult(
            name="wbc",
            value=18.2,
            timestamp=onset_time + timedelta(hours=6),
            unit="K/uL",
            normal_range=(4.5, 11.0),
            critical_range=(20.0, 50.0)
        ),
        LabResult(
            name="lactate",
            value=3.8,
            timestamp=onset_time + timedelta(hours=6),
            unit="mmol/L",
            normal_range=(0.5, 2.0),
            critical_range=(4.0, 20.0)
        )
    ]
    
    # Clinical events (interventions)
    events: List[ClinicalEvent] = [
        ClinicalEvent(
            event_type="medication",
            name="broad_spectrum_antibiotics",
            timestamp=onset_time + timedelta(minutes=45),
            details={"type": "empiric", "route": "iv"}
        ),
        ClinicalEvent(
            event_type="medication",
            name="iv_fluids",
            timestamp=onset_time + timedelta(minutes=30),
            details={"type": "crystalloid", "volume_ml": 2000}
        ),
        ClinicalEvent(
            event_type="procedure",
            name="blood_cultures",
            timestamp=onset_time + timedelta(minutes=20)
        )
    ]
    
    return {
        "case_id": case_id,
        "onset_time": onset_time,
        "vitals": vitals,
        "labs": labs,
        "events": events,
        "outcomes": {
            "mortality_risk": 0.3,
            "organ_dysfunction": True,
            "icu_admission": True
        }
    }

def load_test_sepsis_cohort(size: int = 10) -> List[Dict[str, Any]]:
    """Load a test cohort of sepsis cases."""
    base_time = datetime(2025, 1, 1, 12, 0, 0)
    return [
        create_sepsis_case(
            f"SEPSIS_{i:03d}",
            base_time + timedelta(days=i, hours=i % 24)
        )
        for i in range(size)
    ]
