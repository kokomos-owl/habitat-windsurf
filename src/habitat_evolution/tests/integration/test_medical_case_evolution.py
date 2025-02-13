"""Test medical case evolution through field patterns and knowledge emergence."""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

from habitat_evolution.core.field import FieldGradients, FlowMetrics
from habitat_evolution.core.field.gradient import GradientFlowController
from habitat_evolution.visualization.pattern_id import PatternAdaptiveID

class TestMedicalCaseEvolution:
    """Test evolution of medical case knowledge through field patterns."""
    
    @pytest.fixture
    def initial_case_data(self) -> Dict[str, Any]:
        """Initial medical case presentation."""
        return {
            "case_id": "MC2025_001",
            "presentation": {
                "symptoms": ["fatigue", "joint_pain", "fever"],
                "vitals": {
                    "temperature": 38.5,
                    "heart_rate": 88,
                    "blood_pressure": "130/85"
                },
                "lab_results": {
                    "wbc": 11.2,
                    "crp": 45,
                    "rf": "positive"
                }
            },
            "timestamps": {
                "initial": "2025-02-13T10:00:00",
                "lab_results": "2025-02-13T11:30:00"
            }
        }

    @pytest.fixture
    def field_config(self) -> Dict[str, float]:
        """Configuration for medical field dynamics."""
        return {
            "coherence_threshold": 0.6,
            "energy_decay": 0.1,
            "interaction_range": 2.0,
            "turbulence_factor": 0.3
        }

    def test_pattern_emergence_from_symptoms(self, initial_case_data, field_config):
        """Test how symptom patterns emerge in the field."""
        # Create pattern for symptom cluster
        symptom_pattern = PatternAdaptiveID(
            pattern_type="symptom_cluster",
            hazard_type="clinical_presentation",
            creator_id="test_medical_evolution"
        )
        
        # Initialize field state from symptoms
        field_state = self._symptoms_to_field_state(
            initial_case_data["presentation"]["symptoms"],
            initial_case_data["presentation"]["vitals"]
        )
        
        # Calculate initial metrics
        initial_coherence = self._calculate_symptom_coherence(
            initial_case_data["presentation"]["symptoms"]
        )
        
        # Update pattern with initial state
        symptom_pattern.update_metrics(
            position=(0, 0),  # Initial position in field
            field_state=field_state,
            coherence=initial_coherence,
            energy_state=0.7  # Initial energy based on symptom severity
        )
        
        # Verify pattern properties
        assert symptom_pattern.pattern_type == "symptom_cluster"
        assert symptom_pattern.spatial_context["field_state"] is not None
        assert 0.5 <= symptom_pattern.versions[symptom_pattern.version_id]["data"]["coherence"] <= 1.0

    def test_lab_result_field_interaction(self, initial_case_data, field_config):
        """Test how lab results interact with existing symptom patterns."""
        # Create gradient controller
        gradient_controller = GradientFlowController()
        
        # Create lab result pattern
        lab_pattern = PatternAdaptiveID(
            pattern_type="lab_cluster",
            hazard_type="diagnostic_data",
            creator_id="test_medical_evolution"
        )
        
        # Calculate lab metrics
        lab_coherence = self._calculate_lab_coherence(
            initial_case_data["presentation"]["lab_results"]
        )
        
        # Update lab pattern
        lab_pattern.update_metrics(
            position=(1, 0),  # Adjacent to symptom pattern
            field_state=0.65,  # Normalized lab severity
            coherence=lab_coherence,
            energy_state=0.8  # High energy state for significant labs
        )
        
        # Calculate field gradients
        gradients = FieldGradients(
            coherence=0.7,
            energy=0.75,
            density=0.6,
            turbulence=0.2
        )
        
        # Calculate flow metrics
        flow = gradient_controller.calculate_flow(
            gradients=gradients,
            pattern=lab_pattern.to_dict(),
            related_patterns=[]
        )
        
        # Verify field interactions
        assert flow.viscosity > 0
        assert flow.back_pressure > 0
        assert 0.2 <= flow.volume <= 1.0

    def _symptoms_to_field_state(self, symptoms: List[str], vitals: Dict[str, Any]) -> float:
        """Convert symptoms and vitals to field state."""
        # Normalize vital signs
        temp_severity = (vitals["temperature"] - 37.0) / 3.0  # Max fever considered 40°
        
        # Count severe symptoms
        symptom_count = len(symptoms)
        
        # Combine into field state (0-1 scale)
        return min(1.0, (temp_severity + symptom_count/5) / 2)

    def _calculate_symptom_coherence(self, symptoms: List[str]) -> float:
        """Calculate coherence of symptom cluster."""
        # More symptoms in recognized patterns = higher coherence
        base_coherence = min(1.0, len(symptoms) / 5)
        
        # Add random variation (in practice, would be based on medical knowledge)
        variation = np.random.uniform(-0.1, 0.1)
        
        return min(1.0, max(0.0, base_coherence + variation))

    def _calculate_lab_coherence(self, lab_results: Dict[str, Any]) -> float:
        """Calculate coherence of lab results."""
        # Count abnormal results
        abnormal_count = sum(1 for value in lab_results.values() 
                           if isinstance(value, (int, float)) and value > 10)
        
        # Base coherence on number of related abnormalities
        base_coherence = min(1.0, abnormal_count / 3)
        
        return base_coherence
