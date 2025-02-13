"""Verification framework for sepsis pattern detection based on Sepsis-3 criteria."""

from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import numpy as np

class SepsisVerifier:
    """Verifies pattern detection against Sepsis-3 clinical criteria."""
    
    def __init__(self):
        """Initialize sepsis verification criteria."""
        self.sofa_criteria = {
            "respiratory": {
                "measure": "pao2_fio2_ratio",
                "thresholds": [(400, 0), (300, 1), (200, 2), (100, 3), (0, 4)]
            },
            "coagulation": {
                "measure": "platelets",
                "thresholds": [(150, 0), (100, 1), (50, 2), (20, 3), (0, 4)]
            },
            "liver": {
                "measure": "bilirubin",
                "thresholds": [(1.2, 0), (2.0, 1), (6.0, 2), (12.0, 3)]
            },
            "cardiovascular": {
                "measure": "mean_arterial_pressure",
                "thresholds": [(70, 0), (65, 1), (0, 2)]
            },
            "cns": {
                "measure": "gcs",
                "thresholds": [(15, 0), (13, 1), (10, 2), (6, 3), (0, 4)]
            },
            "renal": {
                "measure": "creatinine",
                "thresholds": [(1.2, 0), (2.0, 1), (3.5, 2), (5.0, 3)]
            }
        }
        
        self.qsofa_criteria = {
            "respiratory_rate": 22,
            "systolic_bp": 100,
            "altered_mental_status": True
        }
    
    def verify_pattern(self, 
                      pattern: Dict[str, Any],
                      clinical_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if detected pattern matches sepsis criteria.
        
        Args:
            pattern: Detected pattern from field
            clinical_data: Actual clinical measurements
            
        Returns:
            Tuple of (matches_criteria, details)
        """
        # Check timing accuracy
        timing_accuracy = self._verify_timing(
            pattern.get("onset_time"),
            clinical_data.get("onset_time")
        )
        
        # Check SOFA score accuracy
        sofa_accuracy = self._verify_sofa_score(
            pattern.get("sofa_indicators", {}),
            clinical_data
        )
        
        # Check qSOFA accuracy
        qsofa_accuracy = self._verify_qsofa(
            pattern.get("qsofa_indicators", {}),
            clinical_data
        )
        
        # Check organ dysfunction patterns
        organ_accuracy = self._verify_organ_dysfunction(
            pattern.get("organ_patterns", {}),
            clinical_data
        )
        
        # Calculate overall accuracy
        weights = {
            "timing": 0.3,
            "sofa": 0.3,
            "qsofa": 0.2,
            "organ": 0.2
        }
        
        total_accuracy = (
            weights["timing"] * timing_accuracy +
            weights["sofa"] * sofa_accuracy +
            weights["qsofa"] * qsofa_accuracy +
            weights["organ"] * organ_accuracy
        )
        
        matches_criteria = total_accuracy >= 0.8
        
        return matches_criteria, {
            "total_accuracy": total_accuracy,
            "timing_accuracy": timing_accuracy,
            "sofa_accuracy": sofa_accuracy,
            "qsofa_accuracy": qsofa_accuracy,
            "organ_accuracy": organ_accuracy
        }
    
    def _verify_timing(self, 
                      pattern_time: datetime,
                      actual_time: datetime) -> float:
        """Verify accuracy of sepsis onset timing detection."""
        if not pattern_time or not actual_time:
            return 0.0
            
        time_diff = abs((pattern_time - actual_time).total_seconds() / 3600)
        
        # Score based on how close the detection is
        if time_diff <= 1:  # Within 1 hour
            return 1.0
        elif time_diff <= 3:  # Within 3 hours
            return 0.8
        elif time_diff <= 6:  # Within 6 hours
            return 0.6
        elif time_diff <= 12:  # Within 12 hours
            return 0.3
        else:
            return 0.0
    
    def _verify_sofa_score(self,
                          pattern_indicators: Dict[str, Any],
                          clinical_data: Dict[str, Any]) -> float:
        """Verify accuracy of SOFA score component detection."""
        if not pattern_indicators:
            return 0.0
            
        accuracies = []
        
        for system, criteria in self.sofa_criteria.items():
            if system in pattern_indicators and system in clinical_data:
                pattern_value = pattern_indicators[system]
                actual_value = clinical_data[system]
                
                # Calculate accuracy based on threshold matching
                pattern_score = self._get_sofa_score(pattern_value, criteria["thresholds"])
                actual_score = self._get_sofa_score(actual_value, criteria["thresholds"])
                
                accuracies.append(1.0 - abs(pattern_score - actual_score) / 4)
        
        return np.mean(accuracies) if accuracies else 0.0
    
    def _verify_qsofa(self,
                     pattern_indicators: Dict[str, Any],
                     clinical_data: Dict[str, Any]) -> float:
        """Verify accuracy of qSOFA criteria detection."""
        if not pattern_indicators:
            return 0.0
            
        accuracies = []
        
        for criterion, threshold in self.qsofa_criteria.items():
            if criterion in pattern_indicators and criterion in clinical_data:
                pattern_value = pattern_indicators[criterion]
                actual_value = clinical_data[criterion]
                
                if isinstance(threshold, bool):
                    accuracies.append(1.0 if pattern_value == actual_value else 0.0)
                else:
                    # For numeric criteria, calculate scaled accuracy
                    diff = abs(pattern_value - actual_value)
                    accuracies.append(max(0.0, 1.0 - diff / threshold))
        
        return np.mean(accuracies) if accuracies else 0.0
    
    def _verify_organ_dysfunction(self,
                                pattern_organs: Dict[str, Any],
                                clinical_data: Dict[str, Any]) -> float:
        """Verify accuracy of organ dysfunction pattern detection."""
        if not pattern_organs or "organ_dysfunction" not in clinical_data:
            return 0.0
            
        # Check if pattern correctly identified presence of organ dysfunction
        pattern_has_dysfunction = any(
            dysfunction.get("severity", 0) > 0.5
            for dysfunction in pattern_organs.values()
        )
        
        actual_has_dysfunction = clinical_data["organ_dysfunction"]
        
        return 1.0 if pattern_has_dysfunction == actual_has_dysfunction else 0.0
    
    def _get_sofa_score(self, value: float, thresholds: List[Tuple[float, int]]) -> int:
        """Calculate SOFA score for a given value and thresholds."""
        for threshold, score in sorted(thresholds, reverse=True):
            if value <= threshold:
                return score
        return 0
