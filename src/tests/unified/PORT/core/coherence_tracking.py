# coherence_tracking.py

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import uuid4

from core.core_evolution import (
    EvolutionEnhancer,
    FeedbackProcessor,
    EvolutionContext,
    EvolutionResult,
    FeedbackLoopMetrics
)

class CoherenceLevel(Enum):
    """Coherence assessment levels"""
    HIGH = "high"          # Strong alignment
    MODERATE = "moderate"  # Acceptable alignment
    LOW = "low"           # Potential issues
    WARNING = "warning"    # Needs attention

@dataclass
class CoherenceMetrics:
    """Metrics for coherence assessment"""
    structure_meaning_alignment: float = 0.0
    pattern_alignment: float = 0.0
    temporal_consistency: float = 0.0
    domain_consistency: float = 0.0
    overall_coherence: float = 0.0
    assessment_level: CoherenceLevel = CoherenceLevel.MODERATE

class CoherenceAssessment:
    """
    Light coherence assessment for structure-meaning alignment.
    Provides warning flags without enforcing constraints.
    """
    
    def __init__(self, threshold_config: Optional[Dict[str, float]] = None):
        self.thresholds = threshold_config or {
            "warning": 0.3,
            "low": 0.5,
            "moderate": 0.7,
            "high": 0.85
        }
        self.assessment_history: List[Dict[str, Any]] = []
        self.current_assessment: Optional[CoherenceMetrics] = None

    def assess_coherence(
        self,
        structure_data: Dict[str, Any],
        meaning_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> CoherenceMetrics:
        """
        Assess coherence between structure and meaning.
        Returns metrics without enforcing changes.
        """
        try:
            # Calculate individual metrics
            structure_meaning = self._assess_structure_meaning_alignment(
                structure_data,
                meaning_data
            )
            
            pattern = self._assess_pattern_alignment(
                structure_data,
                meaning_data,
                context
            )
            
            temporal = self._assess_temporal_consistency(
                structure_data,
                meaning_data,
                context
            )
            
            domain = self._assess_domain_consistency(
                structure_data,
                meaning_data,
                context
            )
            
            # Calculate overall coherence
            overall = self._calculate_overall_coherence([
                structure_meaning,
                pattern,
                temporal,
                domain
            ])
            
            # Determine assessment level
            level = self._determine_coherence_level(overall)
            
            # Create metrics
            metrics = CoherenceMetrics(
                structure_meaning_alignment=structure_meaning,
                pattern_alignment=pattern,
                temporal_consistency=temporal,
                domain_consistency=domain,
                overall_coherence=overall,
                assessment_level=level
            )
            
            # Update tracking
            self._update_assessment_tracking(metrics)
            
            return metrics
            
        except Exception as e:
            # Log error and return default metrics
            return CoherenceMetrics()

    def _assess_structure_meaning_alignment(
        self,
        structure_data: Dict[str, Any],
        meaning_data: Dict[str, Any]
    ) -> float:
        """Assess alignment between structure and meaning"""
        try:
            # Compare structural elements with meaning
            structural_concepts = set(structure_data.get("nodes", {}).keys())
            meaning_concepts = set(meaning_data.get("concepts", {}).keys())
            
            # Calculate overlap
            if not structural_concepts or not meaning_concepts:
                return 0.5  # Default moderate alignment
                
            overlap = len(structural_concepts & meaning_concepts)
            total = len(structural_concepts | meaning_concepts)
            
            return overlap / total
            
        except Exception as e:
            return 0.5  # Default on error

    def _assess_pattern_alignment(
        self,
        structure_data: Dict[str, Any],
        meaning_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Assess alignment with observed patterns"""
        try:
            if not context or "pattern_suggestions" not in context:
                return 0.5  # No pattern data
                
            suggestions = context["pattern_suggestions"]
            applied_count = 0
            total_count = 0
            
            # Check structural suggestions
            struct_suggestions = {
                k: v for k, v in suggestions.items()
                if k.startswith("structure_")
            }
            for key, suggestion in struct_suggestions.items():
                total_count += 1
                if key in structure_data and structure_data[key] == suggestion["value"]:
                    applied_count += 1
                    
            # Check meaning suggestions
            meaning_suggestions = {
                k: v for k, v in suggestions.items()
                if k.startswith("meaning_")
            }
            for key, suggestion in meaning_suggestions.items():
                total_count += 1
                if key in meaning_data and meaning_data[key] == suggestion["value"]:
                    applied_count += 1
                    
            return applied_count / total_count if total_count > 0 else 0.5
            
        except Exception as e:
            return 0.5  # Default on error

    def _assess_temporal_consistency(
        self,
        structure_data: Dict[str, Any],
        meaning_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Assess temporal consistency"""
        try:
            structure_time = structure_data.get("timestamp")
            meaning_time = meaning_data.get("timestamp")
            
            if not structure_time or not meaning_time:
                return 0.5  # No temporal data
                
            # Convert to datetime if needed
            if isinstance(structure_time, str):
                structure_time = datetime.fromisoformat(structure_time)
            if isinstance(meaning_time, str):
                meaning_time = datetime.fromisoformat(meaning_time)
                
            # Calculate temporal difference
            time_diff = abs((structure_time - meaning_time).total_seconds())
            
            # Score based on difference (within 1 hour = high consistency)
            return max(0.0, min(1.0, 1.0 - (time_diff / 3600)))
            
        except Exception as e:
            return 0.5  # Default on error

    def _assess_domain_consistency(
        self,
        structure_data: Dict[str, Any],
        meaning_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Assess consistency with domain rules"""
        try:
            if not context or "domain_rules" not in context:
                return 0.5  # No domain data
                
            rules = context["domain_rules"]
            passed_rules = 0
            total_rules = len(rules)
            
            for rule in rules:
                if self._check_domain_rule(rule, structure_data, meaning_data):
                    passed_rules += 1
                    
            return passed_rules / total_rules if total_rules > 0 else 0.5
            
        except Exception as e:
            return 0.5  # Default on error

    def _check_domain_rule(
        self,
        rule: Dict[str, Any],
        structure_data: Dict[str, Any],
        meaning_data: Dict[str, Any]
    ) -> bool:
        """Check if data complies with a domain rule"""
        # Basic rule checking for POC
        # Can be extended for specific domain rules
        return True

    def _calculate_overall_coherence(
        self,
        metrics: List[float]
    ) -> float:
        """Calculate overall coherence score"""
        if not metrics:
            return 0.5
            
        # Weight the metrics
        weights = [0.4, 0.2, 0.2, 0.2]  # Structure-meaning alignment weighted higher
        
        weighted_sum = sum(m * w for m, w in zip(metrics, weights))
        return weighted_sum

    def _determine_coherence_level(
        self,
        overall_score: float
    ) -> CoherenceLevel:
        """Determine coherence level from overall score"""
        if overall_score >= self.thresholds["high"]:
            return CoherenceLevel.HIGH
        elif overall_score >= self.thresholds["moderate"]:
            return CoherenceLevel.MODERATE
        elif overall_score >= self.thresholds["low"]:
            return CoherenceLevel.LOW
        else:
            return CoherenceLevel.WARNING

    def _update_assessment_tracking(
        self,
        metrics: CoherenceMetrics
    ) -> None:
        """Update assessment history and current assessment"""
        assessment = {
            "timestamp": datetime.utcnow(),
            "metrics": metrics.__dict__,
            "assessment_id": str(uuid4())
        }
        
        self.assessment_history.append(assessment)
        self.current_assessment = metrics

class CoherenceEnhancer(EvolutionEnhancer):
    """
    Enhances evolution based on coherence assessment.
    Provides suggestions without enforcing changes.
    """
    
    def __init__(self):
        self.coherence_assessor = CoherenceAssessment()

    def enhance_evolution(
        self,
        context: EvolutionContext[Dict[str, Any]],
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance evolution based on coherence assessment.
        Adds coherence information without modifying state.
        """
        enhanced_state = current_state.copy()
        
        # Extract structure and meaning data
        structure_data = self._extract_structure_data(current_state)
        meaning_data = self._extract_meaning_data(current_state)
        
        # Assess coherence
        metrics = self.coherence_assessor.assess_coherence(
            structure_data,
            meaning_data,
            context.data
        )
        
        # Add coherence information
        enhanced_state["coherence_assessment"] = {
            "metrics": metrics.__dict__,
            "timestamp": datetime.utcnow()
        }
        
        # Add warnings if needed
        if metrics.assessment_level in [CoherenceLevel.LOW, CoherenceLevel.WARNING]:
            enhanced_state["coherence_warnings"] = self._generate_warnings(metrics)
            
        return enhanced_state

    def validate_enhancement(
        self,
        original: Dict[str, Any],
        enhanced: Dict[str, Any]
    ) -> float:
        """Validate coherence enhancement"""
        if "coherence_assessment" not in enhanced:
            return 1.0
            
        return enhanced["coherence_assessment"]["metrics"]["overall_coherence"]

    def _extract_structure_data(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract structural data from state"""
        return {
            "nodes": state.get("nodes", {}),
            "relationships": state.get("relationships", {}),
            "timestamp": state.get("timestamp")
        }

    def _extract_meaning_data(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract meaning data from state"""
        return {
            "concepts": state.get("concepts", {}),
            "semantic_changes": state.get("semantic_changes", {}),
            "domain_changes": state.get("domain_changes", {}),
            "timestamp": state.get("timestamp")
        }

    def _generate_warnings(
        self,
        metrics: CoherenceMetrics
    ) -> List[Dict[str, Any]]:
        """Generate coherence warnings"""
        warnings = []
        
        if metrics.structure_meaning_alignment < 0.5:
            warnings.append({
                "type": "structure_meaning_misalignment",
                "severity": "high",
                "message": "Low alignment between structure and meaning"
            })
            
        if metrics.pattern_alignment < 0.3:
            warnings.append({
                "type": "pattern_misalignment",
                "severity": "medium",
                "message": "Low pattern alignment detected"
            })
            
        if metrics.temporal_consistency < 0.3:
            warnings.append({
                "type": "temporal_inconsistency",
                "severity": "medium",
                "message": "Temporal inconsistency detected"
            })
            
        if metrics.domain_consistency < 0.5:
            warnings.append({
                "type": "domain_inconsistency",
                "severity": "high",
                "message": "Low domain consistency detected"
            })
            
        return warnings

class CoherenceFeedbackProcessor(FeedbackProcessor):
    """
    Processes evolution feedback for coherence tracking.
    """
    
    def __init__(self):
        self.coherence_assessor = CoherenceAssessment()
        self.feedback_history: List[Dict[str, Any]] = []

    def process_feedback(
        self,
        evolution_result: EvolutionResult[Dict[str, Any]],
        current_state: Dict[str, Any]
    ) -> FeedbackLoopMetrics:
        """Process evolution results for coherence feedback"""
        try:
            if not evolution_result.evolved_data:
                return FeedbackLoopMetrics()
                
            # Get coherence assessment
            assessment = current_state.get("coherence_assessment")
            if not assessment:
                return FeedbackLoopMetrics()
                
            # Record feedback
            feedback = {
                "timestamp": datetime.utcnow(),
                "evolution_success": evolution_result.success,
                "coherence_metrics": assessment["metrics"],
                "evolution_id": str(uuid4())
            }
            self.feedback_history.append(feedback)
            
            # Calculate feedback metrics
            metrics = assessment["metrics"]
            return FeedbackLoopMetrics(
                structure_meaning_coherence=metrics["overall_coherence"]
            )
            
        except Exception as e:
            return FeedbackLoopMetrics()