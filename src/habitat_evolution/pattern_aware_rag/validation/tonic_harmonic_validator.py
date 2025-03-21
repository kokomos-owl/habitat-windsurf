"""
Validation framework for the Vector + Tonic-Harmonic approach.

This module implements the "Piano Tuner's Approach" to testing tonic-harmonic systems,
providing rigorous metrics for validating performance claims and evaluating the
computational complexity of the approach.
"""

import json
import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from datetime import datetime
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class TonicHarmonicValidator:
    """
    Validation framework for the Vector + Tonic-Harmonic approach.
    
    This class implements the "Piano Tuner's Approach" to testing tonic-harmonic systems,
    focusing on holistic observation, embracing constructive dissonance, and
    context sensitivity.
    """
    
    def __init__(
        self,
        vector_only_detector: Optional[Any] = None,
        resonance_detector: Optional[Any] = None
    ):
        """
        Initialize the validator.
        
        Args:
            vector_only_detector: Detector using vector-only approach
            resonance_detector: Detector using Vector + Tonic-Harmonic approach
        """
        self.vector_only_detector = vector_only_detector
        self.resonance_detector = resonance_detector
        self.test_datasets = {}
        self.results = {}
        
    def register_test_dataset(self, name: str, dataset: Any) -> None:
        """
        Register a test dataset.
        
        Args:
            name: Name of the dataset
            dataset: The dataset to test with
        """
        self.test_datasets[name] = dataset
        logger.info(f"Registered test dataset: {name}")
        
    def run_comparative_tests(self) -> Dict[str, Dict[str, float]]:
        """
        Run comparative tests between vector-only and tonic-harmonic approaches.
        
        Returns:
            Dictionary of test results by dataset
        """
        results = {}
        
        for dataset_name, dataset in self.test_datasets.items():
            logger.info(f"Running comparative tests on dataset: {dataset_name}")
            
            # Run vector-only detection
            vector_patterns = self.vector_only_detector.detect_patterns(dataset) if self.vector_only_detector else []
            
            # Run resonance-based detection
            resonance_patterns = self.resonance_detector.detect_patterns(dataset) if self.resonance_detector else []
            
            # Calculate metrics
            metrics = self._calculate_metrics(vector_patterns, resonance_patterns, dataset)
            results[dataset_name] = metrics
            
            logger.info(f"Completed tests for {dataset_name}: {metrics}")
            
        self.results = results
        return results
    
    def _calculate_metrics(
        self,
        vector_patterns: List[Any],
        resonance_patterns: List[Any],
        dataset: Any
    ) -> Dict[str, float]:
        """
        Calculate comparative metrics between approaches.
        
        Args:
            vector_patterns: Patterns detected by vector-only approach
            resonance_patterns: Patterns detected by tonic-harmonic approach
            dataset: The test dataset
            
        Returns:
            Dictionary of metrics
        """
        # Ensure we don't divide by zero
        vector_pattern_count = max(1, len(vector_patterns))
        
        # Calculate pattern detection ratio
        pattern_detection_ratio = len(resonance_patterns) / vector_pattern_count
        
        # Calculate average pattern size
        vector_avg_size = self._calc_avg_size(vector_patterns)
        resonance_avg_size = self._calc_avg_size(resonance_patterns)
        avg_pattern_size_ratio = resonance_avg_size / max(0.1, vector_avg_size)
        
        # Calculate pattern type diversity
        vector_type_diversity = len(self._get_pattern_types(vector_patterns))
        resonance_type_diversity = len(self._get_pattern_types(resonance_patterns))
        
        # Calculate edge detection precision
        edge_detection_precision = self._calc_edge_precision(resonance_patterns, dataset)
        
        # Calculate dimensional resonance count
        dimensional_resonance_count = self._count_dimensional_resonance(resonance_patterns)
        
        # Calculate computational complexity ratio
        complexity_ratio = self._calc_complexity_ratio(vector_patterns, resonance_patterns)
        
        # Calculate harmonic adaptation score
        harmonic_adaptation = self._calc_harmonic_adaptation(resonance_patterns)
        
        # Calculate constructive dissonance utilization
        constructive_dissonance = self._calc_constructive_dissonance(resonance_patterns)
        
        return {
            "pattern_detection_ratio": pattern_detection_ratio,
            "avg_pattern_size_ratio": avg_pattern_size_ratio,
            "vector_type_diversity": vector_type_diversity,
            "resonance_type_diversity": resonance_type_diversity,
            "edge_detection_precision": edge_detection_precision,
            "dimensional_resonance_count": dimensional_resonance_count,
            "complexity_ratio": complexity_ratio,
            "harmonic_adaptation": harmonic_adaptation,
            "constructive_dissonance": constructive_dissonance
        }
    
    def _calc_avg_size(self, patterns: List[Any]) -> float:
        """Calculate average pattern size."""
        if not patterns:
            return 0.0
            
        total_size = 0
        for pattern in patterns:
            # Get pattern size based on available attributes
            if hasattr(pattern, 'size'):
                total_size += pattern.size
            elif hasattr(pattern, 'text_fragments') and isinstance(pattern.text_fragments, list):
                total_size += len(pattern.text_fragments)
            elif hasattr(pattern, 'relationships') and isinstance(pattern.relationships, (list, dict)):
                if isinstance(pattern.relationships, list):
                    total_size += len(pattern.relationships)
                else:
                    total_size += len(pattern.relationships.keys())
            else:
                total_size += 1  # Default size
                
        return total_size / len(patterns)
    
    def _get_pattern_types(self, patterns: List[Any]) -> Set[str]:
        """Get unique pattern types."""
        types = set()
        for pattern in patterns:
            if hasattr(pattern, 'pattern_type'):
                types.add(pattern.pattern_type)
            elif hasattr(pattern, 'type'):
                types.add(pattern.type)
        return types or {"default"}  # Return at least one type
    
    def _calc_edge_precision(self, patterns: List[Any], dataset: Any) -> float:
        """Calculate edge detection precision."""
        # This is a placeholder - actual implementation would compare
        # detected boundaries with ground truth
        boundary_patterns = [p for p in patterns if self._is_boundary_pattern(p)]
        
        # Default precision if no implementation is provided
        return len(boundary_patterns) / max(1, len(patterns)) if patterns else 0.0
    
    def _is_boundary_pattern(self, pattern: Any) -> bool:
        """Determine if a pattern is a boundary pattern."""
        # Check for explicit boundary type
        if hasattr(pattern, 'pattern_type') and pattern.pattern_type == 'boundary':
            return True
        if hasattr(pattern, 'type') and pattern.type == 'boundary':
            return True
            
        # Check for boundary properties
        if hasattr(pattern, 'properties'):
            props = pattern.properties
            if isinstance(props, dict) and props.get('is_boundary', False):
                return True
                
        # Check for dimensional coordinates spanning multiple domains
        if hasattr(pattern, 'dimensional_coordinates') and hasattr(pattern, 'primary_dimensions'):
            # Complex logic to determine if pattern spans domains
            # This is a simplified placeholder
            return False
            
        return False
    
    def _count_dimensional_resonance(self, patterns: List[Any]) -> int:
        """Count patterns with significant dimensional resonance."""
        count = 0
        for pattern in patterns:
            # Check for explicit dimensional resonance type
            if hasattr(pattern, 'pattern_type') and pattern.pattern_type == 'dimensional_resonance':
                count += 1
            elif hasattr(pattern, 'type') and pattern.type == 'dimensional_resonance':
                count += 1
                
            # Check for dimensional resonance properties
            elif hasattr(pattern, 'properties'):
                props = pattern.properties
                if isinstance(props, dict) and props.get('dimensional_resonance', False):
                    count += 1
                    
            # Check for high dimensional alignment
            elif hasattr(pattern, 'dimensional_alignment') and pattern.dimensional_alignment > 0.8:
                count += 1
                
        return count
    
    def _calc_complexity_ratio(self, vector_patterns: List[Any], resonance_patterns: List[Any]) -> float:
        """Calculate computational complexity ratio between approaches."""
        # This is a simplified placeholder - actual implementation would
        # measure computational resources used by each approach
        
        # Assume complexity scales with pattern count and dimensionality
        vector_complexity = len(vector_patterns) * 1.0  # Base dimensionality
        
        # Resonance complexity includes additional dimensions for tonic-harmonic properties
        resonance_complexity = len(resonance_patterns) * 3.0  # Higher dimensionality
        
        # Return ratio of complexities
        return resonance_complexity / max(1.0, vector_complexity)
    
    def _calc_harmonic_adaptation(self, patterns: List[Any]) -> float:
        """Calculate harmonic adaptation score."""
        # This measures how well the system adapts its structure based on harmonic state
        if not patterns:
            return 0.0
            
        adaptation_scores = []
        for pattern in patterns:
            score = 0.0
            
            # Check for structural adaptation indicators
            if hasattr(pattern, 'phase_stability'):
                score += pattern.phase_stability
                
            if hasattr(pattern, 'harmonic_value'):
                score += min(1.0, pattern.harmonic_value)
                
            if hasattr(pattern, 'tonic_value') and hasattr(pattern, 'stability'):
                score += min(1.0, pattern.tonic_value * pattern.stability)
                
            adaptation_scores.append(min(1.0, score / 3.0))
            
        return sum(adaptation_scores) / len(patterns) if adaptation_scores else 0.0
    
    def _calc_constructive_dissonance(self, patterns: List[Any]) -> float:
        """Calculate constructive dissonance utilization."""
        # This measures how well the system leverages apparent inconsistencies
        if not patterns:
            return 0.0
            
        dissonance_scores = []
        for pattern in patterns:
            score = 0.0
            
            # Check for wave interference properties
            if hasattr(pattern, 'wave_interference_type'):
                if pattern.wave_interference_type == 'DESTRUCTIVE':
                    score += 1.0
                elif pattern.wave_interference_type == 'PARTIAL':
                    score += 0.5
                    
            # Check for relationships with dissonance
            if hasattr(pattern, 'relationships'):
                relationships = pattern.relationships
                if isinstance(relationships, dict):
                    for rel_id, rel_data in relationships.items():
                        if isinstance(rel_data, list):
                            for rel in rel_data:
                                if isinstance(rel, dict) and rel.get('type') == 'dissonant':
                                    score += 0.5
                                    
            dissonance_scores.append(min(1.0, score))
            
        return sum(dissonance_scores) / len(patterns) if dissonance_scores else 0.0
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive validation report.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No test results available. Run comparative_tests first."
            
        report = []
        report.append("# Vector + Tonic-Harmonic Approach Validation Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Overall summary
        avg_metrics = self._calculate_average_metrics()
        report.append("## Overall Performance")
        report.append(f"- Pattern Detection Improvement: {avg_metrics['pattern_detection_ratio']:.2f}x")
        report.append(f"- Pattern Size Improvement: {avg_metrics['avg_pattern_size_ratio']:.2f}x")
        report.append(f"- Pattern Type Diversity: {avg_metrics['resonance_type_diversity']} vs {avg_metrics['vector_type_diversity']} types")
        report.append(f"- Dimensional Resonance Detection: {avg_metrics['dimensional_resonance_count']:.1f} patterns per dataset")
        report.append(f"- Computational Complexity Ratio: {avg_metrics['complexity_ratio']:.2f}x")
        report.append("")
        
        # Dataset-specific results
        report.append("## Dataset Results")
        for dataset_name, metrics in self.results.items():
            report.append(f"### {dataset_name}")
            for metric_name, value in metrics.items():
                report.append(f"- {metric_name}: {value:.2f}")
            report.append("")
            
        # Piano Tuner's Approach insights
        report.append("## Piano Tuner's Approach Insights")
        report.append(f"- Harmonic Adaptation Score: {avg_metrics['harmonic_adaptation']:.2f}")
        report.append(f"- Constructive Dissonance Utilization: {avg_metrics['constructive_dissonance']:.2f}")
        report.append("")
        
        # Conclusions
        report.append("## Conclusions")
        
        # Determine if the approach meets the 4x improvement claim
        meets_claim = avg_metrics['pattern_detection_ratio'] >= 4.0
        report.append(f"The Vector + Tonic-Harmonic approach {'meets' if meets_claim else 'does not meet'} the claimed 4x improvement in pattern detection.")
        
        # Overall assessment
        overall_score = (
            avg_metrics['pattern_detection_ratio'] / 4.0 +  # Normalized to claimed 4x improvement
            avg_metrics['avg_pattern_size_ratio'] / 2.0 +   # Normalized to expected 2x improvement
            avg_metrics['resonance_type_diversity'] / 5.0 +  # Normalized to expected 5 types
            avg_metrics['harmonic_adaptation'] +
            avg_metrics['constructive_dissonance']
        ) / 5.0  # Average of normalized scores
        
        report.append(f"Overall effectiveness score: {overall_score:.2f} (0.0-1.0 scale)")
        
        return "\n".join(report)
    
    def _calculate_average_metrics(self) -> Dict[str, float]:
        """Calculate average metrics across all datasets."""
        if not self.results:
            return {}
            
        avg_metrics = defaultdict(float)
        for metrics in self.results.values():
            for metric_name, value in metrics.items():
                avg_metrics[metric_name] += value
                
        # Calculate averages
        for metric_name in avg_metrics:
            avg_metrics[metric_name] /= len(self.results)
            
        return dict(avg_metrics)
    
    def save_report(self, filepath: str) -> None:
        """
        Save the validation report to a file.
        
        Args:
            filepath: Path to save the report
        """
        report = self.generate_report()
        with open(filepath, 'w') as f:
            f.write(report)
        logger.info(f"Saved validation report to {filepath}")
        
    def save_results_json(self, filepath: str) -> None:
        """
        Save the validation results as JSON.
        
        Args:
            filepath: Path to save the results
        """
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved validation results to {filepath}")
