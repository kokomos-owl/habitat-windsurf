"""
Meta-Pattern Propensity Module

This module implements a calculator that analyzes meta-patterns to generate
probability distributions for future pattern emergence, building on Habitat's
concepts of capaciousness and supposition.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import logging
from datetime import datetime
from collections import defaultdict

class MetaPatternPropensityCalculator:
    """
    Analyzes meta-patterns to generate probability distributions for future pattern emergence.
    
    This calculator builds on Habitat's concepts of capaciousness (ability to contain meaning
    through change) and directionality (the "supposing" within meaning) while adding
    meta-pattern awareness.
    
    The calculator tracks meta-pattern occurrences and their evolution, calculates propensity
    scores for potential future patterns, and generates probability distributions for pattern
    emergence that adjust based on field metrics like coherence and stability.
    """
    
    def __init__(self, field_metrics: Dict[str, float] = None):
        """
        Initialize the meta-pattern propensity calculator.
        
        Args:
            field_metrics: Initial field metrics (coherence, stability, etc.)
        """
        self.meta_patterns = {}  # Meta-patterns by ID
        self.pattern_history = []  # Historical patterns
        self.meta_pattern_history = []  # Historical meta-patterns
        self.pattern_transitions = defaultdict(list)  # Transitions between patterns
        self.field_metrics = field_metrics or {
            "coherence": 0.7,
            "stability": 0.7,
            "turbulence": 0.3,
            "density": 0.5
        }
        
        # Thresholds and weights
        self.propensity_threshold = 0.3  # Minimum propensity to consider
        self.history_weight = 0.4  # Weight for historical pattern frequency
        self.meta_pattern_weight = 0.6  # Weight for meta-pattern influence
        self.recency_factor = 0.8  # Decay factor for older patterns
        
        # Density tracking
        self.domain_predicate_density = defaultdict(float)  # Tracks domain<->predicate density
        
        logging.info("MetaPatternPropensityCalculator initialized")
    
    def register_pattern(self, pattern: Dict[str, Any]) -> None:
        """
        Register a detected pattern.
        
        Args:
            pattern: The pattern to register
        """
        # Add timestamp if not present
        if "detection_timestamp" not in pattern:
            pattern["detection_timestamp"] = datetime.now().isoformat()
            
        # Store pattern in history
        self.pattern_history.append({
            "pattern": pattern,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update domain-predicate density
        domain = pattern.get("domain", "default")
        predicate = pattern.get("predicate")
        if predicate:
            density_key = (domain, predicate)
            self.domain_predicate_density[density_key] += 0.1
            # Cap at 1.0
            self.domain_predicate_density[density_key] = min(
                1.0, self.domain_predicate_density[density_key]
            )
        
        # Check for transitions from previous patterns
        if len(self.pattern_history) > 1:
            prev_pattern = self.pattern_history[-2]["pattern"]
            self._register_transition(prev_pattern, pattern)
        
        logging.debug(f"Registered pattern: {pattern.get('id', 'unknown')}")
    
    def register_meta_pattern(self, meta_pattern: Dict[str, Any]) -> None:
        """
        Register a detected meta-pattern.
        
        Args:
            meta_pattern: The meta-pattern to register
        """
        meta_pattern_id = meta_pattern.get("id")
        if not meta_pattern_id:
            meta_pattern_id = f"meta_pattern_{len(self.meta_patterns)}"
            meta_pattern["id"] = meta_pattern_id
            
        self.meta_patterns[meta_pattern_id] = meta_pattern
        self.meta_pattern_history.append({
            "meta_pattern": meta_pattern,
            "timestamp": datetime.now().isoformat()
        })
        
        logging.info(f"Registered meta-pattern: {meta_pattern_id}")
        logging.info(f"  Evolution type: {meta_pattern.get('evolution_type')}")
        logging.info(f"  Frequency: {meta_pattern.get('frequency')}")
        logging.info(f"  Confidence: {meta_pattern.get('confidence')}")
    
    def update_field_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Update field metrics.
        
        Args:
            metrics: New field metrics
        """
        self.field_metrics.update(metrics)
        logging.debug(f"Updated field metrics: {metrics}")
    
    def _register_transition(self, from_pattern: Dict[str, Any], to_pattern: Dict[str, Any]) -> None:
        """
        Register a transition between patterns.
        
        Args:
            from_pattern: Source pattern
            to_pattern: Target pattern
        """
        from_id = from_pattern.get("id")
        if from_id:
            self.pattern_transitions[from_id].append({
                "to_pattern": to_pattern,
                "timestamp": datetime.now().isoformat()
            })
    
    def calculate_pattern_propensities(self) -> Dict[str, float]:
        """
        Calculate propensity scores for potential future patterns.
        
        Returns:
            Dictionary mapping pattern templates to propensity scores
        """
        propensities = {}
        
        # Generate potential patterns based on meta-patterns
        potential_patterns = self._generate_potential_patterns()
        
        for pattern_template, base_score in potential_patterns.items():
            # Start with base score
            propensity = base_score
            
            # Adjust based on domain-predicate density
            domain, predicate, _ = self._parse_pattern_template(pattern_template)
            density = self.domain_predicate_density.get((domain, predicate), 0.0)
            propensity *= (0.5 + (density * 0.5))
            
            # Adjust based on field metrics
            coherence = self.field_metrics.get("coherence", 0.5)
            stability = self.field_metrics.get("stability", 0.5)
            field_factor = (coherence * 0.6) + (stability * 0.4)
            propensity *= field_factor
            
            # Only include if above threshold
            if propensity >= self.propensity_threshold:
                propensities[pattern_template] = propensity
        
        # Normalize propensities to form a probability distribution
        total = sum(propensities.values())
        if total > 0:
            propensities = {p: score/total for p, score in propensities.items()}
        
        return propensities
    
    def _generate_potential_patterns(self) -> Dict[str, float]:
        """
        Generate potential patterns based on meta-patterns and history.
        
        Returns:
            Dictionary mapping pattern templates to base propensity scores
        """
        potential_patterns = {}
        
        # Consider recent patterns (with recency weighting)
        recent_patterns = self.pattern_history[-20:] if len(self.pattern_history) > 20 else self.pattern_history
        for i, pattern_entry in enumerate(reversed(recent_patterns)):
            pattern = pattern_entry["pattern"]
            recency_weight = self.recency_factor ** i  # Decay for older patterns
            
            # Generate potential follow-up patterns based on transitions
            pattern_id = pattern.get("id")
            if pattern_id and pattern_id in self.pattern_transitions:
                for transition in self.pattern_transitions[pattern_id]:
                    to_pattern = transition["to_pattern"]
                    template = self._create_pattern_template(to_pattern)
                    
                    # Add to potential patterns with recency-weighted score
                    base_score = 0.2 * recency_weight * self.history_weight
                    potential_patterns[template] = potential_patterns.get(template, 0) + base_score
        
        # Apply meta-pattern influence
        for meta_pattern_entry in self.meta_pattern_history:
            meta_pattern = meta_pattern_entry["meta_pattern"]
            evolution_type = meta_pattern.get("evolution_type")
            confidence = meta_pattern.get("confidence", 0.5)
            
            # Generate potential patterns based on meta-pattern type
            if evolution_type == "object_evolution":
                self._apply_object_evolution_meta_pattern(meta_pattern, potential_patterns, confidence)
            elif evolution_type == "subject_evolution":
                self._apply_subject_evolution_meta_pattern(meta_pattern, potential_patterns, confidence)
            elif evolution_type == "predicate_evolution":
                self._apply_predicate_evolution_meta_pattern(meta_pattern, potential_patterns, confidence)
        
        return potential_patterns
    
    def _apply_object_evolution_meta_pattern(self, meta_pattern: Dict[str, Any], 
                                            potential_patterns: Dict[str, float],
                                            confidence: float) -> None:
        """
        Apply object evolution meta-pattern to generate potential patterns.
        
        Args:
            meta_pattern: The meta-pattern to apply
            potential_patterns: Dictionary of potential patterns to update
            confidence: Confidence score of the meta-pattern
        """
        # Get examples from meta-pattern
        examples = meta_pattern.get("examples", [])
        
        for example in examples:
            details = example.get("details", {})
            from_object = details.get("from_object")
            to_object = details.get("to_object")
            
            if from_object and to_object:
                # Look for recent patterns with the same object as from_object
                recent_patterns = self.pattern_history[-10:] if len(self.pattern_history) > 10 else self.pattern_history
                for pattern_entry in reversed(recent_patterns):
                    pattern = pattern_entry["pattern"]
                    if pattern.get("target") == from_object:
                        # Create a new pattern template with the same source and predicate but new target
                        source = pattern.get("source")
                        predicate = pattern.get("predicate")
                        if source and predicate:
                            template = f"{source}_{predicate}_{to_object}"
                            
                            # Add to potential patterns with meta-pattern-weighted score
                            base_score = 0.3 * confidence * self.meta_pattern_weight
                            potential_patterns[template] = potential_patterns.get(template, 0) + base_score
    
    def _apply_subject_evolution_meta_pattern(self, meta_pattern: Dict[str, Any], 
                                             potential_patterns: Dict[str, float],
                                             confidence: float) -> None:
        """
        Apply subject evolution meta-pattern to generate potential patterns.
        
        Args:
            meta_pattern: The meta-pattern to apply
            potential_patterns: Dictionary of potential patterns to update
            confidence: Confidence score of the meta-pattern
        """
        # Similar implementation to object evolution but for subject
        examples = meta_pattern.get("examples", [])
        
        for example in examples:
            details = example.get("details", {})
            from_subject = details.get("from_subject")
            to_subject = details.get("to_subject")
            
            if from_subject and to_subject:
                # Look for recent patterns with the same subject as from_subject
                recent_patterns = self.pattern_history[-10:] if len(self.pattern_history) > 10 else self.pattern_history
                for pattern_entry in reversed(recent_patterns):
                    pattern = pattern_entry["pattern"]
                    if pattern.get("source") == from_subject:
                        # Create a new pattern template with new source but same predicate and target
                        predicate = pattern.get("predicate")
                        target = pattern.get("target")
                        if predicate and target:
                            template = f"{to_subject}_{predicate}_{target}"
                            
                            base_score = 0.3 * confidence * self.meta_pattern_weight
                            potential_patterns[template] = potential_patterns.get(template, 0) + base_score
    
    def _apply_predicate_evolution_meta_pattern(self, meta_pattern: Dict[str, Any], 
                                               potential_patterns: Dict[str, float],
                                               confidence: float) -> None:
        """
        Apply predicate evolution meta-pattern to generate potential patterns.
        
        Args:
            meta_pattern: The meta-pattern to apply
            potential_patterns: Dictionary of potential patterns to update
            confidence: Confidence score of the meta-pattern
        """
        # Similar implementation to object evolution but for predicate
        examples = meta_pattern.get("examples", [])
        
        for example in examples:
            details = example.get("details", {})
            from_predicate = details.get("from_predicate")
            to_predicate = details.get("to_predicate")
            
            if from_predicate and to_predicate:
                # Look for recent patterns with the same predicate as from_predicate
                recent_patterns = self.pattern_history[-10:] if len(self.pattern_history) > 10 else self.pattern_history
                for pattern_entry in reversed(recent_patterns):
                    pattern = pattern_entry["pattern"]
                    if pattern.get("predicate") == from_predicate:
                        # Create a new pattern template with same source and target but new predicate
                        source = pattern.get("source")
                        target = pattern.get("target")
                        if source and target:
                            template = f"{source}_{to_predicate}_{target}"
                            
                            base_score = 0.3 * confidence * self.meta_pattern_weight
                            potential_patterns[template] = potential_patterns.get(template, 0) + base_score
    
    def _create_pattern_template(self, pattern: Dict[str, Any]) -> str:
        """
        Create a pattern template string from a pattern.
        
        Args:
            pattern: The pattern to create a template from
            
        Returns:
            Pattern template string
        """
        source = pattern.get("source", "unknown")
        predicate = pattern.get("predicate", "unknown")
        target = pattern.get("target", "unknown")
        return f"{source}_{predicate}_{target}"
    
    def _parse_pattern_template(self, template: str) -> Tuple[str, str, str]:
        """
        Parse a pattern template into components.
        
        Args:
            template: Pattern template string
            
        Returns:
            Tuple of (source, predicate, target)
        """
        parts = template.split("_")
        if len(parts) >= 3:
            source = parts[0]
            predicate = parts[1]
            target = "_".join(parts[2:])  # Handle targets with underscores
            return source, predicate, target
        return "unknown", "unknown", "unknown"
    
    def get_top_propensities(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top N pattern propensities.
        
        Args:
            n: Number of top propensities to return
            
        Returns:
            List of dictionaries with pattern and propensity
        """
        propensities = self.calculate_pattern_propensities()
        sorted_propensities = sorted(propensities.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for template, propensity in sorted_propensities[:n]:
            source, predicate, target = self._parse_pattern_template(template)
            result.append({
                "source": source,
                "predicate": predicate,
                "target": target,
                "template": template,
                "propensity": propensity
            })
        
        return result
