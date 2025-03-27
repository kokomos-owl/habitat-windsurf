"""
Emergent Pattern Detector

This module implements the EmergentPatternDetector class, which detects patterns
that emerge naturally from semantic observations without imposing predefined categories.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import uuid
import logging
import math
from collections import defaultdict

from ..id.adaptive_id import AdaptiveID
from .semantic_current_observer import SemanticCurrentObserver


class EmergentPatternDetector:
    """
    Detects patterns that emerge naturally from semantic observations.
    
    This class identifies recurring patterns based on frequency and consistency,
    without categorizing them in advance. It allows patterns to function as
    first-class entities that can influence the system's behavior.
    """
    
    def __init__(self, semantic_observer: SemanticCurrentObserver, threshold: int = 3):
        """
        Initialize an emergent pattern detector.
        
        Args:
            semantic_observer: Observer for semantic currents
            threshold: Minimum frequency threshold for pattern detection
        """
        self.semantic_observer = semantic_observer
        self.threshold = threshold
        self.potential_patterns = []
        self.pattern_history = []
        self.pattern_evolution = {}
        
        # Create an AdaptiveID for this detector
        self.adaptive_id = AdaptiveID(
            base_concept="emergent_pattern_detector",
            creator_id="system"
        )
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
    
    def detect_patterns(self) -> List[Dict[str, Any]]:
        """
        Detect potential patterns based on observed frequencies.
        
        Returns:
            List of detected patterns
        """
        # Reset potential patterns for this detection cycle
        self.potential_patterns = []
        
        # Get all observations from the semantic observer
        observations = self.semantic_observer.observed_relationships
        frequencies = self.semantic_observer.relationship_frequency
        
        # Find relationships that exceed the threshold
        for rel_key, count in frequencies.items():
            if count >= self.threshold:
                rel_data = observations[rel_key]
                # Extract source, predicate, target from the relationship data
                # instead of trying to parse from the key
                source = rel_data["source"]
                predicate = rel_data["predicate"]
                target = rel_data["target"]
                
                # Calculate confidence based on frequency and recency
                confidence = self._calculate_confidence(rel_data)
                
                # Create pattern
                pattern = {
                    "id": f"pattern_{len(self.pattern_history)}_{rel_key}",
                    "source": source,
                    "predicate": predicate,
                    "target": target,
                    "frequency": count,
                    "confidence": confidence,
                    "first_observed": rel_data["first_observed"],
                    "last_observed": rel_data["last_observed"],
                    "detection_timestamp": datetime.now().isoformat()
                }
                
                # Check if this is an evolution of an existing pattern
                evolved_from = self._check_pattern_evolution(pattern)
                if evolved_from:
                    pattern["evolved_from"] = evolved_from
                
                # Update the AdaptiveID with this pattern
                pattern_key = f"pattern_detected_{rel_key}"
                pattern_data = {
                    "pattern_id": pattern["id"],
                    "source": pattern["source"],
                    "predicate": pattern["predicate"],
                    "target": pattern["target"],
                    "frequency": pattern["frequency"],
                    "confidence": pattern["confidence"],
                    "detection_timestamp": pattern["detection_timestamp"]
                }
                self.adaptive_id.update_temporal_context(pattern_key, pattern_data, "pattern_detection")
                
                self.potential_patterns.append(pattern)
                
                # Add to pattern history
                self.pattern_history.append(pattern)
        
        # Detect meta-patterns (patterns of patterns)
        self._detect_meta_patterns()
        
        return self.potential_patterns
    
    def _calculate_confidence(self, rel_data: Dict[str, Any]) -> float:
        """
        Calculate confidence score for a pattern.
        
        Args:
            rel_data: Relationship data
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence on frequency
        frequency = rel_data["frequency"]
        frequency_factor = min(1.0, frequency / 10.0)  # Cap at 1.0
        
        # Factor in recency
        last_observed = datetime.fromisoformat(rel_data["last_observed"])
        now = datetime.now()
        hours_since = (now - last_observed).total_seconds() / 3600
        recency_factor = math.exp(-hours_since / 24)  # Decay over 24 hours
        
        # Factor in consistency of contexts
        context_similarity = self._calculate_context_similarity(rel_data["contexts"])
        
        # Combine factors
        confidence = 0.4 * frequency_factor + 0.3 * recency_factor + 0.3 * context_similarity
        
        return min(1.0, confidence)
    
    def _calculate_context_similarity(self, contexts: List[Dict[str, Any]]) -> float:
        """
        Calculate similarity between contexts.
        
        Args:
            contexts: List of context dictionaries
            
        Returns:
            Similarity score between 0 and 1
        """
        if len(contexts) <= 1:
            return 0.5  # Neutral score for insufficient data
        
        # Extract context keys
        all_keys = set()
        for context_data in contexts:
            if "context" in context_data:
                all_keys.update(context_data["context"].keys())
        
        if not all_keys:
            return 0.5  # Neutral score for empty contexts
        
        # Count key occurrences
        key_counts = defaultdict(int)
        for context_data in contexts:
            if "context" in context_data:
                for key in context_data["context"].keys():
                    key_counts[key] += 1
        
        # Calculate average consistency
        consistency_sum = sum(count / len(contexts) for count in key_counts.values())
        avg_consistency = consistency_sum / len(key_counts) if key_counts else 0.5
        
        return avg_consistency
    
    def _check_pattern_evolution(self, pattern: Dict[str, Any]) -> Optional[str]:
        """
        Check if this pattern is an evolution of an existing pattern.
        
        Args:
            pattern: The pattern to check
            
        Returns:
            ID of the pattern this evolved from, or None
        """
        # Look for similar patterns in history
        for hist_pattern in reversed(self.pattern_history):  # Check most recent first
            # Skip if this is the same pattern
            if hist_pattern["source"] == pattern["source"] and \
               hist_pattern["predicate"] == pattern["predicate"] and \
               hist_pattern["target"] == pattern["target"]:
                continue
            
            # Check for evolution relationships
            if hist_pattern["source"] == pattern["source"] and \
               hist_pattern["target"] == pattern["target"]:
                # Predicate evolution
                evolved_id = hist_pattern["id"]
                
                # Record evolution
                if evolved_id not in self.pattern_evolution:
                    self.pattern_evolution[evolved_id] = []
                
                self.pattern_evolution[evolved_id].append({
                    "evolution_type": "predicate_evolution",
                    "from_pattern": evolved_id,
                    "to_pattern": pattern["id"],
                    "from_predicate": hist_pattern["predicate"],
                    "to_predicate": pattern["predicate"],
                    "timestamp": datetime.now().isoformat()
                })
                
                return evolved_id
            
            elif hist_pattern["predicate"] == pattern["predicate"] and \
                 hist_pattern["target"] == pattern["target"]:
                # Subject evolution
                evolved_id = hist_pattern["id"]
                
                # Record evolution
                if evolved_id not in self.pattern_evolution:
                    self.pattern_evolution[evolved_id] = []
                
                self.pattern_evolution[evolved_id].append({
                    "evolution_type": "subject_evolution",
                    "from_pattern": evolved_id,
                    "to_pattern": pattern["id"],
                    "from_subject": hist_pattern["source"],
                    "to_subject": pattern["source"],
                    "timestamp": datetime.now().isoformat()
                })
                
                return evolved_id
            
            elif hist_pattern["source"] == pattern["source"] and \
                 hist_pattern["predicate"] == pattern["predicate"]:
                # Object evolution
                evolved_id = hist_pattern["id"]
                
                # Record evolution
                if evolved_id not in self.pattern_evolution:
                    self.pattern_evolution[evolved_id] = []
                
                self.pattern_evolution[evolved_id].append({
                    "evolution_type": "object_evolution",
                    "from_pattern": evolved_id,
                    "to_pattern": pattern["id"],
                    "from_object": hist_pattern["target"],
                    "to_object": pattern["target"],
                    "timestamp": datetime.now().isoformat()
                })
                
                return evolved_id
        
        return None
    
    def _detect_meta_patterns(self) -> List[Dict[str, Any]]:
        """
        Detect meta-patterns (patterns of patterns).
        
        Returns:
            List of meta-patterns
        """
        meta_patterns = []
        
        # Need at least a few patterns to detect meta-patterns
        if len(self.pattern_history) < 5:
            return meta_patterns
        
        # Look for common evolution types
        evolution_types = defaultdict(int)
        evolution_examples = defaultdict(list)  # Store examples of each evolution type
        
        for pattern_id, evolutions in self.pattern_evolution.items():
            for evolution in evolutions:
                evolution_type = evolution["evolution_type"]
                evolution_types[evolution_type] += 1
                
                # Store detailed example (limiting to 5 examples per type to avoid excessive logging)
                if len(evolution_examples[evolution_type]) < 5:
                    # Extract specific fields based on evolution type
                    details = {}
                    if evolution_type == "object_evolution":
                        details = {
                            "from_object": evolution.get("from_object"),
                            "to_object": evolution.get("to_object")
                        }
                    elif evolution_type == "subject_evolution":
                        details = {
                            "from_subject": evolution.get("from_subject"),
                            "to_subject": evolution.get("to_subject")
                        }
                    elif evolution_type == "predicate_evolution":
                        details = {
                            "from_predicate": evolution.get("from_predicate"),
                            "to_predicate": evolution.get("to_predicate")
                        }
                    
                    evolution_examples[evolution_type].append({
                        "from_pattern": evolution.get("from_pattern"),
                        "to_pattern": evolution.get("to_pattern"),
                        "details": details
                    })
        
        # Log detailed evolution type statistics
        for evolution_type, count in evolution_types.items():
            logging.info(f"Evolution type: {evolution_type}, Count: {count}")
            if evolution_examples[evolution_type]:
                logging.info(f"Examples of {evolution_type}:")
                for i, example in enumerate(evolution_examples[evolution_type]):
                    logging.info(f"  Example {i+1}: {example}")
        
        # Detect meta-patterns based on frequent evolution types
        for evolution_type, count in evolution_types.items():
            if count >= self.threshold:
                meta_pattern = {
                    "id": f"meta_pattern_{len(meta_patterns)}_{evolution_type}",
                    "type": "evolution_meta_pattern",
                    "evolution_type": evolution_type,
                    "frequency": count,
                    "confidence": min(1.0, count / 10.0),  # Simple confidence calculation
                    "detection_timestamp": datetime.now().isoformat(),
                    "examples": evolution_examples[evolution_type]  # Include examples in the meta-pattern
                }
                
                # Log detailed meta-pattern information
                logging.info(f"Detected meta-pattern: {meta_pattern['id']}")
                logging.info(f"  Evolution type: {evolution_type}")
                logging.info(f"  Frequency: {count}")
                logging.info(f"  Confidence: {meta_pattern['confidence']}")
                logging.info(f"  Examples: {len(meta_pattern['examples'])} instances")
                
                # Update the AdaptiveID with this meta-pattern
                timestamp = datetime.now().isoformat()
                
                # Use update_temporal_context for each value that should be tracked over time
                self.adaptive_id.update_temporal_context(
                    "meta_pattern_detected", 
                    {
                        "type": evolution_type,
                        "data": meta_pattern,
                        "timestamp": timestamp
                    },
                    "pattern_detection"
                )
                
                meta_patterns.append(meta_pattern)
        
        # Add meta-patterns to potential patterns
        self.potential_patterns.extend(meta_patterns)
        
        return meta_patterns
    
    def get_pattern_evolution(self, pattern_id: str) -> List[Dict[str, Any]]:
        """
        Get the evolution history for a pattern.
        
        Args:
            pattern_id: ID of the pattern
            
        Returns:
            List of evolution events
        """
        if pattern_id in self.pattern_evolution:
            return self.pattern_evolution[pattern_id]
        
        # Check if this pattern evolved from another
        for pattern in self.pattern_history:
            if pattern["id"] == pattern_id and "evolved_from" in pattern:
                return self.get_pattern_evolution(pattern["evolved_from"])
        
        return []
    
    def register_with_field_observer(self, field_observer) -> None:
        """
        Register this detector with a field observer.
        
        Args:
            field_observer: The field observer to register with
        """
        self.adaptive_id.register_with_field_observer(field_observer)
