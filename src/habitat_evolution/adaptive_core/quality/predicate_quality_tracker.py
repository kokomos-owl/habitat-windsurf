"""
Predicate Quality Tracker for Habitat Evolution.

This module implements quality tracking for predicates (relationship types) in the
Habitat Evolution system, enabling predicates to evolve in quality similar to entities.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import json
import os
from pathlib import Path

# Quality states for predicates
QUALITY_POOR = "poor"
QUALITY_UNCERTAIN = "uncertain"
QUALITY_GOOD = "good"

# Valid transitions between quality states
VALID_TRANSITIONS = {
    QUALITY_POOR: {QUALITY_UNCERTAIN, QUALITY_GOOD},
    QUALITY_UNCERTAIN: {QUALITY_POOR, QUALITY_GOOD},
    QUALITY_GOOD: {QUALITY_UNCERTAIN, QUALITY_POOR}
}

class PredicateQualityTracker:
    """
    Tracks quality states and transitions for predicates (relationship types).
    
    This class enables predicates to evolve in quality similar to entities,
    creating a more complete co-evolutionary system where both entities and
    their relationships can improve through contextual reinforcement.
    """
    
    def __init__(self, event_bus=None, logger=None):
        """
        Initialize the predicate quality tracker.
        
        Args:
            event_bus: Event bus for publishing quality transition events
            logger: Logger instance for tracking quality changes
        """
        self.logger = logger or logging.getLogger(__name__)
        self.event_bus = event_bus
        
        # Dictionary mapping predicates to their current quality
        self.predicate_quality: Dict[str, str] = {}
        
        # Dictionary mapping domain pairs to predicate specializations
        # Format: (source_domain, target_domain) -> {predicate: confidence}
        self.domain_predicate_specialization: Dict[Tuple[str, str], Dict[str, float]] = {}
        
        # History of quality transitions for each predicate
        self.quality_transition_history: Dict[str, List[Dict]] = {}
        
        # Confidence scores for each predicate
        self.predicate_confidence: Dict[str, float] = {}
        
        # Initialize with default predicates as uncertain
        default_predicates = [
            "causes", "affects", "damages", "mitigates", 
            "part_of", "contains", "component_of", 
            "protects_against", "analyzes", "evaluates",
            "precedes", "concurrent_with", "follows", "during",
            "located_in", "adjacent_to", "implements", "monitors"
        ]
        
        for predicate in default_predicates:
            self.predicate_quality[predicate] = QUALITY_UNCERTAIN
            self.predicate_confidence[predicate] = 0.5  # Start with neutral confidence
            self.quality_transition_history[predicate] = []
    
    def get_predicate_quality(self, predicate: str) -> str:
        """
        Get the current quality state of a predicate.
        
        Args:
            predicate: The predicate to check
            
        Returns:
            The quality state (poor, uncertain, good) of the predicate
        """
        return self.predicate_quality.get(predicate, QUALITY_UNCERTAIN)
    
    def get_predicate_confidence(self, predicate: str) -> float:
        """
        Get the confidence score for a predicate.
        
        Args:
            predicate: The predicate to check
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        return self.predicate_confidence.get(predicate, 0.5)
    
    def get_domain_specific_predicates(self, source_domain: str, target_domain: str) -> Dict[str, float]:
        """
        Get predicates specialized for a specific domain pair with their confidence scores.
        
        Args:
            source_domain: Source entity domain
            target_domain: Target entity domain
            
        Returns:
            Dictionary mapping predicates to confidence scores for this domain pair
        """
        return self.domain_predicate_specialization.get((source_domain, target_domain), {})
    
    def transition_predicate_quality(self, predicate: str, to_quality: str, 
                                    source_domain: Optional[str] = None, 
                                    target_domain: Optional[str] = None,
                                    evidence: Optional[str] = None) -> bool:
        """
        Transition a predicate to a new quality state.
        
        Args:
            predicate: The predicate to transition
            to_quality: The target quality state
            source_domain: Optional source domain for domain-specific transitions
            target_domain: Optional target domain for domain-specific transitions
            evidence: Optional evidence supporting this transition
            
        Returns:
            True if transition was successful, False otherwise
        """
        current_quality = self.get_predicate_quality(predicate)
        
        # Check if transition is valid
        if to_quality not in VALID_TRANSITIONS.get(current_quality, set()):
            self.logger.warning(f"Invalid quality transition for predicate {predicate}: {current_quality} -> {to_quality}")
            return False
        
        # Update quality
        self.predicate_quality[predicate] = to_quality
        
        # Update confidence based on quality
        quality_confidence = {
            QUALITY_POOR: 0.2,
            QUALITY_UNCERTAIN: 0.5,
            QUALITY_GOOD: 0.8
        }
        self.predicate_confidence[predicate] = quality_confidence.get(to_quality, 0.5)
        
        # Record transition in history
        transition = {
            'timestamp': datetime.now().isoformat(),
            'from_quality': current_quality,
            'to_quality': to_quality,
            'source_domain': source_domain,
            'target_domain': target_domain,
            'evidence': evidence
        }
        
        if predicate not in self.quality_transition_history:
            self.quality_transition_history[predicate] = []
        
        self.quality_transition_history[predicate].append(transition)
        
        # Update domain-specific specialization if domains provided
        if source_domain and target_domain:
            domain_pair = (source_domain, target_domain)
            if domain_pair not in self.domain_predicate_specialization:
                self.domain_predicate_specialization[domain_pair] = {}
            
            # Increase confidence for this domain pair
            current_confidence = self.domain_predicate_specialization[domain_pair].get(predicate, 0.5)
            
            # Adjust confidence based on transition direction
            if to_quality == QUALITY_GOOD:
                new_confidence = min(1.0, current_confidence + 0.1)
            elif to_quality == QUALITY_UNCERTAIN:
                new_confidence = 0.5  # Reset to neutral
            else:  # QUALITY_POOR
                new_confidence = max(0.1, current_confidence - 0.1)
                
            self.domain_predicate_specialization[domain_pair][predicate] = new_confidence
        
        # Log the transition
        self.logger.info(f"Predicate quality transition: {predicate} from {current_quality} to {to_quality}")
        
        # Publish event if event bus is available
        if self.event_bus:
            event_data = {
                'predicate': predicate,
                'from_quality': current_quality,
                'to_quality': to_quality,
                'source_domain': source_domain,
                'target_domain': target_domain,
                'confidence': self.predicate_confidence[predicate]
            }
            
            event = {
                'event_type': 'predicate.quality.transition',
                'source': 'predicate_quality_tracker',
                'data': event_data
            }
            
            self.event_bus.publish(event)
        
        return True
    
    def reinforce_predicate_from_context(self, predicate: str, 
                                        source_entity: str, source_domain: str,
                                        target_entity: str, target_domain: str,
                                        context_evidence: str) -> bool:
        """
        Reinforce a predicate based on contextual evidence.
        
        Args:
            predicate: The predicate to reinforce
            source_entity: Source entity in the relationship
            source_domain: Domain of the source entity
            target_entity: Target entity in the relationship
            target_domain: Domain of the target entity
            context_evidence: Contextual evidence supporting this reinforcement
            
        Returns:
            True if reinforcement led to a quality transition, False otherwise
        """
        current_quality = self.get_predicate_quality(predicate)
        
        # Determine if evidence is sufficient for a quality transition
        if current_quality == QUALITY_UNCERTAIN:
            # Transition to good if we have strong contextual evidence
            return self.transition_predicate_quality(
                predicate, 
                QUALITY_GOOD, 
                source_domain, 
                target_domain, 
                f"Contextual evidence from {source_entity} to {target_entity}: {context_evidence}"
            )
        
        # No transition needed
        return False
    
    def save_to_json(self, output_dir: str) -> str:
        """
        Save the predicate quality state to a JSON file.
        
        Args:
            output_dir: Directory to save the JSON file
            
        Returns:
            Path to the saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predicate_quality_state_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        data = {
            'predicate_quality': self.predicate_quality,
            'predicate_confidence': self.predicate_confidence,
            'domain_predicate_specialization': {str(k): v for k, v in self.domain_predicate_specialization.items()},
            'quality_transition_history': self.quality_transition_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved predicate quality state to {filepath}")
        return filepath
    
    @classmethod
    def load_from_json(cls, filepath: str, event_bus=None, logger=None):
        """
        Load predicate quality state from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            event_bus: Event bus for publishing quality transition events
            logger: Logger instance for tracking quality changes
            
        Returns:
            PredicateQualityTracker instance with loaded state
        """
        instance = cls(event_bus, logger)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        instance.predicate_quality = data.get('predicate_quality', {})
        instance.predicate_confidence = data.get('predicate_confidence', {})
        
        # Convert string tuple keys back to actual tuples
        domain_spec = data.get('domain_predicate_specialization', {})
        instance.domain_predicate_specialization = {}
        for k, v in domain_spec.items():
            # Parse the string tuple representation "(domain1, domain2)"
            domains = k.strip('()').split(', ')
            if len(domains) == 2:
                instance.domain_predicate_specialization[(domains[0], domains[1])] = v
        
        instance.quality_transition_history = data.get('quality_transition_history', {})
        
        instance.logger.info(f"Loaded predicate quality state from {filepath}")
        return instance
