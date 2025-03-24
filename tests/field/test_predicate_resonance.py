"""
Test module for predicate resonance and transformation detection.
Implements a minimal version of the topological syntax approach.
"""

import unittest
import json
import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Mock data structures for testing
@dataclass
class Actant:
    """Represents an entity that appears as subject or object in predicates."""
    id: str
    name: str
    aliases: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []

@dataclass
class Predicate:
    """Represents a subject-verb-object structure."""
    id: str
    subject: str  # Actant ID
    verb: str
    object: str   # Actant ID
    text: str
    domain_id: str
    position: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "subject": self.subject,
            "verb": self.verb,
            "object": self.object,
            "text": self.text,
            "domain_id": self.domain_id,
            "position": self.position
        }

@dataclass
class Domain:
    """Represents a semantic domain."""
    id: str
    name: str
    predicates: List[str] = None  # List of predicate IDs
    
    def __post_init__(self):
        if self.predicates is None:
            self.predicates = []

@dataclass
class PredicateTransformation:
    """Represents a transformation between predicates."""
    source_id: str
    target_id: str
    carrying_actants: List[str]  # Actant IDs
    
    # Oscillation properties
    amplitude: float = 0.5       # Intensity (0-1)
    frequency: float = 0.5       # How often this occurs (0-1)
    phase: float = 0.0           # Relative timing (0-1)
    
    # Feedback properties
    is_feedback_loop: bool = False
    gain: float = 1.0            # >1 = amplifying, <1 = dampening
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "carrying_actants": self.carrying_actants,
            "oscillation": {
                "amplitude": self.amplitude,
                "frequency": self.frequency,
                "phase": self.phase
            },
            "feedback": {
                "is_loop": self.is_feedback_loop,
                "gain": self.gain
            }
        }


class PredicateResonanceDetector:
    """Detects resonance and transformations between predicates."""
    
    def __init__(self):
        self.actants = {}  # id -> Actant
        self.predicates = {}  # id -> Predicate
        self.domains = {}  # id -> Domain
        self.transformations = []  # List of PredicateTransformation
    
    def add_actant(self, actant: Actant) -> None:
        """Add an actant to the detector."""
        self.actants[actant.id] = actant
    
    def add_predicate(self, predicate: Predicate) -> None:
        """Add a predicate to the detector."""
        self.predicates[predicate.id] = predicate
        
        # Add predicate to its domain
        if predicate.domain_id in self.domains:
            if predicate.id not in self.domains[predicate.domain_id].predicates:
                self.domains[predicate.domain_id].predicates.append(predicate.id)
    
    def add_domain(self, domain: Domain) -> None:
        """Add a domain to the detector."""
        self.domains[domain.id] = domain
    
    def detect_transformations(self) -> List[PredicateTransformation]:
        """Detect transformations between predicates across domains."""
        transformations = []
        
        # Get all pairs of domains
        domain_ids = list(self.domains.keys())
        for i in range(len(domain_ids)):
            for j in range(i+1, len(domain_ids)):
                source_domain_id = domain_ids[i]
                target_domain_id = domain_ids[j]
                
                # Find transformations between these domains
                domain_transformations = self._detect_between_domains(
                    source_domain_id, target_domain_id)
                transformations.extend(domain_transformations)
        
        self.transformations = transformations
        return transformations
    
    def _detect_between_domains(
        self, source_domain_id: str, target_domain_id: str
    ) -> List[PredicateTransformation]:
        """Detect transformations between two specific domains."""
        transformations = []
        
        # Get predicates for each domain
        source_predicates = [self.predicates[pid] for pid in 
                            self.domains[source_domain_id].predicates]
        target_predicates = [self.predicates[pid] for pid in 
                            self.domains[target_domain_id].predicates]
        
        # Find shared actants between domains
        source_actants = set()
        for pred in source_predicates:
            source_actants.add(pred.subject)
            source_actants.add(pred.object)
            
        target_actants = set()
        for pred in target_predicates:
            target_actants.add(pred.subject)
            target_actants.add(pred.object)
            
        shared_actants = source_actants.intersection(target_actants)
        
        # For each shared actant, find predicate transformations
        for actant_id in shared_actants:
            # Find predicates involving this actant in each domain
            source_actant_predicates = [p for p in source_predicates 
                                      if p.subject == actant_id or p.object == actant_id]
            target_actant_predicates = [p for p in target_predicates 
                                      if p.subject == actant_id or p.object == actant_id]
            
            # Compare each pair of predicates
            for source_pred in source_actant_predicates:
                for target_pred in target_actant_predicates:
                    # Calculate resonance between predicates
                    resonance = self._calculate_resonance(source_pred, target_pred)
                    
                    # Always create a transformation with the calculated resonance
                    # (We'll let visualization filter by amplitude if needed)
                    transformation = PredicateTransformation(
                        source_id=source_pred.id,
                        target_id=target_pred.id,
                        carrying_actants=[actant_id],
                        amplitude=resonance["amplitude"],
                        frequency=resonance["frequency"],
                        phase=resonance["phase"]
                    )
                    transformations.append(transformation)
        
        return transformations
    
    def _calculate_resonance(
        self, source_pred: Predicate, target_pred: Predicate
    ) -> Dict[str, float]:
        """
        Calculate resonance properties between two predicates.
        
        This implementation captures both direct relationships and emergent wordforms
        at transition points between domains.
        """
        # Get actant names for more meaningful analysis
        source_subject_id = source_pred.subject
        source_object_id = source_pred.object
        target_subject_id = target_pred.subject
        target_object_id = target_pred.object
        
        # Identify shared actants
        shared_actants = []
        if source_subject_id == target_subject_id or source_subject_id == target_object_id:
            shared_actants.append(source_subject_id)
        if source_object_id == target_subject_id or source_object_id == target_object_id:
            shared_actants.append(source_object_id)
        
        # Calculate verb relationship
        # Instead of binary similarity, use a spectrum to capture emergent meanings
        verb_relationship = 0.0
        
        # Direct verb relationships
        if source_pred.verb == target_pred.verb:
            verb_relationship = 1.0  # Same verb
        elif source_pred.verb in ["rises", "increases"] and target_pred.verb in ["threatens", "erodes"]:
            verb_relationship = 0.8  # Causal relationship
        elif source_pred.verb in ["threatens", "erodes"] and target_pred.verb in ["protects", "adapts"]:
            verb_relationship = 0.7  # Response relationship
        elif source_pred.verb in ["adapts", "relocates"] and target_pred.verb in ["protects"]:
            verb_relationship = 0.6  # Strategic relationship
        else:
            # Base level relationship for all predicates sharing actants
            verb_relationship = 0.4
        
        # Role transformation patterns
        role_pattern = "stable"  # Default
        
        # Check for role inversions (subject becomes object or vice versa)
        if source_subject_id == target_object_id and source_object_id == target_subject_id:
            role_pattern = "inversion"
            verb_relationship = max(verb_relationship, 0.7)  # Inversions have high resonance
        
        # Check for role shifts (actant changes from subject to object or vice versa)
        elif source_subject_id == target_object_id or source_object_id == target_subject_id:
            role_pattern = "shift"
            verb_relationship = max(verb_relationship, 0.6)  # Shifts have moderate-high resonance
        
        # Calculate amplitude based on verb relationship and role patterns
        amplitude = verb_relationship
        
        # Calculate frequency - how common this transformation pattern is
        # For now use a simplified approach
        frequency = 0.5
        if role_pattern == "inversion":
            frequency = 0.3  # Inversions are less common
        elif role_pattern == "shift":
            frequency = 0.4  # Shifts are moderately common
        
        # Calculate phase - relative position in transformation sequence
        # For simplicity, use domain positions as a proxy
        source_domain_id = source_pred.domain_id
        target_domain_id = target_pred.domain_id
        
        # Extract domain numbers from IDs (assuming format "d1", "d2", etc.)
        try:
            source_num = int(source_domain_id[1:])
            target_num = int(target_domain_id[1:])
            total_domains = max(source_num, target_num)
            
            # Normalize to 0-1 range
            phase = abs(target_num - source_num) / total_domains
        except:
            phase = 0.0
        
        return {
            "amplitude": amplitude,
            "frequency": frequency,
            "phase": phase
        }
    
    def detect_feedback_loops(self) -> List[Dict[str, Any]]:
        """Detect feedback loops in the transformation network."""
        if not self.transformations:
            self.detect_transformations()
            
        # Build a directed graph from transformations
        graph = {}
        for t in self.transformations:
            if t.source_id not in graph:
                graph[t.source_id] = []
            graph[t.source_id].append(t.target_id)
        
        # Find all cycles in the graph
        cycles = self._find_cycles(graph)
        
        # Convert cycles to feedback loops
        feedback_loops = []
        for cycle in cycles:
            # Get the transformations in this cycle
            cycle_transformations = []
            for i in range(len(cycle)):
                source_id = cycle[i]
                target_id = cycle[(i + 1) % len(cycle)]
                
                # Find the transformation between these predicates
                for t in self.transformations:
                    if t.source_id == source_id and t.target_id == target_id:
                        cycle_transformations.append(t)
                        break
            
            # Calculate cycle properties
            cycle_amplitude = np.mean([t.amplitude for t in cycle_transformations])
            cycle_gain = np.prod([t.gain for t in cycle_transformations])
            
            # Mark transformations as part of a feedback loop
            for t in cycle_transformations:
                t.is_feedback_loop = True
                t.gain = cycle_gain ** (1 / len(cycle_transformations))
            
            feedback_loops.append({
                "cycle": cycle,
                "transformations": cycle_transformations,
                "amplitude": cycle_amplitude,
                "gain": cycle_gain,
                "is_amplifying": cycle_gain > 1.0
            })
        
        return feedback_loops
    
    def _find_cycles(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Find all cycles in a directed graph using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            if node in graph:
                for neighbor in graph[node]:
                    dfs(neighbor, path + [neighbor])
            
            rec_stack.remove(node)
        
        for node in graph:
            dfs(node, [node])
        
        return cycles


class TestPredicateResonance(unittest.TestCase):
    """Test cases for predicate resonance detection."""
    
    def setUp(self):
        """Set up test data."""
        self.detector = PredicateResonanceDetector()
        
        # Create test domains
        self.domains = [
            Domain(id="d1", name="Climate Science"),
            Domain(id="d2", name="Coastal Infrastructure"),
            Domain(id="d3", name="Community Planning")
        ]
        
        # Create test actants
        self.actants = [
            Actant(id="a1", name="sea level", aliases=["ocean level"]),
            Actant(id="a2", name="coastline", aliases=["shore", "coast"]),
            Actant(id="a3", name="community", aliases=["town", "residents"]),
            Actant(id="a4", name="infrastructure", aliases=["buildings", "roads"])
        ]
        
        # Create test predicates
        self.predicates = [
            # Domain 1: Climate Science
            Predicate(id="p1", subject="a1", verb="rises", object="a2", 
                     text="Sea level rises along the coastline", domain_id="d1"),
            Predicate(id="p2", subject="a1", verb="threatens", object="a4", 
                     text="Sea level threatens infrastructure", domain_id="d1"),
            
            # Domain 2: Coastal Infrastructure
            Predicate(id="p3", subject="a2", verb="erodes", object="a4", 
                     text="Coastline erodes infrastructure", domain_id="d2"),
            Predicate(id="p4", subject="a4", verb="protects", object="a2", 
                     text="Infrastructure protects coastline", domain_id="d2"),
            
            # Domain 3: Community Planning
            Predicate(id="p5", subject="a3", verb="adapts", object="a1", 
                     text="Community adapts to sea level", domain_id="d3"),
            Predicate(id="p6", subject="a3", verb="relocates", object="a2", 
                     text="Community relocates from coastline", domain_id="d3")
        ]
        
        # Add all test data to the detector
        for domain in self.domains:
            self.detector.add_domain(domain)
            
        for actant in self.actants:
            self.detector.add_actant(actant)
            
        for predicate in self.predicates:
            self.detector.add_predicate(predicate)
    
    def test_transformation_detection(self):
        """Test basic transformation detection."""
        transformations = self.detector.detect_transformations()
        
        # We should have detected some transformations
        self.assertTrue(len(transformations) > 0)
        
        # Print transformation details for inspection
        print(f"\nDetected {len(transformations)} transformations:")
        for i, t in enumerate(transformations):
            source_pred = self.detector.predicates[t.source_id]
            target_pred = self.detector.predicates[t.target_id]
            
            # Get actant names
            carrying_actant_names = [self.detector.actants[a_id].name 
                                   for a_id in t.carrying_actants]
            
            print(f"  {i+1}. {source_pred.text} → {target_pred.text}")
            print(f"     Carrying actants: {', '.join(carrying_actant_names)}")
            print(f"     Amplitude: {t.amplitude:.2f}, Frequency: {t.frequency:.2f}")
    
    def test_role_inversion(self):
        """Test detection of role inversions."""
        transformations = self.detector.detect_transformations()
        
        # Find transformations with role inversions
        inversions = []
        for t in transformations:
            source_pred = self.detector.predicates[t.source_id]
            target_pred = self.detector.predicates[t.target_id]
            
            # Check if there's a complete role inversion (coastline erodes infrastructure → infrastructure protects coastline)
            if ((source_pred.subject == target_pred.object and source_pred.object == target_pred.subject) or
                # Also check for partial inversions where one actant changes role
                (source_pred.subject == target_pred.object or source_pred.object == target_pred.subject)):
                inversions.append(t)
        
        # We should have detected inversions
        self.assertTrue(len(inversions) > 0)
        
        # Print inversion details
        print(f"\nDetected {len(inversions)} role inversions:")
        for i, t in enumerate(inversions):
            source_pred = self.detector.predicates[t.source_id]
            target_pred = self.detector.predicates[t.target_id]
            print(f"  {i+1}. {source_pred.text} → {target_pred.text}")
            print(f"     Amplitude: {t.amplitude:.2f}")
    
    def test_feedback_loops(self):
        """Test detection of feedback loops."""
        # First detect transformations
        self.detector.detect_transformations()
        
        # Then detect feedback loops
        feedback_loops = self.detector.detect_feedback_loops()
        
        # Print feedback loop details
        print(f"\nDetected {len(feedback_loops)} feedback loops:")
        for i, loop in enumerate(feedback_loops):
            cycle_predicates = [self.detector.predicates[pid] for pid in loop["cycle"]]
            cycle_text = " → ".join([p.text for p in cycle_predicates])
            
            print(f"  {i+1}. {cycle_text}")
            print(f"     Amplitude: {loop['amplitude']:.2f}, Gain: {loop['gain']:.2f}")
            print(f"     {'Amplifying' if loop['is_amplifying'] else 'Dampening'} feedback")
    
    def test_impedance_detection(self):
        """Test detection of impedance in transformation flow."""
        # Add a predicate that creates impedance
        impedance_predicate = Predicate(
            id="p7", 
            subject="a4", 
            verb="fails", 
            object="a3", 
            text="Infrastructure fails to protect community", 
            domain_id="d2"
        )
        self.detector.add_predicate(impedance_predicate)
        
        # Detect transformations
        transformations = self.detector.detect_transformations()
        
        # Find transformations involving the impedance predicate
        impedance_transformations = [t for t in transformations 
                                   if t.source_id == "p7" or t.target_id == "p7"]
        
        # We should have detected some transformations with the impedance predicate
        self.assertTrue(len(impedance_transformations) > 0)
        
        # Print impedance details
        print(f"\nDetected {len(impedance_transformations)} impedance transformations:")
        for i, t in enumerate(impedance_transformations):
            source_pred = self.detector.predicates[t.source_id]
            target_pred = self.detector.predicates[t.target_id]
            print(f"  {i+1}. {source_pred.text} → {target_pred.text}")
            print(f"     Amplitude: {t.amplitude:.2f}")


if __name__ == "__main__":
    unittest.main()
