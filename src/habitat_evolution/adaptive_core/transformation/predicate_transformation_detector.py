"""
Predicate Transformation Detector

Detects transformations of predicates across domain boundaries, with special
attention to emergent wordforms at transition points.

This module implements the topological syntax approach that balances structure and
emergence, treating predicate edges as resonant oscillators rather than discrete
"tiddly winks" entities.
"""

from typing import Dict, List, Tuple, Any, Optional, Set, Union
from dataclasses import dataclass, field
import uuid
import math
import numpy as np
from datetime import datetime

from ..id.adaptive_id import AdaptiveID


@dataclass
class EmergentForm:
    """
    Represents an emergent wordform that appears at transition points between predicates.
    
    These are not explicitly stated in the text but emerge from the relationship
    between predicates across domain boundaries.
    """
    id: str
    source_predicate_id: str
    target_predicate_id: str
    form_text: str
    confidence: float
    adaptive_id: Optional[AdaptiveID] = None
    
    @classmethod
    def create(cls, source_predicate_id: str, target_predicate_id: str, 
               form_text: str, confidence: float = 0.5):
        """Create a new emergent form."""
        return cls(
            id=str(uuid.uuid4()),
            source_predicate_id=source_predicate_id,
            target_predicate_id=target_predicate_id,
            form_text=form_text,
            confidence=confidence
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "source_predicate_id": self.source_predicate_id,
            "target_predicate_id": self.target_predicate_id,
            "form_text": self.form_text,
            "confidence": self.confidence
        }


@dataclass
class TransformationEdge:
    """
    Represents a transformation edge between predicates.
    
    Unlike simple connections, these edges have oscillatory properties and can
    participate in feedback loops. They also track emergent forms that appear
    at the transition points.
    """
    id: str
    source_id: str
    target_id: str
    carrying_actants: List[str]
    
    # Oscillation properties
    amplitude: float = 0.5       # Intensity (0-1)
    frequency: float = 0.5       # How often this occurs (0-1)
    phase: float = 0.0           # Relative timing (0-1)
    harmonics: List[float] = field(default_factory=list)  # Secondary resonance patterns
    
    # Feedback properties
    is_feedback_loop: bool = False
    gain: float = 1.0            # >1 = amplifying, <1 = dampening
    
    # Emergent forms at this transition point
    emergent_forms: List[EmergentForm] = field(default_factory=list)
    
    # Role transformation patterns
    role_pattern: str = "stable"  # stable, shift, inversion
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "carrying_actants": self.carrying_actants,
            "oscillation": {
                "amplitude": self.amplitude,
                "frequency": self.frequency,
                "phase": self.phase,
                "harmonics": self.harmonics
            },
            "feedback": {
                "is_loop": self.is_feedback_loop,
                "gain": self.gain
            },
            "role_pattern": self.role_pattern,
            "emergent_forms": [ef.to_dict() for ef in self.emergent_forms]
        }
    
    def add_emergent_form(self, form_text: str, confidence: float = 0.5) -> EmergentForm:
        """Add an emergent form to this transformation edge."""
        form = EmergentForm.create(
            source_predicate_id=self.source_id,
            target_predicate_id=self.target_id,
            form_text=form_text,
            confidence=confidence
        )
        self.emergent_forms.append(form)
        return form


class PredicateTransformationDetector:
    """
    Detects transformations between predicates across domain boundaries.
    
    This implementation focuses on detecting not just direct connections between
    predicates, but also the emergent wordforms that appear at transition points.
    It treats predicate edges as resonant oscillators rather than discrete entities.
    """
    
    def __init__(self):
        """Initialize the detector."""
        self.actants = {}  # id -> Actant
        self.predicates = {}  # id -> Predicate
        self.domains = {}  # id -> Domain
        self.transformations = []  # List of TransformationEdge
        self.emergent_forms = []  # List of EmergentForm
        
        # Transformation patterns that can generate emergent forms
        self.transformation_patterns = {
            # Pattern: (source_verb, target_verb, role_pattern) -> (emergent_form_template, confidence)
            ("rises", "threatens", "stable"): ("increasing risk of {object}", 0.7),
            ("threatens", "erodes", "stable"): ("accelerating damage to {object}", 0.6),
            ("erodes", "protects", "inversion"): ("defensive response to {subject} loss", 0.8),
            ("threatens", "adapts", "shift"): ("adaptive strategy against {subject}", 0.7),
            ("rises", "relocates", "shift"): ("managed retreat from {object}", 0.6),
        }
    
    def add_actant(self, actant) -> None:
        """Add an actant to the detector."""
        self.actants[actant.id] = actant
    
    def add_predicate(self, predicate) -> None:
        """Add a predicate to the detector."""
        self.predicates[predicate.id] = predicate
        
        # Add predicate to its domain
        if predicate.domain_id in self.domains:
            if predicate.id not in self.domains[predicate.domain_id].predicates:
                self.domains[predicate.domain_id].predicates.append(predicate.id)
    
    def add_domain(self, domain) -> None:
        """Add a domain to the detector."""
        self.domains[domain.id] = domain
    
    def detect_transformations(self) -> List[TransformationEdge]:
        """
        Detect transformations between predicates across domains.
        
        Returns a list of TransformationEdge objects representing the detected
        transformations, including their oscillatory properties and emergent forms.
        """
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
        
        # Detect emergent forms at transition points
        for transformation in transformations:
            self._detect_emergent_forms(transformation)
        
        self.transformations = transformations
        return transformations
    
    def _detect_between_domains(
        self, source_domain_id: str, target_domain_id: str
    ) -> List[TransformationEdge]:
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
                    # Calculate resonance and role pattern
                    resonance, role_pattern = self._calculate_resonance(source_pred, target_pred)
                    
                    # Create transformation edge
                    transformation = TransformationEdge(
                        id=str(uuid.uuid4()),
                        source_id=source_pred.id,
                        target_id=target_pred.id,
                        carrying_actants=[actant_id],
                        amplitude=resonance["amplitude"],
                        frequency=resonance["frequency"],
                        phase=resonance["phase"],
                        harmonics=resonance.get("harmonics", []),
                        role_pattern=role_pattern
                    )
                    transformations.append(transformation)
        
        return transformations
    
    def _calculate_resonance(
        self, source_pred, target_pred
    ) -> Tuple[Dict[str, Any], str]:
        """
        Calculate resonance properties and role pattern between two predicates.
        
        Returns a tuple of (resonance_properties, role_pattern).
        """
        # Get actant IDs
        source_subject_id = source_pred.subject
        source_object_id = source_pred.object
        target_subject_id = target_pred.subject
        target_object_id = target_pred.object
        
        # Calculate verb relationship
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
        frequency = 0.5
        if role_pattern == "inversion":
            frequency = 0.3  # Inversions are less common
        elif role_pattern == "shift":
            frequency = 0.4  # Shifts are moderately common
        
        # Calculate phase - relative position in transformation sequence
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
        
        # Calculate harmonics - secondary resonance patterns
        harmonics = []
        
        # Add harmonic for shared actants
        if source_subject_id == target_subject_id or source_subject_id == target_object_id:
            harmonics.append(0.6)
        if source_object_id == target_subject_id or source_object_id == target_object_id:
            harmonics.append(0.5)
        
        return {
            "amplitude": amplitude,
            "frequency": frequency,
            "phase": phase,
            "harmonics": harmonics
        }, role_pattern
    
    def _detect_emergent_forms(self, transformation: TransformationEdge) -> List[EmergentForm]:
        """
        Detect emergent forms at the transition point between predicates.
        
        These are implicit concepts that emerge from the relationship between
        predicates but aren't explicitly stated in either predicate.
        """
        source_pred = self.predicates[transformation.source_id]
        target_pred = self.predicates[transformation.target_id]
        
        # Get actant names for template filling
        actant_names = {}
        for actant_id in transformation.carrying_actants:
            if actant_id in self.actants:
                actant_names[actant_id] = self.actants[actant_id].name
        
        source_subject_name = self.actants[source_pred.subject].name if source_pred.subject in self.actants else "subject"
        source_object_name = self.actants[source_pred.object].name if source_pred.object in self.actants else "object"
        
        # Check for known transformation patterns
        pattern_key = (source_pred.verb, target_pred.verb, transformation.role_pattern)
        if pattern_key in self.transformation_patterns:
            template, confidence = self.transformation_patterns[pattern_key]
            
            # Fill in the template with actant names
            form_text = template.format(
                subject=source_subject_name,
                object=source_object_name
            )
            
            # Add the emergent form to the transformation
            emergent_form = transformation.add_emergent_form(form_text, confidence)
            self.emergent_forms.append(emergent_form)
        
        # For role inversions, always add an emergent form about the relationship change
        if transformation.role_pattern == "inversion":
            form_text = f"reciprocal relationship between {source_subject_name} and {source_object_name}"
            emergent_form = transformation.add_emergent_form(form_text, 0.8)
            self.emergent_forms.append(emergent_form)
        
        # For predicates with high amplitude, suggest a causal relationship
        if transformation.amplitude > 0.7 and source_pred.verb != target_pred.verb:
            form_text = f"{source_pred.verb} leads to {target_pred.verb}"
            emergent_form = transformation.add_emergent_form(form_text, 0.6)
            self.emergent_forms.append(emergent_form)
        
        return transformation.emergent_forms
    
    def detect_feedback_loops(self) -> List[Dict[str, Any]]:
        """
        Detect feedback loops in the transformation network.
        
        Feedback loops represent cycles in the predicate transformation network
        where predicates influence each other in a circular pattern.
        """
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
            
            # Detect emergent forms for the entire cycle
            cycle_emergent_forms = self._detect_cycle_emergent_forms(cycle, cycle_transformations)
            
            feedback_loops.append({
                "cycle": cycle,
                "transformations": cycle_transformations,
                "amplitude": cycle_amplitude,
                "gain": cycle_gain,
                "is_amplifying": cycle_gain > 1.0,
                "emergent_forms": cycle_emergent_forms
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
    
    def _detect_cycle_emergent_forms(
        self, cycle: List[str], transformations: List[TransformationEdge]
    ) -> List[EmergentForm]:
        """
        Detect emergent forms for an entire feedback cycle.
        
        These represent higher-order concepts that emerge from the cycle as a whole,
        not just from individual transformations.
        """
        emergent_forms = []
        
        # Get the predicates in the cycle
        cycle_predicates = [self.predicates[pid] for pid in cycle]
        
        # Get all actants involved in the cycle
        cycle_actants = set()
        for pred in cycle_predicates:
            cycle_actants.add(pred.subject)
            cycle_actants.add(pred.object)
        
        # Get actant names
        actant_names = {}
        for actant_id in cycle_actants:
            if actant_id in self.actants:
                actant_names[actant_id] = self.actants[actant_id].name
        
        # Create emergent forms based on cycle properties
        if len(cycle) >= 3:
            # For longer cycles, suggest a systemic relationship
            actant_list = ", ".join(actant_names.values())
            form = EmergentForm.create(
                source_predicate_id=cycle[0],
                target_predicate_id=cycle[0],  # Self-referential for cycles
                form_text=f"systemic interaction involving {actant_list}",
                confidence=0.7
            )
            emergent_forms.append(form)
        
        # Check if the cycle represents a causal chain
        verbs = [pred.verb for pred in cycle_predicates]
        if "rises" in verbs and "threatens" in verbs and "adapts" in verbs:
            form = EmergentForm.create(
                source_predicate_id=cycle[0],
                target_predicate_id=cycle[0],
                form_text="climate adaptation feedback loop",
                confidence=0.8
            )
            emergent_forms.append(form)
        
        # Check for role inversions within the cycle
        role_inversions = [t for t in transformations if t.role_pattern == "inversion"]
        if role_inversions:
            form = EmergentForm.create(
                source_predicate_id=cycle[0],
                target_predicate_id=cycle[0],
                form_text="reciprocal influence cycle",
                confidence=0.7
            )
            emergent_forms.append(form)
        
        return emergent_forms


# Example usage:
if __name__ == "__main__":
    # This would be replaced with actual data from documents
    from dataclasses import dataclass
    
    @dataclass
    class Actant:
        id: str
        name: str
        aliases: List[str] = None
        
        def __post_init__(self):
            if self.aliases is None:
                self.aliases = []
    
    @dataclass
    class Predicate:
        id: str
        subject: str
        verb: str
        object: str
        text: str
        domain_id: str
        position: int = 0
    
    @dataclass
    class Domain:
        id: str
        name: str
        predicates: List[str] = None
        
        def __post_init__(self):
            if self.predicates is None:
                self.predicates = []
    
    # Create detector
    detector = PredicateTransformationDetector()
    
    # Create test domains
    domains = [
        Domain(id="d1", name="Climate Science"),
        Domain(id="d2", name="Coastal Infrastructure"),
        Domain(id="d3", name="Community Planning")
    ]
    
    # Create test actants
    actants = [
        Actant(id="a1", name="sea level", aliases=["ocean level"]),
        Actant(id="a2", name="coastline", aliases=["shore", "coast"]),
        Actant(id="a3", name="community", aliases=["town", "residents"]),
        Actant(id="a4", name="infrastructure", aliases=["buildings", "roads"])
    ]
    
    # Create test predicates
    predicates = [
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
    for domain in domains:
        detector.add_domain(domain)
        
    for actant in actants:
        detector.add_actant(actant)
        
    for predicate in predicates:
        detector.add_predicate(predicate)
    
    # Detect transformations
    transformations = detector.detect_transformations()
    
    # Print transformation details
    print(f"Detected {len(transformations)} transformations:")
    for i, t in enumerate(transformations):
        source_pred = detector.predicates[t.source_id]
        target_pred = detector.predicates[t.target_id]
        
        print(f"  {i+1}. {source_pred.text} → {target_pred.text}")
        print(f"     Role pattern: {t.role_pattern}")
        print(f"     Amplitude: {t.amplitude:.2f}, Frequency: {t.frequency:.2f}")
        
        if t.emergent_forms:
            print(f"     Emergent forms:")
            for ef in t.emergent_forms:
                print(f"       - {ef.form_text} (confidence: {ef.confidence:.2f})")
    
    # Detect feedback loops
    feedback_loops = detector.detect_feedback_loops()
    
    # Print feedback loop details
    print(f"\nDetected {len(feedback_loops)} feedback loops:")
    for i, loop in enumerate(feedback_loops):
        cycle_predicates = [detector.predicates[pid] for pid in loop["cycle"]]
        cycle_text = " → ".join([p.text for p in cycle_predicates])
        
        print(f"  {i+1}. {cycle_text}")
        print(f"     Amplitude: {loop['amplitude']:.2f}, Gain: {loop['gain']:.2f}")
        print(f"     {'Amplifying' if loop['is_amplifying'] else 'Dampening'} feedback")
        
        if loop.get("emergent_forms"):
            print(f"     Emergent forms:")
            for ef in loop["emergent_forms"]:
                print(f"       - {ef.form_text} (confidence: {ef.confidence:.2f})")
