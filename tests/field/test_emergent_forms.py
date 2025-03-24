"""
Test for detecting emergent forms at predicate transformation transition points.

This test validates that our system can identify not just direct transformations
between predicates, but also the emergent wordforms that appear at transition points.
"""

import unittest
import sys
import os
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from habitat_evolution.adaptive_core.transformation.predicate_transformation_detector import (
    PredicateTransformationDetector, TransformationEdge, EmergentForm
)


class TestEmergentForms(unittest.TestCase):
    """Test suite for emergent forms detection."""
    
    def setUp(self):
        """Set up test data."""
        # Create detector
        self.detector = PredicateTransformationDetector()
        
        # Create test domains
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
            self.detector.add_domain(domain)
            
        for actant in actants:
            self.detector.add_actant(actant)
            
        for predicate in predicates:
            self.detector.add_predicate(predicate)
    
    def test_emergent_forms_detection(self):
        """Test that emergent forms are detected at transformation transition points."""
        # Detect transformations
        transformations = self.detector.detect_transformations()
        
        # Check that we have transformations
        self.assertTrue(len(transformations) > 0)
        
        # Check that at least some transformations have emergent forms
        transformations_with_forms = [t for t in transformations if t.emergent_forms]
        self.assertTrue(len(transformations_with_forms) > 0)
        
        # Print emergent forms for inspection
        print(f"\nDetected {len(transformations)} transformations, {len(transformations_with_forms)} with emergent forms:")
        for i, t in enumerate(transformations_with_forms):
            source_pred = self.detector.predicates[t.source_id]
            target_pred = self.detector.predicates[t.target_id]
            
            print(f"  {i+1}. {source_pred.text} → {target_pred.text}")
            print(f"     Role pattern: {t.role_pattern}")
            print(f"     Amplitude: {t.amplitude:.2f}, Frequency: {t.frequency:.2f}")
            
            print(f"     Emergent forms:")
            for ef in t.emergent_forms:
                print(f"       - {ef.form_text} (confidence: {ef.confidence:.2f})")
    
    def test_feedback_loop_emergent_forms(self):
        """Test that emergent forms are detected for feedback loops."""
        # Create a feedback loop by adding two new predicates that complete a cycle
        from dataclasses import dataclass
        
        @dataclass
        class Predicate:
            id: str
            subject: str
            verb: str
            object: str
            text: str
            domain_id: str
            position: int = 0
        
        # Add predicates that create a cycle: p1 → p3 → p7 → p1
        # Sea level rises along coastline → Coastline erodes infrastructure → Infrastructure affects sea level → (back to) Sea level rises along coastline
        feedback_predicate = Predicate(
            id="p7", 
            subject="a4", 
            verb="affects", 
            object="a1", 
            text="Infrastructure affects sea level", 
            domain_id="d2"
        )
        
        self.detector.add_predicate(feedback_predicate)
        if "p7" not in self.detector.domains["d2"].predicates:
            self.detector.domains["d2"].predicates.append("p7")
        
        # Detect transformations and feedback loops
        transformations = self.detector.detect_transformations()
        
        # Manually create a cycle for testing purposes
        # This simulates the graph structure that would be detected
        p1_to_p3 = next((t for t in transformations 
                         if t.source_id == "p1" and t.target_id == "p3"), None)
        p3_to_p7 = next((t for t in transformations 
                         if t.source_id == "p3" and t.target_id == "p7"), None)
        p7_to_p1 = next((t for t in transformations 
                         if t.source_id == "p7" and t.target_id == "p1"), None)
        
        # If any of these transformations don't exist, create them
        if not p1_to_p3:
            p1_to_p3 = TransformationEdge(
                id="t_p1_p3",
                source_id="p1",
                target_id="p3",
                carrying_actants=["a2"],  # coastline
                amplitude=0.7,
                frequency=0.4,
                role_pattern="shift"
            )
            transformations.append(p1_to_p3)
            self.detector.transformations.append(p1_to_p3)
            
        if not p3_to_p7:
            p3_to_p7 = TransformationEdge(
                id="t_p3_p7",
                source_id="p3",
                target_id="p7",
                carrying_actants=["a4"],  # infrastructure
                amplitude=0.6,
                frequency=0.5,
                role_pattern="shift"
            )
            transformations.append(p3_to_p7)
            self.detector.transformations.append(p3_to_p7)
            
        if not p7_to_p1:
            p7_to_p1 = TransformationEdge(
                id="t_p7_p1",
                source_id="p7",
                target_id="p1",
                carrying_actants=["a1"],  # sea level
                amplitude=0.8,
                frequency=0.3,
                role_pattern="stable"
            )
            transformations.append(p7_to_p1)
            self.detector.transformations.append(p7_to_p1)
        
        # Now detect feedback loops
        feedback_loops = self.detector.detect_feedback_loops()
        
        # Check that we have at least one feedback loop
        self.assertTrue(len(feedback_loops) > 0, "No feedback loops detected")
        
        # Check that the feedback loop has emergent forms
        loops_with_forms = [loop for loop in feedback_loops if loop.get("emergent_forms")]
        self.assertTrue(len(loops_with_forms) > 0, "No emergent forms detected in feedback loops")
        
        # Print feedback loop emergent forms for inspection
        print(f"\nDetected {len(feedback_loops)} feedback loops, {len(loops_with_forms)} with emergent forms:")
        for i, loop in enumerate(loops_with_forms):
            cycle_predicates = [self.detector.predicates[pid] for pid in loop["cycle"]]
            cycle_text = " → ".join([p.text for p in cycle_predicates])
            
            print(f"  {i+1}. {cycle_text}")
            print(f"     Amplitude: {loop['amplitude']:.2f}, Gain: {loop['gain']:.2f}")
            print(f"     {'Amplifying' if loop['is_amplifying'] else 'Dampening'} feedback")
            
            print(f"     Emergent forms:")
            for ef in loop["emergent_forms"]:
                print(f"       - {ef.form_text} (confidence: {ef.confidence:.2f})")


if __name__ == "__main__":
    unittest.main()
