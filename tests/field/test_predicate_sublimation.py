"""
Test for detecting predicate sublimation in semantic networks.

This test validates that our system can identify critical thresholds where
predicate networks undergo qualitative shifts in meaning, resulting in
entirely new conceptual frameworks.
"""

import unittest
import sys
import os
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from habitat_evolution.adaptive_core.transformation.predicate_sublimation import (
    PredicateSublimationDetector, ConceptualFramework
)


class TestPredicateSublimation(unittest.TestCase):
    """Test suite for predicate sublimation detection."""
    
    def setUp(self):
        """Set up test data."""
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
        class TransformationEdge:
            id: str
            source_id: str
            target_id: str
            carrying_actants: List[str]
            amplitude: float = 0.5
            frequency: float = 0.5
            phase: float = 0.0
            role_pattern: str = "stable"
        
        # Create test data
        self.actants = {
            "a1": Actant(id="a1", name="sea level", aliases=["ocean level"]),
            "a2": Actant(id="a2", name="coastline", aliases=["shore", "coast"]),
            "a3": Actant(id="a3", name="community", aliases=["town", "residents"]),
            "a4": Actant(id="a4", name="infrastructure", aliases=["buildings", "roads"]),
            "a5": Actant(id="a5", name="policy", aliases=["regulation", "law"]),
            "a6": Actant(id="a6", name="economy", aliases=["market", "business"])
        }
        
        self.predicates = {
            # Domain 1: Climate Science
            "p1": Predicate(id="p1", subject="a1", verb="rises", object="a2", 
                           text="Sea level rises along the coastline", domain_id="d1"),
            "p2": Predicate(id="p2", subject="a1", verb="threatens", object="a4", 
                           text="Sea level threatens infrastructure", domain_id="d1"),
            
            # Domain 2: Coastal Infrastructure
            "p3": Predicate(id="p3", subject="a2", verb="erodes", object="a4", 
                           text="Coastline erodes infrastructure", domain_id="d2"),
            "p4": Predicate(id="p4", subject="a4", verb="protects", object="a2", 
                           text="Infrastructure protects coastline", domain_id="d2"),
            "p5": Predicate(id="p5", subject="a4", verb="fails", object="a3", 
                           text="Infrastructure fails to protect community", domain_id="d2"),
            
            # Domain 3: Community Planning
            "p6": Predicate(id="p6", subject="a3", verb="adapts", object="a1", 
                           text="Community adapts to sea level", domain_id="d3"),
            "p7": Predicate(id="p7", subject="a3", verb="relocates", object="a2", 
                           text="Community relocates from coastline", domain_id="d3"),
            "p8": Predicate(id="p8", subject="a3", verb="demands", object="a5", 
                           text="Community demands policy changes", domain_id="d3"),
            
            # Domain 4: Policy Response
            "p9": Predicate(id="p9", subject="a5", verb="regulates", object="a4", 
                           text="Policy regulates infrastructure development", domain_id="d4"),
            "p10": Predicate(id="p10", subject="a5", verb="funds", object="a4", 
                            text="Policy funds infrastructure improvements", domain_id="d4"),
            "p11": Predicate(id="p11", subject="a5", verb="supports", object="a3", 
                            text="Policy supports community adaptation", domain_id="d4"),
            
            # Domain 5: Economic Impact
            "p12": Predicate(id="p12", subject="a1", verb="damages", object="a6", 
                            text="Sea level damages economy", domain_id="d5"),
            "p13": Predicate(id="p13", subject="a6", verb="influences", object="a5", 
                            text="Economy influences policy decisions", domain_id="d5"),
            "p14": Predicate(id="p14", subject="a6", verb="invests", object="a4", 
                            text="Economy invests in infrastructure", domain_id="d5"),
            "p15": Predicate(id="p15", subject="a3", verb="participates", object="a6", 
                            text="Community participates in economy", domain_id="d5")
        }
        
        # Create transformations between predicates
        self.transformations = [
            # Climate Science → Coastal Infrastructure
            TransformationEdge(id="t1", source_id="p1", target_id="p3", 
                             carrying_actants=["a2"], amplitude=0.8, role_pattern="shift"),
            TransformationEdge(id="t2", source_id="p2", target_id="p4", 
                             carrying_actants=["a4"], amplitude=0.7, role_pattern="shift"),
            TransformationEdge(id="t3", source_id="p2", target_id="p5", 
                             carrying_actants=["a4"], amplitude=0.6, role_pattern="stable"),
            
            # Coastal Infrastructure → Community Planning
            TransformationEdge(id="t4", source_id="p3", target_id="p7", 
                             carrying_actants=["a2"], amplitude=0.7, role_pattern="shift"),
            TransformationEdge(id="t5", source_id="p5", target_id="p6", 
                             carrying_actants=["a3"], amplitude=0.8, role_pattern="shift"),
            TransformationEdge(id="t6", source_id="p5", target_id="p8", 
                             carrying_actants=["a3"], amplitude=0.7, role_pattern="stable"),
            
            # Community Planning → Policy Response
            TransformationEdge(id="t7", source_id="p8", target_id="p9", 
                             carrying_actants=["a5"], amplitude=0.9, role_pattern="shift"),
            TransformationEdge(id="t8", source_id="p8", target_id="p10", 
                             carrying_actants=["a5"], amplitude=0.8, role_pattern="shift"),
            TransformationEdge(id="t9", source_id="p6", target_id="p11", 
                             carrying_actants=["a3"], amplitude=0.7, role_pattern="stable"),
            
            # Climate Science → Economic Impact
            TransformationEdge(id="t10", source_id="p2", target_id="p12", 
                             carrying_actants=["a1"], amplitude=0.8, role_pattern="shift"),
            
            # Economic Impact → Policy Response
            TransformationEdge(id="t11", source_id="p13", target_id="p9", 
                             carrying_actants=["a5"], amplitude=0.7, role_pattern="shift"),
            TransformationEdge(id="t12", source_id="p14", target_id="p10", 
                             carrying_actants=["a4"], amplitude=0.6, role_pattern="stable"),
            
            # Community Planning → Economic Impact
            TransformationEdge(id="t13", source_id="p7", target_id="p15", 
                             carrying_actants=["a3"], amplitude=0.5, role_pattern="shift"),
            
            # Feedback loops
            TransformationEdge(id="t14", source_id="p14", target_id="p4", 
                             carrying_actants=["a4"], amplitude=0.7, role_pattern="stable"),
            TransformationEdge(id="t15", source_id="p11", target_id="p6", 
                             carrying_actants=["a3"], amplitude=0.8, role_pattern="stable"),
            TransformationEdge(id="t16", source_id="p12", target_id="p13", 
                             carrying_actants=["a6"], amplitude=0.7, role_pattern="shift"),
            TransformationEdge(id="t17", source_id="p9", target_id="p14", 
                             carrying_actants=["a4"], amplitude=0.6, role_pattern="shift")
        ]
        
        # Create detector
        self.detector = PredicateSublimationDetector(
            predicates=self.predicates,
            actants=self.actants,
            transformations=self.transformations
        )
    
    def test_threshold_detection(self):
        """Test detection of critical thresholds in predicate networks."""
        # Detect sublimations
        frameworks = self.detector.detect_sublimations()
        
        # Check that we have detected some frameworks
        self.assertTrue(len(frameworks) > 0)
        
        # Print results
        print(f"\nDetected {len(frameworks)} conceptual frameworks:")
        for i, framework in enumerate(frameworks):
            print(f"\n{i+1}. {framework.name}")
            print(f"   Description: {framework.description}")
            print(f"   Confidence: {framework.emergence_confidence:.2f}")
            print(f"   Stability: {framework.stability_index:.2f}")
            print(f"   Capaciousness: {framework.capaciousness_index:.2f}")
            print(f"   Constituent predicates: {len(framework.constituent_predicates)}")
            print(f"   Constituent actants: {len(framework.constituent_actants)}")
            
            # Print semantic directionality (the "supposing" within meaning)
            print("   Semantic directionality (where meaning wants to flow):")
            for direction, strength in sorted(framework.semantic_directionality.items(), 
                                             key=lambda x: x[1], reverse=True):
                print(f"     - {direction}: {strength:.2f}")
            
            # Print a few sample predicates
            print("   Sample predicates:")
            for pred_id in framework.constituent_predicates[:3]:
                if pred_id in self.predicates:
                    print(f"     - {self.predicates[pred_id].text}")
            if len(framework.constituent_predicates) > 3:
                print(f"     - ... and {len(framework.constituent_predicates) - 3} more")
    
    def test_climate_adaptation_framework(self):
        """Test that a climate adaptation framework emerges from the network."""
        # Detect sublimations
        frameworks = self.detector.detect_sublimations()
        
        # Look for a framework related to climate adaptation
        adaptation_frameworks = [f for f in frameworks 
                               if any(term in f.name.lower() for term in 
                                     ["adapt", "climate", "sea level", "community"])]
        
        # Check that we found at least one adaptation framework
        self.assertTrue(len(adaptation_frameworks) > 0)
        
        # Print the adaptation frameworks
        print(f"\nDetected {len(adaptation_frameworks)} adaptation-related frameworks:")
        for i, framework in enumerate(adaptation_frameworks):
            print(f"\n{i+1}. {framework.name}")
            print(f"   Description: {framework.description}")
            print(f"   Capaciousness: {framework.capaciousness_index:.2f}")
            
            # Print semantic directionality (the "supposing" within meaning)
            print("   Semantic directionality (where meaning wants to flow):")
            for direction, strength in sorted(framework.semantic_directionality.items(), 
                                             key=lambda x: x[1], reverse=True):
                print(f"     - {direction}: {strength:.2f}")
            
            # Print all predicates in this framework
            print("   All constituent predicates:")
            for pred_id in framework.constituent_predicates:
                if pred_id in self.predicates:
                    print(f"     - {self.predicates[pred_id].text}")
    
    def test_policy_economic_framework(self):
        """Test that a policy-economic framework emerges from the network."""
        # Detect sublimations
        frameworks = self.detector.detect_sublimations()
        
        # Look for a framework related to policy and economy
        policy_frameworks = []
        
        # Check each framework's description and constituent predicates
        for f in frameworks:
            # Check the name
            if any(term in f.name.lower() for term in ["policy", "econom", "fund", "invest"]):
                policy_frameworks.append(f)
                continue
                
            # Check the predicates
            has_policy_predicates = False
            for pred_id in f.constituent_predicates:
                if pred_id in self.predicates:
                    pred_text = self.predicates[pred_id].text.lower()
                    if any(term in pred_text for term in ["policy", "economy", "fund", "invest"]):
                        has_policy_predicates = True
                        break
            
            if has_policy_predicates:
                policy_frameworks.append(f)
        
        # Check that we found at least one policy framework
        self.assertTrue(len(policy_frameworks) > 0)
        
        # Print the policy frameworks
        print(f"\nDetected {len(policy_frameworks)} policy-related frameworks:")
        for i, framework in enumerate(policy_frameworks):
            print(f"\n{i+1}. {framework.name}")
            print(f"   Description: {framework.description}")
            print(f"   Capaciousness: {framework.capaciousness_index:.2f}")
            
            # Print semantic directionality (the "supposing" within meaning)
            print("   Semantic directionality (where meaning wants to flow):")
            for direction, strength in sorted(framework.semantic_directionality.items(), 
                                             key=lambda x: x[1], reverse=True):
                print(f"     - {direction}: {strength:.2f}")
            
            # Print all predicates in this framework
            print("   All constituent predicates:")
            for pred_id in framework.constituent_predicates:
                if pred_id in self.predicates:
                    print(f"     - {self.predicates[pred_id].text}")


    def test_semantic_flow_analysis(self):
        """Test the analysis of semantic flow directionality and capaciousness."""
        # Detect sublimations
        frameworks = self.detector.detect_sublimations()
        
        # Check that all frameworks have semantic directionality and capaciousness
        for framework in frameworks:
            self.assertTrue(hasattr(framework, 'semantic_directionality'))
            self.assertTrue(hasattr(framework, 'capaciousness_index'))
            self.assertIsInstance(framework.semantic_directionality, dict)
            self.assertIsInstance(framework.capaciousness_index, float)
            
            # There should be at least one direction for frameworks with enough predicates
            if len(framework.constituent_predicates) >= 2:
                self.assertTrue(len(framework.semantic_directionality) > 0)
            
            # Capaciousness should be between 0 and 1
            self.assertTrue(0 <= framework.capaciousness_index <= 1)
        
        # Find the framework with highest capaciousness
        if frameworks:
            most_capacious = max(frameworks, key=lambda f: f.capaciousness_index)
            print(f"\nMost capacious framework: {most_capacious.name}")
            print(f"   Capaciousness: {most_capacious.capaciousness_index:.2f}")
            print("   Semantic directionality (where meaning wants to flow):")
            for direction, strength in sorted(most_capacious.semantic_directionality.items(), 
                                             key=lambda x: x[1], reverse=True):
                print(f"     - {direction}: {strength:.2f}")
            
            # Find the framework with most diverse semantic directions
            most_diverse = max(frameworks, key=lambda f: len(f.semantic_directionality))
            print(f"\nFramework with most diverse semantic directions: {most_diverse.name}")
            print(f"   Number of directions: {len(most_diverse.semantic_directionality)}")
            print("   Semantic directionality (where meaning wants to flow):")
            for direction, strength in sorted(most_diverse.semantic_directionality.items(), 
                                             key=lambda x: x[1], reverse=True):
                print(f"     - {direction}: {strength:.2f}")


if __name__ == "__main__":
    unittest.main()
