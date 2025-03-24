"""
Tests for the semantic proposition patterns module.

This demonstrates how patterns can be instantiated as code and
negatively instantiated as conductive structures.
"""

import unittest
from typing import Dict, List, Set, Any
import networkx as nx
import numpy as np
from dataclasses import dataclass

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from habitat_evolution.adaptive_core.transformation.predicate_sublimation import (
    ConceptualFramework, PredicateSublimationDetector
)
from habitat_evolution.adaptive_core.transformation.semantic_proposition_patterns import (
    SemanticProposition, PropositionInstance, ConductiveGap, PropositionPatternRegistry
)


@dataclass
class Predicate:
    """Simple predicate class for testing."""
    id: str
    subject: str
    verb: str
    object: str
    text: str
    domain_id: str = ""
    position: int = 0


class TestSemanticPropositionPatterns(unittest.TestCase):
    """Test suite for semantic proposition patterns."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample predicates
        self.predicates = {
            "p1": Predicate(id="p1", text="Community adapts to sea level", 
                           subject="Community", verb="adapts", object="sea level"),
            "p2": Predicate(id="p2", text="Infrastructure protects coastline", 
                           subject="Infrastructure", verb="protects", object="coastline"),
            "p3": Predicate(id="p3", text="Policy supports community adaptation", 
                           subject="Policy", verb="supports", object="community adaptation"),
            "p4": Predicate(id="p4", text="Infrastructure fails to protect community", 
                           subject="Infrastructure", verb="fails", object="protect community"),
            "p5": Predicate(id="p5", text="Economy invests in infrastructure", 
                           subject="Economy", verb="invests", object="infrastructure"),
            "p6": Predicate(id="p6", text="Policy funds infrastructure improvements", 
                           subject="Policy", verb="funds", object="infrastructure improvements"),
            "p7": Predicate(id="p7", text="Sea level threatens infrastructure", 
                           subject="Sea level", verb="threatens", object="infrastructure"),
            "p8": Predicate(id="p8", text="Community demands policy changes", 
                           subject="Community", verb="demands", object="policy changes"),
        }
        
        # Create sample conceptual frameworks
        self.community_framework = ConceptualFramework(
            id="cf1",
            name="Community-policy pattern",
            description="A conceptual framework emerging from the interaction of 3 predicates involving community, policy, infrastructure and 1 other actants.",
            constituent_predicates=["p1", "p3", "p4"],
            constituent_actants=["community", "policy", "infrastructure", "sea level"],
            emergence_confidence=0.77,
            stability_index=0.56,
            capaciousness_index=0.74,
            semantic_directionality={
                "adapts": 0.49,
                "supports": 0.46,
                "fails": 0.05
            }
        )
        
        self.infrastructure_framework = ConceptualFramework(
            id="cf2",
            name="Community-infrastructure pattern",
            description="A conceptual framework emerging from the interaction of 4 predicates involving community, infrastructure, economy and 1 other actants.",
            constituent_predicates=["p5", "p6", "p7", "p8"],
            constituent_actants=["community", "infrastructure", "economy", "policy"],
            emergence_confidence=0.73,
            stability_index=0.26,
            capaciousness_index=0.71,
            semantic_directionality={
                "funds": 0.41,
                "invests": 0.28,
                "regulates": 0.18,
                "demands": 0.13
            }
        )
        
        # Create sample domains
        self.coastal_domain = {
            "name": "Coastal management",
            "actants": ["coastline", "community", "sea level", "infrastructure"],
            "verbs": ["protects", "adapts", "threatens"],
            "predicates": [
                {"subject": "Infrastructure", "verb": "protects", "object": "coastline"},
                {"subject": "Sea level", "verb": "threatens", "object": "infrastructure"}
            ],
            "context": {
                "description": "Coastal communities facing rising sea levels and infrastructure challenges",
                "emphasis": {
                    "adaptation": 0.7,
                    "protection": 0.6,
                    "vulnerability": 0.5
                }
            },
            "semantic_flows": {
                "protects": 0.4,
                "threatens": 0.4,
                "adapts": 0.2
            }
        }
        
        self.urban_domain = {
            "name": "Urban development",
            "actants": ["community", "infrastructure", "policy", "economy"],
            "verbs": ["invests", "funds", "demands"],
            "predicates": [
                {"subject": "Economy", "verb": "invests", "object": "infrastructure"},
                {"subject": "Community", "verb": "demands", "object": "policy changes"}
            ],
            "context": {
                "description": "Urban development focusing on infrastructure investment and policy changes",
                "emphasis": {
                    "economic_growth": 0.8,
                    "policy_reform": 0.6,
                    "community_engagement": 0.4
                }
            },
            "semantic_flows": {
                "invests": 0.5,
                "demands": 0.3,
                "funds": 0.2
            }
        }
        
        self.rural_domain = {
            "name": "Rural adaptation",
            "actants": ["community", "infrastructure", "economy"],
            "verbs": ["invests"],
            "predicates": [
                {"subject": "Economy", "verb": "invests", "object": "infrastructure"}
            ],
            "context": "Rural communities with limited infrastructure investment",
            "semantic_flows": {
                "invests": 0.7,
                "develops": 0.3
            }
        }
        
        # Create registry
        self.registry = PropositionPatternRegistry()
    
    def test_proposition_creation(self):
        """Test creating a semantic proposition from a conceptual framework."""
        # Create proposition
        proposition = SemanticProposition(self.community_framework)
        
        # Check basic properties
        self.assertEqual(proposition.name, "Community-policy pattern")
        self.assertEqual(proposition.capaciousness, 0.74)
        self.assertEqual(proposition.directionality["adapts"], 0.49)
        
        # Check derived properties
        self.assertIsNotNone(proposition.flow_signature)
        self.assertIsNotNone(proposition.proposition_code)
        
        # Check primary verb
        self.assertEqual(proposition.proposition_code["primary_verb"], "adapts")
        
        # Print proposition code for inspection
        print("\nProposition Code:")
        for key, value in proposition.proposition_code.items():
            print(f"  {key}: {value}")
    
    def test_proposition_instantiation(self):
        """Test instantiating a proposition in a domain."""
        # Create proposition
        proposition = SemanticProposition(self.community_framework)
        
        # Instantiate in coastal domain
        instance = proposition.instantiate(self.coastal_domain)
        
        # Check instance properties
        self.assertIsInstance(instance, PropositionInstance)
        self.assertTrue(0 <= instance.compatibility <= 1)
        self.assertTrue(0 <= instance.instantiation_strength <= 1)
        
        # Execute instance
        execution_result = instance.execute()
        
        # Print execution results
        print("\nProposition Execution Results:")
        print(f"  Status: {execution_result['status']}")
        print(f"  Strength: {execution_result.get('instantiation_strength', 'N/A')}")
        print("  Effects:")
        for verb, effect in execution_result.get('effects', {}).items():
            print(f"    {verb}: {effect['strength']:.2f} → {', '.join(effect['targets'])}")
    
    def test_negative_instantiation(self):
        """Test negative instantiation to find conductive gaps."""
        # Create proposition
        proposition = SemanticProposition(self.infrastructure_framework)
        
        # Negative instantiate in rural domain
        gap = proposition.negative_instantiate(self.rural_domain)
        
        # Check gap properties
        self.assertIsInstance(gap, ConductiveGap)
        self.assertTrue(0 <= gap.conductivity <= 1)
        
        # Induce behavior propensities
        propensities = gap.induce_behavior_propensities()
        
        # Print gap and propensities
        print("\nConductive Gap:")
        print(f"  Conductivity: {gap.conductivity:.2f}")
        print("  Primary Gap:")
        if gap.primary_gap[0]:
            print(f"    {gap.primary_gap[0]}: expected={gap.primary_gap[1]['expected']:.2f}, "
                  f"actual={gap.primary_gap[1]['actual']:.2f}, "
                  f"gap={gap.primary_gap[1]['gap_size']:.2f}")
        
        print("\nInduced Behavior Propensities:")
        print(f"  Status: {propensities['status']}")
        for verb, propensity in propensities.get('propensities', {}).items():
            print(f"    {verb}: {propensity['strength']:.2f} → {', '.join(propensity['targets'])}")
    
    def test_registry_operations(self):
        """Test proposition registry operations."""
        # Register propositions
        self.registry.register_proposition(self.community_framework)
        self.registry.register_proposition(self.infrastructure_framework)
        
        # Check registry size
        self.assertEqual(len(self.registry.propositions), 2)
        
        # Find propositions for coastal domain
        coastal_props = self.registry.find_propositions_for_domain(self.coastal_domain)
        
        # Print compatible propositions
        print("\nCompatible Propositions for Coastal Domain:")
        for key, prop, compatibility in coastal_props:
            print(f"  {prop.name}: {compatibility:.2f}")
        
        # Find conductive gaps in rural domain
        gaps = self.registry.find_conductive_gaps(self.rural_domain)
        
        # Print conductive gaps
        print("\nConductive Gaps in Rural Domain:")
        for gap in gaps:
            print(f"  From {gap.proposition.name}: {gap.conductivity:.2f}")
            if gap.primary_gap[0]:
                print(f"    Primary gap: {gap.primary_gap[0]} "
                      f"(gap size: {gap.primary_gap[1]['gap_size']:.2f})")
    
    def test_multi_domain_analysis(self):
        """Test analyzing multiple domains with propositions."""
        # Register propositions
        community_prop = self.registry.register_proposition(self.community_framework)
        infra_prop = self.registry.register_proposition(self.infrastructure_framework)
        
        # Analyze domains
        domains = [self.coastal_domain, self.urban_domain, self.rural_domain]
        
        print("\nMulti-Domain Analysis:")
        for domain in domains:
            print(f"\nDomain: {domain['name']}")
            
            # Instantiate propositions
            instances = self.registry.instantiate_in_domain(domain)
            
            print(f"  Compatible Propositions: {len(instances)}")
            for instance in instances:
                print(f"    {instance.proposition.name}: "
                      f"strength={instance.instantiation_strength:.2f}")
            
            # Find gaps
            gaps = self.registry.find_conductive_gaps(domain)
            
            print(f"  Conductive Gaps: {len(gaps)}")
            for gap in gaps:
                print(f"    {gap.proposition.name}: "
                      f"conductivity={gap.conductivity:.2f}")
                if gap.primary_gap[0]:
                    print(f"      Primary gap: {gap.primary_gap[0]}")


if __name__ == "__main__":
    unittest.main()
