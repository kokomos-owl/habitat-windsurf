"""
Test Emergent Propensity

This module tests the EmergentPropensity class, which captures the emergent properties
that arise from the interaction between semantic propositions and domains.
"""

import unittest
from typing import Dict, List, Set, Any
import numpy as np
from dataclasses import dataclass

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from habitat_evolution.adaptive_core.transformation.predicate_sublimation import ConceptualFramework
from habitat_evolution.adaptive_core.transformation.semantic_proposition_patterns import SemanticProposition
from habitat_evolution.adaptive_core.emergence.emergent_propensity import EmergentPropensity


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


class TestEmergentPropensity(unittest.TestCase):
    """Test suite for emergent propensity patterns."""
    
    def setUp(self):
        """Set up test fixtures."""
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
        
        # Create propositions from frameworks
        self.community_proposition = SemanticProposition(self.community_framework)
    
    def test_emergent_quotients(self):
        """Test that emergent quotients are properly calculated."""
        # Create emergent propensity for coastal domain
        coastal_propensity = EmergentPropensity(self.community_proposition, self.coastal_domain)
        
        # Get emergent quotients
        quotients = coastal_propensity.get_emergent_quotients()
        
        # Verify quotients exist and are within expected ranges
        self.assertIn('manifestation_potential', quotients)
        self.assertTrue(0 <= quotients['manifestation_potential'] <= 1)
        
        # Check for state change vectors
        self.assertIn('adapts_vector', quotients)
        self.assertIn('supports_vector', quotients)
        
        # Check for community condition indices
        self.assertIn('adapts_index', quotients)
        self.assertIn('supports_index', quotients)
        self.assertIn('resilience_index', quotients)
        
        print("\nEmergent Quotients:")
        for name, value in quotients.items():
            print(f"  {name}: {value:.2f}")
    
    def test_state_change_projection(self):
        """Test projection of state changes in patterns."""
        # Create emergent propensity for coastal domain
        coastal_propensity = EmergentPropensity(self.community_proposition, self.coastal_domain)
        
        # Project state change
        state_change = coastal_propensity.project_state_change()
        
        # Verify state change projection
        self.assertIn('primary_direction', state_change)
        self.assertIn('transformation_magnitude', state_change)
        self.assertIn('transformation_type', state_change)
        self.assertIn('state_change_description', state_change)
        
        print("\nPattern State Change Projection:")
        print(f"  Primary Direction: {state_change['primary_direction']}")
        print(f"  Original Weight: {state_change['original_weight']:.2f}")
        print(f"  Transformed Weight: {state_change['transformed_weight']:.2f}")
        print(f"  Transformation Type: {state_change['transformation_type']}")
        print(f"  Description: {state_change['state_change_description']}")
    
    def test_community_condition_projection(self):
        """Test projection of community conditions."""
        # Create emergent propensity for urban domain
        urban_propensity = EmergentPropensity(self.community_proposition, self.urban_domain)
        
        # Project community condition
        condition = urban_propensity.project_community_condition()
        
        # Verify community condition projection
        self.assertIn('primary_condition', condition)
        self.assertIn('condition_strength', condition)
        self.assertIn('strength_category', condition)
        self.assertIn('condition_description', condition)
        
        print("\nCommunity Condition Projection:")
        print(f"  Primary Condition: {condition['primary_condition']}")
        print(f"  Strength: {condition['condition_strength']:.2f}")
        print(f"  Category: {condition['strength_category']}")
        print(f"  Description: {condition['condition_description']}")
    
    def test_cross_domain_comparison(self):
        """Test comparing emergent propensities across domains."""
        # Create emergent propensities for both domains
        coastal_propensity = EmergentPropensity(self.community_proposition, self.coastal_domain)
        urban_propensity = EmergentPropensity(self.community_proposition, self.urban_domain)
        
        # Compare manifestation potentials
        coastal_potential = coastal_propensity.manifestation_potential
        urban_potential = urban_propensity.manifestation_potential
        
        print("\nCross-Domain Comparison:")
        print(f"  Coastal Domain Manifestation Potential: {coastal_potential:.2f}")
        print(f"  Urban Domain Manifestation Potential: {urban_potential:.2f}")
        print(f"  Difference: {abs(coastal_potential - urban_potential):.2f}")
        
        # Compare state change vectors
        coastal_vectors = coastal_propensity.state_change_vectors
        urban_vectors = urban_propensity.state_change_vectors
        
        print("\nState Change Vector Comparison:")
        for direction in set(coastal_vectors.keys()).union(urban_vectors.keys()):
            coastal_weight = coastal_vectors.get(direction, 0)
            urban_weight = urban_vectors.get(direction, 0)
            print(f"  {direction}: Coastal={coastal_weight:.2f}, Urban={urban_weight:.2f}, Diff={abs(coastal_weight - urban_weight):.2f}")


if __name__ == "__main__":
    unittest.main()
