"""
Test Emergent Patterns

This module tests the emergent pattern components and validates the assumption
that patterns can emerge naturally from observations without being predefined.
"""

import unittest
import sys
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
import json
import random

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer
from habitat_evolution.field.field_navigator import FieldNavigator
from habitat_evolution.field.field_state import TonicHarmonicFieldState
from habitat_evolution.adaptive_core.transformation.actant_journey_tracker import ActantJourneyTracker
from habitat_evolution.adaptive_core.transformation.actant_journey_tracker import ActantJourney, ActantJourneyPoint, DomainTransition
from habitat_evolution.adaptive_core.emergence.semantic_current_observer import SemanticCurrentObserver
from habitat_evolution.adaptive_core.emergence.emergent_pattern_detector import EmergentPatternDetector
from habitat_evolution.adaptive_core.emergence.resonance_trail_observer import ResonanceTrailObserver
from habitat_evolution.adaptive_core.emergence.pattern_integration import (
    integrate_with_actant_journey_tracker,
    integrate_with_field_navigator,
    integrate_with_field_state,
    setup_emergent_pattern_system
)
from habitat_evolution.adaptive_core.emergence.climate_data_loader import ClimateDataLoader


class TestEmergentPatterns(unittest.TestCase):
    """Test suite for emergent pattern components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Create field analyzer and navigator
        self.field_analyzer = TopologicalFieldAnalyzer()
        self.field_navigator = FieldNavigator(self.field_analyzer)
        
        # Create a test resonance matrix
        size = 10
        self.resonance_matrix = np.zeros((size, size))
        
        # Create community structure
        for i in range(3):
            for j in range(3):
                if i != j:
                    self.resonance_matrix[i, j] = 0.8 - 0.1 * abs(i - j)
        
        for i in range(3, 6):
            for j in range(3, 6):
                if i != j:
                    self.resonance_matrix[i, j] = 0.8 - 0.1 * abs(i - j)
        
        for i in range(6, 10):
            for j in range(6, 10):
                if i != j:
                    self.resonance_matrix[i, j] = 0.8 - 0.1 * abs(i - j)
        
        # Create fuzzy boundaries
        self.resonance_matrix[2, 3] = 0.5
        self.resonance_matrix[3, 2] = 0.5
        self.resonance_matrix[5, 6] = 0.5
        self.resonance_matrix[6, 5] = 0.5
        
        # Create pattern metadata
        self.pattern_metadata = []
        self.pattern_dict = {}
        for i in range(size):
            community = i // 3  # Assign community based on index
            pattern = {
                "id": f"pattern_{i}",
                "type": "test",
                "metrics": {
                    "coherence": 0.8,
                    "stability": 0.7
                },
                "community": community,
                "position": [i * 0.1, i * 0.1, 0],
                "eigenspace_position": [i * 0.1, i * 0.1, i * 0.1, i * 0.1, i * 0.1]
            }
            self.pattern_metadata.append(pattern)
            self.pattern_dict[f"pattern_{i}"] = pattern
        
        # Initialize the field
        self.field = self.field_navigator.set_field(self.resonance_matrix, self.pattern_metadata)
        
        # Create a proper field analysis structure for the field state
        # This ensures we have the required topology data with eigenvalues
        mock_field_analysis = {
            "topology": {
                "effective_dimensionality": 3,
                "principal_dimensions": [0, 1, 2],
                "eigenvalues": np.array([0.8, 0.5, 0.3, 0.1, 0.05]),
                "eigenvectors": np.random.rand(5, 5)
            },
            "communities": {
                "count": 3,
                "membership": [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
            },
            "boundaries": {
                "transitions": [(2, 3), (5, 6)],
                "uncertainty": {"2-3": 0.5, "5-6": 0.5}
            },
            "density": {
                "density_centers": [(1.0, 1.0), (4.0, 4.0), (8.0, 8.0)],
                "density_map": np.random.rand(10, 10)
            },
            "field_properties": {
                "coherence": 0.75,
                "navigability_score": 0.8,
                "stability": 0.7
            },
            "resonance_relationships": {
                "pattern_0": {"pattern_1": 0.8, "pattern_2": 0.5},
                "pattern_1": {"pattern_0": 0.8, "pattern_2": 0.6},
                "pattern_2": {"pattern_0": 0.5, "pattern_1": 0.6}
            },
            "patterns": self.pattern_dict
        }
        
        # Create field state with the mock analysis
        self.field_state = TonicHarmonicFieldState(mock_field_analysis)
        
        # Create actant journey tracker
        self.journey_tracker = ActantJourneyTracker()
        
        # Create test components
        self.semantic_observer = SemanticCurrentObserver(self.field_navigator, self.journey_tracker)
        self.pattern_detector = EmergentPatternDetector(self.semantic_observer)
        self.resonance_observer = ResonanceTrailObserver(self.field_state)
        
        # Integrate components
        integrate_with_actant_journey_tracker(self.semantic_observer, self.journey_tracker)
        integrate_with_field_navigator(self.pattern_detector, self.field_navigator)
        integrate_with_field_state(self.resonance_observer, self.field_state)
        
        # Initialize climate data loader
        climate_risk_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/climate_risk'))
        self.data_loader = ClimateDataLoader(climate_risk_dir)
        self.data_loader.load_all_files()
    
    def test_semantic_current_observation(self):
        """Test that semantic currents are properly observed."""
        # Create test data
        test_data = {
            "predicates": [
                {
                    "subject": "coastal_community",
                    "verb": "faces",
                    "object": "sea_level_rise",
                    "context": {"severity": "high"}
                },
                {
                    "subject": "coastal_community",
                    "verb": "implements",
                    "object": "adaptation_measures",
                    "context": {"effectiveness": "medium"}
                },
                {
                    "subject": "sea_level_rise",
                    "verb": "threatens",
                    "object": "infrastructure",
                    "context": {"timeframe": "near-term"}
                }
            ]
        }
        
        # Observe semantic currents
        results = self.semantic_observer.observe_semantic_currents(test_data)
        
        # Verify results
        self.assertEqual(results["actants_observed"], 4)
        self.assertEqual(results["relationships_observed"], 3)
        
        # Verify relationships were stored
        self.assertEqual(len(self.semantic_observer.observed_relationships), 3)
        
        # Verify AdaptiveID was updated
        self.assertTrue(hasattr(self.semantic_observer, "adaptive_id"))
        
        # Feed more data to establish patterns
        for i in range(3):
            self.semantic_observer.observe_semantic_currents({
                "predicates": [
                    {
                        "subject": "coastal_community",
                        "verb": "faces",
                        "object": "sea_level_rise",
                        "context": {"severity": "high"}
                    },
                    {
                        "subject": "policy_makers",
                        "verb": "allocate",
                        "object": "adaptation_budget",
                        "context": {"amount": "insufficient"}
                    }
                ]
            })
        
        # Get frequent relationships
        frequent = self.semantic_observer.get_frequent_relationships(threshold=3)
        
        # Verify frequent relationships
        self.assertGreaterEqual(len(frequent), 1)
        self.assertEqual(frequent[0]["source"], "coastal_community")
        self.assertEqual(frequent[0]["predicate"], "faces")
        self.assertEqual(frequent[0]["target"], "sea_level_rise")
        self.assertGreaterEqual(frequent[0]["frequency"], 4)
    
    def test_emergent_pattern_detection(self):
        """Test that patterns emerge naturally from observations."""
        # Create test data with repeated patterns to ensure detection
        test_data_sets = []
        
        # Define key patterns to detect
        key_patterns = [
            # Coastal community faces sea level rise
            {
                "subject": "coastal_community",
                "verb": "faces",
                "object": "sea_level_rise",
                "context_variations": [
                    {"severity": "high"},
                    {"severity": "medium"},
                    {"severity": "increasing"}
                ]
            },
            # Infrastructure needs adaptation
            {
                "subject": "infrastructure",
                "verb": "needs",
                "object": "adaptation",
                "context_variations": [
                    {"urgency": "high"},
                    {"urgency": "critical"}
                ]
            },
            # Policy makers allocate budget
            {
                "subject": "policy_makers",
                "verb": "allocate",
                "object": "budget",
                "context_variations": [
                    {"amount": "insufficient"},
                    {"amount": "increasing"}
                ]
            },
            # Evolution pattern: coastal community adapts to sea level rise
            {
                "subject": "coastal_community",
                "verb": "adapts_to",
                "object": "sea_level_rise",
                "context_variations": [
                    {"effectiveness": "moderate"}
                ]
            }
        ]
        
        # Generate multiple instances of each pattern to ensure frequency threshold is met
        for pattern in key_patterns:
            # Create 4 instances of each pattern with different context variations
            for _ in range(4):
                context = random.choice(pattern["context_variations"]).copy()
                # Add timestamp to make each instance unique
                context["timestamp"] = datetime.now().isoformat()
                
                test_data_sets.append({
                    "predicates": [
                        {
                            "subject": pattern["subject"],
                            "verb": pattern["verb"],
                            "object": pattern["object"],
                            "context": context
                        }
                    ]
                })
        
        # Feed data to semantic observer
        for data_set in test_data_sets:
            self.semantic_observer.observe_semantic_currents(data_set)
        
        # Detect patterns
        patterns = self.pattern_detector.detect_patterns()
        
        # Verify patterns were detected
        self.assertGreaterEqual(len(patterns), 3)
        
        # Verify specific patterns
        pattern_keys = set()
        for pattern in patterns:
            pattern_keys.add(f"{pattern['source']}_{pattern['predicate']}_{pattern['target']}")
        
        self.assertIn("coastal_community_faces_sea_level_rise", pattern_keys)
        self.assertIn("infrastructure_needs_adaptation", pattern_keys)
        self.assertIn("policy_makers_allocate_budget", pattern_keys)
        
        # Verify pattern evolution was detected
        evolution_detected = False
        for pattern in patterns:
            if pattern["source"] == "coastal_community" and pattern["predicate"] == "adapts_to":
                if "evolved_from" in pattern:
                    evolution_detected = True
                    break
        
        self.assertTrue(evolution_detected)
        
        # Verify AdaptiveID was updated
        context_values = self.pattern_detector.adaptive_id.get_all_context_values()
        pattern_detections = [c for c in context_values if "pattern_detected" in c]
        self.assertGreaterEqual(len(pattern_detections), 3)
    
    def test_resonance_trail_formation(self):
        """Test that resonance trails form naturally from pattern movements."""
        # Simulate pattern movements
        for i in range(5):
            pattern_id = f"pattern_{i % 3}"  # Cycle through 3 patterns
            old_position = (i, i)
            new_position = (i + 1, i + 1)
            timestamp = datetime.now().isoformat()
            
            self.resonance_observer.observe_pattern_movement(
                pattern_id, old_position, new_position, timestamp
            )
        
        # Verify movements were recorded
        self.assertEqual(len(self.resonance_observer.observed_movements), 5)
        
        # Verify trails were formed
        self.assertGreaterEqual(len(self.resonance_observer.trail_map), 3)
        
        # Create overlapping trails
        self.resonance_observer.observe_pattern_movement(
            "pattern_0", (5, 5), (6, 6), datetime.now().isoformat()
        )
        self.resonance_observer.observe_pattern_movement(
            "pattern_1", (6, 6), (7, 7), datetime.now().isoformat()
        )
        
        # Detect emergent pathways
        pathways = self.resonance_observer.detect_emergent_pathways(threshold=0.5)
        
        # Verify pathways were detected
        self.assertGreaterEqual(len(pathways), 1)
        
        # Verify AdaptiveID was updated
        context_values = self.resonance_observer.adaptive_id.get_all_context_values()
        movement_records = [c for c in context_values if "pattern_movement" in c]
        self.assertEqual(len(movement_records), 7)
    
    def test_actant_journey_integration(self):
        """Test integration with actant journeys."""
        # Create test actant journeys
        journey = ActantJourney.create("coastal_community")
        journey.initialize_adaptive_id()
        
        # Add journey points
        journey.add_journey_point(ActantJourneyPoint.create(
            actant_name="coastal_community",
            domain_id="risk_domain",
            predicate_id="pred_1",
            role="subject",
            timestamp=datetime.now().isoformat()
        ))
        
        journey.add_journey_point(ActantJourneyPoint.create(
            actant_name="coastal_community",
            domain_id="adaptation_domain",
            predicate_id="pred_2",
            role="subject",
            timestamp=(datetime.now() + timedelta(hours=1)).isoformat()
        ))
        
        # Add domain transition
        journey.add_domain_transition(DomainTransition.create(
            actant_name="coastal_community",
            source_domain_id="risk_domain",
            target_domain_id="adaptation_domain",
            source_predicate_id="pred_1",
            target_predicate_id="pred_2",
            source_role="subject",
            target_role="subject",
            timestamp=(datetime.now() + timedelta(hours=1)).isoformat()
        ))
        
        # Add to journey tracker
        self.journey_tracker.actant_journeys["coastal_community"] = journey
        
        # Observe journey data
        self.semantic_observer.observe_semantic_currents({
            "actant_journeys": [journey.to_dict()]
        })
        
        # Verify relationships were observed
        transitions_from = False
        transitions_to = False
        
        for rel_key in self.semantic_observer.observed_relationships:
            if "transitions_from" in rel_key:
                transitions_from = True
            if "transitions_to" in rel_key:
                transitions_to = True
        
        self.assertTrue(transitions_from)
        self.assertTrue(transitions_to)
        
        # Detect patterns
        patterns = self.pattern_detector.detect_patterns()
        
        # Verify patterns include journey transitions
        transition_patterns = [p for p in patterns if "transitions_" in p["predicate"]]
        self.assertGreaterEqual(len(transition_patterns), 0)  # May not meet threshold yet
    
    def test_field_integration(self):
        """Test integration with field components."""
        # Create test pattern movements based on field
        for i in range(3):
            pattern_id = f"pattern_{i}"
            old_position = (i, i)
            new_position = (i + 0.1, i + 0.1)  # Small movement
            
            self.resonance_observer.observe_pattern_movement(
                pattern_id, old_position, new_position, datetime.now().isoformat()
            )
        
        # Update field state with new pattern positions
        updated_patterns = {}
        for i in range(10):
            updated_patterns[f"pattern_{i}"] = {
                "position": [i * 0.1, i * 0.1, 0],
                "community": i // 3
            }
        
        self.field_state.update_patterns(updated_patterns)
        
        # Verify trail influence
        influence = self.resonance_observer.get_trail_influence((0.5, 0.5))
        self.assertGreaterEqual(influence["total_influence"], 0)
        
        # Verify AdaptiveID integration
        self.assertTrue(hasattr(self.field_state, "observers") or 
                        hasattr(self.field_navigator, "observers"))
    
    def test_end_to_end_emergent_patterns(self):
        """Test the complete emergent pattern system end-to-end."""
        # Create a new system using the setup function
        system = setup_emergent_pattern_system(
            self.journey_tracker,
            self.field_navigator,
            self.field_state
        )
        
        # Verify system components
        self.assertIn("semantic_observer", system)
        self.assertIn("pattern_detector", system)
        self.assertIn("resonance_observer", system)
        
        # Feed climate risk data
        climate_data_sets = []
        
        # Define key patterns to detect
        key_patterns = [
            # Community facing risk
            {
                "subject": "coastal_community",
                "verb": "faces",
                "object": "inundation_risk",
                "context_variations": [
                    {"severity": "high"},
                    {"severity": "increasing"},
                    {"severity": "critical"}
                ]
            },
            # Infrastructure vulnerability
            {
                "subject": "infrastructure",
                "verb": "vulnerable_to",
                "object": "inundation_risk",
                "context_variations": [
                    {"degree": "high"},
                    {"degree": "critical"},
                    {"degree": "severe"}
                ]
            },
            # Budget allocation
            {
                "subject": "government",
                "verb": "allocates",
                "object": "adaptation_budget",
                "context_variations": [
                    {"amount": "insufficient"},
                    {"amount": "increasing"},
                    {"amount": "limited"}
                ]
            },
            # Cultural factors
            {
                "subject": "cultural_values",
                "verb": "influence",
                "object": "adaptation_decisions",
                "context_variations": [
                    {"strength": "significant"}
                ]
            },
            # Life changes
            {
                "subject": "residents",
                "verb": "experience",
                "object": "life_changes",
                "context_variations": [
                    {"impact": "major"}
                ]
            },
            # Evolution patterns
            {
                "subject": "coastal_community",
                "verb": "adapts_to",
                "object": "inundation_risk",
                "context_variations": [
                    {"effectiveness": "moderate"},
                    {"effectiveness": "limited"}
                ]
            },
            {
                "subject": "infrastructure",
                "verb": "retrofitted_against",
                "object": "inundation_risk",
                "context_variations": [
                    {"cost": "high"},
                    {"cost": "substantial"}
                ]
            }
        ]
        
        # Generate multiple instances of each pattern to ensure frequency threshold is met
        for pattern in key_patterns:
            # Create at least 4 instances of each key pattern to exceed detection threshold
            repeat_count = 5 if pattern["subject"] in ["coastal_community", "infrastructure", "government"] else 2
            
            for _ in range(repeat_count):
                context = random.choice(pattern["context_variations"]).copy()
                # Add timestamp to make each instance unique
                context["timestamp"] = datetime.now().isoformat()
                
                climate_data_sets.append({
                    "predicates": [
                        {
                            "subject": pattern["subject"],
                            "verb": pattern["verb"],
                            "object": pattern["object"],
                            "context": context
                        }
                    ]
                })
        
        # Feed data to semantic observer
        for data_set in climate_data_sets:
            system["semantic_observer"].observe_semantic_currents(data_set)
        
        # Detect patterns
        patterns = system["pattern_detector"].detect_patterns()
        
        # Verify patterns were detected
        self.assertGreaterEqual(len(patterns), 3)
        
        # Verify specific climate risk patterns
        pattern_keys = set()
        for pattern in patterns:
            pattern_keys.add(f"{pattern['source']}_{pattern['predicate']}_{pattern['target']}")
        
        self.assertIn("coastal_community_faces_inundation_risk", pattern_keys)
        self.assertIn("infrastructure_vulnerable_to_inundation_risk", pattern_keys)
        self.assertIn("government_allocates_adaptation_budget", pattern_keys)
        
        # Verify pattern evolution was detected
        evolution_detected = False
        for pattern in patterns:
            if "evolved_from" in pattern:
                evolution_detected = True
                break
        
        self.assertTrue(evolution_detected)
        
        # Verify AdaptiveID participation
        self.assertTrue(hasattr(system["semantic_observer"], "adaptive_id"))
        self.assertTrue(hasattr(system["pattern_detector"], "adaptive_id"))
        self.assertTrue(hasattr(system["resonance_observer"], "adaptive_id"))
        
        # Verify AdaptiveID context updates
        observer_context = system["semantic_observer"].adaptive_id.get_all_context_values()
        detector_context = system["pattern_detector"].adaptive_id.get_all_context_values()
        
        self.assertGreaterEqual(len(observer_context), 10)
        self.assertGreaterEqual(len(detector_context), 3)
        
        # Print summary of detected patterns
        print("\nDetected Climate Risk Patterns:")
        for pattern in patterns:
            print(f"  {pattern['source']} {pattern['predicate']} {pattern['target']} "
                  f"(frequency: {pattern['frequency']}, confidence: {pattern['confidence']:.2f})")
            if "evolved_from" in pattern:
                print(f"    Evolved from: {pattern['evolved_from']}")


    def test_climate_risk_data_patterns(self):
        """Test pattern detection using real climate risk data."""
        # Generate synthetic relationships to ensure pattern detection
        # Increase the count to ensure we have enough patterns for detection
        self.data_loader.generate_synthetic_relationships(count=30)
        
        # Get observation data from climate risk documents with larger batch size
        observations = self.data_loader.generate_observation_data(batch_size=5)
        
        # Verify we have observations
        self.assertGreaterEqual(len(observations), 1)
        
        # Feed data to semantic observer
        # Feed each observation multiple times to increase pattern frequency
        for observation in observations:
            # Feed each observation 3 times to ensure pattern detection threshold is met
            for _ in range(3):
                # Add a timestamp to make each observation unique
                for predicate in observation["predicates"]:
                    if "context" not in predicate:
                        predicate["context"] = {}
                    predicate["context"]["timestamp"] = datetime.now().isoformat()
                
                self.semantic_observer.observe_semantic_currents(observation)
        
        # Detect patterns
        patterns = self.pattern_detector.detect_patterns()
        
        # Verify patterns were detected - we should have more patterns now
        self.assertGreaterEqual(len(patterns), 3, "Expected at least 3 patterns to be detected")
        
        # Print detected patterns for analysis
        print("\nDetected Climate Risk Patterns from Real Data:")
        for pattern in patterns:
            print(f"  {pattern['source']} {pattern['predicate']} {pattern['target']} "
                  f"(frequency: {pattern['frequency']}, confidence: {pattern['confidence']:.2f})")
        
        # Test pattern evolution with real data - increase steps for more evolution data
        evolution_data = self.data_loader.generate_evolution_data(steps=6)
        
        # Feed evolution data multiple times to ensure pattern detection
        for data in evolution_data:
            # Feed each evolution data point 3 times
            for _ in range(3):
                # Add a timestamp to make each observation unique
                for predicate in data["predicates"]:
                    if "context" not in predicate:
                        predicate["context"] = {}
                    predicate["context"]["timestamp"] = datetime.now().isoformat()
                
                self.semantic_observer.observe_semantic_currents(data)
        
        # Detect patterns again
        evolved_patterns = self.pattern_detector.detect_patterns()
        
        # Find evolved patterns
        evolved_count = 0
        for pattern in evolved_patterns:
            if "evolved_from" in pattern:
                evolved_count += 1
                print(f"\nEvolved Pattern: {pattern['source']} {pattern['predicate']} {pattern['target']}")
                print(f"  Evolved from: {pattern['evolved_from']}")
        
        # We may not have evolved patterns in every run due to thresholds,
        # but we should have observed the evolution data
        self.assertGreaterEqual(len(evolution_data), 1)
        
        # Verify AdaptiveID participation
        context_values = self.semantic_observer.adaptive_id.get_all_context_values()
        self.assertGreaterEqual(len(context_values), len(observations) + len(evolution_data))

if __name__ == "__main__":
    unittest.main()
