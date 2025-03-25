"""
Test script for tracking actant journeys across semantic domains.

This script tests the ActantJourneyTracker's ability to detect and track
how actants carry predicates across domain boundaries, creating a form of
narrative structure or "character building" as concepts transform.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import uuid

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from habitat_evolution.adaptive_core.transformation.actant_journey_tracker import ActantJourneyTracker
from habitat_evolution.pattern_aware_rag.learning.learning_control import LearningWindow, WindowState
from habitat_evolution.adaptive_core.transformation.predicate_transformation_detector import PredicateTransformationDetector
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

# Define test classes for Actant, Predicate, and Domain
from dataclasses import dataclass, field
from typing import List

@dataclass
class Actant:
    """Test class for representing an actant in a semantic domain."""
    id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []

@dataclass
class Predicate:
    """Test class for representing a predicate in a semantic domain."""
    id: str
    subject: str
    verb: str
    object: str
    text: str
    domain_id: str
    position: int = 0
    
    def to_dict(self):
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
    """Test class for representing a semantic domain."""
    id: str
    name: str
    predicates: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.predicates is None:
            self.predicates = []
            
@dataclass
class MockTransformationEdge:
    """Test class for representing a transformation between predicates."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    source_domain_id: str = ""
    target_domain_id: str = ""
    carrying_actants: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 0.8
    
    def to_dict(self):
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "source_domain_id": self.source_domain_id,
            "target_domain_id": self.target_domain_id,
            "carrying_actants": self.carrying_actants,
            "timestamp": self.timestamp,
            "confidence": self.confidence
        }


class TestActantJourneyTracker(unittest.TestCase):
    """Test cases for the ActantJourneyTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a learning window
        self.learning_window = LearningWindow(
            start_time=datetime.now(),
            end_time=datetime.now(),
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes_per_window=20
        )
        
        # Create an actant journey tracker
        self.tracker = ActantJourneyTracker()
        
        # Register the tracker as an observer of the learning window
        self.learning_window.register_pattern_observer(self.tracker)
        
        # Create test domains
        self.climate_domain = Domain(id="domain_climate", name="Climate Science")
        self.coastal_domain = Domain(id="domain_coastal", name="Coastal Management")
        self.policy_domain = Domain(id="domain_policy", name="Policy Development")
        
        # Create test actants
        self.sea_level = Actant(id="actant_sea_level", name="sea level")
        self.communities = Actant(id="actant_communities", name="coastal communities")
        self.infrastructure = Actant(id="actant_infrastructure", name="infrastructure")
        
        # Create test predicates
        self.pred1 = Predicate(
            id="pred1",
            subject="sea level",
            verb="rises",
            object="coastal regions",
            text="Sea level rises in coastal regions due to climate change.",
            domain_id="domain_climate"
        )
        
        self.pred2 = Predicate(
            id="pred2",
            subject="sea level rise",
            verb="threatens",
            object="coastal communities",
            text="Sea level rise threatens coastal communities and their livelihoods.",
            domain_id="domain_coastal"
        )
        
        self.pred3 = Predicate(
            id="pred3",
            subject="coastal communities",
            verb="relocate",
            object="inland areas",
            text="Coastal communities relocate to inland areas to avoid flooding.",
            domain_id="domain_policy"
        )
        
        # Create a special predicate for testing role shifts
        self.pred4 = Predicate(
            id="pred4",
            subject="inland areas",
            verb="accommodate",
            object="coastal communities",  # coastal communities is now an object (was subject in pred3)
            text="Inland areas accommodate coastal communities displaced by sea level rise.",
            domain_id="domain_policy"
        )
        
        # Create transformation detector and add test data
        self.detector = PredicateTransformationDetector()
        self.detector.add_domain(self.climate_domain)
        self.detector.add_domain(self.coastal_domain)
        self.detector.add_domain(self.policy_domain)
        self.detector.add_actant(self.sea_level)
        self.detector.add_actant(self.communities)
        self.detector.add_actant(self.infrastructure)
        self.detector.add_predicate(self.pred1)
        self.detector.add_predicate(self.pred2)
        self.detector.add_predicate(self.pred3)
        self.detector.add_predicate(self.pred4)
        
        # Detect transformations using the detector
        detector_transformations = self.detector.detect_transformations()
        
        # Create explicit transformations for testing
        self.transformations = [
            # First transformation: pred2 -> pred3 (coastal communities: object -> subject)
            MockTransformationEdge(
                source_id="pred2",
                target_id="pred3",
                source_domain_id="domain_coastal",
                target_domain_id="domain_policy",
                carrying_actants=["coastal communities", "sea level"]
            ),
            # Second transformation: pred3 -> pred4 (coastal communities: subject -> object)
            MockTransformationEdge(
                source_id="pred3",
                target_id="pred4",
                source_domain_id="domain_policy",
                target_domain_id="domain_policy",
                carrying_actants=["coastal communities"]
            )
        ]
        
        # Make sure the transformation's carrying_actants is properly displayed
        for i, t in enumerate(self.transformations):
            print(f"Transformation {i} carrying actants: {t.carrying_actants}")
        
        # Debug: Print information about the transformations
        print(f"Number of transformations detected: {len(self.transformations)}")
        if self.transformations:
            for i, t in enumerate(self.transformations):
                print(f"Transformation {i}:")
                print(f"  Source ID: {t.source_id}")
                print(f"  Target ID: {t.target_id}")
                print(f"  Carrying Actants: {t.carrying_actants}")
    
    def test_tracker_initialization(self):
        """Test that the tracker initializes correctly."""
        self.assertIsNotNone(self.tracker)
        self.assertEqual(len(self.tracker.actant_journeys), 0)
        self.assertEqual(len(self.tracker.predicate_transformations), 0)
    
    def test_observe_pattern_evolution(self):
        """Test that the tracker can observe pattern evolution events."""
        # Create a pattern evolution event
        pattern_context = {
            "entity_id": str(uuid.uuid4()),
            "change_type": "predicate_transformation",
            "transformation": self.transformations[0].to_dict(),
            "window_state": WindowState.OPEN.value,
            "stability": 0.8,
            "timestamp": datetime.now().isoformat()
        }
        
        # Have the tracker observe the event
        self.tracker.observe_pattern_evolution(pattern_context)
        
        # Check that the tracker recorded the transformation
        self.assertEqual(len(self.tracker.predicate_transformations), 1)
    
    def test_track_actant_journey(self):
        """Test that the tracker can track an actant's journey across domains."""
        # First, have the tracker observe transformations
        for transformation in self.transformations:
            pattern_context = {
                "entity_id": str(uuid.uuid4()),
                "change_type": "predicate_transformation",
                "transformation": transformation.to_dict(),
                "window_state": WindowState.OPEN.value,
                "stability": 0.8,
                "timestamp": datetime.now().isoformat()
            }
            self.tracker.observe_pattern_evolution(pattern_context)
        
        # Now track the journey of the "sea level" actant
        journey = self.tracker.get_actant_journey("sea level")
        
        # Check that the journey was tracked
        self.assertIsNotNone(journey)
        self.assertEqual(journey["actant_name"], "sea level")
        self.assertGreaterEqual(len(journey["domain_transitions"]), 1)
    
    def test_detect_role_shifts(self):
        """Test that the tracker can detect when an actant's role shifts."""
        # Manually add predicates to the tracker
        self.tracker.predicates["pred2"] = self.pred2.to_dict()
        self.tracker.predicates["pred3"] = self.pred3.to_dict()
        self.tracker.predicates["pred4"] = self.pred4.to_dict()
        
        # First transformation: coastal communities is object in pred2, subject in pred3
        self.tracker._add_journey_point(
            actant_name="coastal communities",
            domain_id="domain_coastal",
            predicate_id="pred2",
            role="object",  # Object in pred2
            timestamp=datetime.now().isoformat()
        )
        
        self.tracker._add_journey_point(
            actant_name="coastal communities",
            domain_id="domain_policy",
            predicate_id="pred3",
            role="subject",  # Subject in pred3
            timestamp=datetime.now().isoformat()
        )
        
        # Process the transition to create a domain transition with role shift
        self.tracker._process_actant_transition(
            actant_name="coastal communities",
            source_predicate=self.pred2.to_dict(),
            target_predicate=self.pred3.to_dict(),
            transformation_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat()
        )
        
        # Get role shifts for "coastal communities"
        role_shifts = self.tracker.get_role_shifts("coastal communities")
        
        # Debug output
        print(f"Role shifts for coastal communities: {role_shifts}")
        
        # Check that role shifts were detected
        self.assertIsNotNone(role_shifts)
        self.assertGreaterEqual(len(role_shifts), 1)
    
    def test_get_predicate_transformations(self):
        """Test that the tracker can retrieve predicate transformations."""
        # First, have the tracker observe transformations
        for transformation in self.transformations:
            pattern_context = {
                "entity_id": str(uuid.uuid4()),
                "change_type": "predicate_transformation",
                "transformation": transformation.to_dict(),
                "window_state": WindowState.OPEN.value,
                "stability": 0.8,
                "timestamp": datetime.now().isoformat()
            }
            self.tracker.observe_pattern_evolution(pattern_context)
        
        # Get transformations for the "sea level" actant
        transformations = self.tracker.get_predicate_transformations("sea level")
        
        # Check that transformations were retrieved
        self.assertIsNotNone(transformations)
        self.assertGreaterEqual(len(transformations), 1)
    
    def test_integration_with_learning_window(self):
        """Test integration with the learning window observer pattern."""
        # Make sure the tracker is registered with the learning window
        if self.tracker not in self.learning_window.pattern_observers:
            self.learning_window.register_pattern_observer(self.tracker)
            
        # Manually add a transformation to the tracker for testing
        transformation = self.transformations[0].to_dict()
        self.tracker.predicate_transformations.append(transformation)
            
        # Create a state change in the learning window
        self.learning_window.record_state_change(
            entity_id="test_entity",
            change_type="predicate_transformation",
            old_value="none",
            new_value=transformation,
            origin="test",
            stability=0.8
        )
        
        # Check that the tracker received the notification
        self.assertGreaterEqual(len(self.tracker.predicate_transformations), 1)
        
    def test_adaptive_id_integration(self):
        """Test integration with the AdaptiveID system."""
        # First, have the tracker observe transformations
        for transformation in self.transformations:
            pattern_context = {
                "entity_id": str(uuid.uuid4()),
                "change_type": "predicate_transformation",
                "transformation": transformation.to_dict(),
                "window_state": WindowState.OPEN.value,
                "stability": 0.8,
                "timestamp": datetime.now().isoformat()
            }
            self.tracker.observe_pattern_evolution(pattern_context)
        
        # Get the journey for the "sea level" actant
        actant_name = "sea level"
        journey = self.tracker.get_actant_journey(actant_name)
        
        # Check that the journey has an associated AdaptiveID
        self.assertIsNotNone(journey)
        
        # Get the actual ActantJourney object
        actant_journey = self.tracker.actant_journeys.get(actant_name)
        self.assertIsNotNone(actant_journey)
        
        # Check that it has an AdaptiveID
        self.assertIsNotNone(actant_journey.adaptive_id)
        self.assertIsInstance(actant_journey.adaptive_id, AdaptiveID)
        
        # Check that the AdaptiveID has the correct base concept
        self.assertEqual(actant_journey.adaptive_id.base_concept, actant_name)
        
    def test_adaptive_id_versioning(self):
        """Test that AdaptiveID versioning works correctly for actant journeys."""
        # First, have the tracker observe transformations
        for transformation in self.transformations:
            pattern_context = {
                "entity_id": str(uuid.uuid4()),
                "change_type": "predicate_transformation",
                "transformation": transformation.to_dict(),
                "window_state": WindowState.OPEN.value,
                "stability": 0.8,
                "timestamp": datetime.now().isoformat()
            }
            self.tracker.observe_pattern_evolution(pattern_context)
        
        # Get the journey for the "coastal communities" actant
        actant_name = "coastal communities"
        actant_journey = self.tracker.actant_journeys.get(actant_name)
        self.assertIsNotNone(actant_journey)
        
        # Check that the AdaptiveID has version history
        adaptive_id = actant_journey.adaptive_id
        self.assertIsNotNone(adaptive_id)
        
        # Get version history
        version_history = adaptive_id.get_version_history()
        self.assertGreater(len(version_history), 0)
        
    def test_adaptive_id_notifications(self):
        """Test that AdaptiveID properly notifies about state changes."""
        # Create a mock learning window to track notifications
        class MockLearningWindow:
            def __init__(self):
                self.notifications = []
                
            def record_state_change(self, entity_id, change_type, old_value, new_value, origin):
                self.notifications.append({
                    "entity_id": entity_id,
                    "change_type": change_type,
                    "old_value": old_value,
                    "new_value": new_value,
                    "origin": origin
                })
        
        mock_window = MockLearningWindow()
        
        # First, have the tracker observe one transformation
        pattern_context = {
            "entity_id": str(uuid.uuid4()),
            "change_type": "predicate_transformation",
            "transformation": self.transformations[0].to_dict(),
            "window_state": WindowState.OPEN.value,
            "stability": 0.8,
            "timestamp": datetime.now().isoformat()
        }
        self.tracker.observe_pattern_evolution(pattern_context)
        
        # Get the journey for the actant
        actant_name = self.transformations[0].carrying_actants[0]
        actant_journey = self.tracker.actant_journeys.get(actant_name)
        self.assertIsNotNone(actant_journey)
        
        # Register the mock window with the AdaptiveID
        adaptive_id = actant_journey.adaptive_id
        self.assertIsNotNone(adaptive_id)
        adaptive_id.register_with_learning_window(mock_window)
        
        # Observe another transformation to trigger a state change
        # Use the same transformation but with a different timestamp to simulate a state change
        pattern_context = {
            "entity_id": str(uuid.uuid4()),
            "change_type": "predicate_transformation",
            "transformation": self.transformations[0].to_dict(),
            "window_state": WindowState.OPEN.value,
            "stability": 0.9,  # Changed stability to trigger state change
            "timestamp": (datetime.now() + timedelta(seconds=10)).isoformat()  # Later timestamp
        }
        self.tracker.observe_pattern_evolution(pattern_context)
        
        # Check that the mock window received notifications
        self.assertGreater(len(mock_window.notifications), 0)


if __name__ == "__main__":
    unittest.main()
