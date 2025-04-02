"""
Test suite for Vector-Tonic Persistence Connector.

This test suite defines the expected behavior of the connector between
the vector-tonic-window system and the ArangoDB persistence layer.
"""

import unittest
import logging
import os
import sys
from unittest.mock import MagicMock, patch
from datetime import datetime
import uuid

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from src.habitat_evolution.adaptive_core.emergence.persistence_integration import VectorTonicPersistenceIntegration
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import VectorTonicWindowIntegrator
from src.habitat_evolution.adaptive_core.emergence.interfaces.learning_window_observer import LearningWindowState
from src.habitat_evolution.adaptive_core.emergence.interfaces.field_state_repository import FieldStateRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.pattern_repository import PatternRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.relationship_repository import RelationshipRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.topology_repository import TopologyRepositoryInterface

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] %(message)s')

logger = logging.getLogger(__name__)

# Import the modules we want to test
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector import (
    VectorTonicPersistenceConnector,
    create_connector
)


class TestVectorTonicPersistenceConnector(unittest.TestCase):
    """Test the VectorTonicPersistenceConnector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_event_bus = MagicMock(spec=LocalEventBus)
        self.mock_db = MagicMock()
        self.mock_persistence_integration = MagicMock()
        
        # Mock repositories
        self.mock_field_state_repository = MagicMock(spec=FieldStateRepositoryInterface)
        self.mock_pattern_repository = MagicMock(spec=PatternRepositoryInterface)
        self.mock_relationship_repository = MagicMock(spec=RelationshipRepositoryInterface)
        self.mock_topology_repository = MagicMock(spec=TopologyRepositoryInterface)
        
        # Create connector with mocks
        self.connector = VectorTonicPersistenceConnector(
            event_bus=self.mock_event_bus,
            db=self.mock_db,
            field_state_repository=self.mock_field_state_repository,
            pattern_repository=self.mock_pattern_repository,
            relationship_repository=self.mock_relationship_repository,
            topology_repository=self.mock_topology_repository
        )
        self.connector.persistence_integration = self.mock_persistence_integration
        self.connector.pattern_service = self.mock_persistence_integration.pattern_service
        self.connector.field_state_service = self.mock_persistence_integration.field_state_service
        self.connector.relationship_service = self.mock_persistence_integration.relationship_service
    
    def test_initialize(self):
        """Test that initialize initializes the persistence integration and subscribes to events."""
        # Call initialize
        self.connector.initialize()
        
        # Verify persistence integration was initialized
        self.mock_persistence_integration.initialize.assert_called_once()
        
        # Verify pattern event subscriptions
        self.mock_event_bus.subscribe.assert_any_call("pattern.detected", self.connector._on_pattern_detected)
        self.mock_event_bus.subscribe.assert_any_call("pattern.evolved", self.connector._on_pattern_evolution)
        self.mock_event_bus.subscribe.assert_any_call("pattern.quality.changed", self.connector._on_pattern_quality_change)
        self.mock_event_bus.subscribe.assert_any_call("pattern.relationship.detected", self.connector._on_pattern_relationship_detected)
        self.mock_event_bus.subscribe.assert_any_call("pattern.merged", self.connector._on_pattern_merge)
        self.mock_event_bus.subscribe.assert_any_call("pattern.split", self.connector._on_pattern_split)
        
        # Verify field state event subscriptions
        self.mock_event_bus.subscribe.assert_any_call("field.state.changed", self.connector._on_field_state_change)
        self.mock_event_bus.subscribe.assert_any_call("field.coherence.changed", self.connector._on_field_coherence_change)
        self.mock_event_bus.subscribe.assert_any_call("field.stability.changed", self.connector._on_field_stability_change)
        self.mock_event_bus.subscribe.assert_any_call("field.density.centers.shifted", self.connector._on_density_center_shift)
        self.mock_event_bus.subscribe.assert_any_call("field.eigenspace.changed", self.connector._on_eigenspace_change)
        self.mock_event_bus.subscribe.assert_any_call("field.topology.changed", self.connector._on_topology_change)
        
        # Verify learning window event subscriptions
        self.mock_event_bus.subscribe.assert_any_call("learning.window.state.changed", self.connector._on_window_state_change)
        self.mock_event_bus.subscribe.assert_any_call("learning.window.opened", self.connector._on_window_open)
        self.mock_event_bus.subscribe.assert_any_call("learning.window.closed", self.connector._on_learning_window_closed)
        self.mock_event_bus.subscribe.assert_any_call("learning.window.back.pressure", self.connector._on_back_pressure)
        
        # Verify legacy event subscriptions
        self.mock_event_bus.subscribe.assert_any_call("document.processed", self.connector._on_document_processed)
        self.mock_event_bus.subscribe.assert_any_call("vector.gradient.updated", self.connector._on_vector_gradient_updated)
        
        # Verify initialization state
        self.assertTrue(self.connector.initialized)
    
    def test_connect_to_integrator(self):
        """Test connecting to a VectorTonicWindowIntegrator."""
        # Create a mock integrator
        mock_integrator = MagicMock(spec=VectorTonicWindowIntegrator)
        mock_integrator.initialized = False
        
        # Connect to the integrator
        self.connector.connect_to_integrator(mock_integrator)
        
        # Verify integrator was initialized if not already
        mock_integrator.initialize.assert_called_once()
        
        # Verify the connector registered as an observer
        mock_integrator.register_learning_window_observer.assert_called_once_with(self.connector)
        mock_integrator.register_pattern_observer.assert_called_once_with(self.connector)
        mock_integrator.register_field_observer.assert_called_once_with(self.connector)
        
        # Verify connection event was published
        self.mock_event_bus.publish.assert_called_once()
        
        # Verify event data
        args, _ = self.mock_event_bus.publish.call_args
        event = args[0]
        self.assertEqual(event.type, "persistence.connected")
        self.assertEqual(event.data["integrator_id"], id(mock_integrator))
    
    def test_process_document(self):
        """Test processing a document."""
        # Create a test document
        document = {
            "id": "test_doc",
            "content": "This is a test document about climate change."
        }
        
        # Configure mock persistence integration
        self.mock_persistence_integration.process_document.return_value = "doc_id"
        
        # Process the document
        result = self.connector.process_document(document)
        
        # Verify document was processed
        self.mock_persistence_integration.process_document.assert_called_with(document)
        
        # Verify result
        self.assertEqual(result, "doc_id")
    
    # Tests for LearningWindowObserverInterface methods
    def test_on_window_state_change(self):
        """Test handling window state change events."""
        # Create test data
        window_id = str(uuid.uuid4())
        previous_state = LearningWindowState.CLOSED
        new_state = LearningWindowState.OPENING
        metadata = {"reason": "test"}
        
        # Call the observer method
        self.connector.on_window_state_change(window_id, previous_state, new_state, metadata)
        
        # Verify the active window was updated
        self.assertIn(window_id, self.connector.active_windows)
        self.assertEqual(self.connector.active_windows[window_id]["state"], new_state)
        
        # Verify event was published
        self.mock_event_bus.publish.assert_called_once()
        args, _ = self.mock_event_bus.publish.call_args
        event = args[0]
        self.assertEqual(event.type, "learning.window.state.changed")
        self.assertEqual(event.data["window_id"], window_id)
        self.assertEqual(event.data["previous_state"], previous_state.value)
        self.assertEqual(event.data["new_state"], new_state.value)
    
    def test_on_window_open(self):
        """Test handling window open events."""
        # Create test data
        window_id = str(uuid.uuid4())
        metadata = {"reason": "test"}
        
        # Call the observer method
        self.connector.on_window_open(window_id, metadata)
        
        # Verify the active window was updated
        self.assertIn(window_id, self.connector.active_windows)
        self.assertEqual(self.connector.active_windows[window_id]["state"], LearningWindowState.OPEN)
        
        # Verify event was published
        self.mock_event_bus.publish.assert_called_once()
        args, _ = self.mock_event_bus.publish.call_args
        event = args[0]
        self.assertEqual(event.type, "learning.window.opened")
        self.assertEqual(event.data["window_id"], window_id)
    
    def test_on_window_close(self):
        """Test handling window close events."""
        # Create test data
        window_id = str(uuid.uuid4())
        patterns_detected = {
            "pattern1": {"id": "pattern1", "name": "Test Pattern 1", "confidence": 0.8},
            "pattern2": {"id": "pattern2", "name": "Test Pattern 2", "confidence": 0.7}
        }
        metadata = {"field_state": {"id": "field1", "coherence": 0.9, "stability": 0.8}}
        
        # Setup active window
        self.connector.active_windows[window_id] = {
            "state": LearningWindowState.OPEN,
            "opened_at": datetime.now().isoformat()
        }
        
        # Call the observer method
        self.connector.on_window_close(window_id, patterns_detected, metadata)
        
        # Verify the active window was updated
        self.assertEqual(self.connector.active_windows[window_id]["state"], LearningWindowState.CLOSED)
        self.assertIn("closed_at", self.connector.active_windows[window_id])
        
        # Verify pattern cache was updated
        for pattern_id, pattern_data in patterns_detected.items():
            self.assertIn(pattern_id, self.connector.pattern_cache)
            self.assertEqual(self.connector.pattern_cache[pattern_id], pattern_data)
        
        # Verify events were published (one for each pattern + one for field state)
        self.assertEqual(self.mock_event_bus.publish.call_count, len(patterns_detected) + 1)
    
    def test_on_back_pressure(self):
        """Test handling back pressure events."""
        # Create test data
        window_id = str(uuid.uuid4())
        pressure_level = 0.75
        metadata = {"reason": "high load"}
        
        # Setup active window
        self.connector.active_windows[window_id] = {
            "state": LearningWindowState.OPEN,
            "opened_at": datetime.now().isoformat()
        }
        
        # Call the observer method
        self.connector.on_back_pressure(window_id, pressure_level, metadata)
        
        # Verify the active window was updated
        self.assertEqual(self.connector.active_windows[window_id]["pressure_level"], pressure_level)
        self.assertIn("pressure_detected_at", self.connector.active_windows[window_id])
        
        # Verify event was published
        self.mock_event_bus.publish.assert_called_once()
        args, _ = self.mock_event_bus.publish.call_args
        event = args[0]
        self.assertEqual(event.type, "learning.window.back.pressure")
        self.assertEqual(event.data["window_id"], window_id)
        self.assertEqual(event.data["pressure_level"], pressure_level)
    
    # Tests for PatternObserverInterface methods
    def test_on_pattern_detected(self):
        """Test handling pattern detected events."""
        # Create test data
        pattern_id = str(uuid.uuid4())
        pattern_data = {
            "id": pattern_id,
            "name": "Test Pattern",
            "confidence": 0.9,
            "vector": [0.1, 0.2, 0.3]
        }
        metadata = {"window_id": str(uuid.uuid4())}
        
        # Call the observer method
        self.connector.on_pattern_detected(pattern_id, pattern_data, metadata)
        
        # Verify pattern was persisted
        self.mock_pattern_repository.save.assert_called_once()
        args, _ = self.mock_pattern_repository.save.call_args
        saved_pattern = args[0]
        self.assertEqual(saved_pattern["id"], pattern_id)
        
        # Verify pattern cache was updated
        self.assertIn(pattern_id, self.connector.pattern_cache)
        self.assertEqual(self.connector.pattern_cache[pattern_id], pattern_data)
        
        # Verify event was published
        self.mock_event_bus.publish.assert_called_once()
        args, _ = self.mock_event_bus.publish.call_args
        event = args[0]
        self.assertEqual(event.type, "pattern.detected")
        self.assertEqual(event.data["pattern_id"], pattern_id)
    
    def test_on_pattern_evolution(self):
        """Test handling pattern evolution events."""
        # Create test data
        pattern_id = str(uuid.uuid4())
        previous_version = {
            "id": pattern_id,
            "version": 1,
            "confidence": 0.8,
            "vector": [0.1, 0.2, 0.3]
        }
        new_version = {
            "id": pattern_id,
            "version": 2,
            "confidence": 0.9,
            "vector": [0.15, 0.25, 0.35]
        }
        metadata = {"window_id": str(uuid.uuid4())}
        
        # Call the observer method
        self.connector.on_pattern_evolution(pattern_id, previous_version, new_version, metadata)
        
        # Verify pattern was persisted
        self.mock_pattern_repository.save.assert_called_once()
        args, _ = self.mock_pattern_repository.save.call_args
        saved_pattern = args[0]
        self.assertEqual(saved_pattern["id"], pattern_id)
        self.assertEqual(saved_pattern["version"], new_version["version"])
        
        # Verify pattern cache was updated
        self.assertIn(pattern_id, self.connector.pattern_cache)
        self.assertEqual(self.connector.pattern_cache[pattern_id], new_version)
        
        # Verify event was published
        self.mock_event_bus.publish.assert_called_once()
        args, _ = self.mock_event_bus.publish.call_args
        event = args[0]
        self.assertEqual(event.type, "pattern.evolved")
        self.assertEqual(event.data["pattern_id"], pattern_id)
    
    def test_on_pattern_quality_change(self):
        """Test handling pattern quality change events."""
        # Create test data
        pattern_id = str(uuid.uuid4())
        previous_quality = 0.8
        new_quality = 0.9
        quality_data = {
            "confidence": new_quality,
            "stability": 0.85,
            "coherence": 0.95
        }
        metadata = {"window_id": str(uuid.uuid4())}
        
        # Setup pattern cache
        self.connector.pattern_cache[pattern_id] = {
            "id": pattern_id,
            "confidence": previous_quality
        }
        
        # Call the observer method
        self.connector.on_pattern_quality_change(pattern_id, previous_quality, new_quality, quality_data, metadata)
        
        # Verify pattern was persisted
        self.mock_pattern_repository.update_quality.assert_called_once()
        args, _ = self.mock_pattern_repository.update_quality.call_args
        self.assertEqual(args[0], pattern_id)
        self.assertEqual(args[1], quality_data)
        
        # Verify pattern cache was updated
        self.assertEqual(self.connector.pattern_cache[pattern_id]["confidence"], new_quality)
        
        # Verify event was published
        self.mock_event_bus.publish.assert_called_once()
        args, _ = self.mock_event_bus.publish.call_args
        event = args[0]
        self.assertEqual(event.type, "pattern.quality.changed")
        self.assertEqual(event.data["pattern_id"], pattern_id)
        self.assertEqual(event.data["new_quality"], new_quality)
    
    def test_on_pattern_relationship_detected(self):
        """Test handling pattern relationship detected events."""
        # Create test data
        source_id = str(uuid.uuid4())
        target_id = str(uuid.uuid4())
        relationship_data = {
            "type": "SIMILAR_TO",
            "strength": 0.85
        }
        metadata = {"window_id": str(uuid.uuid4())}
        
        # Call the observer method
        self.connector.on_pattern_relationship_detected(source_id, target_id, relationship_data, metadata)
        
        # Verify relationship was persisted
        self.mock_relationship_repository.save.assert_called_once()
        args, _ = self.mock_relationship_repository.save.call_args
        saved_relationship = args[0]
        self.assertEqual(saved_relationship["source_id"], source_id)
        self.assertEqual(saved_relationship["target_id"], target_id)
        
        # Verify event was published
        self.mock_event_bus.publish.assert_called_once()
        args, _ = self.mock_event_bus.publish.call_args
        event = args[0]
        self.assertEqual(event.type, "pattern.relationship.detected")
        self.assertEqual(event.data["source_id"], source_id)
        self.assertEqual(event.data["target_id"], target_id)
    
    # Tests for FieldObserverInterface methods
    def test_on_field_state_change(self):
        """Test handling field state change events."""
        # Create test data
        field_id = str(uuid.uuid4())
        previous_state = {
            "id": field_id,
            "coherence": 0.7,
            "stability": 0.6
        }
        new_state = {
            "id": field_id,
            "coherence": 0.8,
            "stability": 0.7
        }
        metadata = {"window_id": str(uuid.uuid4())}
        
        # Call the observer method
        self.connector.on_field_state_change(field_id, previous_state, new_state, metadata)
        
        # Verify field state was persisted
        self.mock_field_state_repository.save.assert_called_once()
        args, _ = self.mock_field_state_repository.save.call_args
        saved_state = args[0]
        self.assertEqual(saved_state["id"], field_id)
        
        # Verify event was published
        self.mock_event_bus.publish.assert_called_once()
        args, _ = self.mock_event_bus.publish.call_args
        event = args[0]
        self.assertEqual(event.type, "field.state.changed")
        self.assertEqual(event.data["field_id"], field_id)
    
    def test_on_field_coherence_change(self):
        """Test handling field coherence change events."""
        # Create test data
        field_id = str(uuid.uuid4())
        previous_coherence = 0.7
        new_coherence = 0.8
        metadata = {"window_id": str(uuid.uuid4())}
        
        # Call the observer method
        self.connector.on_field_coherence_change(field_id, previous_coherence, new_coherence, metadata)
        
        # Verify field state was persisted
        self.mock_field_state_repository.update_coherence.assert_called_once()
        args, _ = self.mock_field_state_repository.update_coherence.call_args
        self.assertEqual(args[0], field_id)
        self.assertEqual(args[1], new_coherence)
        
        # Verify event was published
        self.mock_event_bus.publish.assert_called_once()
        args, _ = self.mock_event_bus.publish.call_args
        event = args[0]
        self.assertEqual(event.type, "field.coherence.changed")
        self.assertEqual(event.data["field_id"], field_id)
        self.assertEqual(event.data["new_coherence"], new_coherence)
    
    def test_on_field_stability_change(self):
        """Test handling field stability change events."""
        # Create test data
        field_id = str(uuid.uuid4())
        previous_stability = 0.6
        new_stability = 0.7
        metadata = {"window_id": str(uuid.uuid4())}
        
        # Call the observer method
        self.connector.on_field_stability_change(field_id, previous_stability, new_stability, metadata)
        
        # Verify field state was persisted
        self.mock_field_state_repository.update_stability.assert_called_once()
        args, _ = self.mock_field_state_repository.update_stability.call_args
        self.assertEqual(args[0], field_id)
        self.assertEqual(args[1], new_stability)
        
        # Verify event was published
        self.mock_event_bus.publish.assert_called_once()
        args, _ = self.mock_event_bus.publish.call_args
        event = args[0]
        self.assertEqual(event.type, "field.stability.changed")
        self.assertEqual(event.data["field_id"], field_id)
        self.assertEqual(event.data["new_stability"], new_stability)
    
    def test_on_density_center_shift(self):
        """Test handling density center shift events."""
        # Create test data
        field_id = str(uuid.uuid4())
        previous_centers = [[0.1, 0.2, 0.3]]
        new_centers = [[0.15, 0.25, 0.35]]
        metadata = {"window_id": str(uuid.uuid4())}
        
        # Call the observer method
        self.connector.on_density_center_shift(field_id, previous_centers, new_centers, metadata)
        
        # Verify field state was persisted
        self.mock_field_state_repository.update_density_centers.assert_called_once()
        args, _ = self.mock_field_state_repository.update_density_centers.call_args
        self.assertEqual(args[0], field_id)
        self.assertEqual(args[1], new_centers)
        
        # Verify event was published
        self.mock_event_bus.publish.assert_called_once()
        args, _ = self.mock_event_bus.publish.call_args
        event = args[0]
        self.assertEqual(event.type, "field.density.centers.shifted")
        self.assertEqual(event.data["field_id"], field_id)
    
    def test_on_eigenspace_change(self):
        """Test handling eigenspace change events."""
        # Create test data
        field_id = str(uuid.uuid4())
        previous_eigenspace = {
            "eigenvalues": [1.0, 0.5, 0.2],
            "eigenvectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        }
        new_eigenspace = {
            "eigenvalues": [1.1, 0.6, 0.3],
            "eigenvectors": [[0.15, 0.25, 0.35], [0.45, 0.55, 0.65], [0.75, 0.85, 0.95]]
        }
        metadata = {"window_id": str(uuid.uuid4())}
        
        # Call the observer method
        self.connector.on_eigenspace_change(field_id, previous_eigenspace, new_eigenspace, metadata)
        
        # Verify field state was persisted
        self.mock_field_state_repository.update_eigenspace.assert_called_once()
        args, _ = self.mock_field_state_repository.update_eigenspace.call_args
        self.assertEqual(args[0], field_id)
        self.assertEqual(args[1], new_eigenspace)
        
        # Verify event was published
        self.mock_event_bus.publish.assert_called_once()
        args, _ = self.mock_event_bus.publish.call_args
        event = args[0]
        self.assertEqual(event.type, "field.eigenspace.changed")
        self.assertEqual(event.data["field_id"], field_id)
    
    def test_on_topology_change(self):
        """Test handling topology change events."""
        # Create test data
        field_id = str(uuid.uuid4())
        previous_topology = {
            "clusters": 3,
            "connectivity": 0.6
        }
        new_topology = {
            "clusters": 4,
            "connectivity": 0.7
        }
        metadata = {"window_id": str(uuid.uuid4())}
        
        # Call the observer method
        self.connector.on_topology_change(field_id, previous_topology, new_topology, metadata)
        
        # Verify topology was persisted
        self.mock_topology_repository.save.assert_called_once()
        args, _ = self.mock_topology_repository.save.call_args
        saved_topology = args[0]
        self.assertEqual(saved_topology["field_id"], field_id)
        self.assertEqual(saved_topology["topology"], new_topology)
        
        # Verify event was published
        self.mock_event_bus.publish.assert_called_once()
        args, _ = self.mock_event_bus.publish.call_args
        event = args[0]
        self.assertEqual(event.type, "field.topology.changed")
        self.assertEqual(event.data["field_id"], field_id)
    
    def test_on_document_processed(self):
        """Test handling document processed events."""
        # Create a mock event
        mock_event = MagicMock()
        mock_event.data = {
            "document_id": "doc_id",
            "entities": [
                {
                    "id": "entity_1",
                    "type": "CONCEPT",
                    "text": "Climate Change",
                    "confidence": 0.9
                },
                {
                    "id": "entity_2",
                    "type": "CONCEPT",
                    "text": "Food Security",
                    "confidence": 0.85
                }
            ],
            "relationships": [
                {
                    "source": "Climate Change",
                    "predicate": "impacts",
                    "target": "Food Security",
                    "confidence": 0.8
                }
            ]
        }
        
        # Call the event handler
        self.connector._on_document_processed(mock_event)
        
        # Verify entity events were published
        self.assertEqual(self.mock_event_bus.publish.call_count, 3)  # 2 entities + 1 relationship
        
        # Verify entity event data
        entity_calls = [
            call for call in self.mock_event_bus.publish.call_args_list 
            if call[0][0].type == "entity.detected"
        ]
        self.assertEqual(len(entity_calls), 2)
        
        # Verify relationship event data
        rel_calls = [
            call for call in self.mock_event_bus.publish.call_args_list 
            if call[0][0].type == "relationship.detected"
        ]
        self.assertEqual(len(rel_calls), 1)
        rel_event = rel_calls[0][0][0]
        self.assertEqual(rel_event.data["source"], "Climate Change")
        self.assertEqual(rel_event.data["predicate"], "impacts")
        self.assertEqual(rel_event.data["target"], "Food Security")
    
    def test_on_vector_gradient_updated(self):
        """Test handling vector gradient updated events."""
        # Create a mock event
        mock_event = MagicMock()
        mock_event.data = {
            "gradient": {
                "field_state_id": "field_state_1",
                "metrics": {
                    "density": 0.65,
                    "turbulence": 0.35,
                    "coherence": 0.75,
                    "stability": 0.8,
                    "pattern_count": 12
                }
            }
        }
        
        # Call the event handler
        self.connector._on_vector_gradient_updated(mock_event)
        
        # Verify field state event was published
        self.mock_event_bus.publish.assert_called_once()
        
        # Verify event data
        args, _ = self.mock_event_bus.publish.call_args
        event = args[0]
        self.assertEqual(event.type, "field.state.updated")
        self.assertEqual(event.data["field_state"]["id"], "field_state_1")
        self.assertEqual(event.data["field_state"]["metrics"]["pattern_count"], 12)
    
    def test_on_learning_window_closed(self):
        """Test handling learning window closed events."""
        # Create a mock event
        mock_event = MagicMock()
        mock_event.data = {
            "window_id": "window_1",
            "patterns": [
                {
                    "id": "pattern_1",
                    "confidence": 0.85,
                    "description": "Climate change pattern"
                },
                {
                    "id": "pattern_2",
                    "confidence": 0.75,
                    "description": "Food security pattern"
                }
            ],
            "field_state": {
                "id": "field_state_1",
                "density": 0.65,
                "turbulence": 0.35
            }
        }
        
        # Call the event handler
        self.connector._on_learning_window_closed(mock_event)
        
        # Verify events were published
        self.assertEqual(self.mock_event_bus.publish.call_count, 3)  # 2 patterns + 1 field state
        
        # Verify pattern event data
        pattern_calls = [
            call for call in self.mock_event_bus.publish.call_args_list 
            if call[0][0].type == "pattern.detected"
        ]
        self.assertEqual(len(pattern_calls), 2)
        
        # Verify field state event data
        field_calls = [
            call for call in self.mock_event_bus.publish.call_args_list 
            if call[0][0].type == "field.state.updated"
        ]
        self.assertEqual(len(field_calls), 1)
        field_event = field_calls[0][0][0]
        self.assertEqual(field_event.data["field_state"]["id"], "field_state_1")
        self.assertEqual(field_event.data["window_id"], "window_1")


class TestVectorTonicPersistenceConnectorFactory(unittest.TestCase):
    """Test the factory function for creating VectorTonicPersistenceConnector instances."""
    
    def test_create_connector(self):
        """Test creating a connector with the factory function."""
        # Mock dependencies
        mock_event_bus = MagicMock(spec=LocalEventBus)
        mock_db = MagicMock()
        
        # Mock the connector class
        mock_connector = MagicMock()
        
        # Patch the connector constructor
        with patch('src.habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector.VectorTonicPersistenceConnector', return_value=mock_connector):
            # Import the factory function
            from src.habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector import create_connector
            
            # Create a connector
            connector = create_connector(mock_event_bus, mock_db)
            
            # Verify connector was created with correct arguments
            from src.habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector import VectorTonicPersistenceConnector
            VectorTonicPersistenceConnector.assert_called_with(mock_event_bus, mock_db)
            
            # Verify connector was initialized
            mock_connector.initialize.assert_called_once()
            
            # Verify the result
            self.assertEqual(connector, mock_connector)


if __name__ == "__main__":
    unittest.main()
