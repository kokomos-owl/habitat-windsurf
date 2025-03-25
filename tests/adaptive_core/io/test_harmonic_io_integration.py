"""
Integration tests for the Harmonic I/O system.

These tests validate:
1. Harmonic I/O service functionality
2. Integration with field components
3. Integration with repositories
4. AdaptiveID integration
5. Eigenspace preservation
"""

import sys
import os
import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time
import threading
import uuid

# Add src to path if not already there
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import harmonic I/O components
from habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService, OperationType
from habitat_evolution.adaptive_core.io.harmonic_repository_mixin import HarmonicRepositoryMixin
from habitat_evolution.field.harmonic_field_io_bridge import HarmonicFieldIOBridge
from habitat_evolution.adaptive_core.persistence.arangodb.harmonic_actant_journey_repository import HarmonicActantJourneyRepository

# Import field components
from habitat_evolution.field.semantic_boundary_detector import SemanticBoundaryDetector
from habitat_evolution.field.field_navigator import FieldNavigator
from habitat_evolution.field.field_adaptive_id_bridge import FieldAdaptiveIDBridge

# Import adaptive core components
from habitat_evolution.adaptive_core.transformation.actant_journey_tracker import ActantJourneyTracker, ActantJourney
from habitat_evolution.pattern_aware_rag.learning.learning_control import LearningWindow, WindowState
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

# Mock classes for testing
class MockRepository:
    """Mock repository for testing harmonic I/O."""
    
    def __init__(self):
        """Initialize the mock repository."""
        self.operations = []
        self.data = {}
        
    def _direct_save(self, entity_id, data):
        """Direct save method."""
        self.operations.append(("save", entity_id, data))
        self.data[entity_id] = data
        return entity_id
        
    def _direct_get(self, entity_id):
        """Direct get method."""
        self.operations.append(("get", entity_id))
        return self.data.get(entity_id)
        
    def _direct_update(self, entity_id, data):
        """Direct update method."""
        self.operations.append(("update", entity_id, data))
        if entity_id in self.data:
            self.data[entity_id].update(data)
        else:
            self.data[entity_id] = data
        return entity_id
        
    def _direct_delete(self, entity_id):
        """Direct delete method."""
        self.operations.append(("delete", entity_id))
        if entity_id in self.data:
            del self.data[entity_id]
        return True


class MockHarmonicRepository(MockRepository, HarmonicRepositoryMixin):
    """Mock repository with harmonic I/O capabilities."""
    
    def __init__(self, io_service):
        """Initialize the mock harmonic repository."""
        MockRepository.__init__(self)
        HarmonicRepositoryMixin.__init__(self, io_service)
        
    def save(self, entity_id, data):
        """Save with harmonic timing."""
        data_context = self._create_data_context(data, "save")
        return self._harmonic_write("save", entity_id, data, _data_context=data_context)
        
    def get(self, entity_id):
        """Get with harmonic timing."""
        data_context = {"entity_id": entity_id}
        return self._harmonic_read("get", entity_id, _data_context=data_context)
        
    def update(self, entity_id, data):
        """Update with harmonic timing."""
        data_context = self._create_data_context(data, "update")
        return self._harmonic_update("update", entity_id, data, _data_context=data_context)
        
    def delete(self, entity_id):
        """Delete with harmonic timing."""
        data_context = {"entity_id": entity_id}
        return self._harmonic_delete("delete", entity_id, _data_context=data_context)


class MockFieldComponent:
    """Mock field component for testing harmonic I/O integration."""
    
    def __init__(self, name):
        """Initialize the mock field component."""
        self.name = name
        self.observers = []
        self.eigenspace_stability = 0.5
        self.pattern_coherence = 0.5
        self.state_changes = []
        
    def register_observer(self, observer):
        """Register an observer."""
        if observer not in self.observers:
            self.observers.append(observer)
            
    def set_eigenspace_stability(self, stability):
        """Set eigenspace stability and notify observers."""
        old_stability = self.eigenspace_stability
        self.eigenspace_stability = stability
        
        # Record state change
        self.state_changes.append({
            "type": "eigenspace_stability",
            "old_value": old_stability,
            "new_value": stability,
            "timestamp": datetime.now().isoformat()
        })
        
        # Notify observers
        self._notify_observers()
        
    def set_pattern_coherence(self, coherence):
        """Set pattern coherence and notify observers."""
        old_coherence = self.pattern_coherence
        self.pattern_coherence = coherence
        
        # Record state change
        self.state_changes.append({
            "type": "pattern_coherence",
            "old_value": old_coherence,
            "new_value": coherence,
            "timestamp": datetime.now().isoformat()
        })
        
        # Notify observers
        self._notify_observers()
        
    def _notify_observers(self):
        """Notify observers of state changes."""
        field_state = {
            "eigenspace": {
                "stability": self.eigenspace_stability
            },
            "patterns": {
                "coherence": self.pattern_coherence
            },
            "timestamp": datetime.now().isoformat()
        }
        
        for observer in self.observers:
            if hasattr(observer, "observe_field_state"):
                observer.observe_field_state(field_state)


class MockLearningWindow:
    """Mock learning window for testing harmonic I/O integration."""
    
    def __init__(self):
        """Initialize the mock learning window."""
        self.state_changes = []
        
    def record_state_change(self, entity_id, change_type, old_value, new_value, origin, adaptive_id=None):
        """Record a state change."""
        self.state_changes.append({
            "entity_id": entity_id,
            "change_type": change_type,
            "old_value": old_value,
            "new_value": new_value,
            "origin": origin,
            "adaptive_id": adaptive_id,
            "timestamp": datetime.now().isoformat()
        })


@pytest.fixture
def io_service():
    """Create a harmonic I/O service for testing."""
    service = HarmonicIOService(base_frequency=0.1, harmonics=3)
    service.start()
    yield service
    service.stop()


@pytest.fixture
def mock_repository(io_service):
    """Create a mock harmonic repository for testing."""
    return MockHarmonicRepository(io_service)


@pytest.fixture
def field_io_bridge(io_service):
    """Create a field I/O bridge for testing."""
    return HarmonicFieldIOBridge(io_service)


@pytest.fixture
def mock_field_component():
    """Create a mock field component for testing."""
    return MockFieldComponent("test_field")


@pytest.fixture
def mock_learning_window():
    """Create a mock learning window for testing."""
    return MockLearningWindow()


class TestHarmonicIOIntegration:
    """Test suite for harmonic I/O integration."""
    
    def test_harmonic_repository_operations(self, io_service, mock_repository):
        """Test basic harmonic repository operations."""
        # Perform operations
        entity_id = "test_entity"
        data = {"name": "Test Entity", "value": 42, "stability": 0.8}
        
        # Save
        mock_repository.save(entity_id, data)
        
        # Allow time for processing
        time.sleep(0.2)
        
        # Verify operation was recorded
        assert len(mock_repository.operations) > 0
        assert mock_repository.operations[0][0] == "save"
        
        # Get
        result = mock_repository.get(entity_id)
        
        # Allow time for processing
        time.sleep(0.2)
        
        # Verify get operation
        assert "get" in [op[0] for op in mock_repository.operations]
        
        # Update
        update_data = {"value": 43, "stability": 0.9}
        mock_repository.update(entity_id, update_data)
        
        # Allow time for processing
        time.sleep(0.2)
        
        # Verify update operation
        assert "update" in [op[0] for op in mock_repository.operations]
        
        # Delete
        mock_repository.delete(entity_id)
        
        # Allow time for processing
        time.sleep(0.2)
        
        # Verify delete operation
        assert "delete" in [op[0] for op in mock_repository.operations]
    
    def test_field_io_bridge_integration(self, io_service, field_io_bridge, mock_field_component):
        """Test integration between field components and I/O service."""
        # Register bridge with field component
        field_io_bridge.register_with_field_navigator(mock_field_component)
        
        # Initial stability
        initial_stability = io_service.eigenspace_stability
        
        # Change field component stability
        new_stability = 0.8
        mock_field_component.set_eigenspace_stability(new_stability)
        
        # Allow time for propagation
        time.sleep(0.2)
        
        # Verify I/O service was updated
        assert io_service.eigenspace_stability == new_stability
        
        # Change field component coherence
        new_coherence = 0.7
        mock_field_component.set_pattern_coherence(new_coherence)
        
        # Allow time for propagation
        time.sleep(0.2)
        
        # Verify I/O service was updated
        assert io_service.pattern_coherence == new_coherence
        
        # Verify field metrics were recorded
        metrics = field_io_bridge.get_field_metrics()
        assert "eigenspace_stability" in metrics
        assert metrics["eigenspace_stability"] == new_stability
    
    def test_harmonic_actant_journey_repository(self, io_service, mock_learning_window):
        """Test harmonic actant journey repository with AdaptiveID integration."""
        # Create repository
        repository = HarmonicActantJourneyRepository(io_service)
        
        # Register learning window
        repository.register_learning_window(mock_learning_window)
        
        # Create actant journey
        journey = ActantJourney.create("test_actant")
        
        # Initialize AdaptiveID
        journey.initialize_adaptive_id()
        
        # Save journey
        journey_id = repository.save_journey(journey)
        
        # Allow time for processing
        time.sleep(0.3)
        
        # Create journey point
        point = {
            "id": str(uuid.uuid4()),
            "actant_name": "test_actant",
            "domain_id": "test_domain",
            "predicate_id": "test_predicate",
            "role": "subject",
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.8
        }
        
        # Save journey point
        repository.save_journey_point(journey_id, point)
        
        # Allow time for processing
        time.sleep(0.3)
        
        # Verify learning window was notified
        assert len(mock_learning_window.state_changes) > 0
        
        # Create domain transition
        transition = {
            "id": str(uuid.uuid4()),
            "actant_name": "test_actant",
            "source_domain_id": "test_domain",
            "target_domain_id": "new_domain",
            "source_predicate_id": "test_predicate",
            "target_predicate_id": "new_predicate",
            "source_role": "subject",
            "target_role": "object",  # Role shift
            "timestamp": datetime.now().isoformat()
        }
        
        # Save domain transition
        repository.save_domain_transition(journey_id, transition)
        
        # Allow time for processing
        time.sleep(0.3)
        
        # Verify learning window was notified again
        assert len(mock_learning_window.state_changes) > 1
        
        # Check for role shift notification
        role_shift_notifications = [
            change for change in mock_learning_window.state_changes
            if change.get("change_type") == "domain_transition_added" and
            change.get("new_value", {}).get("source_role") != change.get("new_value", {}).get("target_role")
        ]
        
        assert len(role_shift_notifications) > 0
    
    def test_harmonic_timing_with_eigenspace_evolution(self, io_service, mock_repository, mock_field_component, field_io_bridge):
        """Test harmonic timing adapts to eigenspace evolution."""
        # Register bridge with field component
        field_io_bridge.register_with_field_navigator(mock_field_component)
        
        # Perform operations with varying stability
        entity_id = "test_entity"
        
        # Test with low stability (evolving eigenspace)
        mock_field_component.set_eigenspace_stability(0.2)
        
        # Allow time for propagation
        time.sleep(0.2)
        
        # Clear previous operations
        mock_repository.operations = []
        
        # Perform write operation
        start_time = time.time()
        mock_repository.save(entity_id, {"name": "Test Entity", "value": 42, "stability": 0.2})
        
        # Allow time for processing
        time.sleep(0.3)
        
        # Record time for low stability
        low_stability_time = time.time() - start_time
        
        # Test with high stability (stable eigenspace)
        mock_field_component.set_eigenspace_stability(0.9)
        
        # Allow time for propagation
        time.sleep(0.2)
        
        # Clear previous operations
        mock_repository.operations = []
        
        # Perform write operation
        start_time = time.time()
        mock_repository.save(entity_id, {"name": "Test Entity", "value": 43, "stability": 0.9})
        
        # Allow time for processing
        time.sleep(0.3)
        
        # Record time for high stability
        high_stability_time = time.time() - start_time
        
        # Verify timing differences
        # Note: This is a probabilistic test, so it may occasionally fail
        # The key is that harmonic timing should be influenced by eigenspace stability
        print(f"Low stability time: {low_stability_time}, High stability time: {high_stability_time}")
        
        # The actual timing difference may vary, but there should be some difference
        # reflecting the adaptive timing based on eigenspace stability
        assert abs(low_stability_time - high_stability_time) > 0.01


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
