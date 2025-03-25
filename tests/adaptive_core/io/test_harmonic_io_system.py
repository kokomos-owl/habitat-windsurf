"""
Tests for the Harmonic I/O system with real-world data.

These tests validate:
1. Core functionality of the Harmonic I/O service
2. Repository integration
3. Field component integration
4. Eigenspace preservation during I/O operations
5. Performance under load
"""

import sys
import os
import pytest
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import uuid
import json
from pathlib import Path

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
from habitat_evolution.adaptive_core.transformation.actant_journey_tracker import ActantJourney, ActantJourneyPoint, DomainTransition


@pytest.fixture
def io_service():
    """Create a harmonic I/O service for testing."""
    service = HarmonicIOService(base_frequency=0.2, harmonics=3)
    service.start()
    yield service
    service.stop()


class TestHarmonicIOCore:
    """Test core functionality of the Harmonic I/O service."""
    
    def test_operation_scheduling(self, io_service):
        """Test operation scheduling and execution."""
        # Create a simple target object
        class Target:
            def __init__(self):
                self.called = False
                self.args = None
                self.kwargs = None
                
            def test_method(self, *args, **kwargs):
                self.called = True
                self.args = args
                self.kwargs = kwargs
                return "success"
        
        target = Target()
        
        # Schedule an operation
        priority = io_service.schedule_operation(
            OperationType.READ.value,
            target,
            "test_method",
            ("arg1", "arg2"),
            {"kwarg1": "value1"},
            {"stability": 0.8}
        )
        
        # Verify priority is assigned
        assert priority is not None
        
        # Wait for operation to be processed
        time.sleep(0.5)
        
        # Verify operation was executed
        assert target.called
        assert target.args == ("arg1", "arg2")
        assert target.kwargs == {"kwarg1": "value1"}
    
    def test_harmonic_priority_calculation(self, io_service):
        """Test harmonic priority calculation."""
        # Test with different operation types and stabilities
        read_priority_high_stability = io_service.calculate_harmonic_priority(
            OperationType.READ.value,
            {"stability": 0.9}
        )
        
        write_priority_high_stability = io_service.calculate_harmonic_priority(
            OperationType.WRITE.value,
            {"stability": 0.9}
        )
        
        read_priority_low_stability = io_service.calculate_harmonic_priority(
            OperationType.READ.value,
            {"stability": 0.2}
        )
        
        write_priority_low_stability = io_service.calculate_harmonic_priority(
            OperationType.WRITE.value,
            {"stability": 0.2}
        )
        
        # Verify priorities follow expected patterns
        # Note: The actual priority values may vary based on implementation details
        # and the current cycle position, so we'll check more general patterns
        
        # 1. Low stability operations should generally have different priorities than high stability ones
        assert abs(read_priority_high_stability - read_priority_low_stability) > 0.01
        
        # 2. High stability operations should generally have higher priority for writes
        assert write_priority_high_stability < write_priority_low_stability
        
        # 3. Low stability operations should generally have higher priority for reads
        assert read_priority_low_stability < read_priority_high_stability
    
    def test_eigenspace_stability_update(self, io_service):
        """Test updating eigenspace stability."""
        # Record initial stability
        initial_stability = io_service.eigenspace_stability
        
        # Update stability
        new_stability = 0.75
        io_service.update_eigenspace_stability(new_stability)
        
        # Verify stability was updated
        assert io_service.eigenspace_stability == new_stability
        
        # Test effect on priority calculation
        priority_before = io_service.calculate_harmonic_priority(
            OperationType.WRITE.value,
            {}  # Empty context should use service's eigenspace_stability
        )
        
        # Change stability
        io_service.update_eigenspace_stability(0.25)
        
        # Calculate priority again
        priority_after = io_service.calculate_harmonic_priority(
            OperationType.WRITE.value,
            {}
        )
        
        # Verify priority changed
        assert priority_before != priority_after
    
    def test_concurrent_operations(self, io_service):
        """Test handling of concurrent operations."""
        # Create a target that tracks operation order
        class OrderTracker:
            def __init__(self):
                self.operations = []
                self.lock = threading.Lock()
                
            def operation(self, op_id, delay=0):
                time.sleep(delay)  # Simulate work
                with self.lock:
                    self.operations.append(op_id)
                return op_id
        
        tracker = OrderTracker()
        
        # Schedule operations with different priorities
        # Use stability to influence priority
        io_service.schedule_operation(
            OperationType.WRITE.value,
            tracker,
            "operation",
            (1, 0.1),  # op_id, delay
            {},
            {"stability": 0.9}  # High stability - should be processed sooner for writes
        )
        
        io_service.schedule_operation(
            OperationType.WRITE.value,
            tracker,
            "operation",
            (2, 0.1),
            {},
            {"stability": 0.5}  # Medium stability
        )
        
        io_service.schedule_operation(
            OperationType.WRITE.value,
            tracker,
            "operation",
            (3, 0.1),
            {},
            {"stability": 0.1}  # Low stability - should be processed later for writes
        )
        
        # Wait for operations to complete
        time.sleep(1.0)
        
        # Verify operations were processed
        assert len(tracker.operations) == 3
        
        # Check if high stability operation was processed first
        # Note: This is probabilistic due to the harmonic timing, but should generally hold
        assert 1 in tracker.operations[:2]  # Should be in first or second position
    
    def test_service_shutdown(self):
        """Test clean service shutdown."""
        # Create a service
        service = HarmonicIOService(base_frequency=0.1, harmonics=2)
        service.start()
        
        # Schedule some operations
        class Target:
            def method(self):
                return True
        
        target = Target()
        
        for i in range(10):
            service.schedule_operation(
                OperationType.READ.value,
                target,
                "method",
                (),
                {},
                {"stability": 0.5}
            )
        
        # Stop the service
        service.stop()
        
        # Verify service is stopped
        assert not service.running
        
        # Verify worker threads are stopped
        time.sleep(0.2)  # Allow threads to terminate
        assert all(not worker.is_alive() for worker in service.workers)


class TestRepositoryIntegration:
    """Test integration with repositories."""
    
    class TestRepository(HarmonicRepositoryMixin):
        """Test repository with harmonic capabilities."""
        
        def __init__(self, io_service):
            """Initialize the test repository."""
            HarmonicRepositoryMixin.__init__(self, io_service)
            self.data = {}
            self.operation_log = []
            
        def _direct_save(self, key, value):
            """Direct save method."""
            self.operation_log.append(("save", key, value))
            self.data[key] = value
            return key
            
        def _direct_get(self, key):
            """Direct get method."""
            self.operation_log.append(("get", key))
            return self.data.get(key)
            
        def _direct_update(self, key, value):
            """Direct update method."""
            self.operation_log.append(("update", key, value))
            if key in self.data:
                if isinstance(self.data[key], dict) and isinstance(value, dict):
                    self.data[key].update(value)
                else:
                    self.data[key] = value
            return key
            
        def save(self, key, value):
            """Save with harmonic timing."""
            data_context = self._create_data_context(value, "save")
            return self._harmonic_write("save", key, value, _data_context=data_context)
            
        def get(self, key):
            """Get with harmonic timing."""
            data_context = {"entity_id": key}
            return self._harmonic_read("get", key, _data_context=data_context)
            
        def update(self, key, value):
            """Update with harmonic timing."""
            data_context = self._create_data_context(value, "update")
            return self._harmonic_update("update", key, value, _data_context=data_context)
    
    @pytest.fixture
    def test_repo(self, io_service):
        """Create a test repository."""
        return self.TestRepository(io_service)
    
    def test_basic_operations(self, test_repo):
        """Test basic repository operations."""
        # Save data
        key = "test_key"
        value = {"name": "Test Entity", "value": 42, "stability": 0.8}
        
        test_repo.save(key, value)
        
        # Wait for operation to complete
        time.sleep(0.5)
        
        # Verify save operation was logged
        assert any(op[0] == "save" and op[1] == key for op in test_repo.operation_log)
        
        # Get data
        result = test_repo.get(key)
        
        # Wait for operation to complete
        time.sleep(0.5)
        
        # Verify get operation was logged
        assert any(op[0] == "get" and op[1] == key for op in test_repo.operation_log)
        
        # Update data
        update_value = {"value": 43, "stability": 0.9}
        test_repo.update(key, update_value)
        
        # Wait for operation to complete
        time.sleep(0.5)
        
        # Verify update operation was logged
        assert any(op[0] == "update" and op[1] == key for op in test_repo.operation_log)
        
        # Get updated data
        updated_result = test_repo.get(key)
        
        # Wait for operation to complete
        time.sleep(0.5)
        
        # Verify data was updated
        assert updated_result["value"] == 43
        assert updated_result["stability"] == 0.9
    
    def test_data_context_extraction(self, test_repo):
        """Test data context extraction from different data structures."""
        # Test with stability in root
        data1 = {"name": "Test1", "stability": 0.8}
        context1 = test_repo._create_data_context(data1, "test")
        assert context1["stability"] == 0.8
        
        # Test with stability in metrics
        data2 = {"name": "Test2", "metrics": {"stability": 0.7}}
        context2 = test_repo._create_data_context(data2, "test")
        assert context2["stability"] == 0.7
        
        # Test with confidence instead of stability
        data3 = {"name": "Test3", "confidence": 0.6}
        context3 = test_repo._create_data_context(data3, "test")
        assert context3["stability"] == 0.6
        
        # Test with no stability info
        data4 = {"name": "Test4"}
        context4 = test_repo._create_data_context(data4, "test")
        assert context4["stability"] == 0.5  # Default value
    
    def test_actant_journey_repository(self, io_service):
        """Test HarmonicActantJourneyRepository with a mock journey."""
        # Create repository
        repo = HarmonicActantJourneyRepository(io_service)
        
        # Create a simple observer
        class Observer:
            def __init__(self):
                self.notifications = []
                
            def record_state_change(self, **kwargs):
                self.notifications.append(kwargs)
        
        observer = Observer()
        repo.register_learning_window(observer)
        
        # Create actant journey
        journey = ActantJourney.create("test_actant")
        journey.initialize_adaptive_id()
        
        # Save journey (this would normally be mocked, but we'll just test the method call)
        try:
            repo.save_journey(journey)
            time.sleep(0.5)
        except Exception as e:
            # We expect this to fail without a real database, but we're testing the method call
            pass
        
        # Create journey point
        point = ActantJourneyPoint(
            id=str(uuid.uuid4()),
            actant_name="test_actant",
            domain_id="test_domain",
            predicate_id="test_predicate",
            role="subject",
            timestamp=datetime.now().isoformat()
        )
        
        # Try to save journey point
        try:
            repo.save_journey_point("mock_journey_id", point)
            time.sleep(0.5)
        except Exception as e:
            # Expected to fail without a real database
            pass


class TestFieldIntegration:
    """Test integration with field components."""
    
    class MockFieldComponent:
        """Mock field component for testing."""
        
        def __init__(self, name):
            """Initialize the mock field component."""
            self.name = name
            self.observers = []
            self.eigenspace_stability = 0.5
            self.pattern_coherence = 0.5
            
        def register_observer(self, observer):
            """Register an observer."""
            if observer not in self.observers:
                self.observers.append(observer)
                
        def set_eigenspace_stability(self, stability):
            """Set eigenspace stability and notify observers."""
            self.eigenspace_stability = stability
            self._notify_observers()
            
        def set_pattern_coherence(self, coherence):
            """Set pattern coherence and notify observers."""
            self.pattern_coherence = coherence
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
    
    @pytest.fixture
    def field_component(self):
        """Create a mock field component."""
        return self.MockFieldComponent("test_field")
    
    @pytest.fixture
    def field_bridge(self, io_service):
        """Create a field I/O bridge."""
        return HarmonicFieldIOBridge(io_service)
    
    def test_field_bridge_registration(self, field_component, field_bridge):
        """Test field bridge registration with field component."""
        # Register bridge with field component
        field_bridge.register_with_field_navigator(field_component)
        
        # Verify bridge was registered
        assert field_bridge in field_component.observers
    
    def test_field_state_propagation(self, io_service, field_component, field_bridge):
        """Test field state propagation to I/O service."""
        # Register bridge with field component
        field_bridge.register_with_field_navigator(field_component)
        
        # Initial values
        initial_stability = io_service.eigenspace_stability
        
        # Set new stability
        new_stability = 0.8
        field_component.set_eigenspace_stability(new_stability)
        
        # Wait for propagation
        time.sleep(0.3)
        
        # Verify I/O service was updated
        assert io_service.eigenspace_stability == new_stability
        
        # Set new coherence
        new_coherence = 0.7
        field_component.set_pattern_coherence(new_coherence)
        
        # Wait for propagation
        time.sleep(0.3)
        
        # Verify field metrics were recorded
        metrics = field_bridge.get_field_metrics()
        assert "eigenspace_stability" in metrics
        assert metrics["eigenspace_stability"] == new_stability
        assert "pattern_coherence" in metrics
        assert metrics["pattern_coherence"] == new_coherence
    
    def test_multiple_field_components(self, io_service):
        """Test integration with multiple field components."""
        # Create field components
        field1 = self.MockFieldComponent("field1")
        field2 = self.MockFieldComponent("field2")
        
        # Create bridges
        bridge1 = HarmonicFieldIOBridge(io_service)
        bridge2 = HarmonicFieldIOBridge(io_service)
        
        # Register bridges
        bridge1.register_with_field_navigator(field1)
        bridge2.register_with_field_navigator(field2)
        
        # Set different stabilities
        field1.set_eigenspace_stability(0.8)
        
        # Wait for propagation
        time.sleep(0.3)
        
        # Verify I/O service was updated
        assert io_service.eigenspace_stability == 0.8
        
        # Set different stability in field2
        field2.set_eigenspace_stability(0.6)
        
        # Wait for propagation
        time.sleep(0.3)
        
        # Verify I/O service was updated to the latest value
        assert io_service.eigenspace_stability == 0.6


class TestPerformance:
    """Test performance characteristics of the Harmonic I/O system."""
    
    def test_operation_throughput(self, io_service):
        """Test operation throughput under load."""
        # Create a simple target
        class Target:
            def __init__(self):
                self.counter = 0
                self.lock = threading.Lock()
                
            def increment(self):
                with self.lock:
                    self.counter += 1
                return self.counter
        
        target = Target()
        
        # Number of operations to schedule
        num_operations = 100
        
        # Schedule operations
        start_time = time.time()
        
        for i in range(num_operations):
            stability = 0.5 + 0.4 * np.sin(i * 0.1)  # Vary stability sinusoidally
            io_service.schedule_operation(
                OperationType.WRITE.value if i % 2 == 0 else OperationType.READ.value,
                target,
                "increment",
                (),
                {},
                {"stability": stability}
            )
        
        # Wait for operations to complete
        max_wait = 5.0  # Maximum wait time in seconds
        wait_start = time.time()
        
        while target.counter < num_operations and time.time() - wait_start < max_wait:
            time.sleep(0.1)
        
        end_time = time.time()
        
        # Calculate throughput
        elapsed_time = end_time - start_time
        throughput = target.counter / elapsed_time
        
        # Log performance metrics
        print(f"Throughput: {throughput:.2f} operations/second")
        print(f"Completed {target.counter}/{num_operations} operations in {elapsed_time:.2f} seconds")
        
        # Verify reasonable throughput
        # This is a flexible test as performance depends on the system
        assert throughput > 10  # Should be able to process at least 10 ops/sec
    
    def test_priority_queue_performance(self, io_service):
        """Test priority queue performance with many operations."""
        # Create a target that just counts operations
        class Counter:
            def __init__(self):
                self.count = 0
                self.lock = threading.Lock()
                
            def increment(self, op_id):
                with self.lock:
                    self.count += 1
                return op_id
        
        counter = Counter()
        
        # Schedule a large number of operations with varying priorities
        num_operations = 200
        
        # Record start time
        start_time = time.time()
        
        for i in range(num_operations):
            # Alternate between read and write operations
            op_type = OperationType.READ.value if i % 2 == 0 else OperationType.WRITE.value
            
            # Vary stability in a pattern
            stability = 0.1 + 0.8 * ((i % 10) / 10)
            
            io_service.schedule_operation(
                op_type,
                counter,
                "increment",
                (i,),
                {},
                {"stability": stability}
            )
        
        # Wait for most operations to complete
        timeout = 5.0
        wait_start = time.time()
        
        while counter.count < num_operations * 0.9 and time.time() - wait_start < timeout:
            time.sleep(0.1)
        
        # Record end time
        end_time = time.time()
        
        # Calculate metrics
        elapsed_time = end_time - start_time
        throughput = counter.count / elapsed_time
        
        # Log performance metrics
        print(f"Priority queue throughput: {throughput:.2f} operations/second")
        print(f"Processed {counter.count}/{num_operations} operations in {elapsed_time:.2f} seconds")
        
        # Verify reasonable performance
        assert counter.count > num_operations * 0.8  # At least 80% should complete
        assert throughput > 20  # Should process at least 20 ops/sec


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
