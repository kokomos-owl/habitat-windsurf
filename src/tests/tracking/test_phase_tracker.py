"""Tests for phase tracking system."""

import pytest
from datetime import datetime, timedelta
from src.core.tracking.phase_tracker import (
    PhaseTrackingSystem,
    TaskStatus,
    ValidationLevel,
    ValidationResult,
    PhaseMetrics,
    TransitionMetrics
)

@pytest.fixture
def tracking_system():
    """Create a fresh tracking system for each test."""
    return PhaseTrackingSystem()

def test_phase_tracking(tracking_system):
    """Test basic phase tracking."""
    # Start phase with metrics
    phase_id = "phase_1"
    metrics = {
        'confidence': 0.8,
        'stability': 0.7,
        'validation': 0.9
    }
    tracking_system.start_phase(phase_id, metrics)
    
    # Verify metrics
    phase_metrics = tracking_system.feedback_tracker.phase_metrics[phase_id]
    assert phase_metrics.confidence == 0.8
    assert phase_metrics.stability == 0.7
    assert phase_metrics.validation_score == 0.9

def test_task_dependencies(tracking_system):
    """Test task dependency management."""
    # Create tasks with dependencies
    tracking_system.register_task("task1", "phase1", [])
    tracking_system.register_task("task2", "phase1", ["task1"])
    tracking_system.register_task("task3", "phase2", ["task1", "task2"])
    
    # Initially task2 and task3 should be blocked
    assert tracking_system.task_tracker.active_tasks["task2"].status == TaskStatus.BLOCKED
    assert tracking_system.task_tracker.active_tasks["task3"].status == TaskStatus.BLOCKED
    
    # Complete task1
    tracking_system.update_task("task1", TaskStatus.COMPLETED)
    
    # task2 should now be unblocked
    assert tracking_system.task_tracker.active_tasks["task2"].status == TaskStatus.PENDING
    # task3 should still be blocked
    assert tracking_system.task_tracker.active_tasks["task3"].status == TaskStatus.BLOCKED

def test_validation_chain(tracking_system):
    """Test validation chain functionality."""
    phase_id = "phase_1"
    tracking_system.start_phase(phase_id, {'confidence': 0.8})
    
    # Add validation results
    validation = ValidationResult(
        level=ValidationLevel.WARNING,
        message="Test warning",
        score=0.7
    )
    tracking_system.error_chain.add_validation_result(phase_id, validation)
    
    # Check phase status
    status = tracking_system.error_chain.get_phase_status(phase_id)
    assert status[ValidationLevel.WARNING.value] == 0.7
    assert status[ValidationLevel.CRITICAL.value] == 0.0

def test_phase_transition(tracking_system):
    """Test phase transition analysis."""
    # Setup initial phase
    phase1 = "phase_1"
    phase2 = "phase_2"
    
    tracking_system.start_phase(phase1, {
        'confidence': 0.7,
        'stability': 0.6,
        'validation': 0.8
    })
    
    tracking_system.start_phase(phase2, {
        'confidence': 0.8,
        'stability': 0.7,
        'validation': 0.9
    })
    
    # Analyze transition
    transition = tracking_system.feedback_tracker.analyze_phase_transition(phase1, phase2)
    
    assert transition.confidence_delta == 0.1
    assert transition.stability_delta == 0.1
    assert transition.validation_delta == 0.1

def test_comprehensive_phase_metrics(tracking_system):
    """Test getting comprehensive phase metrics."""
    phase_id = "test_phase"
    
    # Setup phase
    tracking_system.start_phase(phase_id, {
        'confidence': 0.8,
        'stability': 0.7,
        'validation': 0.9
    })
    
    # Add tasks
    tracking_system.register_task("task1", phase_id, [])
    tracking_system.register_task("task2", phase_id, ["task1"])
    
    # Add validation
    validation = ValidationResult(
        level=ValidationLevel.INFO,
        message="Test info",
        score=0.9
    )
    tracking_system.error_chain.add_validation_result(phase_id, validation)
    
    # Get comprehensive metrics
    metrics = tracking_system.get_phase_metrics(phase_id)
    
    assert metrics['metrics'].confidence == 0.8
    assert len(metrics['tasks']) == 2
    assert metrics['error_status'][ValidationLevel.INFO.value] == 0.9

def test_error_propagation(tracking_system):
    """Test error propagation through tasks."""
    phase_id = "error_phase"
    tracking_system.start_phase(phase_id, {'confidence': 0.8})
    
    # Create task
    task_id = "error_task"
    tracking_system.register_task(task_id, phase_id, [])
    
    # Add critical validation error
    validation = ValidationResult(
        level=ValidationLevel.CRITICAL,
        message="Critical error",
        score=0.3  # Below critical threshold
    )
    
    # Update task with error
    tracking_system.update_task(task_id, TaskStatus.FAILED, validation)
    
    # Check error is recorded
    task = tracking_system.task_tracker.active_tasks[task_id]
    assert task.status == TaskStatus.FAILED
    assert len(task.validation_results) == 1
    assert task.validation_results[0].level == ValidationLevel.CRITICAL

def test_task_status_updates(tracking_system):
    """Test task status update workflow."""
    # Create independent tasks
    task_ids = ["task1", "task2", "task3"]
    for task_id in task_ids:
        tracking_system.register_task(task_id, "phase1", [])
    
    # Update statuses
    status_flow = [
        (TaskStatus.IN_PROGRESS, "Started task"),
        (TaskStatus.COMPLETED, "Completed task"),
        (TaskStatus.FAILED, "Task failed")
    ]
    
    for task_id in task_ids:
        for status, message in status_flow:
            validation = ValidationResult(
                level=ValidationLevel.INFO,
                message=message,
                score=0.9
            )
            tracking_system.update_task(task_id, status, validation)
            
            task = tracking_system.task_tracker.active_tasks[task_id]
            assert task.status == status
            assert len(task.validation_results) == status_flow.index((status, message)) + 1
