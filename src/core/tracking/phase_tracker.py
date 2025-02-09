"""Phase tracking system for monitoring development progress."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"

class ValidationLevel(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

@dataclass
class PhaseMetrics:
    """Metrics for a development phase."""
    confidence: float = 0.0
    stability: float = 0.0
    validation_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ValidationResult:
    """Result of a validation check."""
    level: ValidationLevel
    message: str
    score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class TaskState:
    """State of a development task."""
    task_id: str
    phase: str
    status: TaskStatus
    dependencies: List[str]
    validation_results: List[ValidationResult] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class TransitionMetrics:
    """Metrics for phase transitions."""
    from_phase: str
    to_phase: str
    confidence_delta: float
    stability_delta: float
    validation_delta: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

class InterimFeedbackTracker:
    """Tracks interim feedback during development."""
    
    def __init__(self):
        self.phase_metrics: Dict[str, PhaseMetrics] = {}
        self.feedback_chain: List[TransitionMetrics] = []
        
    def track_phase(self, phase_id: str, metrics: Dict[str, float]) -> PhaseMetrics:
        """Track metrics for a development phase."""
        phase_metrics = PhaseMetrics(
            confidence=metrics.get('confidence', 0.0),
            stability=metrics.get('stability', 0.0),
            validation_score=metrics.get('validation', 0.0)
        )
        self.phase_metrics[phase_id] = phase_metrics
        return phase_metrics
        
    def analyze_phase_transition(self, from_phase: str, to_phase: str) -> Optional[TransitionMetrics]:
        """Analyze metrics change during phase transition."""
        if from_phase not in self.phase_metrics or to_phase not in self.phase_metrics:
            return None
            
        from_metrics = self.phase_metrics[from_phase]
        to_metrics = self.phase_metrics[to_phase]
        
        transition = TransitionMetrics(
            from_phase=from_phase,
            to_phase=to_phase,
            confidence_delta=to_metrics.confidence - from_metrics.confidence,
            stability_delta=to_metrics.stability - from_metrics.stability,
            validation_delta=to_metrics.validation_score - from_metrics.validation_score
        )
        
        self.feedback_chain.append(transition)
        return transition

class ErrorMetricsChain:
    """Tracks error metrics through development phases."""
    
    def __init__(self):
        self.validation_chain: List[ValidationResult] = []
        self.error_thresholds = {
            ValidationLevel.CRITICAL: 0.8,
            ValidationLevel.WARNING: 0.6,
            ValidationLevel.INFO: 0.4
        }
        self.phase_results: Dict[str, List[ValidationResult]] = defaultdict(list)
    
    def add_validation_result(self, phase: str, result: ValidationResult):
        """Add a validation result to the chain."""
        self.validation_chain.append(result)
        self.phase_results[phase].append(result)
        
        if result.level == ValidationLevel.CRITICAL and result.score < self.error_thresholds[ValidationLevel.CRITICAL]:
            logger.error(f"Critical validation failure in phase {phase}: {result.message}")
    
    def get_phase_status(self, phase: str) -> Dict[str, float]:
        """Get current phase status with error metrics."""
        if phase not in self.phase_results:
            return {level.value: 1.0 for level in ValidationLevel}
            
        results = self.phase_results[phase]
        status = {level.value: 0.0 for level in ValidationLevel}
        
        for result in results:
            status[result.level.value] = max(
                status[result.level.value],
                result.score
            )
            
        return status

class TaskPhaseTracker:
    """Tracks tasks through development phases."""
    
    def __init__(self):
        self.active_tasks: Dict[str, TaskState] = {}
        self.phase_dependencies: List[tuple] = []
        self.phase_tasks: Dict[str, List[str]] = defaultdict(list)
        
    def register_task(self, task_id: str, phase: str, dependencies: List[str]) -> TaskState:
        """Register a new task with phase dependencies."""
        task = TaskState(
            task_id=task_id,
            phase=phase,
            status=TaskStatus.PENDING,
            dependencies=dependencies
        )
        self.active_tasks[task_id] = task
        self.phase_tasks[phase].append(task_id)
        
        # Add phase dependencies
        for dep in dependencies:
            if dep in self.active_tasks:
                dep_task = self.active_tasks[dep]
                self.phase_dependencies.append((dep_task.phase, phase))
                
        return task
        
    def update_task_status(self, task_id: str, status: TaskStatus, validation_result: Optional[ValidationResult] = None):
        """Update task status and propagate changes."""
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")
            
        task = self.active_tasks[task_id]
        task.status = status
        task.last_updated = datetime.utcnow()
        
        if validation_result:
            task.validation_results.append(validation_result)
            
        # Check dependent tasks
        self._update_dependent_tasks(task_id)
        
    def _update_dependent_tasks(self, task_id: str):
        """Update status of dependent tasks."""
        task = self.active_tasks[task_id]
        
        # Find all tasks that depend on this one
        dependent_tasks = [
            t for t in self.active_tasks.values()
            if task_id in t.dependencies
        ]
        
        for dep_task in dependent_tasks:
            # Check if all dependencies are complete
            deps_completed = all(
                self.active_tasks[dep].status == TaskStatus.COMPLETED
                for dep in dep_task.dependencies
            )
            
            if deps_completed and dep_task.status == TaskStatus.BLOCKED:
                dep_task.status = TaskStatus.PENDING
                logger.info(f"Task {dep_task.task_id} unblocked")
            elif not deps_completed and dep_task.status == TaskStatus.PENDING:
                dep_task.status = TaskStatus.BLOCKED
                logger.info(f"Task {dep_task.task_id} blocked")

class PhaseTrackingSystem:
    """Combined system for phase tracking."""
    
    def __init__(self):
        self.feedback_tracker = InterimFeedbackTracker()
        self.error_chain = ErrorMetricsChain()
        self.task_tracker = TaskPhaseTracker()
        
    def start_phase(self, phase_id: str, initial_metrics: Dict[str, float]):
        """Start tracking a new phase."""
        self.feedback_tracker.track_phase(phase_id, initial_metrics)
        
    def register_task(self, task_id: str, phase: str, dependencies: List[str]):
        """Register a new task."""
        return self.task_tracker.register_task(task_id, phase, dependencies)
        
    def update_task(self, task_id: str, status: TaskStatus, validation_result: Optional[ValidationResult] = None):
        """Update task status with validation."""
        self.task_tracker.update_task_status(task_id, status, validation_result)
        
        if validation_result:
            phase = self.task_tracker.active_tasks[task_id].phase
            self.error_chain.add_validation_result(phase, validation_result)
            
    def get_phase_metrics(self, phase_id: str) -> Dict[str, any]:
        """Get comprehensive metrics for a phase."""
        return {
            'metrics': self.feedback_tracker.phase_metrics.get(phase_id),
            'error_status': self.error_chain.get_phase_status(phase_id),
            'tasks': [
                task for task in self.task_tracker.active_tasks.values()
                if task.phase == phase_id
            ]
        }
