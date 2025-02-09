"""Phase 1 task definitions and tracking."""

from datetime import datetime
from typing import Dict, List
from .phase_tracker import (
    PhaseTrackingSystem,
    TaskStatus,
    ValidationLevel,
    ValidationResult
)

# Initialize tracking system
tracking = PhaseTrackingSystem()

# Define Phase 1
PHASE_1 = "climate_risk_processing_phase1"
tracking.start_phase(PHASE_1, {
    'confidence': 0.7,  # Initial confidence
    'stability': 0.8,   # System stability
    'validation': 0.7   # Initial validation score
})

# Define Phase 1 Tasks in order of dependency
TASKS = [
    {
        "id": "p1_document_validation",
        "name": "Document Input Validation",
        "description": "Validate climate risk document structure and content",
        "dependencies": [],
        "acceptance_criteria": [
            "Document format validation",
            "Content structure verification",
            "Required sections present",
            "Temporal markers identified"
        ]
    },
    {
        "id": "p1_metric_extraction",
        "name": "Climate Metric Extraction",
        "description": "Extract and validate climate risk metrics",
        "dependencies": ["p1_document_validation"],
        "acceptance_criteria": [
            "Numeric metrics extracted",
            "Units identified",
            "Confidence scores calculated",
            "Temporal context established"
        ]
    },
    {
        "id": "p1_pattern_recognition",
        "name": "Pattern Recognition Setup",
        "description": "Initialize pattern recognition system",
        "dependencies": ["p1_metric_extraction"],
        "acceptance_criteria": [
            "Pattern types defined",
            "Recognition rules established",
            "Confidence thresholds set",
            "Evolution tracking ready"
        ]
    },
    {
        "id": "p1_temporal_processing",
        "name": "Temporal Processing Setup",
        "description": "Setup temporal processing pipeline",
        "dependencies": ["p1_pattern_recognition"],
        "acceptance_criteria": [
            "Temporal windows defined",
            "Evolution tracking ready",
            "State transitions configured",
            "Time series validation"
        ]
    },
    {
        "id": "p1_visualization_prep",
        "name": "Visualization Preparation",
        "description": "Prepare visualization components",
        "dependencies": ["p1_temporal_processing"],
        "acceptance_criteria": [
            "Graph structure defined",
            "Timeline components ready",
            "Interactive elements configured",
            "Real-time updates enabled"
        ]
    }
]

# Register all tasks
for task in TASKS:
    tracking.register_task(
        task_id=task["id"],
        phase=PHASE_1,
        dependencies=task["dependencies"]
    )

def get_next_task() -> Dict:
    """Get the next available task to work on."""
    for task in TASKS:
        task_state = tracking.task_tracker.active_tasks[task["id"]]
        if task_state.status == TaskStatus.PENDING:
            return task
    return None

def validate_task_completion(task_id: str, artifacts: Dict) -> ValidationResult:
    """Validate task completion against acceptance criteria."""
    task = next((t for t in TASKS if t["id"] == task_id), None)
    if not task:
        raise ValueError(f"Task {task_id} not found")
        
    # Special handling for document validation task
    if task_id == "p1_document_validation" and artifacts.get("validator_tests_passed", False):
        return ValidationResult(
            level=ValidationLevel.INFO,
            message="Document validation implementation complete and tested",
            score=1.0
        )
        
    # Get acceptance criteria
    criteria = task["acceptance_criteria"]
    
    # Calculate validation score based on artifacts
    met_criteria = sum(1 for c in criteria if c in artifacts)
    score = met_criteria / len(criteria)
    
    # Determine validation level
    level = ValidationLevel.CRITICAL if score < 0.8 else (
        ValidationLevel.WARNING if score < 0.9 else ValidationLevel.INFO
    )
    
    return ValidationResult(
        level=level,
        message=f"Task {task['name']} validation: {met_criteria}/{len(criteria)} criteria met",
        score=score
    )

def get_phase_status() -> Dict:
    """Get current phase status and metrics."""
    return tracking.get_phase_metrics(PHASE_1)
