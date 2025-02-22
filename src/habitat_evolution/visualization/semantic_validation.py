"""Semantic validation and logging framework for pattern visualization."""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s - %(context)s'
)

class ValidationStatus(Enum):
    """Validation status for semantic operations."""
    GREEN = "success"
    YELLOW = "warning"
    RED = "error"
    
    def to_color_code(self) -> str:
        """Convert status to color code for UI."""
        return {
            "success": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545"
        }[self.value]

@dataclass
class ValidationResult:
    """Result of semantic validation operation."""
    status: ValidationStatus
    message: str
    context: Dict[str, Any]
    timestamp: datetime = datetime.now()
    
    def to_log_context(self) -> Dict[str, Any]:
        """Convert to logging context."""
        return {
            "validation_status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            **self.context
        }

class SemanticValidator:
    """Validates semantic structures and operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_history: List[ValidationResult] = []
    
    def validate_node_structure(self, node: Any) -> ValidationResult:
        """Validate basic node structure."""
        try:
            required_fields = ["id", "type", "created_at"]
            missing_fields = [f for f in required_fields if not hasattr(node, f)]
            
            if missing_fields:
                return ValidationResult(
                    status=ValidationStatus.RED,
                    message=f"Missing required fields: {missing_fields}",
                    context={"node_type": getattr(node, "type", "unknown")}
                )
            
            # Validate event type if it's an event node
            if node.type == "event":
                valid_event_types = ["extreme_precipitation", "drought", "wildfire", "extratropical_storms", "storm_surge"]
                if not hasattr(node, "event_type") or node.event_type not in valid_event_types:
                    return ValidationResult(
                        status=ValidationStatus.RED,
                        message=f"Invalid event type: {getattr(node, 'event_type', 'missing')}",
                        context={"valid_types": valid_event_types}
                    )
            
            return ValidationResult(
                status=ValidationStatus.GREEN,
                message="Node structure valid",
                context={"node_id": node.id, "node_type": node.type}
            )
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.RED,
                message=f"Node validation error: {str(e)}",
                context={"error_type": type(e).__name__}
            )
    
    def validate_relationship(self, relation: Any) -> ValidationResult:
        """Validate relationship structure and references."""
        try:
            if not all(hasattr(relation, f) for f in ["source_id", "target_id", "relation_type"]):
                return ValidationResult(
                    status=ValidationStatus.RED,
                    message="Invalid relationship structure",
                    context={"relation": str(relation)}
                )
            
            if relation.strength < 0 or relation.strength > 1:
                return ValidationResult(
                    status=ValidationStatus.YELLOW,
                    message="Relationship strength out of bounds",
                    context={"strength": relation.strength}
                )
            
            return ValidationResult(
                status=ValidationStatus.GREEN,
                message="Relationship valid",
                context={
                    "source": relation.source_id,
                    "target": relation.target_id,
                    "type": relation.relation_type
                }
            )
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.RED,
                message=f"Relationship validation error: {str(e)}",
                context={"error_type": type(e).__name__}
            )
    
    def validate_temporal_sequence(self, nodes: List[Any]) -> ValidationResult:
        """Validate temporal sequence consistency."""
        try:
            # Sort nodes by year
            sorted_nodes = sorted(nodes, key=lambda x: x.year)
            
            # Check for gaps or inconsistencies
            for i in range(len(sorted_nodes) - 1):
                if sorted_nodes[i+1].year - sorted_nodes[i].year > 50:
                    return ValidationResult(
                        status=ValidationStatus.YELLOW,
                        message="Large temporal gap detected",
                        context={
                            "gap_start": sorted_nodes[i].year,
                            "gap_end": sorted_nodes[i+1].year
                        }
                    )
            
            return ValidationResult(
                status=ValidationStatus.GREEN,
                message="Temporal sequence valid",
                context={"years": [n.year for n in sorted_nodes]}
            )
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.RED,
                message=f"Temporal validation error: {str(e)}",
                context={"error_type": type(e).__name__}
            )
    
    def log_validation(self, result: ValidationResult):
        """Log validation result and update history."""
        self.validation_history.append(result)
        
        log_method = {
            ValidationStatus.GREEN: self.logger.info,
            ValidationStatus.YELLOW: self.logger.warning,
            ValidationStatus.RED: self.logger.error
        }[result.status]
        
        log_method(
            result.message,
            extra={"context": result.to_log_context()}
        )
    
    def get_ui_status(self) -> Dict[str, Any]:
        """Get current validation status for UI."""
        if not self.validation_history:
            return {"status": ValidationStatus.GREEN, "message": "No validations run"}
            
        latest = self.validation_history[-1]
        return {
            "status": latest.status,
            "message": latest.message,
            "color": latest.status.to_color_code(),
            "timestamp": latest.timestamp.isoformat(),
            "context": latest.context
        }
