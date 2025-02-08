from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

@dataclass
class BaseProjectState:
    """Base class for project state objects."""
    state_id: str
    timestamp: datetime
    metadata: Dict[str, Any]
    parent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "state_id": self.state_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "parent_id": self.parent_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseProjectState':
        """Create state from dictionary."""
        return cls(
            state_id=data["state_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data["metadata"],
            parent_id=data.get("parent_id")
        )
