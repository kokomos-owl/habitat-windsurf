"""
UserID component for managing user identity and relationships in knowledge media contexts.
"""

import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class UserContext:
    """Represents user context in knowledge media interactions."""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_active: str = field(default_factory=lambda: datetime.now().isoformat())
    preferences: Dict[str, Any] = field(default_factory=dict)
    collaborations: List[str] = field(default_factory=list)
    shared_patterns: List[str] = field(default_factory=list)
    discovered_patterns: List[str] = field(default_factory=list)

class UserID:
    """
    Manages user identity and relationships in knowledge media contexts.
    
    Key aspects:
    - User identity and authentication
    - Pattern creation and ownership
    - Collaboration tracking
    - Knowledge sharing permissions
    """
    
    def __init__(self, username: str, email: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.username = username
        self.email = email
        self.created_at = datetime.now().isoformat()
        self.context = UserContext()
        self._owned_patterns: Dict[str, Dict[str, Any]] = {}
        self._shared_access: Dict[str, List[str]] = {}
        
    def register_pattern(self, pattern_id: str, metadata: Dict[str, Any]) -> None:
        """Register a pattern as created/owned by this user."""
        self._owned_patterns[pattern_id] = {
            "created_at": datetime.now().isoformat(),
            "metadata": metadata
        }
        self.context.discovered_patterns.append(pattern_id)
        
    def share_pattern(self, pattern_id: str, user_ids: List[str]) -> None:
        """Share a pattern with other users."""
        if pattern_id not in self._owned_patterns:
            raise ValueError(f"Pattern {pattern_id} not owned by user {self.id}")
            
        if pattern_id not in self._shared_access:
            self._shared_access[pattern_id] = []
            
        self._shared_access[pattern_id].extend(user_ids)
        self.context.shared_patterns.append(pattern_id)
        
    def can_access_pattern(self, pattern_id: str, user_id: str) -> bool:
        """Check if a user has access to a pattern."""
        if pattern_id in self._owned_patterns:
            return True
            
        return (pattern_id in self._shared_access and 
                user_id in self._shared_access[pattern_id])
                
    def add_collaboration(self, pattern_id: str, user_id: str) -> None:
        """Record a collaboration on a pattern."""
        if pattern_id not in self._owned_patterns:
            raise ValueError(f"Pattern {pattern_id} not owned by user {self.id}")
            
        self.context.collaborations.append(user_id)
        
    def get_user_context(self) -> Dict[str, Any]:
        """Get the current user context."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at,
            "context": self.context
        }
        
    def update_preferences(self, preferences: Dict[str, Any]) -> None:
        """Update user preferences."""
        self.context.preferences.update(preferences)
        self.context.last_active = datetime.now().isoformat()
