"""
UserPattern: Represents the evolutionary relationship between user identity and knowledge patterns.
"""

from typing import Dict, Any, List
from datetime import datetime
import uuid



class UserPattern:
    """
    Represents the co-evolution of user identity through knowledge patterns.
    
    Key concepts:
    - Patterns as identity markers
    - Pattern similarity networks
    - Evolutionary bonds between users
    - Knowledge creation signatures
    """
    
    def __init__(self, user_id: str):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.created_at = datetime.now().isoformat()
        
        # Core tracking only
        self.pattern_interactions: List[Dict[str, Any]] = []
        self.contexts: Dict[str, List[Dict[str, Any]]] = {}
        
    def record_interaction(
        self,
        pattern_id: str,
        interaction_type: str,
        context: Dict[str, Any]
    ) -> None:
        """Record raw interaction data for future pattern emergence."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "pattern_id": pattern_id,
            "type": interaction_type,
            "context": context
        }
        self.pattern_interactions.append(interaction)
        
        # Index by context for future analysis
        for context_type, value in context.items():
            if context_type not in self.contexts:
                self.contexts[context_type] = []
            self.contexts[context_type].append(interaction)
        
    def get_interaction_history(
        self,
        context_type: str = None,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """Get raw interaction history, optionally filtered by context type."""
        if context_type:
            interactions = self.contexts.get(context_type, [])
        else:
            interactions = self.pattern_interactions
            
        if limit:
            return interactions[-limit:]
        return interactions
        

