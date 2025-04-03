"""
Repository adapter for Habitat Evolution.

This module provides adapters for repository interfaces to support
both naming conventions used in the codebase.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from src.habitat_evolution.adaptive_core.persistence.interfaces.pattern_repository import PatternRepositoryInterface
from src.habitat_evolution.adaptive_core.persistence.adapters.in_memory_pattern_repository import InMemoryPatternRepository as _InMemoryPatternRepository

# Aliases for backward compatibility
PatternRepository = PatternRepositoryInterface

# Re-export the InMemoryPatternRepository with the new interface
class InMemoryPatternRepository(_InMemoryPatternRepository):
    """Adapter for InMemoryPatternRepository to work with PatternRepository interface."""
    
    def find_by_coherence_range(self, min_coherence: float, max_coherence: float) -> List[Dict[str, Any]]:
        """Find patterns by coherence range.
        
        Args:
            min_coherence: Minimum coherence value
            max_coherence: Maximum coherence value
            
        Returns:
            List of patterns within the coherence range
        """
        return [p for p in self.patterns.values() 
                if 'coherence' in p and min_coherence <= p['coherence'] <= max_coherence]
    
    def find_by_eigenspace_position(self, position: Dict[str, float], radius: float) -> List[Dict[str, Any]]:
        """Find patterns by eigenspace position.
        
        Args:
            position: Eigenspace position
            radius: Search radius
            
        Returns:
            List of patterns within the radius of the position
        """
        # Simplified implementation for testing
        return []
    
    def find_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find a pattern by its name.
        
        Args:
            name: The name of the pattern to find
            
        Returns:
            The pattern if found, None otherwise
        """
        for pattern in self.patterns.values():
            if 'name' in pattern and pattern['name'] == name:
                return pattern
        return None
    
    def find_by_relationship(self, relationship_type: str, target_id: str) -> List[Dict[str, Any]]:
        """Find patterns by relationship.
        
        Args:
            relationship_type: Type of relationship
            target_id: ID of the target pattern
            
        Returns:
            List of patterns with the specified relationship
        """
        # Simplified implementation for testing
        return []
    
    def find_by_resonance(self, min_resonance: float) -> List[Dict[str, Any]]:
        """Find patterns by minimum resonance.
        
        Args:
            min_resonance: Minimum resonance value
            
        Returns:
            List of patterns with resonance >= min_resonance
        """
        return [p for p in self.patterns.values() 
                if 'resonance' in p and p['resonance'] >= min_resonance]
    
    def find_by_stability_range(self, min_stability: float, max_stability: float) -> List[Dict[str, Any]]:
        """Find patterns by stability range.
        
        Args:
            min_stability: Minimum stability value
            max_stability: Maximum stability value
            
        Returns:
            List of patterns within the stability range
        """
        return [p for p in self.patterns.values() 
                if 'stability' in p and min_stability <= p['stability'] <= max_stability]
    
    def find_by_time_range(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Find patterns by time range.
        
        Args:
            start_time: Start of the time range
            end_time: End of the time range
            
        Returns:
            List of patterns created within the time range
        """
        from datetime import datetime
        
        # Convert ISO format strings to datetime objects for comparison
        def _parse_time(time_str):
            try:
                return datetime.fromisoformat(time_str)
            except (ValueError, TypeError):
                return None
        
        return [p for p in self.patterns.values() 
                if 'created_at' in p 
                and _parse_time(p['created_at']) is not None
                and start_time <= _parse_time(p['created_at']) <= end_time]
