"""
In-memory pattern repository for testing and development.

This module provides an in-memory implementation of the PatternRepository
interface for use in testing and development.
"""

from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime

from src.habitat_evolution.adaptive_core.persistence.interfaces.pattern_repository import PatternRepositoryInterface
from src.habitat_evolution.adaptive_core.models import Pattern, Relationship


class InMemoryPatternRepository(PatternRepositoryInterface):
    """In-memory implementation of the PatternRepository interface.
    
    This class provides an in-memory implementation of the PatternRepository
    interface for use in testing and development. It does not persist patterns
    to disk or a database.
    """
    
    def __init__(self):
        """Initialize the in-memory pattern repository."""
        self.patterns = {}  # id -> pattern
        self.patterns_by_text = {}  # text -> pattern
    
    def save(self, pattern: Dict[str, Any]) -> str:
        """Save a pattern to the repository.
        
        Args:
            pattern: The pattern to save.
            
        Returns:
            The ID of the saved pattern.
        """
        if 'id' not in pattern:
            pattern['id'] = str(uuid.uuid4())
        
        pattern['last_modified'] = datetime.now().isoformat()
        self.patterns[pattern['id']] = pattern
        
        if 'text' in pattern:
            self.patterns_by_text[pattern['text']] = pattern
        
        return pattern['id']
    
    def find_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Find a pattern by its ID.
        
        Args:
            id: The ID of the pattern to find.
            
        Returns:
            The pattern if found, None otherwise.
        """
        return self.patterns.get(id)
    
    def find_by_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Find a pattern by its text.
        
        Args:
            text: The text of the pattern to find.
            
        Returns:
            The pattern if found, None otherwise.
        """
        return self.patterns_by_text.get(text)
    
    def find_all(self) -> List[Dict[str, Any]]:
        """Find all patterns in the repository.
        
        Returns:
            A list of all patterns in the repository.
        """
        return list(self.patterns.values())
    
    def update(self, pattern: Dict[str, Any]) -> bool:
        """Update a pattern in the repository.
        
        Args:
            pattern: The pattern to update.
            
        Returns:
            True if the pattern was updated, False otherwise.
        """
        if 'id' not in pattern or pattern['id'] not in self.patterns:
            return False
        
        pattern['last_modified'] = datetime.now().isoformat()
        self.patterns[pattern['id']] = pattern
        
        if 'text' in pattern:
            self.patterns_by_text[pattern['text']] = pattern
        
        return True
    
    def delete(self, id: str) -> bool:
        """Delete a pattern from the repository.
        
        Args:
            id: The ID of the pattern to delete.
            
        Returns:
            True if the pattern was deleted, False otherwise.
        """
        if id not in self.patterns:
            return False
        
        pattern = self.patterns[id]
        if 'text' in pattern and pattern['text'] in self.patterns_by_text:
            del self.patterns_by_text[pattern['text']]
        
        del self.patterns[id]
        return True
    
    def find_by_property(self, property_name: str, property_value: Any) -> List[Dict[str, Any]]:
        """Find patterns by a property value.
        
        Args:
            property_name: The name of the property to search by.
            property_value: The value of the property to search for.
            
        Returns:
            A list of patterns with the specified property value.
        """
        return [p for p in self.patterns.values() 
                if property_name in p and p[property_name] == property_value]
    
    def store(self, pattern: Any) -> str:
        """Store a Pattern object in the repository.
        
        Args:
            pattern: The Pattern object to store.
            
        Returns:
            The ID of the stored pattern.
        """
        # Convert Pattern object to dictionary
        pattern_dict = self._pattern_to_dict(pattern)
        return self.save(pattern_dict)
    
    def _pattern_to_dict(self, pattern: Any) -> Dict[str, Any]:
        """Convert a Pattern object to a dictionary.
        
        Args:
            pattern: The Pattern object to convert.
            
        Returns:
            Dictionary representation of the pattern.
        """
        # Handle both our Pattern class and the existing Pattern class
        if hasattr(pattern, 'to_dict'):
            return pattern.to_dict()
        
        # Basic conversion for our simple Pattern class
        return {
            'id': getattr(pattern, 'id', str(uuid.uuid4())),
            'text': getattr(pattern, 'text', ''),
            'pattern_type': getattr(pattern, 'pattern_type', 'entity'),
            'metadata': getattr(pattern, 'metadata', {}),
            'created_at': getattr(pattern, 'created_at', datetime.now().isoformat()),
            'last_modified': datetime.now().isoformat()
        }
