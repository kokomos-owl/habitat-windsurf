"""
Context-aware pattern extraction with sliding windows.

This module provides the ContextAwareExtractor class which implements
a sliding window approach to extract entities with contextual awareness.
"""

from typing import Dict, List, Any, Optional, Set
import re
import logging
from pathlib import Path

from .entity_context_manager import EntityContextManager
from .quality_assessment import QualityAssessment

logger = logging.getLogger(__name__)

class ContextAwareExtractor:
    """Extract entities with contextual awareness using sliding windows.
    
    This class implements a self-reinforcing feedback mechanism for pattern extraction
    that combines sliding window approaches with context awareness and quality assessment.
    """
    
    def __init__(
        self, 
        window_sizes: List[int] = None,
        quality_threshold: float = 0.7,
        data_dir: Optional[Path] = None
    ):
        """Initialize the context-aware extractor.
        
        Args:
            window_sizes: List of window sizes to use for extraction
            quality_threshold: Threshold for "good" quality state
            data_dir: Optional directory for climate risk data
        """
        self.window_sizes = window_sizes or [2, 3, 4, 5]
        self.quality_threshold = quality_threshold
        self.data_dir = data_dir
        
        # Initialize components
        self.entity_context_manager = EntityContextManager()
        self.quality_assessor = QualityAssessment(threshold=quality_threshold)
        
        # Common stopwords to filter out
        self.stopwords = {
            "The", "A", "An", "This", "That", "These", "Those", "It", "They",
            "I", "We", "You", "He", "She", "His", "Her", "Their", "Our", "Its"
        }
        
        logger.info(f"Initialized ContextAwareExtractor with window sizes {self.window_sizes}")
    
    def extract_with_sliding_window(self, text: str) -> List[str]:
        """Extract entities using variable-sized sliding windows.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities
        """
        entities = []
        words = text.split()
        
        # Use multiple window sizes to capture multi-word entities
        for size in self.window_sizes:
            for i in range(len(words) - size + 1):
                candidate = " ".join(words[i:i+size])
                if self._is_potential_entity(candidate):
                    entities.append(candidate)
                    # Store context around the entity
                    self.entity_context_manager.store_context(
                        candidate, 
                        words, 
                        i, 
                        size
                    )
        
        return list(set(entities))  # Remove duplicates
    
    def _is_potential_entity(self, candidate: str) -> bool:
        """Determine if a candidate string is likely to be an entity.
        
        Args:
            candidate: Candidate entity string
            
        Returns:
            True if candidate is a potential entity, False otherwise
        """
        # Basic validation rules
        if len(candidate) < 3:
            return False
            
        # Check if starts with capital letter
        if not candidate[0].isupper():
            return False
            
        # Check against stopwords
        if candidate in self.stopwords:
            return False
            
        return True
    
    def process_document(self, document: str) -> Dict[str, Any]:
        """Process a document and extract context-aware entities.
        
        Args:
            document: Document text to process
            
        Returns:
            Dictionary with extraction results
        """
        # Split into paragraphs
        paragraphs = [p for p in document.split('\n\n') if p.strip()]
        
        all_entities = []
        for paragraph in paragraphs:
            # Extract entities with sliding window
            entities = self.extract_with_sliding_window(paragraph)
            all_entities.extend(entities)
        
        # Update quality assessments
        self.quality_assessor.assess_entities(
            all_entities,
            self.entity_context_manager
        )
        
        # Identify relationships between entities
        relationships = self.identify_relationships()
        
        logger.info(f"Processed document: {len(all_entities)} entities, {len(relationships)} relationships")
        
        return {
            "entities": all_entities,
            "quality_states": self.quality_assessor.get_quality_states(),
            "relationships": relationships,
            "quality_summary": self.quality_assessor.get_quality_summary()
        }
    
    def identify_relationships(self) -> List[Dict[str, Any]]:
        """Identify relationships between entities based on context.
        
        Returns:
            List of identified relationships
        """
        return self.entity_context_manager.identify_relationships(
            self.quality_assessor.get_quality_states()
        )
    
    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a file and extract context-aware entities.
        
        Args:
            file_path: Path to file to process
            
        Returns:
            Dictionary with extraction results
        """
        with open(file_path, 'r') as f:
            content = f.read()
        
        result = self.process_document(content)
        result["filename"] = file_path.name
        
        return result
    
    def get_quality_states(self) -> Dict[str, Dict[str, float]]:
        """Get the current quality states.
        
        Returns:
            Dictionary of quality states
        """
        return self.quality_assessor.get_quality_states()
    
    def get_entity_contexts(self, entity: str) -> List[Dict[str, str]]:
        """Get contexts for an entity.
        
        Args:
            entity: Entity to get contexts for
            
        Returns:
            List of contexts for the entity
        """
        return self.entity_context_manager.get_contexts(entity)
    
    def get_transitions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get quality transitions for all entities.
        
        Returns:
            Dictionary of transitions
        """
        return self.quality_assessor.get_transitions()
