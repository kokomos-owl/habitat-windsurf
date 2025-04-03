"""
Entity context management for context-aware pattern extraction.

This module provides the EntityContextManager class which stores and manages
contextual information for entities extracted from text.
"""

from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class EntityContextManager:
    """Manage context information for entities.
    
    This class stores and manages the contextual information surrounding entities
    extracted from text, enabling context-aware pattern extraction and relationship
    identification.
    """
    
    def __init__(self):
        """Initialize the entity context manager."""
        self.entity_contexts = {}  # entity -> list of contexts
    
    def store_context(self, entity: str, words: List[str], start_idx: int, size: int) -> None:
        """Store the context around an entity.
        
        Args:
            entity: The entity text
            words: List of words in the document
            start_idx: Starting index of the entity in the words list
            size: Size of the entity in words
        """
        # Get words before and after the entity
        context_before = " ".join(words[max(0, start_idx-3):start_idx])
        context_after = " ".join(words[start_idx+size:min(len(words), start_idx+size+3)])
        
        if entity not in self.entity_contexts:
            self.entity_contexts[entity] = []
            
        self.entity_contexts[entity].append({
            "before": context_before,
            "after": context_after,
            "full_text": " ".join(words)
        })
        
        logger.debug(f"Stored context for entity '{entity}': {context_before} | {context_after}")
    
    def get_contexts(self, entity: str) -> List[Dict[str, str]]:
        """Get all contexts for an entity.
        
        Args:
            entity: The entity to get contexts for
            
        Returns:
            List of context dictionaries for the entity
        """
        return self.entity_contexts.get(entity, [])
    
    def identify_relationships(self, quality_states: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Identify relationships between entities based on context.
        
        Args:
            quality_states: Dictionary of quality states for entities
            
        Returns:
            List of identified relationships
        """
        relationships = []
        
        # Get all good entities
        good_entities = list(quality_states["good"].keys())
        uncertain_entities = list(quality_states["uncertain"].keys())
        
        # Check for part-of relationships
        for uncertain in uncertain_entities:
            for good in good_entities:
                if uncertain in good and uncertain != good:
                    # Create part-of relationship
                    relationships.append({
                        "source": uncertain,
                        "predicate": "part_of",
                        "target": good,
                        "confidence": quality_states["good"][good]
                    })
                    logger.info(f"Identified part-of relationship: {uncertain} -> {good}")
        
        # Check for co-occurrence relationships
        for i, entity1 in enumerate(good_entities):
            contexts1 = self.get_contexts(entity1)
            for entity2 in good_entities[i+1:]:
                contexts2 = self.get_contexts(entity2)
                
                # Check if entities appear in similar contexts
                for ctx1 in contexts1:
                    for ctx2 in contexts2:
                        if ctx1["full_text"] == ctx2["full_text"]:
                            # Create co-occurrence relationship
                            relationships.append({
                                "source": entity1,
                                "predicate": "co_occurs_with",
                                "target": entity2,
                                "confidence": 0.8  # Default confidence for co-occurrence
                            })
                            logger.info(f"Identified co-occurrence relationship: {entity1} <-> {entity2}")
                            break
        
        return relationships
    
    def get_entity_with_context(self, entity: str) -> Dict[str, Any]:
        """Get an entity with all its contextual information.
        
        Args:
            entity: The entity to get information for
            
        Returns:
            Dictionary with entity and its contexts
        """
        return {
            "entity": entity,
            "contexts": self.get_contexts(entity),
            "context_count": len(self.get_contexts(entity))
        }
    
    def get_all_entities(self) -> List[str]:
        """Get all entities that have context information.
        
        Returns:
            List of all entities
        """
        return list(self.entity_contexts.keys())
