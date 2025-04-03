"""
Semantic Current Observer

This module provides the SemanticCurrentObserver class which observes and records
semantic relationships between entities as they flow through the Habitat Evolution system.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
import uuid
from datetime import datetime

from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

logger = logging.getLogger(__name__)

class SemanticRelationship:
    """Represents a semantic relationship between entities."""
    
    def __init__(
        self, 
        source: str, 
        predicate: str, 
        target: str, 
        context: Dict[str, Any] = None,
        quality: str = "uncertain",
        source_category: str = None,
        target_category: str = None
    ):
        """Initialize a semantic relationship."""
        self.id = str(uuid.uuid4())
        self.source = source
        self.predicate = predicate
        self.target = target
        self.context = context or {}
        self.quality = quality
        self.source_category = source_category
        self.target_category = target_category
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.observations = 1
        
    def update_quality(self, new_quality: str):
        """Update the quality of the relationship."""
        self.quality = new_quality
        self.updated_at = datetime.now()
        
    def increment_observations(self):
        """Increment the observation count."""
        self.observations += 1
        self.updated_at = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'source': self.source,
            'predicate': self.predicate,
            'target': self.target,
            'context': self.context,
            'quality': self.quality,
            'source_category': self.source_category,
            'target_category': self.target_category,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'observations': self.observations
        }
        
    def __str__(self) -> str:
        return f"{self.source} --[{self.predicate}]--> {self.target} ({self.quality})"

class SemanticCurrentObserver:
    """
    Observes and records semantic relationships between entities.
    
    This class is responsible for tracking the semantic relationships between
    entities as they flow through the Habitat Evolution system. It maintains
    a record of all observed relationships and provides methods for querying
    and analyzing these relationships.
    """
    
    def __init__(self, observer_id: AdaptiveID):
        """Initialize the semantic current observer."""
        self.observer_id = observer_id
        self.relationships: Dict[str, SemanticRelationship] = {}
        self.entities: Set[str] = set()
        self.entity_categories: Dict[str, str] = {}
        self.entity_quality: Dict[str, str] = {}
        self.quality_transitions: List[Dict[str, Any]] = []
        
    def observe_relationship(
        self, 
        source: str, 
        predicate: str, 
        target: str, 
        context: Dict[str, Any] = None,
        quality: str = "uncertain",
        source_category: str = None,
        target_category: str = None
    ) -> SemanticRelationship:
        """
        Observe a semantic relationship between entities.
        
        Args:
            source: The source entity
            predicate: The relationship predicate
            target: The target entity
            context: Additional context for the relationship
            quality: The quality of the relationship (poor, uncertain, good)
            source_category: The category of the source entity
            target_category: The category of the target entity
            
        Returns:
            The observed relationship
        """
        # Add entities to the set of known entities
        self.entities.add(source)
        self.entities.add(target)
        
        # Update entity categories if provided
        if source_category:
            self.entity_categories[source] = source_category
        
        if target_category:
            self.entity_categories[target] = target_category
        
        # Create a unique key for the relationship
        rel_key = f"{source}|{predicate}|{target}"
        
        # Check if we've seen this relationship before
        if rel_key in self.relationships:
            # Update existing relationship
            relationship = self.relationships[rel_key]
            relationship.increment_observations()
            
            # Update quality if it has improved
            quality_rank = {"poor": 0, "uncertain": 1, "good": 2}
            if quality_rank.get(quality, 0) > quality_rank.get(relationship.quality, 0):
                old_quality = relationship.quality
                relationship.update_quality(quality)
                
                # Record quality transition
                self.quality_transitions.append({
                    'entity': source if source == target else f"{source}|{target}",
                    'relationship_id': relationship.id,
                    'from_quality': old_quality,
                    'to_quality': quality,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"Relationship quality transition: {relationship} from {old_quality} to {quality}")
        else:
            # Create new relationship
            relationship = SemanticRelationship(
                source=source,
                predicate=predicate,
                target=target,
                context=context,
                quality=quality,
                source_category=source_category,
                target_category=target_category
            )
            self.relationships[rel_key] = relationship
            logger.info(f"New relationship observed: {relationship}")
        
        return relationship
    
    def update_entity_quality(self, entity: str, quality: str):
        """
        Update the quality of an entity.
        
        Args:
            entity: The entity to update
            quality: The new quality (poor, uncertain, good)
        """
        if entity in self.entities:
            old_quality = self.entity_quality.get(entity, "poor")
            
            # Only record transition if quality has changed
            if old_quality != quality:
                self.entity_quality[entity] = quality
                
                # Record quality transition
                self.quality_transitions.append({
                    'entity': entity,
                    'from_quality': old_quality,
                    'to_quality': quality,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"Entity quality transition: {entity} from {old_quality} to {quality}")
    
    def get_entity_relationships(self, entity: str) -> List[SemanticRelationship]:
        """
        Get all relationships involving an entity.
        
        Args:
            entity: The entity to get relationships for
            
        Returns:
            List of relationships involving the entity
        """
        return [
            rel for rel_key, rel in self.relationships.items()
            if rel.source == entity or rel.target == entity
        ]
    
    def get_relationships_by_quality(self, quality: str) -> List[SemanticRelationship]:
        """
        Get all relationships with a specific quality.
        
        Args:
            quality: The quality to filter by (poor, uncertain, good)
            
        Returns:
            List of relationships with the specified quality
        """
        return [
            rel for rel_key, rel in self.relationships.items()
            if rel.quality == quality
        ]
    
    def get_relationships_by_predicate(self, predicate: str) -> List[SemanticRelationship]:
        """
        Get all relationships with a specific predicate.
        
        Args:
            predicate: The predicate to filter by
            
        Returns:
            List of relationships with the specified predicate
        """
        return [
            rel for rel_key, rel in self.relationships.items()
            if rel.predicate == predicate
        ]
    
    def get_entities_by_category(self, category: str) -> List[str]:
        """
        Get all entities in a specific category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of entities in the specified category
        """
        return [
            entity for entity, cat in self.entity_categories.items()
            if cat == category
        ]
    
    def get_entities_by_quality(self, quality: str) -> List[str]:
        """
        Get all entities with a specific quality.
        
        Args:
            quality: The quality to filter by (poor, uncertain, good)
            
        Returns:
            List of entities with the specified quality
        """
        return [
            entity for entity, q in self.entity_quality.items()
            if q == quality
        ]
    
    def get_quality_transitions(self, entity: str = None) -> List[Dict[str, Any]]:
        """
        Get quality transitions, optionally filtered by entity.
        
        Args:
            entity: The entity to filter by (optional)
            
        Returns:
            List of quality transitions
        """
        if entity:
            return [
                transition for transition in self.quality_transitions
                if transition['entity'] == entity
            ]
        return self.quality_transitions
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about observed relationships and entities.
        
        Returns:
            Dictionary of statistics
        """
        # Count entities by quality
        entities_by_quality = {
            'poor': len(self.get_entities_by_quality('poor')),
            'uncertain': len(self.get_entities_by_quality('uncertain')),
            'good': len(self.get_entities_by_quality('good'))
        }
        
        # Count entities by category
        entities_by_category = {}
        for category in set(self.entity_categories.values()):
            entities_by_category[category] = len(self.get_entities_by_category(category))
        
        # Count relationships by quality
        relationships_by_quality = {
            'poor': len(self.get_relationships_by_quality('poor')),
            'uncertain': len(self.get_relationships_by_quality('uncertain')),
            'good': len(self.get_relationships_by_quality('good'))
        }
        
        # Count relationships by predicate
        relationships_by_predicate = {}
        for rel in self.relationships.values():
            if rel.predicate not in relationships_by_predicate:
                relationships_by_predicate[rel.predicate] = 0
            relationships_by_predicate[rel.predicate] += 1
        
        # Count quality transitions
        quality_transitions = {
            'poor_to_uncertain': len([t for t in self.quality_transitions if t['from_quality'] == 'poor' and t['to_quality'] == 'uncertain']),
            'uncertain_to_good': len([t for t in self.quality_transitions if t['from_quality'] == 'uncertain' and t['to_quality'] == 'good']),
            'poor_to_good': len([t for t in self.quality_transitions if t['from_quality'] == 'poor' and t['to_quality'] == 'good'])
        }
        
        return {
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'entities_by_quality': entities_by_quality,
            'entities_by_category': entities_by_category,
            'relationships_by_quality': relationships_by_quality,
            'relationships_by_predicate': relationships_by_predicate,
            'quality_transitions': quality_transitions
        }
