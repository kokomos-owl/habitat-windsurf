"""
Neo4j pattern repository implementation for the Adaptive Core system.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from ...models.pattern import Pattern
from ..interfaces import PatternRepository
from .base_repository import Neo4jBaseRepository

class Neo4jPatternRepository(Neo4jBaseRepository[Pattern], PatternRepository):
    """
    Neo4j-specific implementation of the pattern repository.
    Handles pattern persistence and relationships in Neo4j.
    """
    
    def __init__(self):
        super().__init__()
        self.node_label = "Pattern"

    def _create_entity(self, properties: Dict[str, Any]) -> Pattern:
        """Create a Pattern instance from Neo4j properties"""
        return Pattern(**properties)

    def get_by_concept(self, base_concept: str) -> List[Pattern]:
        """Get patterns by base concept"""
        query = (
            f"MATCH (n:{self.node_label} {{base_concept: $base_concept}}) "
            "RETURN n"
        )
        
        with self.connection_manager.get_session() as session:
            result = session.run(query, base_concept=base_concept)
            patterns = []
            for record in result:
                properties = self._from_node_properties(dict(record["n"]))
                pattern = self._create_entity(properties)
                patterns.append(pattern)
            return patterns

    def get_by_creator(self, creator_id: str) -> List[Pattern]:
        """Get patterns by creator"""
        query = (
            f"MATCH (n:{self.node_label} {{creator_id: $creator_id}}) "
            "RETURN n"
        )
        
        with self.connection_manager.get_session() as session:
            result = session.run(query, creator_id=creator_id)
            patterns = []
            for record in result:
                properties = self._from_node_properties(dict(record["n"]))
                pattern = self._create_entity(properties)
                patterns.append(pattern)
            return patterns

    def get_by_coherence_range(self, min_coherence: float, max_coherence: float) -> List[Pattern]:
        """Get patterns within a coherence range"""
        query = (
            f"MATCH (n:{self.node_label}) "
            "WHERE n.coherence >= $min_coherence AND n.coherence <= $max_coherence "
            "RETURN n"
        )
        
        with self.connection_manager.get_session() as session:
            result = session.run(
                query,
                min_coherence=min_coherence,
                max_coherence=max_coherence
            )
            patterns = []
            for record in result:
                properties = self._from_node_properties(dict(record["n"]))
                pattern = self._create_entity(properties)
                patterns.append(pattern)
            return patterns

    def create_relationship(self, source_id: str, target_id: str, rel_type: str, properties: Dict[str, Any]) -> str:
        """Create a relationship between patterns"""
        properties['created_at'] = datetime.now().isoformat()
        
        query = (
            f"MATCH (source:{self.node_label} {{id: $source_id}}), "
            f"(target:{self.node_label} {{id: $target_id}}) "
            f"CREATE (source)-[r:{rel_type} $properties]->(target) "
            "RETURN id(r) as rel_id"
        )
        
        with self.connection_manager.get_session() as session:
            result = session.run(
                query,
                source_id=source_id,
                target_id=target_id,
                properties=properties
            )
            record = result.single()
            if not record:
                raise ValueError("Failed to create relationship")
            return str(record["rel_id"])

    def get_related_patterns(self, pattern_id: str, rel_type: Optional[str] = None) -> List[Pattern]:
        """Get patterns related to the given pattern"""
        rel_type_clause = f":{rel_type}" if rel_type else ""
        query = (
            f"MATCH (n:{self.node_label} {{id: $pattern_id}})"
            f"-[r{rel_type_clause}]->(related:{self.node_label}) "
            "RETURN related"
        )
        
        with self.connection_manager.get_session() as session:
            result = session.run(query, pattern_id=pattern_id)
            patterns = []
            for record in result:
                properties = self._from_node_properties(dict(record["related"]))
                pattern = self._create_entity(properties)
                patterns.append(pattern)
            return patterns

    def update_pattern_metrics(self, pattern_id: str, metrics: Dict[str, float]) -> None:
        """Update pattern metrics"""
        metrics['last_modified'] = datetime.now().isoformat()
        
        query = (
            f"MATCH (n:{self.node_label} {{id: $pattern_id}}) "
            "SET n.metrics = $metrics, n.last_modified = $last_modified"
        )
        
        with self.connection_manager.get_session() as session:
            result = session.run(
                query,
                pattern_id=pattern_id,
                metrics=metrics,
                last_modified=metrics['last_modified']
            )
            if result.consume().counters.properties_set == 0:
                raise ValueError(f"Pattern with id {pattern_id} not found")
