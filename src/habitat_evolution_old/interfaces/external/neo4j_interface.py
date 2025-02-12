"""
Neo4j interface for pattern storage and relationships.
"""
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase
from datetime import datetime

from ...core.pattern.types import Pattern
from ...core.pattern.metrics import PatternMetrics

class Neo4jPatternStore:
    """Neo4j-based pattern storage."""
    
    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
    
    async def store_pattern(self, pattern: Pattern) -> str:
        """Store pattern in Neo4j.
        
        Args:
            pattern: Pattern to store
            
        Returns:
            Pattern ID
        """
        with self._driver.session() as session:
            # Create pattern node
            result = session.write_transaction(
                self._create_pattern_node,
                pattern
            )
            
            # Create relationships
            if pattern.get('relationships'):
                session.write_transaction(
                    self._create_pattern_relationships,
                    pattern['id'],
                    pattern['relationships']
                )
            
            return result
    
    async def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Retrieve pattern from Neo4j."""
        with self._driver.session() as session:
            result = session.read_transaction(
                self._get_pattern_node,
                pattern_id
            )
            return result if result else None
    
    async def get_related_patterns(self,
                                 pattern_id: str,
                                 relationship_type: Optional[str] = None
                                 ) -> List[Pattern]:
        """Get patterns related to the given pattern."""
        with self._driver.session() as session:
            return session.read_transaction(
                self._get_related_patterns,
                pattern_id,
                relationship_type
            )
    
    def _create_pattern_node(self, tx, pattern: Pattern) -> str:
        """Create pattern node in Neo4j."""
        query = """
        CREATE (p:Pattern {
            id: $id,
            coherence: $coherence,
            energy: $energy,
            state: $state,
            metrics: $metrics,
            created_at: $created_at
        })
        RETURN p.id
        """
        
        result = tx.run(
            query,
            id=pattern['id'],
            coherence=pattern['coherence'],
            energy=pattern['energy'],
            state=pattern['state'],
            metrics=pattern['metrics'],
            created_at=datetime.now().isoformat()
        )
        
        return result.single()[0]
    
    def _create_pattern_relationships(self, tx, pattern_id: str,
                                    relationships: List[str]) -> None:
        """Create pattern relationships in Neo4j."""
        query = """
        MATCH (p1:Pattern {id: $pattern_id})
        MATCH (p2:Pattern {id: $related_id})
        CREATE (p1)-[:RELATED_TO]->(p2)
        """
        
        for related_id in relationships:
            tx.run(query, pattern_id=pattern_id, related_id=related_id)
    
    def _get_pattern_node(self, tx, pattern_id: str) -> Optional[Pattern]:
        """Get pattern node from Neo4j."""
        query = """
        MATCH (p:Pattern {id: $pattern_id})
        RETURN p
        """
        
        result = tx.run(query, pattern_id=pattern_id)
        record = result.single()
        
        if record:
            node = record[0]
            return {
                'id': node['id'],
                'coherence': node['coherence'],
                'energy': node['energy'],
                'state': node['state'],
                'metrics': node['metrics'],
                'relationships': []  # Relationships loaded separately
            }
        
        return None
    
    def _get_related_patterns(self, tx, pattern_id: str,
                            relationship_type: Optional[str]) -> List[Pattern]:
        """Get related pattern nodes from Neo4j."""
        query = """
        MATCH (p1:Pattern {id: $pattern_id})-[:RELATED_TO]->(p2:Pattern)
        RETURN p2
        """
        
        result = tx.run(query, pattern_id=pattern_id)
        patterns = []
        
        for record in result:
            node = record[0]
            patterns.append({
                'id': node['id'],
                'coherence': node['coherence'],
                'energy': node['energy'],
                'state': node['state'],
                'metrics': node['metrics'],
                'relationships': []
            })
        
        return patterns
