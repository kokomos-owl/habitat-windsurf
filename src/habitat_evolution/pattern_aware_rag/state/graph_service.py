"""Pattern-aware graph service for Neo4j integration."""

from typing import Dict, Any, List, Optional
from datetime import datetime

# Use relative imports to avoid module path issues
from src.habitat_evolution.adaptive_core.persistence.neo4j.connection import Neo4jConnectionManager
from src.habitat_evolution.adaptive_core.persistence.neo4j.pattern_repository import Neo4jPatternRepository
from src.habitat_evolution.adaptive_core.models import Pattern

class GraphService:
    """
    Graph service for pattern-aware RAG integration with Neo4j.
    Provides bidirectional sync between patterns and graph database.
    """
    
    def __init__(self):
        self.connection = Neo4jConnectionManager()
        self.pattern_repo = Neo4jPatternRepository()
        
    async def store_pattern(self, pattern: Pattern) -> str:
        """
        Store a pattern in Neo4j with its metrics and relationships.
        Returns the pattern ID.
        """
        # Create pattern node
        pattern_id = await self.pattern_repo.create(pattern)
        
        # Store pattern metrics
        await self.pattern_repo.update_pattern_metrics(
            pattern_id,
            pattern.metrics.__dict__
        )
        
        return pattern_id
        
    async def track_relationships(
        self,
        pattern_id: str,
        related_patterns: List[str],
        relationship_type: str = "RELATED_TO"
    ) -> None:
        """Track relationships between patterns."""
        for related_id in related_patterns:
            await self.pattern_repo.create_relationship(
                pattern_id,
                related_id,
                relationship_type,
                {
                    "created_at": datetime.now().isoformat(),
                    "strength": 1.0  # Initial relationship strength
                }
            )
            
    async def map_density_centers(
        self,
        coherence_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Map and track density centers based on pattern coherence
        and relationships.
        """
        # Get high-coherence patterns
        patterns = await self.pattern_repo.get_by_coherence_range(
            coherence_threshold,
            1.0
        )
        
        # Group by relationship clusters
        centers = []
        for pattern in patterns:
            related = await self.pattern_repo.get_related_patterns(pattern.id)
            if len(related) > 2:  # Minimum cluster size
                centers.append({
                    "center_id": pattern.id,
                    "coherence": pattern.metrics.coherence,
                    "related_count": len(related),
                    "related_patterns": [r.id for r in related]
                })
                
        return centers
        
    async def get_context(
        self,
        query_pattern_id: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get graph context for a query pattern, including:
        - Related patterns up to max_depth
        - Local density center if part of one
        - Cross-pattern paths
        """
        # Get pattern and relationships
        pattern = await self.pattern_repo.read(query_pattern_id)
        if not pattern:
            return {}
            
        # Get related patterns up to max_depth
        related = await self._get_related_recursive(
            query_pattern_id,
            max_depth
        )
        
        # Find density center if part of one
        centers = await self.map_density_centers()
        center = next(
            (c for c in centers if query_pattern_id in c["related_patterns"]),
            None
        )
        
        return {
            "pattern": pattern.__dict__,
            "related_patterns": related,
            "density_center": center,
            "timestamp": datetime.now().isoformat()
        }
        
    async def _get_related_recursive(
        self,
        pattern_id: str,
        depth: int,
        visited: Optional[set] = None
    ) -> List[Dict[str, Any]]:
        """Recursively get related patterns up to max depth."""
        if depth <= 0:
            return []
            
        if visited is None:
            visited = set()
            
        visited.add(pattern_id)
        related = []
        
        # Get direct relationships
        patterns = await self.pattern_repo.get_related_patterns(pattern_id)
        
        for pattern in patterns:
            if pattern.id not in visited:
                pattern_dict = pattern.__dict__
                pattern_dict["depth"] = depth
                related.append(pattern_dict)
                
                # Recurse for next level
                next_related = await self._get_related_recursive(
                    pattern.id,
                    depth - 1,
                    visited
                )
                related.extend(next_related)
                
        return related
