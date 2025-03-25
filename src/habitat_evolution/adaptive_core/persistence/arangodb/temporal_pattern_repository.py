"""
Temporal Pattern Repository for ArangoDB.

Handles persistence of temporal patterns and their evolution to ArangoDB.
Implements pattern evolution tracking across learning windows.
"""

from typing import Dict, List, Any, Optional, Union
import logging
import uuid
from datetime import datetime

from .base_repository import ArangoDBBaseRepository
from .connection import ArangoDBConnectionManager

logger = logging.getLogger(__name__)

class TemporalPatternRepository(ArangoDBBaseRepository):
    """
    Repository for persisting temporal patterns to ArangoDB.
    
    Handles the storage and retrieval of patterns that emerge over time,
    tracking their evolution, stability, and relationships to actants and domains.
    """
    
    def __init__(self):
        """Initialize the TemporalPattern repository."""
        super().__init__()
        self.collection_name = "TemporalPattern"
        self.connection_manager = ArangoDBConnectionManager()
        self.db = self.connection_manager.get_db()
        
        # Edge collections we'll be working with
        self.pattern_evolves_to_collection = "PatternEvolvesTo"
        self.window_contains_pattern_collection = "WindowContainsPattern"
    
    def _dict_to_entity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a dictionary to a pattern entity.
        
        Since patterns are represented as dictionaries, this method
        primarily ensures that all necessary fields are present.
        
        Args:
            data: Dictionary containing pattern data
            
        Returns:
            Pattern dictionary with standardized fields
        """
        pattern = data.copy()
        
        # Ensure required fields
        if "id" not in pattern and "_key" in pattern:
            pattern["id"] = pattern["_key"]
        elif "id" not in pattern:
            pattern["id"] = str(uuid.uuid4())
        
        # Add timestamps if missing
        if "created_at" not in pattern:
            pattern["created_at"] = datetime.now().isoformat()
        if "updated_at" not in pattern:
            pattern["updated_at"] = datetime.now().isoformat()
        
        # Ensure stability is a float
        if "stability" in pattern and not isinstance(pattern["stability"], float):
            try:
                pattern["stability"] = float(pattern["stability"])
            except:
                pattern["stability"] = 0.0
        
        return pattern
    
    def save(self, pattern: Dict[str, Any]) -> str:
        """
        Save a temporal pattern to ArangoDB.
        
        Args:
            pattern: Dictionary containing pattern information
            
        Returns:
            ID of the saved pattern
        """
        # Ensure pattern has required fields
        pattern = self._dict_to_entity(pattern)
        
        # Check if pattern already exists
        existing_pattern = self.find_by_id(pattern["id"])
        
        if existing_pattern:
            # Update existing pattern
            pattern["updated_at"] = datetime.now().isoformat()
            self.update(existing_pattern["id"], pattern)
            return existing_pattern["id"]
        else:
            # Create new pattern
            pattern["_key"] = pattern["id"]
            result = self.db.collection(self.collection_name).insert(pattern, return_new=True)
            return result["_key"]
    
    def record_pattern_evolution(self, source_pattern_id: str, target_pattern_id: str, 
                                evolution_type: str, similarity: float) -> str:
        """
        Record an evolution relationship between two patterns.
        
        Args:
            source_pattern_id: ID of the source pattern
            target_pattern_id: ID of the target pattern
            evolution_type: Type of evolution (e.g., "refinement", "mutation", "combination")
            similarity: Similarity score between the patterns (0.0 to 1.0)
            
        Returns:
            ID of the created evolution edge
        """
        # Create the evolution edge
        edge = {
            "_from": f"{self.collection_name}/{source_pattern_id}",
            "_to": f"{self.collection_name}/{target_pattern_id}",
            "evolution_type": evolution_type,
            "similarity": similarity,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check if edge already exists
        existing_edge = self._find_evolution_edge(source_pattern_id, target_pattern_id)
        
        if existing_edge:
            # Update existing edge
            edge_id = existing_edge["_key"]
            self.db.collection(self.pattern_evolves_to_collection).update(edge_id, edge)
            return edge_id
        else:
            # Create new edge
            result = self.db.collection(self.pattern_evolves_to_collection).insert(edge, return_new=True)
            return result["_key"]
    
    def _find_evolution_edge(self, source_pattern_id: str, target_pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Find an evolution edge between two patterns.
        
        Args:
            source_pattern_id: ID of the source pattern
            target_pattern_id: ID of the target pattern
            
        Returns:
            Dictionary containing the edge, or None if not found
        """
        query = """
        FOR e IN @@collection
        FILTER e._from == @source AND e._to == @target
        LIMIT 1
        RETURN e
        """
        
        cursor = self.db.aql.execute(query, bind_vars={
            "@collection": self.pattern_evolves_to_collection,
            "source": f"{self.collection_name}/{source_pattern_id}",
            "target": f"{self.collection_name}/{target_pattern_id}"
        })
        
        edges = list(cursor)
        return edges[0] if edges else None
    
    def find_by_id(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a temporal pattern by its ID.
        
        Args:
            pattern_id: ID of the pattern
            
        Returns:
            Pattern dictionary, or None if not found
        """
        try:
            pattern = self.db.collection(self.collection_name).get(pattern_id)
            if not pattern:
                return None
            
            return self._dict_to_entity(pattern)
        except:
            return None
    
    def find_patterns_by_actant(self, actant_name: str) -> List[Dict[str, Any]]:
        """
        Find patterns that involve a specific actant.
        
        Args:
            actant_name: Name of the actant
            
        Returns:
            List of pattern dictionaries
        """
        query = """
        FOR p IN TemporalPattern
        FILTER @actant_name IN p.actants
        SORT p.created_at DESC
        RETURN p
        """
        
        cursor = self.db.aql.execute(query, bind_vars={
            "actant_name": actant_name
        })
        
        patterns = []
        for doc in cursor:
            patterns.append(self._dict_to_entity(doc))
        
        return patterns
    
    def find_patterns_by_domain(self, domain_id: str) -> List[Dict[str, Any]]:
        """
        Find patterns that involve a specific domain.
        
        Args:
            domain_id: ID of the domain
            
        Returns:
            List of pattern dictionaries
        """
        query = """
        FOR p IN TemporalPattern
        FILTER @domain_id IN p.domains
        SORT p.created_at DESC
        RETURN p
        """
        
        cursor = self.db.aql.execute(query, bind_vars={
            "domain_id": domain_id
        })
        
        patterns = []
        for doc in cursor:
            patterns.append(self._dict_to_entity(doc))
        
        return patterns
    
    def find_patterns_by_stability_range(self, min_stability: float, max_stability: float) -> List[Dict[str, Any]]:
        """
        Find patterns within a stability range.
        
        Args:
            min_stability: Minimum stability value (0.0 to 1.0)
            max_stability: Maximum stability value (0.0 to 1.0)
            
        Returns:
            List of pattern dictionaries
        """
        query = """
        FOR p IN TemporalPattern
        FILTER p.stability >= @min_stability AND p.stability <= @max_stability
        SORT p.stability DESC
        RETURN p
        """
        
        cursor = self.db.aql.execute(query, bind_vars={
            "min_stability": min_stability,
            "max_stability": max_stability
        })
        
        patterns = []
        for doc in cursor:
            patterns.append(self._dict_to_entity(doc))
        
        return patterns
    
    def get_pattern_evolution_chain(self, pattern_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """
        Get the evolution chain for a pattern.
        
        This method retrieves both ancestors and descendants of a pattern,
        showing how it evolved from previous patterns and into future patterns.
        
        Args:
            pattern_id: ID of the pattern
            max_depth: Maximum depth to traverse in either direction
            
        Returns:
            Dictionary containing the pattern's evolution chain
        """
        # Get ancestors (patterns that evolved into this one)
        ancestors_query = """
        FOR v, e, p IN 1..@max_depth INBOUND @pattern_id PatternEvolvesTo
        RETURN {
            pattern: v,
            edge: e,
            depth: LENGTH(p.vertices) - 1
        }
        """
        
        ancestors_cursor = self.db.aql.execute(ancestors_query, bind_vars={
            "pattern_id": f"{self.collection_name}/{pattern_id}",
            "max_depth": max_depth
        })
        
        # Get descendants (patterns that evolved from this one)
        descendants_query = """
        FOR v, e, p IN 1..@max_depth OUTBOUND @pattern_id PatternEvolvesTo
        RETURN {
            pattern: v,
            edge: e,
            depth: LENGTH(p.vertices) - 1
        }
        """
        
        descendants_cursor = self.db.aql.execute(descendants_query, bind_vars={
            "pattern_id": f"{self.collection_name}/{pattern_id}",
            "max_depth": max_depth
        })
        
        # Get the pattern itself
        pattern = self.find_by_id(pattern_id)
        
        return {
            "pattern": pattern,
            "ancestors": list(ancestors_cursor),
            "descendants": list(descendants_cursor)
        }
    
    def get_semantic_rhythms(self, time_period: str = "week") -> List[Dict[str, Any]]:
        """
        Get semantic rhythms based on pattern evolution.
        
        Semantic rhythms are temporal patterns in how patterns emerge and evolve,
        such as daily, weekly, or monthly cycles.
        
        Args:
            time_period: Time period to analyze ("day", "week", "month")
            
        Returns:
            List of semantic rhythm dictionaries
        """
        time_unit = ""
        if time_period == "day":
            time_unit = "day"
        elif time_period == "week":
            time_unit = "week"
        elif time_period == "month":
            time_unit = "month"
        else:
            time_unit = "day"  # Default
        
        query = """
        // Group patterns by time unit
        FOR p IN TemporalPattern
        COLLECT time_unit = DATE_TRUNC(@time_unit, DATE_PARSE(p.created_at))
        
        LET patterns_in_unit = (
            FOR p2 IN TemporalPattern
            FILTER DATE_TRUNC(@time_unit, DATE_PARSE(p2.created_at)) == time_unit
            RETURN p2
        )
        
        LET evolutions_in_unit = (
            FOR p2 IN patterns_in_unit
            FOR e IN PatternEvolvesTo
            FILTER e._from == CONCAT('TemporalPattern/', p2._key)
            RETURN e
        )
        
        RETURN {
            time_unit: time_unit,
            pattern_count: LENGTH(patterns_in_unit),
            evolution_count: LENGTH(evolutions_in_unit),
            avg_stability: AVG(patterns_in_unit[*].stability),
            patterns: patterns_in_unit
        }
        """
        
        cursor = self.db.aql.execute(query, bind_vars={
            "time_unit": time_unit
        })
        
        return list(cursor)
    
    def calculate_adaptive_timing(self, num_periods: int = 10) -> Dict[str, Any]:
        """
        Calculate adaptive timing based on observed semantic rhythms.
        
        This method analyzes pattern evolution over time to determine
        optimal timing for learning windows and other temporal operations.
        
        Args:
            num_periods: Number of time periods to analyze
            
        Returns:
            Dictionary containing adaptive timing recommendations
        """
        # Get daily rhythms
        daily_rhythms = self.get_semantic_rhythms("day")
        
        # Get weekly rhythms
        weekly_rhythms = self.get_semantic_rhythms("week")
        
        # Get monthly rhythms
        monthly_rhythms = self.get_semantic_rhythms("month")
        
        # Calculate optimal window sizes based on pattern density
        if not daily_rhythms:
            optimal_window_size = 100  # Default
        else:
            # Calculate average pattern count per day
            avg_patterns_per_day = sum(r["pattern_count"] for r in daily_rhythms) / len(daily_rhythms)
            optimal_window_size = max(50, min(200, int(avg_patterns_per_day * 2)))
        
        # Calculate optimal stability threshold based on average stability
        if not daily_rhythms:
            optimal_stability = 0.7  # Default
        else:
            # Calculate average stability across all rhythms
            stabilities = [r["avg_stability"] for r in daily_rhythms if r["avg_stability"] is not None]
            if not stabilities:
                optimal_stability = 0.7  # Default
            else:
                avg_stability = sum(stabilities) / len(stabilities)
                optimal_stability = max(0.5, min(0.9, avg_stability - 0.1))
        
        # Identify peak evolution times
        peak_evolution_times = []
        if daily_rhythms:
            # Sort by evolution count
            sorted_daily = sorted(daily_rhythms, key=lambda r: r["evolution_count"], reverse=True)
            peak_evolution_times = [r["time_unit"] for r in sorted_daily[:3]]
        
        return {
            "optimal_window_size": optimal_window_size,
            "optimal_stability_threshold": optimal_stability,
            "peak_evolution_times": peak_evolution_times,
            "daily_rhythms": daily_rhythms[:num_periods],
            "weekly_rhythms": weekly_rhythms[:num_periods],
            "monthly_rhythms": monthly_rhythms[:num_periods]
        }
