"""Pattern co-evolution persistence module.

This module provides integration between the pattern co-evolution testing framework
and the Neo4j persistence layer, allowing pattern evolution and co-evolution to be
tracked and analyzed in the graph database.
"""

from typing import Dict, List, Any, Optional
import datetime
from habitat_evolution.field.pattern_coevolution_cypher import PatternCoevolutionExecutor


class PatternCoevolutionPersistence:
    """Persistence layer for pattern co-evolution.
    
    This class provides methods for persisting pattern evolution and co-evolution
    events to a Neo4j database, allowing them to be tracked and analyzed over time.
    """
    
    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str, 
                neo4j_database: str = "neo4j"):
        """Initialize the pattern co-evolution persistence layer.
        
        Args:
            neo4j_uri: URI for the Neo4j database connection
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name (default: "neo4j")
        """
        self.executor = PatternCoevolutionExecutor(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database
        )
    
    def close(self):
        """Close the Neo4j connection."""
        self.executor.close()
    
    def persist_pattern_evolution(self, pattern_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a pattern evolution event to Neo4j.
        
        Args:
            pattern_id: ID of the pattern that evolved
            context: Context of the evolution event, including:
                - old_value: Old value of the pattern
                - new_value: New value of the pattern
                - change_type: Type of change (e.g., "concept_shift", "semantic_shift")
                - origin: Origin of the change (e.g., "window1")
                - tonic_value: Tonic value at the time of evolution
                - stability: Stability value at the time of evolution
                - timestamp: Timestamp of the evolution (optional)
            
        Returns:
            Result of the persistence operation
        """
        # Create a new pattern ID for the evolved pattern
        new_pattern_id = f"{pattern_id}_{datetime.datetime.now().isoformat()}"
        
        # Extract values from context
        old_value = context.get("old_value", {})
        new_value = context.get("new_value", {})
        change_type = context.get("change_type", "unknown")
        origin = context.get("origin", "unknown")
        tonic_value = context.get("tonic_value", 0.5)
        stability = context.get("stability", 0.5)
        timestamp = context.get("timestamp", datetime.datetime.now().isoformat())
        
        # Determine pattern type based on context
        pattern_type = self._determine_pattern_type(context)
        
        # Create the old pattern if it doesn't exist
        self.executor.create_pattern_with_tonic_harmonic(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            tonic_value=tonic_value,
            stability=stability,
            timestamp=timestamp
        )
        
        # Create the new pattern
        self.executor.create_pattern_with_tonic_harmonic(
            pattern_id=new_pattern_id,
            pattern_type=pattern_type,
            tonic_value=tonic_value,
            stability=stability,
            timestamp=timestamp
        )
        
        # Create the evolution relationship
        return self.executor.create_pattern_evolution(
            old_pattern_id=pattern_id,
            new_pattern_id=new_pattern_id,
            change_type=change_type,
            window_id=origin,
            timestamp=timestamp,
            tonic_value=tonic_value,
            stability=stability,
            origin=origin
        )
    
    def persist_pattern_coevolution(self, patterns: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Persist pattern co-evolution relationships to Neo4j.
        
        Args:
            patterns: List of pattern IDs that co-evolved
            context: Context of the co-evolution event, including:
                - window_id: ID of the learning window
                - change_type: Type of change (e.g., "concept_shift", "semantic_shift")
                - tonic_value: Tonic value at the time of co-evolution
                - stability: Stability value at the time of co-evolution
                - timestamp: Timestamp of the co-evolution (optional)
            
        Returns:
            Results of the persistence operations
        """
        results = []
        
        # Extract values from context
        window_id = context.get("origin", "unknown")
        change_type = context.get("change_type", "unknown")
        tonic_value = context.get("tonic_value", 0.5)
        stability = context.get("stability", 0.5)
        harmonic_value = tonic_value * stability
        timestamp = context.get("timestamp", datetime.datetime.now().isoformat())
        
        # Create co-evolution relationships between all pairs of patterns
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                result = self.executor.create_pattern_coevolution(
                    pattern_id1=patterns[i],
                    pattern_id2=patterns[j],
                    window_id=window_id,
                    harmonic_value=harmonic_value,
                    timestamp=timestamp,
                    change_type=change_type
                )
                results.append(result)
        
        return results
    
    def record_pattern_tonic_response(self, pattern_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Record a pattern's response to a tonic value.
        
        Args:
            pattern_id: ID of the pattern
            context: Context of the response, including:
                - change_type: Type of change (e.g., "concept_shift", "semantic_shift")
                - tonic_value: Tonic value
                - response_strength: Strength of the pattern's response (optional)
                - timestamp: Timestamp of the response (optional)
            
        Returns:
            Result of the persistence operation
        """
        # Extract values from context
        change_type = context.get("change_type", "unknown")
        tonic_value = context.get("tonic_value", 0.5)
        
        # Calculate response strength based on pattern type if not provided
        response_strength = context.get("response_strength")
        if response_strength is None:
            pattern_type = self._determine_pattern_type(context)
            response_strength = self._calculate_response_strength(pattern_type, tonic_value, change_type)
        
        timestamp = context.get("timestamp", datetime.datetime.now().isoformat())
        
        return self.executor.record_pattern_tonic_response(
            pattern_id=pattern_id,
            change_type=change_type,
            tonic_value=tonic_value,
            response_strength=response_strength,
            timestamp=timestamp
        )
    
    def find_coevolved_patterns(self, window_id: str) -> List[Dict[str, Any]]:
        """Find patterns that co-evolved in a specific window.
        
        Args:
            window_id: ID of the learning window
            
        Returns:
            List of co-evolved pattern pairs
        """
        return self.executor.find_pattern_coevolution(window_id)
    
    def find_patterns_by_change_type(self, change_type: str) -> List[Dict[str, Any]]:
        """Find patterns by change type.
        
        Args:
            change_type: Type of change (e.g., "concept_shift", "semantic_shift")
            
        Returns:
            List of patterns with the specified change type
        """
        return self.executor.find_patterns_by_change_type(change_type)
    
    def _determine_pattern_type(self, context: Dict[str, Any]) -> str:
        """Determine the pattern type based on context.
        
        Args:
            context: Context of the pattern
            
        Returns:
            Pattern type (e.g., "harmonic", "sequential", "complementary")
        """
        # Extract pattern type from context if available
        if "pattern_type" in context:
            return context["pattern_type"]
        
        # Try to infer pattern type from other context information
        change_type = context.get("change_type", "")
        
        if change_type == "semantic_shift":
            return "harmonic"
        elif change_type == "concept_shift":
            return "sequential"
        elif change_type == "context_shift":
            return "complementary"
        else:
            return "unknown"
    
    def _calculate_response_strength(self, pattern_type: str, tonic_value: float, 
                                   change_type: str) -> float:
        """Calculate the response strength based on pattern type and tonic value.
        
        Args:
            pattern_type: Type of the pattern
            tonic_value: Tonic value
            change_type: Type of change
            
        Returns:
            Response strength
        """
        # Different pattern types respond differently to tonic values
        if pattern_type == "harmonic" and change_type == "semantic_shift":
            return tonic_value * 1.2  # Harmonic patterns respond strongly to semantic shifts
        elif pattern_type == "sequential" and change_type == "concept_shift":
            return tonic_value * 1.3  # Sequential patterns respond strongly to concept shifts
        elif pattern_type == "complementary" and change_type == "context_shift":
            return tonic_value * 1.1  # Complementary patterns respond strongly to context shifts
        else:
            return tonic_value  # Default response
