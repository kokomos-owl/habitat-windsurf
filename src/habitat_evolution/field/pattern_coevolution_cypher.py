"""Cypher query library for pattern co-evolution tracking.

This module provides a collection of Cypher queries specifically designed for tracking
pattern co-evolution in the Neo4j graph database. These queries are designed to work with
the Vector + Tonic-Harmonic pattern schema and extend the functionality in cypher_query_library.py.
"""

from typing import Dict, List, Any, Optional
import json
from neo4j import GraphDatabase


class PatternCoevolutionQueries:
    """Library of Cypher queries for pattern co-evolution tracking.
    
    This class provides a collection of Cypher queries for tracking pattern co-evolution
    in the Neo4j graph database. It includes queries for creating patterns with tonic-harmonic
    properties, tracking pattern evolution and co-evolution, and analyzing pattern responses
    to tonic values.
    """
    
    # Pattern creation with tonic-harmonic properties
    CREATE_PATTERN_WITH_TONIC_HARMONIC = """
        MERGE (p:Pattern {id: $pattern_id, type: $pattern_type})
        SET p.tonic_value = $tonic_value,
            p.stability = $stability,
            p.harmonic_value = $tonic_value * $stability,
            p.timestamp = $timestamp
        RETURN p
    """
    
    # Pattern evolution tracking
    CREATE_PATTERN_EVOLUTION = """
        MATCH (p1:Pattern {id: $old_pattern_id})
        MATCH (p2:Pattern {id: $new_pattern_id})
        MERGE (p1)-[r:EVOLVES_TO]->(p2)
        SET r.change_type = $change_type,
            r.window_id = $window_id,
            r.timestamp = $timestamp,
            r.tonic_value = $tonic_value,
            r.stability = $stability,
            r.harmonic_value = $tonic_value * $stability,
            r.origin = $origin
        RETURN r
    """
    
    # Pattern co-evolution tracking
    CREATE_PATTERN_COEVOLUTION = """
        MATCH (p1:Pattern {id: $pattern_id1})
        MATCH (p2:Pattern {id: $pattern_id2})
        WHERE p1 <> p2
        MERGE (p1)-[r:CO_EVOLVES_WITH]->(p2)
        SET r.window_id = $window_id,
            r.strength = $harmonic_value,
            r.timestamp = $timestamp,
            r.change_type = $change_type
        RETURN r
    """
    
    # Find patterns that co-evolved in a specific window
    FIND_PATTERN_COEVOLUTION = """
        MATCH (p1:Pattern)-[r:CO_EVOLVES_WITH]->(p2:Pattern)
        WHERE r.window_id = $window_id
        RETURN p1.id AS pattern1, p2.id AS pattern2, 
               r.strength AS strength, r.change_type AS change_type
        ORDER BY r.strength DESC
    """
    
    # Find patterns by change type
    FIND_PATTERNS_BY_CHANGE_TYPE = """
        MATCH (p1:Pattern)-[r:EVOLVES_TO]->(p2:Pattern)
        WHERE r.change_type = $change_type
        RETURN p1.id AS old_pattern, p2.id AS new_pattern, 
               r.tonic_value AS tonic_value, r.stability AS stability,
               r.harmonic_value AS harmonic_value, r.origin AS origin
        ORDER BY r.timestamp DESC
    """
    
    # Record pattern response to tonic value
    RECORD_PATTERN_TONIC_RESPONSE = """
        MATCH (p:Pattern {id: $pattern_id})
        MERGE (p)-[r:RESPONDS_TO {change_type: $change_type}]->(t:TonicValue {value: $tonic_value})
        SET r.response_strength = $response_strength,
            r.timestamp = $timestamp
        RETURN r
    """
    
    # Find patterns with similar tonic responses
    FIND_PATTERNS_WITH_SIMILAR_TONIC_RESPONSE = """
        MATCH (p1:Pattern)-[r1:RESPONDS_TO]->(t:TonicValue)
        MATCH (p2:Pattern)-[r2:RESPONDS_TO]->(t)
        WHERE p1 <> p2 AND r1.change_type = r2.change_type
        WITH p1, p2, t, r1, r2, abs(r1.response_strength - r2.response_strength) AS response_diff
        WHERE response_diff < $max_diff
        RETURN p1.id AS pattern1, p2.id AS pattern2, 
               t.value AS tonic_value, r1.change_type AS change_type,
               r1.response_strength AS response1, r2.response_strength AS response2,
               response_diff
        ORDER BY response_diff ASC
    """
    
    # Find patterns with differential tonic responses
    FIND_PATTERNS_WITH_DIFFERENTIAL_RESPONSE = """
        MATCH (p:Pattern)
        MATCH (p)-[r1:RESPONDS_TO]->(t1:TonicValue)
        MATCH (p)-[r2:RESPONDS_TO]->(t2:TonicValue)
        WHERE t1.value <> t2.value AND r1.change_type = r2.change_type
        WITH p, r1.change_type AS change_type, 
             abs(r1.response_strength - r2.response_strength) AS response_diff
        WHERE response_diff > $min_diff
        RETURN p.id AS pattern, p.type AS pattern_type, 
               change_type, response_diff AS differential_response
        ORDER BY response_diff DESC
    """


class PatternCoevolutionExecutor:
    """Executor for pattern co-evolution Cypher queries.
    
    This class provides methods for executing pattern co-evolution Cypher queries
    against a Neo4j database.
    """
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """Initialize the pattern co-evolution query executor.
        
        Args:
            uri: URI for the Neo4j database connection
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name (default: "neo4j")
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query against the Neo4j database.
        
        Args:
            query: Cypher query to execute
            params: Parameters for the query (default: None)
            
        Returns:
            List of records returned by the query
        """
        if params is None:
            params = {}
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, params)
            return [record.data() for record in result]
    
    def create_pattern_with_tonic_harmonic(self, pattern_id: str, pattern_type: str,
                                          tonic_value: float, stability: float,
                                          timestamp: str) -> Dict[str, Any]:
        """Create a pattern with tonic-harmonic properties.
        
        Args:
            pattern_id: ID of the pattern to create
            pattern_type: Type of the pattern (e.g., "harmonic", "sequential", "complementary")
            tonic_value: Tonic value of the pattern
            stability: Stability value of the pattern
            timestamp: Timestamp for the pattern creation
            
        Returns:
            Created pattern
        """
        params = {
            "pattern_id": pattern_id,
            "pattern_type": pattern_type,
            "tonic_value": tonic_value,
            "stability": stability,
            "timestamp": timestamp
        }
        result = self.execute_query(PatternCoevolutionQueries.CREATE_PATTERN_WITH_TONIC_HARMONIC, params)
        return result[0] if result else None
    
    def create_pattern_evolution(self, old_pattern_id: str, new_pattern_id: str,
                               change_type: str, window_id: str, timestamp: str,
                               tonic_value: float, stability: float,
                               origin: str) -> Dict[str, Any]:
        """Create a pattern evolution relationship.
        
        Args:
            old_pattern_id: ID of the old pattern
            new_pattern_id: ID of the new pattern
            change_type: Type of change (e.g., "concept_shift", "semantic_shift")
            window_id: ID of the learning window
            timestamp: Timestamp for the evolution
            tonic_value: Tonic value at the time of evolution
            stability: Stability value at the time of evolution
            origin: Origin of the evolution (e.g., "window1")
            
        Returns:
            Created evolution relationship
        """
        params = {
            "old_pattern_id": old_pattern_id,
            "new_pattern_id": new_pattern_id,
            "change_type": change_type,
            "window_id": window_id,
            "timestamp": timestamp,
            "tonic_value": tonic_value,
            "stability": stability,
            "harmonic_value": tonic_value * stability,
            "origin": origin
        }
        result = self.execute_query(PatternCoevolutionQueries.CREATE_PATTERN_EVOLUTION, params)
        return result[0] if result else None
    
    def create_pattern_coevolution(self, pattern_id1: str, pattern_id2: str,
                                 window_id: str, harmonic_value: float,
                                 timestamp: str, change_type: str) -> Dict[str, Any]:
        """Create a pattern co-evolution relationship.
        
        Args:
            pattern_id1: ID of the first pattern
            pattern_id2: ID of the second pattern
            window_id: ID of the learning window
            harmonic_value: Harmonic value of the co-evolution
            timestamp: Timestamp for the co-evolution
            change_type: Type of change (e.g., "concept_shift", "semantic_shift")
            
        Returns:
            Created co-evolution relationship
        """
        params = {
            "pattern_id1": pattern_id1,
            "pattern_id2": pattern_id2,
            "window_id": window_id,
            "harmonic_value": harmonic_value,
            "timestamp": timestamp,
            "change_type": change_type
        }
        result = self.execute_query(PatternCoevolutionQueries.CREATE_PATTERN_COEVOLUTION, params)
        return result[0] if result else None
    
    def find_pattern_coevolution(self, window_id: str) -> List[Dict[str, Any]]:
        """Find patterns that co-evolved in a specific window.
        
        Args:
            window_id: ID of the learning window
            
        Returns:
            List of co-evolved pattern pairs
        """
        params = {
            "window_id": window_id
        }
        return self.execute_query(PatternCoevolutionQueries.FIND_PATTERN_COEVOLUTION, params)
    
    def find_patterns_by_change_type(self, change_type: str) -> List[Dict[str, Any]]:
        """Find patterns by change type.
        
        Args:
            change_type: Type of change (e.g., "concept_shift", "semantic_shift")
            
        Returns:
            List of patterns with the specified change type
        """
        params = {
            "change_type": change_type
        }
        return self.execute_query(PatternCoevolutionQueries.FIND_PATTERNS_BY_CHANGE_TYPE, params)
    
    def record_pattern_tonic_response(self, pattern_id: str, change_type: str,
                                    tonic_value: float, response_strength: float,
                                    timestamp: str) -> Dict[str, Any]:
        """Record a pattern's response to a tonic value.
        
        Args:
            pattern_id: ID of the pattern
            change_type: Type of change (e.g., "concept_shift", "semantic_shift")
            tonic_value: Tonic value
            response_strength: Strength of the pattern's response
            timestamp: Timestamp for the response
            
        Returns:
            Created response relationship
        """
        params = {
            "pattern_id": pattern_id,
            "change_type": change_type,
            "tonic_value": tonic_value,
            "response_strength": response_strength,
            "timestamp": timestamp
        }
        result = self.execute_query(PatternCoevolutionQueries.RECORD_PATTERN_TONIC_RESPONSE, params)
        return result[0] if result else None
    
    def find_patterns_with_similar_tonic_response(self, max_diff: float = 0.1) -> List[Dict[str, Any]]:
        """Find patterns with similar tonic responses.
        
        Args:
            max_diff: Maximum difference in response strength (default: 0.1)
            
        Returns:
            List of pattern pairs with similar tonic responses
        """
        params = {
            "max_diff": max_diff
        }
        return self.execute_query(PatternCoevolutionQueries.FIND_PATTERNS_WITH_SIMILAR_TONIC_RESPONSE, params)
    
    def find_patterns_with_differential_response(self, min_diff: float = 0.3) -> List[Dict[str, Any]]:
        """Find patterns with differential tonic responses.
        
        Args:
            min_diff: Minimum difference in response strength (default: 0.3)
            
        Returns:
            List of patterns with differential tonic responses
        """
        params = {
            "min_diff": min_diff
        }
        return self.execute_query(PatternCoevolutionQueries.FIND_PATTERNS_WITH_DIFFERENTIAL_RESPONSE, params)
