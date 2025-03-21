"""
Neo4j query prototypes for semantic analysis of patterns.

This module provides Cypher query templates for tracking pattern evolution,
co-evolution, and semantic relationships in Neo4j, leveraging the enhanced
semantic content of patterns.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Neo4jSemanticQueries:
    """
    Provides Cypher query templates for semantic analysis of patterns in Neo4j.
    
    This class implements query templates for tracking pattern evolution,
    co-evolution, and semantic relationships, leveraging the enhanced semantic
    content of patterns.
    """
    
    @staticmethod
    def get_patterns_by_semantic_content(content_fragment: str) -> str:
        """
        Get patterns containing specific semantic content.
        
        Args:
            content_fragment: Fragment of semantic content to search for
            
        Returns:
            Cypher query string
        """
        return f"""
        MATCH (p:Pattern)
        WHERE p.semantic_content CONTAINS '{content_fragment}'
        RETURN p
        ORDER BY p.tonic_value DESC
        LIMIT 10
        """
    
    @staticmethod
    def get_patterns_by_keywords(keywords: List[str], match_all: bool = False) -> str:
        """
        Get patterns matching specific keywords.
        
        Args:
            keywords: List of keywords to search for
            match_all: If True, all keywords must match; otherwise any match is sufficient
            
        Returns:
            Cypher query string
        """
        keyword_conditions = []
        for keyword in keywords:
            keyword_conditions.append(f"p.keywords CONTAINS '{keyword}'")
            
        if match_all:
            keyword_clause = " AND ".join(keyword_conditions)
        else:
            keyword_clause = " OR ".join(keyword_conditions)
            
        return f"""
        MATCH (p:Pattern)
        WHERE {keyword_clause}
        RETURN p
        ORDER BY p.tonic_value DESC
        LIMIT 20
        """
    
    @staticmethod
    def track_pattern_evolution(pattern_id: str, time_window: Optional[int] = None) -> str:
        """
        Track the evolution of a pattern over time.
        
        Args:
            pattern_id: ID of the pattern to track
            time_window: Optional time window in days (default: all time)
            
        Returns:
            Cypher query string
        """
        time_clause = ""
        if time_window:
            # Calculate timestamp for time_window days ago
            time_clause = f"AND r.timestamp > {datetime.now().timestamp() - (time_window * 86400)}"
            
        return f"""
        MATCH (p:Pattern {{id: '{pattern_id}'}})
        MATCH (p)-[r:EVOLVES_TO]->(p2:Pattern)
        WHERE r.evolution_type IS NOT NULL {time_clause}
        RETURN p, r, p2
        ORDER BY r.timestamp
        """
    
    @staticmethod
    def track_pattern_co_evolution(pattern_id: str, time_window: Optional[int] = None) -> str:
        """
        Track the co-evolution of patterns related to a specific pattern.
        
        Args:
            pattern_id: ID of the pattern to track co-evolution with
            time_window: Optional time window in days (default: all time)
            
        Returns:
            Cypher query string
        """
        time_clause = ""
        if time_window:
            # Calculate timestamp for time_window days ago
            time_clause = f"AND r1.timestamp > {datetime.now().timestamp() - (time_window * 86400)}"
            
        return f"""
        MATCH (p:Pattern {{id: '{pattern_id}'}})
        MATCH (p)-[r1:RESONATES_WITH]->(p2:Pattern)
        MATCH (p2)-[r2:EVOLVES_TO]->(p3:Pattern)
        WHERE r1.resonance_type IS NOT NULL {time_clause}
        RETURN p, r1, p2, r2, p3
        ORDER BY r1.timestamp, r2.timestamp
        """
    
    @staticmethod
    def get_semantic_neighborhood(pattern_id: str, max_distance: int = 2) -> str:
        """
        Get the semantic neighborhood of a pattern.
        
        Args:
            pattern_id: ID of the pattern to get neighborhood for
            max_distance: Maximum relationship distance to traverse
            
        Returns:
            Cypher query string
        """
        return f"""
        MATCH (p:Pattern {{id: '{pattern_id}'}})-[r*1..{max_distance}]-(p2:Pattern)
        RETURN p, r, p2
        """
    
    @staticmethod
    def get_semantic_clusters() -> str:
        """
        Get semantic clusters based on pattern relationships.
        
        Returns:
            Cypher query string
        """
        return """
        MATCH (p:Pattern)
        WHERE p.tonic_value > 0.7
        WITH p
        MATCH (p)-[r:RESONATES_WITH]-(p2:Pattern)
        WHERE r.resonance_type = 'CONSTRUCTIVE'
        WITH p, collect(p2) as resonating_patterns
        WHERE size(resonating_patterns) > 2
        RETURN p, resonating_patterns
        ORDER BY p.tonic_value DESC
        LIMIT 10
        """
    
    @staticmethod
    def get_patterns_by_tonic_harmonic_values(
        min_tonic: float = 0.7,
        min_harmonic: float = 0.5
    ) -> str:
        """
        Get patterns with specific tonic and harmonic values.
        
        Args:
            min_tonic: Minimum tonic value
            min_harmonic: Minimum harmonic value
            
        Returns:
            Cypher query string
        """
        return f"""
        MATCH (p:Pattern)
        WHERE p.tonic_value >= {min_tonic} AND p.harmonic_value >= {min_harmonic}
        RETURN p
        ORDER BY p.harmonic_value DESC, p.tonic_value DESC
        LIMIT 20
        """
    
    @staticmethod
    def get_semantic_bridges() -> str:
        """
        Get patterns that act as semantic bridges between frequency domains.
        
        Returns:
            Cypher query string
        """
        return """
        MATCH (d1:FrequencyDomain)-[:CONTAINS]->(p:Pattern)-[:CONTAINED_IN]->(d2:FrequencyDomain)
        WHERE d1.id <> d2.id
        RETURN p, d1, d2
        ORDER BY p.tonic_value DESC
        LIMIT 10
        """
    
    @staticmethod
    def get_semantic_boundaries() -> str:
        """
        Get patterns that define semantic boundaries.
        
        Returns:
            Cypher query string
        """
        return """
        MATCH (b:Boundary)-[:DEFINED_BY]->(p:Pattern)
        RETURN b, p
        ORDER BY p.tonic_value DESC
        LIMIT 10
        """
    
    @staticmethod
    def get_resonance_groups_with_semantic_content() -> str:
        """
        Get resonance groups with their semantic content.
        
        Returns:
            Cypher query string
        """
        return """
        MATCH (rg:ResonanceGroup)-[:CONTAINS]->(p:Pattern)
        WITH rg, collect(p) as patterns
        RETURN rg, patterns,
               apoc.text.join([p in patterns | p.semantic_content], ' | ') as semantic_content,
               avg(p in patterns | p.tonic_value) as avg_tonic,
               avg(p in patterns | p.harmonic_value) as avg_harmonic
        ORDER BY avg_harmonic DESC
        LIMIT 10
        """
    
    @staticmethod
    def get_semantic_evolution_path(
        start_content: str,
        end_content: str,
        max_path_length: int = 5
    ) -> str:
        """
        Get the semantic evolution path between two content fragments.
        
        Args:
            start_content: Starting semantic content fragment
            end_content: Ending semantic content fragment
            max_path_length: Maximum path length to search
            
        Returns:
            Cypher query string
        """
        return f"""
        MATCH (p1:Pattern)
        WHERE p1.semantic_content CONTAINS '{start_content}'
        MATCH (p2:Pattern)
        WHERE p2.semantic_content CONTAINS '{end_content}'
        MATCH path = shortestPath((p1)-[r:EVOLVES_TO|RESONATES_WITH*1..{max_path_length}]-(p2))
        RETURN path
        LIMIT 5
        """
    
    @staticmethod
    def get_semantic_content_distribution() -> str:
        """
        Get the distribution of semantic content across patterns.
        
        Returns:
            Cypher query string
        """
        return """
        MATCH (p:Pattern)
        WHERE p.semantic_content IS NOT NULL
        WITH p.semantic_content as content, count(*) as pattern_count
        RETURN content, pattern_count
        ORDER BY pattern_count DESC
        LIMIT 20
        """
    
    @staticmethod
    def get_keyword_co_occurrence() -> str:
        """
        Get co-occurrence patterns of keywords.
        
        Returns:
            Cypher query string
        """
        return """
        MATCH (p:Pattern)
        WHERE p.keywords IS NOT NULL
        UNWIND apoc.convert.fromJsonList(p.keywords) as keyword
        WITH keyword, count(*) as keyword_count
        MATCH (p:Pattern)
        WHERE p.keywords IS NOT NULL AND p.keywords CONTAINS keyword
        WITH keyword, keyword_count, p
        UNWIND apoc.convert.fromJsonList(p.keywords) as co_keyword
        WHERE keyword <> co_keyword
        WITH keyword, keyword_count, co_keyword, count(*) as co_occurrence
        RETURN keyword, keyword_count, co_keyword, co_occurrence
        ORDER BY keyword_count DESC, co_occurrence DESC
        LIMIT 50
        """
    
    @staticmethod
    def persist_pattern_with_semantic_content(pattern_data: Dict[str, Any]) -> str:
        """
        Generate a query to persist a pattern with semantic content.
        
        Args:
            pattern_data: Pattern data including semantic content
            
        Returns:
            Cypher query string
        """
        # Convert pattern_data to Cypher parameters format
        params = ", ".join([f"{k}: ${k}" for k in pattern_data.keys()])
        
        return f"""
        MERGE (p:Pattern {{id: $id}})
        SET p = {{{params}}}
        RETURN p
        """
    
    @staticmethod
    def create_semantic_relationship(
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Dict[str, Any]
    ) -> str:
        """
        Generate a query to create a semantic relationship between patterns.
        
        Args:
            source_id: Source pattern ID
            target_id: Target pattern ID
            relationship_type: Type of relationship
            properties: Relationship properties
            
        Returns:
            Cypher query string
        """
        # Convert properties to Cypher parameters format
        props = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        
        return f"""
        MATCH (p1:Pattern {{id: '{source_id}'}})
        MATCH (p2:Pattern {{id: '{target_id}'}})
        MERGE (p1)-[r:{relationship_type} {{{props}}}]->(p2)
        RETURN p1, r, p2
        """
    
    @staticmethod
    def get_bidirectional_feedback_paths() -> str:
        """
        Get bidirectional feedback paths between patterns.
        
        Returns:
            Cypher query string
        """
        return """
        MATCH (p1:Pattern)-[r1:INFLUENCES]->(p2:Pattern)-[r2:INFLUENCES]->(p1)
        RETURN p1, r1, p2, r2
        ORDER BY p1.tonic_value + p2.tonic_value DESC
        LIMIT 10
        """
    
    @staticmethod
    def get_health_monitoring_metrics() -> str:
        """
        Get health monitoring metrics for the semantic topology.
        
        Returns:
            Cypher query string
        """
        return """
        MATCH (p:Pattern)
        WHERE p.tonic_value IS NOT NULL AND p.harmonic_value IS NOT NULL
        WITH 
            count(p) as total_patterns,
            avg(p.tonic_value) as avg_tonic,
            avg(p.harmonic_value) as avg_harmonic,
            avg(p.stability) as avg_stability,
            count(p) - count(p.semantic_content) as patterns_without_semantic_content
        MATCH (rg:ResonanceGroup)
        WITH 
            total_patterns, avg_tonic, avg_harmonic, avg_stability, 
            patterns_without_semantic_content, count(rg) as resonance_group_count
        MATCH (:Pattern)-[r:RESONATES_WITH]->(:Pattern)
        RETURN 
            total_patterns, avg_tonic, avg_harmonic, avg_stability,
            patterns_without_semantic_content, resonance_group_count,
            count(r) as resonance_relationship_count,
            patterns_without_semantic_content * 1.0 / total_patterns as semantic_gap_ratio
        """
