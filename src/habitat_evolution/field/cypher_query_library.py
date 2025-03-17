"""Cypher query library for pattern exploration.

This module provides a collection of Cypher queries for exploring patterns and their
relationships in the Neo4j graph database. These queries are designed to work with
the Vector + Tonic-Harmonic pattern schema defined in neo4j_pattern_schema.py.
"""

from typing import Dict, List, Any, Optional
import json
from neo4j import GraphDatabase


class CypherQueryLibrary:
    """Library of Cypher queries for pattern exploration.
    
    This class provides a collection of Cypher queries for exploring patterns and their
    relationships in the Neo4j graph database. It includes queries for finding patterns,
    dimensions, communities, and their relationships.
    """
    
    # Basic pattern queries
    FIND_PATTERN_BY_ID = """
        MATCH (p:Pattern {id: $pattern_id})
        RETURN p
    """
    
    FIND_PATTERNS_BY_TYPE = """
        MATCH (p:Pattern)
        WHERE p.type = $pattern_type
        RETURN p
        ORDER BY p.id
    """
    
    FIND_PATTERNS_BY_PROPERTY = """
        MATCH (p:Pattern)
        WHERE p[$property_name] = $property_value
        RETURN p
        ORDER BY p.id
    """
    
    # Dimension queries
    FIND_DIMENSION_BY_NUMBER = """
        MATCH (d:Dimension {dimension_number: $dimension_number})
        RETURN d
    """
    
    FIND_TOP_DIMENSIONS_BY_STRENGTH = """
        MATCH (d:Dimension)
        MATCH (r:ResonancePattern)-[:BASED_ON]->(d)
        WITH d, count(r) AS pattern_count, sum(r.strength) AS total_strength
        RETURN d.id AS dimension_id, d.dimension_number AS dimension_number,
               pattern_count, total_strength
        ORDER BY total_strength DESC
        LIMIT $limit
    """
    
    # Pattern projection queries
    FIND_PATTERN_PROJECTIONS = """
        MATCH (p:Pattern {id: $pattern_id})-[r:PROJECTS_TO]->(d:Dimension)
        RETURN d.dimension_number AS dimension, r.value AS projection
        ORDER BY abs(r.value) DESC
    """
    
    FIND_PATTERNS_BY_PROJECTION = """
        MATCH (p:Pattern)-[r:PROJECTS_TO]->(d:Dimension {dimension_number: $dimension_number})
        WHERE r.value >= $min_value
        RETURN p.id AS pattern_id, p.type AS pattern_type, r.value AS projection
        ORDER BY r.value DESC
        LIMIT $limit
    """
    
    # Community queries
    FIND_COMMUNITY_BY_ID = """
        MATCH (c:Community {id: $community_id})
        RETURN c
    """
    
    FIND_COMMUNITY_MEMBERS = """
        MATCH (p:Pattern)-[:BELONGS_TO]->(c:Community {id: $community_id})
        RETURN p.id AS pattern_id, p.type AS pattern_type
        ORDER BY p.id
    """
    
    FIND_PATTERN_COMMUNITIES = """
        MATCH (p:Pattern {id: $pattern_id})-[:BELONGS_TO]->(c:Community)
        RETURN c.id AS community_id, c.size AS community_size
    """
    
    # Resonance pattern queries
    FIND_RESONANCE_PATTERN_BY_ID = """
        MATCH (r:ResonancePattern {id: $resonance_id})
        RETURN r
    """
    
    FIND_RESONANCE_PATTERNS_BY_TYPE = """
        MATCH (r:ResonancePattern {pattern_type: $pattern_type})
        RETURN r.id AS resonance_id, r.pattern_type AS pattern_type,
               r.primary_dimension AS primary_dimension, r.strength AS strength,
               r.metadata AS metadata
        ORDER BY r.strength DESC
    """
    
    FIND_RESONANCE_PATTERN_MEMBERS = """
        MATCH (p:Pattern)-[:PARTICIPATES_IN]->(r:ResonancePattern {id: $resonance_id})
        RETURN p.id AS pattern_id, p.type AS pattern_type
        ORDER BY p.id
    """
    
    FIND_PATTERN_RESONANCE_PATTERNS = """
        MATCH (p:Pattern {id: $pattern_id})-[:PARTICIPATES_IN]->(r:ResonancePattern)
        RETURN r.id AS resonance_id, r.pattern_type AS pattern_type,
               r.primary_dimension AS primary_dimension, r.strength AS strength
        ORDER BY r.strength DESC
    """
    
    # Boundary pattern queries
    FIND_BOUNDARY_PATTERNS = """
        MATCH (p:BoundaryPattern)
        RETURN p.id AS pattern_id, p.type AS pattern_type,
               p.boundary_fuzziness AS fuzziness
        ORDER BY p.boundary_fuzziness DESC
        LIMIT $limit
    """
    
    FIND_COMMUNITY_BOUNDARIES = """
        MATCH (c1:Community {id: $community_id})<-[:BELONGS_TO]-(p:BoundaryPattern)-[:BELONGS_TO]->(c2:Community)
        WHERE c1 <> c2
        RETURN p.id AS pattern_id, p.type AS pattern_type,
               p.boundary_fuzziness AS fuzziness,
               collect(distinct c2.id) AS connected_communities
        ORDER BY p.boundary_fuzziness DESC
    """
    
    # Pattern relationship queries
    FIND_PATTERN_NEIGHBORS = """
        MATCH (p1:Pattern {id: $pattern_id})-[r:PROJECTS_TO]->(d:Dimension)<-[:PROJECTS_TO]-(p2:Pattern)
        WHERE p1 <> p2
        WITH p2, d, r
        ORDER BY abs(r.value) DESC
        WITH p2, collect(d.dimension_number)[0..3] AS top_dimensions
        RETURN p2.id AS pattern_id, p2.type AS pattern_type,
               top_dimensions
        LIMIT $limit
    """
    
    FIND_DIMENSIONAL_RESONANCE = """
        MATCH (p1:Pattern {id: $pattern_id})-[r1:PROJECTS_TO]->(d:Dimension)<-[r2:PROJECTS_TO]-(p2:Pattern)
        WHERE p1 <> p2 AND sign(r1.value) = sign(r2.value) AND abs(r1.value) >= 0.3 AND abs(r2.value) >= 0.3
        WITH p2, d, abs(r1.value) * abs(r2.value) AS resonance_strength
        ORDER BY resonance_strength DESC
        WITH p2, collect({dimension: d.dimension_number, strength: resonance_strength})[0..3] AS resonances
        RETURN p2.id AS pattern_id, p2.type AS pattern_type,
               resonances,
               reduce(s = 0, r IN resonances | s + r.strength) AS total_resonance
        ORDER BY total_resonance DESC
        LIMIT $limit
    """
    
    FIND_COMPLEMENTARY_PATTERNS = """
        MATCH (p1:Pattern {id: $pattern_id})-[r1:PROJECTS_TO]->(d:Dimension)<-[r2:PROJECTS_TO]-(p2:Pattern)
        WHERE p1 <> p2 AND sign(r1.value) <> sign(r2.value) AND abs(r1.value) >= 0.3 AND abs(r2.value) >= 0.3
        WITH p2, d, abs(r1.value) * abs(r2.value) AS complementary_strength
        ORDER BY complementary_strength DESC
        WITH p2, collect({dimension: d.dimension_number, strength: complementary_strength})[0..3] AS complementaries
        RETURN p2.id AS pattern_id, p2.type AS pattern_type,
               complementaries,
               reduce(s = 0, c IN complementaries | s + c.strength) AS total_complementary
        ORDER BY total_complementary DESC
        LIMIT $limit
    """
    
    # Path finding queries
    FIND_SHORTEST_PATH = """
        MATCH path = shortestPath((start:Pattern {id: $start_id})-[:PARTICIPATES_IN|BELONGS_TO*..%d]-(end:Pattern {id: $end_id}))
        RETURN [node in nodes(path) | node.id] AS path_nodes,
               [rel in relationships(path) | type(rel)] AS path_relationships,
               length(path) AS path_length
    """
    
    FIND_ALL_PATHS = """
        MATCH path = allShortestPaths((start:Pattern {id: $start_id})-[:PARTICIPATES_IN|BELONGS_TO*..%d]-(end:Pattern {id: $end_id}))
        RETURN [node in nodes(path) | node.id] AS path_nodes,
               [rel in relationships(path) | type(rel)] AS path_relationships,
               length(path) AS path_length
    """
    
    FIND_EIGENSPACE_PATH = """
        MATCH (start:Pattern {id: $start_id}), (end:Pattern {id: $end_id})
        MATCH (start)-[:PROJECTS_TO]->(d:Dimension)<-[:PROJECTS_TO]-(end)
        WITH start, end, collect(d) AS shared_dimensions
        
        // Find intermediate patterns that share projections with both start and end
        MATCH (p:Pattern)-[:PROJECTS_TO]->(d:Dimension)
        WHERE d IN shared_dimensions AND p <> start AND p <> end
        WITH start, end, p, count(d) AS shared_count
        WHERE shared_count >= size(shared_dimensions) / 2
        
        // Calculate eigenspace distances
        MATCH (start)-[r1:PROJECTS_TO]->(d1:Dimension)<-[r2:PROJECTS_TO]-(p)
        WITH start, end, p, sum((r1.value - r2.value)^2) AS start_distance
        
        MATCH (p)-[r1:PROJECTS_TO]->(d2:Dimension)<-[r2:PROJECTS_TO]-(end)
        WITH start, end, p, start_distance, sum((r1.value - r2.value)^2) AS end_distance
        
        // Return path through intermediate patterns
        RETURN start.id AS start_id,
               p.id AS intermediate_id,
               end.id AS end_id,
               sqrt(start_distance) AS start_to_intermediate_distance,
               sqrt(end_distance) AS intermediate_to_end_distance,
               sqrt(start_distance) + sqrt(end_distance) AS total_path_length
        ORDER BY total_path_length ASC
        LIMIT 5
    """
    
    # Advanced exploration queries
    FIND_PATTERN_CLUSTERS = """
        MATCH (p:Pattern)
        OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Community)
        WITH p, collect(c.id) AS communities
        
        MATCH (p)-[:PARTICIPATES_IN]->(r:ResonancePattern)
        WITH p, communities, collect(r.id) AS resonance_patterns
        
        RETURN p.id AS pattern_id, p.type AS pattern_type,
               communities, resonance_patterns,
               size(communities) AS community_count,
               size(resonance_patterns) AS resonance_count
        ORDER BY resonance_count DESC, community_count DESC
        LIMIT $limit
    """
    
    FIND_DIMENSIONAL_STRUCTURE = """
        MATCH (d:Dimension)
        OPTIONAL MATCH (r:ResonancePattern)-[:BASED_ON]->(d)
        WITH d, count(r) AS resonance_count
        
        MATCH (p:Pattern)-[proj:PROJECTS_TO]->(d)
        WITH d, resonance_count, count(p) AS pattern_count,
             sum(case when proj.value > 0 then 1 else 0 end) AS positive_count,
             sum(case when proj.value < 0 then 1 else 0 end) AS negative_count
        
        RETURN d.id AS dimension_id, d.dimension_number AS dimension_number,
               pattern_count, resonance_count,
               positive_count, negative_count,
               1.0 * abs(positive_count - negative_count) / 
               (positive_count + negative_count) AS polarity
        ORDER BY pattern_count DESC
    """
    
    FIND_COMMUNITY_STRUCTURE = """
        MATCH (c:Community)
        OPTIONAL MATCH (c)<-[:BELONGS_TO]-(p:Pattern)
        WITH c, count(p) AS pattern_count
        
        OPTIONAL MATCH (c)<-[:BELONGS_TO]-(b:BoundaryPattern)
        WITH c, pattern_count, count(b) AS boundary_count
        
        OPTIONAL MATCH (c)<-[:BELONGS_TO]-(p1:Pattern)-[:PARTICIPATES_IN]->(r:ResonancePattern)<-[:PARTICIPATES_IN]-(p2:Pattern)-[:BELONGS_TO]->(c)
        WHERE p1 <> p2
        WITH c, pattern_count, boundary_count, count(distinct r) AS internal_resonance_count
        
        RETURN c.id AS community_id, pattern_count, boundary_count,
               internal_resonance_count,
               1.0 * internal_resonance_count / pattern_count AS cohesion
        ORDER BY pattern_count DESC
    """
    
    FIND_TRANSITION_ZONES = """
        MATCH (c1:Community)<-[:BELONGS_TO]-(p:BoundaryPattern)-[:BELONGS_TO]->(c2:Community)
        WHERE c1.id < c2.id
        WITH c1, c2, collect(p) AS boundary_patterns
        
        RETURN c1.id AS community1, c2.id AS community2,
               [p in boundary_patterns | p.id] AS boundary_pattern_ids,
               size(boundary_patterns) AS boundary_size,
               avg([p in boundary_patterns | p.boundary_fuzziness]) AS avg_fuzziness
        ORDER BY boundary_size DESC
    """
    
    FIND_PATTERN_EVOLUTION = """
        // This query assumes patterns have timestamp or sequence properties
        // to track their evolution over time
        MATCH (p:Pattern)
        WHERE p.timestamp IS NOT NULL OR p.sequence IS NOT NULL
        WITH p
        ORDER BY coalesce(p.timestamp, p.sequence)
        
        WITH collect(p) AS patterns
        UNWIND range(0, size(patterns)-2) AS i
        WITH patterns[i] AS p1, patterns[i+1] AS p2
        
        // Calculate dimensional changes
        MATCH (p1)-[r1:PROJECTS_TO]->(d:Dimension)<-[r2:PROJECTS_TO]-(p2)
        WITH p1, p2, d, r1.value AS v1, r2.value AS v2
        
        RETURN p1.id AS from_pattern, p2.id AS to_pattern,
               collect({dimension: d.dimension_number, 
                       from_value: v1, 
                       to_value: v2, 
                       change: v2 - v1}) AS dimension_changes,
               sqrt(sum((v2 - v1)^2)) AS evolution_distance
        ORDER BY evolution_distance DESC
    """


class CypherQueryExecutor:
    """Executor for Cypher queries from the query library.
    
    This class provides methods for executing Cypher queries from the query library
    against a Neo4j database.
    """
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """Initialize the Cypher query executor.
        
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
            result = session.run(query, **params)
            return [record.data() for record in result]
    
    def find_patterns_by_type(self, pattern_type: str) -> List[Dict[str, Any]]:
        """Find patterns by type.
        
        Args:
            pattern_type: Type of patterns to find
            
        Returns:
            List of patterns with the specified type
        """
        return self.execute_query(CypherQueryLibrary.FIND_PATTERNS_BY_TYPE, 
                                 {"pattern_type": pattern_type})
    
    def find_pattern_projections(self, pattern_id: str) -> List[Dict[str, Any]]:
        """Find projections of a pattern onto dimensions.
        
        Args:
            pattern_id: ID of the pattern to find projections for
            
        Returns:
            List of pattern projections onto dimensions
        """
        return self.execute_query(CypherQueryLibrary.FIND_PATTERN_PROJECTIONS,
                                 {"pattern_id": pattern_id})
    
    def find_patterns_by_projection(self, dimension_number: int, min_value: float = 0.5, 
                                   limit: int = 10) -> List[Dict[str, Any]]:
        """Find patterns with strong projections on a specific dimension.
        
        Args:
            dimension_number: Dimension number to query
            min_value: Minimum projection value (default: 0.5)
            limit: Maximum number of patterns to return (default: 10)
            
        Returns:
            List of patterns with strong projections on the dimension
        """
        return self.execute_query(CypherQueryLibrary.FIND_PATTERNS_BY_PROJECTION,
                                 {"dimension_number": dimension_number,
                                  "min_value": min_value,
                                  "limit": limit})
    
    def find_community_members(self, community_id: str) -> List[Dict[str, Any]]:
        """Find members of a community.
        
        Args:
            community_id: ID of the community to find members for
            
        Returns:
            List of patterns that belong to the community
        """
        return self.execute_query(CypherQueryLibrary.FIND_COMMUNITY_MEMBERS,
                                 {"community_id": community_id})
    
    def find_pattern_communities(self, pattern_id: str) -> List[Dict[str, Any]]:
        """Find communities that a pattern belongs to.
        
        Args:
            pattern_id: ID of the pattern to find communities for
            
        Returns:
            List of communities that the pattern belongs to
        """
        return self.execute_query(CypherQueryLibrary.FIND_PATTERN_COMMUNITIES,
                                 {"pattern_id": pattern_id})
    
    def find_resonance_patterns_by_type(self, pattern_type: str) -> List[Dict[str, Any]]:
        """Find resonance patterns by type.
        
        Args:
            pattern_type: Type of resonance patterns to find
            
        Returns:
            List of resonance patterns with the specified type
        """
        return self.execute_query(CypherQueryLibrary.FIND_RESONANCE_PATTERNS_BY_TYPE,
                                 {"pattern_type": pattern_type})
    
    def find_resonance_pattern_members(self, resonance_id: str) -> List[Dict[str, Any]]:
        """Find members of a resonance pattern.
        
        Args:
            resonance_id: ID of the resonance pattern to find members for
            
        Returns:
            List of patterns that participate in the resonance pattern
        """
        return self.execute_query(CypherQueryLibrary.FIND_RESONANCE_PATTERN_MEMBERS,
                                 {"resonance_id": resonance_id})
    
    def find_pattern_resonance_patterns(self, pattern_id: str) -> List[Dict[str, Any]]:
        """Find resonance patterns that a pattern participates in.
        
        Args:
            pattern_id: ID of the pattern to find resonance patterns for
            
        Returns:
            List of resonance patterns that the pattern participates in
        """
        return self.execute_query(CypherQueryLibrary.FIND_PATTERN_RESONANCE_PATTERNS,
                                 {"pattern_id": pattern_id})
    
    def find_boundary_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Find boundary patterns with high fuzziness values.
        
        Args:
            limit: Maximum number of patterns to return (default: 10)
            
        Returns:
            List of boundary patterns with their fuzziness values
        """
        return self.execute_query(CypherQueryLibrary.FIND_BOUNDARY_PATTERNS,
                                 {"limit": limit})
    
    def find_community_boundaries(self, community_id: str) -> List[Dict[str, Any]]:
        """Find boundary patterns between a community and other communities.
        
        Args:
            community_id: ID of the community to find boundaries for
            
        Returns:
            List of boundary patterns between the community and other communities
        """
        return self.execute_query(CypherQueryLibrary.FIND_COMMUNITY_BOUNDARIES,
                                 {"community_id": community_id})
    
    def find_pattern_neighbors(self, pattern_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find neighbors of a pattern in the resonance network.
        
        Args:
            pattern_id: ID of the pattern to find neighbors for
            limit: Maximum number of neighbors to return (default: 10)
            
        Returns:
            List of patterns that are neighbors of the specified pattern
        """
        return self.execute_query(CypherQueryLibrary.FIND_PATTERN_NEIGHBORS,
                                 {"pattern_id": pattern_id, "limit": limit})
    
    def find_dimensional_resonance(self, pattern_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find patterns with dimensional resonance to a specific pattern.
        
        Args:
            pattern_id: ID of the pattern to find resonance with
            limit: Maximum number of patterns to return (default: 10)
            
        Returns:
            List of patterns with dimensional resonance to the specified pattern
        """
        return self.execute_query(CypherQueryLibrary.FIND_DIMENSIONAL_RESONANCE,
                                 {"pattern_id": pattern_id, "limit": limit})
    
    def find_complementary_patterns(self, pattern_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find patterns with complementary dimensional relationships to a specific pattern.
        
        Args:
            pattern_id: ID of the pattern to find complementary patterns for
            limit: Maximum number of patterns to return (default: 10)
            
        Returns:
            List of patterns with complementary dimensional relationships
        """
        return self.execute_query(CypherQueryLibrary.FIND_COMPLEMENTARY_PATTERNS,
                                 {"pattern_id": pattern_id, "limit": limit})
    
    def find_shortest_path(self, start_id: str, end_id: str, max_length: int = 5) -> List[Dict[str, Any]]:
        """Find the shortest path between two patterns in the resonance network.
        
        Args:
            start_id: ID of the start pattern
            end_id: ID of the end pattern
            max_length: Maximum path length (default: 5)
            
        Returns:
            Shortest path between the patterns
        """
        query = CypherQueryLibrary.FIND_SHORTEST_PATH % max_length
        return self.execute_query(query, {"start_id": start_id, "end_id": end_id})
    
    def find_all_paths(self, start_id: str, end_id: str, max_length: int = 5) -> List[Dict[str, Any]]:
        """Find all shortest paths between two patterns in the resonance network.
        
        Args:
            start_id: ID of the start pattern
            end_id: ID of the end pattern
            max_length: Maximum path length (default: 5)
            
        Returns:
            All shortest paths between the patterns
        """
        query = CypherQueryLibrary.FIND_ALL_PATHS % max_length
        return self.execute_query(query, {"start_id": start_id, "end_id": end_id})
    
    def find_eigenspace_path(self, start_id: str, end_id: str) -> List[Dict[str, Any]]:
        """Find a path between two patterns through eigenspace.
        
        Args:
            start_id: ID of the start pattern
            end_id: ID of the end pattern
            
        Returns:
            Path through eigenspace between the patterns
        """
        return self.execute_query(CypherQueryLibrary.FIND_EIGENSPACE_PATH,
                                 {"start_id": start_id, "end_id": end_id})
    
    def find_pattern_clusters(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Find pattern clusters based on community and resonance pattern membership.
        
        Args:
            limit: Maximum number of patterns to return (default: 20)
            
        Returns:
            List of patterns with their cluster membership information
        """
        return self.execute_query(CypherQueryLibrary.FIND_PATTERN_CLUSTERS,
                                 {"limit": limit})
    
    def find_dimensional_structure(self) -> List[Dict[str, Any]]:
        """Find the dimensional structure of the field.
        
        Returns:
            List of dimensions with their structural information
        """
        return self.execute_query(CypherQueryLibrary.FIND_DIMENSIONAL_STRUCTURE)
    
    def find_community_structure(self) -> List[Dict[str, Any]]:
        """Find the community structure of the field.
        
        Returns:
            List of communities with their structural information
        """
        return self.execute_query(CypherQueryLibrary.FIND_COMMUNITY_STRUCTURE)
    
    def find_transition_zones(self) -> List[Dict[str, Any]]:
        """Find transition zones between communities.
        
        Returns:
            List of transition zones with their boundary patterns
        """
        return self.execute_query(CypherQueryLibrary.FIND_TRANSITION_ZONES)
    
    def find_pattern_evolution(self) -> List[Dict[str, Any]]:
        """Find pattern evolution over time or sequence.
        
        Returns:
            List of pattern evolution steps with dimensional changes
        """
        return self.execute_query(CypherQueryLibrary.FIND_PATTERN_EVOLUTION)
