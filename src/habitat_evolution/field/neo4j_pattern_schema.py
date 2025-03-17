"""Neo4j schema for Vector + Tonic-Harmonic patterns.

This module defines the Neo4j schema for storing and querying Vector + Tonic-Harmonic
patterns detected by the field analysis. It includes node types, relationship types,
and properties for representing patterns, dimensions, and their relationships.
"""

from typing import Dict, List, Any, Optional
import json
from neo4j import GraphDatabase


class Neo4jPatternSchema:
    """Neo4j schema for Vector + Tonic-Harmonic patterns.
    
    This class provides methods for creating the Neo4j schema, importing pattern data,
    and querying the graph database for pattern exploration.
    """
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """Initialize the Neo4j pattern schema.
        
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
        
    def create_schema(self):
        """Create the Neo4j schema for Vector + Tonic-Harmonic patterns.
        
        This method creates the necessary constraints and indexes for the pattern schema.
        """
        with self.driver.session(database=self.database) as session:
            # Create constraints
            session.run("""
                CREATE CONSTRAINT pattern_id IF NOT EXISTS
                FOR (p:Pattern) REQUIRE p.id IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT dimension_id IF NOT EXISTS
                FOR (d:Dimension) REQUIRE d.id IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT resonance_pattern_id IF NOT EXISTS
                FOR (r:ResonancePattern) REQUIRE r.id IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT community_id IF NOT EXISTS
                FOR (c:Community) REQUIRE c.id IS UNIQUE
            """)
            
            # Create indexes
            session.run("""
                CREATE INDEX pattern_type_idx IF NOT EXISTS
                FOR (p:Pattern) ON (p.type)
            """)
            
            session.run("""
                CREATE INDEX resonance_pattern_type_idx IF NOT EXISTS
                FOR (r:ResonancePattern) ON (r.pattern_type)
            """)
            
            session.run("""
                CREATE INDEX dimension_number_idx IF NOT EXISTS
                FOR (d:Dimension) ON (d.dimension_number)
            """)
            
    def import_field_analysis(self, field_analysis: Dict[str, Any], pattern_metadata: List[Dict[str, Any]]):
        """Import field analysis results into Neo4j.
        
        Args:
            field_analysis: Field analysis results from TopologicalFieldAnalyzer
            pattern_metadata: Metadata for each pattern
        """
        with self.driver.session(database=self.database) as session:
            # Import patterns
            for i, metadata in enumerate(pattern_metadata):
                pattern_id = metadata.get("id", f"pattern_{i}")
                pattern_type = metadata.get("type", "unknown")
                
                # Create pattern node
                session.run("""
                    MERGE (p:Pattern {id: $id})
                    SET p.type = $type,
                        p.metadata = $metadata
                """, id=pattern_id, type=pattern_type, metadata=json.dumps(metadata))
                
                # Add pattern projections if available
                if "pattern_projections" in field_analysis and i < len(field_analysis["pattern_projections"]):
                    projections = field_analysis["pattern_projections"][i]
                    for dim_key, proj_value in projections.items():
                        dim_number = int(dim_key.split("_")[1])
                        
                        # Create dimension node if it doesn't exist
                        session.run("""
                            MERGE (d:Dimension {id: $dim_id})
                            SET d.dimension_number = $dim_number
                        """, dim_id=f"dimension_{dim_number}", dim_number=dim_number)
                        
                        # Create projection relationship
                        session.run("""
                            MATCH (p:Pattern {id: $pattern_id})
                            MATCH (d:Dimension {id: $dim_id})
                            MERGE (p)-[r:PROJECTS_TO]->(d)
                            SET r.value = $value
                        """, pattern_id=pattern_id, dim_id=f"dimension_{dim_number}", value=proj_value)
            
            # Import communities if available
            if "communities" in field_analysis:
                communities = field_analysis["communities"]
                for community_id, members in communities.items():
                    # Create community node
                    session.run("""
                        MERGE (c:Community {id: $id})
                        SET c.size = $size
                    """, id=f"community_{community_id}", size=len(members))
                    
                    # Connect patterns to communities
                    for member_idx in members:
                        if member_idx < len(pattern_metadata):
                            pattern_id = pattern_metadata[member_idx].get("id", f"pattern_{member_idx}")
                            session.run("""
                                MATCH (p:Pattern {id: $pattern_id})
                                MATCH (c:Community {id: $community_id})
                                MERGE (p)-[:BELONGS_TO]->(c)
                            """, pattern_id=pattern_id, community_id=f"community_{community_id}")
            
            # Import resonance patterns if available
            if "resonance_patterns" in field_analysis:
                for pattern in field_analysis["resonance_patterns"]:
                    pattern_id = pattern.get("id", f"resonance_{len(pattern.get('members', []))}")
                    pattern_type = pattern.get("pattern_type", "unknown")
                    primary_dimension = pattern.get("primary_dimension", -1)
                    strength = pattern.get("strength", 0.0)
                    
                    # Create resonance pattern node
                    session.run("""
                        MERGE (r:ResonancePattern {id: $id})
                        SET r.pattern_type = $pattern_type,
                            r.primary_dimension = $primary_dimension,
                            r.strength = $strength,
                            r.metadata = $metadata
                    """, id=pattern_id, pattern_type=pattern_type, 
                        primary_dimension=primary_dimension, strength=strength,
                        metadata=json.dumps(pattern))
                    
                    # Connect patterns to resonance pattern
                    for member_idx in pattern.get("members", []):
                        if member_idx < len(pattern_metadata):
                            pattern_id_member = pattern_metadata[member_idx].get("id", f"pattern_{member_idx}")
                            session.run("""
                                MATCH (p:Pattern {id: $pattern_id})
                                MATCH (r:ResonancePattern {id: $resonance_id})
                                MERGE (p)-[:PARTICIPATES_IN]->(r)
                            """, pattern_id=pattern_id_member, resonance_id=pattern_id)
                    
                    # Connect resonance pattern to primary dimension
                    if primary_dimension >= 0:
                        session.run("""
                            MATCH (r:ResonancePattern {id: $resonance_id})
                            MATCH (d:Dimension {id: $dim_id})
                            MERGE (r)-[:BASED_ON]->(d)
                        """, resonance_id=pattern_id, dim_id=f"dimension_{primary_dimension}")
            
            # Import boundary information if available
            if "boundary_fuzziness" in field_analysis:
                boundary_fuzziness = field_analysis["boundary_fuzziness"]
                for i, fuzziness in enumerate(boundary_fuzziness):
                    if i < len(pattern_metadata):
                        pattern_id = pattern_metadata[i].get("id", f"pattern_{i}")
                        session.run("""
                            MATCH (p:Pattern {id: $pattern_id})
                            SET p.boundary_fuzziness = $fuzziness
                        """, pattern_id=pattern_id, fuzziness=fuzziness)
                        
                        # If high fuzziness, mark as boundary pattern
                        if fuzziness > 0.5:  # Threshold for boundary pattern
                            session.run("""
                                MATCH (p:Pattern {id: $pattern_id})
                                SET p:BoundaryPattern
                            """, pattern_id=pattern_id)
    
    def query_patterns_by_dimension(self, dimension_number: int, min_projection: float = 0.5) -> List[Dict[str, Any]]:
        """Query patterns with strong projections on a specific dimension.
        
        Args:
            dimension_number: Dimension number to query
            min_projection: Minimum absolute projection value (default: 0.5)
            
        Returns:
            List of patterns with strong projections on the dimension
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (p:Pattern)-[r:PROJECTS_TO]->(d:Dimension {dimension_number: $dim_number})
                WHERE abs(r.value) >= $min_projection
                RETURN p.id AS pattern_id, p.type AS pattern_type, r.value AS projection,
                       p.metadata AS metadata
                ORDER BY abs(r.value) DESC
            """, dim_number=dimension_number, min_projection=min_projection)
            
            return [record.data() for record in result]
    
    def query_resonance_patterns(self, pattern_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query resonance patterns in the database.
        
        Args:
            pattern_type: Optional filter for pattern type (e.g., "harmonic", "complementary")
            
        Returns:
            List of resonance patterns with their members
        """
        with self.driver.session(database=self.database) as session:
            query = """
                MATCH (r:ResonancePattern)
                WHERE $pattern_type IS NULL OR r.pattern_type = $pattern_type
                MATCH (p:Pattern)-[:PARTICIPATES_IN]->(r)
                RETURN r.id AS resonance_id, r.pattern_type AS pattern_type,
                       r.primary_dimension AS primary_dimension, r.strength AS strength,
                       collect(p.id) AS member_patterns, r.metadata AS metadata
                ORDER BY r.strength DESC
            """
            
            result = session.run(query, pattern_type=pattern_type)
            return [record.data() for record in result]
    
    def query_boundary_patterns(self, min_fuzziness: float = 0.5) -> List[Dict[str, Any]]:
        """Query boundary patterns with high fuzziness values.
        
        Args:
            min_fuzziness: Minimum boundary fuzziness value (default: 0.5)
            
        Returns:
            List of boundary patterns with their fuzziness values
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (p:Pattern)
                WHERE p.boundary_fuzziness >= $min_fuzziness
                OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Community)
                RETURN p.id AS pattern_id, p.type AS pattern_type,
                       p.boundary_fuzziness AS fuzziness,
                       collect(distinct c.id) AS communities,
                       p.metadata AS metadata
                ORDER BY p.boundary_fuzziness DESC
            """, min_fuzziness=min_fuzziness)
            
            return [record.data() for record in result]
    
    def query_pattern_path(self, start_pattern_id: str, end_pattern_id: str, max_length: int = 5) -> List[Dict[str, Any]]:
        """Query a path between two patterns through the resonance network.
        
        Args:
            start_pattern_id: ID of the start pattern
            end_pattern_id: ID of the end pattern
            max_length: Maximum path length (default: 5)
            
        Returns:
            List of paths between the patterns
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH path = shortestPath((start:Pattern {id: $start_id})-[:PARTICIPATES_IN|BELONGS_TO|PROJECTS_TO*..%d]-(end:Pattern {id: $end_id}))
                RETURN [node in nodes(path) | node.id] AS path_nodes,
                       [rel in relationships(path) | type(rel)] AS path_relationships,
                       length(path) AS path_length
                ORDER BY path_length ASC
            """ % max_length, start_id=start_pattern_id, end_id=end_pattern_id)
            
            return [record.data() for record in result]
    
    def query_dimensional_resonance(self, pattern_id: str) -> List[Dict[str, Any]]:
        """Query patterns with dimensional resonance to a specific pattern.
        
        Args:
            pattern_id: ID of the pattern to find resonance with
            
        Returns:
            List of patterns with dimensional resonance to the specified pattern
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (p1:Pattern {id: $pattern_id})-[r1:PROJECTS_TO]->(d:Dimension)<-[r2:PROJECTS_TO]-(p2:Pattern)
                WHERE p1 <> p2 AND sign(r1.value) = sign(r2.value) AND abs(r1.value) >= 0.3 AND abs(r2.value) >= 0.3
                WITH p2, d, count(d) AS shared_dimensions
                WHERE shared_dimensions >= 1
                RETURN p2.id AS pattern_id, p2.type AS pattern_type,
                       collect(d.id) AS shared_dimensions,
                       size(collect(d.id)) AS resonance_strength,
                       p2.metadata AS metadata
                ORDER BY resonance_strength DESC
            """, pattern_id=pattern_id)
            
            return [record.data() for record in result]
    
    def query_complementary_patterns(self, pattern_id: str) -> List[Dict[str, Any]]:
        """Query patterns with complementary dimensional relationships to a specific pattern.
        
        Args:
            pattern_id: ID of the pattern to find complementary patterns for
            
        Returns:
            List of patterns with complementary dimensional relationships
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (p1:Pattern {id: $pattern_id})-[r1:PROJECTS_TO]->(d:Dimension)<-[r2:PROJECTS_TO]-(p2:Pattern)
                WHERE p1 <> p2 AND sign(r1.value) <> sign(r2.value) AND abs(r1.value) >= 0.3 AND abs(r2.value) >= 0.3
                WITH p2, d, count(d) AS complementary_dimensions
                WHERE complementary_dimensions >= 1
                RETURN p2.id AS pattern_id, p2.type AS pattern_type,
                       collect(d.id) AS complementary_dimensions,
                       size(collect(d.id)) AS complementary_strength,
                       p2.metadata AS metadata
                ORDER BY complementary_strength DESC
            """, pattern_id=pattern_id)
            
            return [record.data() for record in result]
    
    def query_community_bridges(self) -> List[Dict[str, Any]]:
        """Query patterns that bridge multiple communities.
        
        Returns:
            List of patterns that belong to multiple communities
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (p:Pattern)-[:BELONGS_TO]->(c:Community)
                WITH p, collect(c) AS communities
                WHERE size(communities) > 1
                RETURN p.id AS pattern_id, p.type AS pattern_type,
                       [comm in communities | comm.id] AS community_ids,
                       size(communities) AS community_count,
                       p.boundary_fuzziness AS fuzziness,
                       p.metadata AS metadata
                ORDER BY community_count DESC, p.boundary_fuzziness DESC
            """)
            
            return [record.data() for record in result]
    
    def query_pattern_neighborhood(self, pattern_id: str, max_distance: float = 0.5) -> List[Dict[str, Any]]:
        """Query the neighborhood of a pattern in eigenspace.
        
        Args:
            pattern_id: ID of the pattern to find neighbors for
            max_distance: Maximum eigenspace distance (default: 0.5)
            
        Returns:
            List of patterns in the neighborhood
        """
        with self.driver.session(database=self.database) as session:
            # This query uses a custom procedure that calculates eigenspace distance
            # For simplicity, we'll use a dimension-based approach here
            result = session.run("""
                MATCH (p1:Pattern {id: $pattern_id})-[r1:PROJECTS_TO]->(d:Dimension)<-[r2:PROJECTS_TO]-(p2:Pattern)
                WHERE p1 <> p2
                WITH p2, sum((r1.value - r2.value)^2) AS distance_squared
                WHERE sqrt(distance_squared) <= $max_distance
                RETURN p2.id AS pattern_id, p2.type AS pattern_type,
                       sqrt(distance_squared) AS distance,
                       p2.metadata AS metadata
                ORDER BY distance ASC
            """, pattern_id=pattern_id, max_distance=max_distance)
            
            return [record.data() for record in result]
