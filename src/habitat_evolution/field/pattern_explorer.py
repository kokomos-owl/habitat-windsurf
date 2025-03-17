"""Pattern Explorer for Vector + Tonic-Harmonic patterns.

This module provides a Pattern Explorer that integrates the Field Navigator with
the Neo4j pattern schema to enable seamless pattern exploration and visualization.
It leverages eigenspace navigation capabilities and dimensional resonance detection
to provide a rich exploration experience.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import json
import logging
from pathlib import Path

from habitat_evolution.field.field_navigator import FieldNavigator
from habitat_evolution.field.neo4j_pattern_schema import Neo4jPatternSchema
from habitat_evolution.field.cypher_query_library import CypherQueryExecutor


class PatternExplorer:
    """Pattern Explorer for Vector + Tonic-Harmonic patterns.
    
    This class integrates the Field Navigator with the Neo4j pattern schema to enable
    seamless pattern exploration and visualization. It provides methods for exploring
    patterns, dimensions, communities, and their relationships.
    """
    
    def __init__(self, field_navigator: Optional[FieldNavigator] = None,
                neo4j_config: Optional[Dict[str, str]] = None):
        """Initialize the Pattern Explorer.
        
        Args:
            field_navigator: Field Navigator instance (optional)
            neo4j_config: Neo4j configuration (optional)
        """
        self.navigator = field_navigator or FieldNavigator()
        self.neo4j_schema = None
        self.query_executor = None
        
        if neo4j_config:
            self._setup_neo4j(neo4j_config)
            
        self.logger = logging.getLogger(__name__)
    
    def _setup_neo4j(self, config: Dict[str, str]):
        """Set up Neo4j connection and schema.
        
        Args:
            config: Neo4j configuration with uri, username, password, and database
        """
        try:
            uri = config.get("uri", "bolt://localhost:7687")
            username = config.get("username", "neo4j")
            password = config.get("password", "password")
            database = config.get("database", "neo4j")
            
            self.neo4j_schema = Neo4jPatternSchema(uri, username, password, database)
            self.query_executor = CypherQueryExecutor(uri, username, password, database)
            
            # Create schema if it doesn't exist
            self.neo4j_schema.create_schema()
            self.logger.info("Neo4j schema created successfully")
        except Exception as e:
            self.logger.error(f"Failed to set up Neo4j: {str(e)}")
            self.neo4j_schema = None
            self.query_executor = None
    
    def set_field(self, field_analysis: Dict[str, Any], pattern_metadata: List[Dict[str, Any]]):
        """Set the field for exploration.
        
        Args:
            field_analysis: Field analysis results from TopologicalFieldAnalyzer
            pattern_metadata: Metadata for each pattern
        """
        self.navigator.set_field(field_analysis, pattern_metadata)
        
        # Import field analysis into Neo4j if available
        if self.neo4j_schema:
            try:
                self.neo4j_schema.import_field_analysis(field_analysis, pattern_metadata)
                self.logger.info("Field analysis imported into Neo4j")
            except Exception as e:
                self.logger.error(f"Failed to import field analysis into Neo4j: {str(e)}")
    
    def explore_pattern(self, pattern_idx: int) -> Dict[str, Any]:
        """Explore a pattern and its relationships.
        
        Args:
            pattern_idx: Index of the pattern to explore
            
        Returns:
            Dictionary with pattern exploration results
        """
        # Get pattern information from navigator
        pattern_info = self.navigator.get_pattern_info(pattern_idx)
        
        # Get related patterns through dimensional resonance
        resonant_patterns = []
        for i in range(len(self.navigator.pattern_metadata)):
            if i != pattern_idx:
                resonance = self.navigator._detect_dimensional_resonance(pattern_idx, i)
                if resonance and resonance["strength"] > 0.3:
                    resonant_patterns.append({
                        "index": i,
                        "id": self.navigator.pattern_metadata[i].get("id", f"pattern_{i}"),
                        "type": self.navigator.pattern_metadata[i].get("type", "unknown"),
                        "resonance": resonance
                    })
        
        # Get boundary information
        boundary_info = self.navigator.detect_fuzzy_boundaries()
        is_boundary = False
        boundary_fuzziness = 0.0
        
        for boundary in boundary_info.get("boundaries", []):
            if boundary["pattern_idx"] == pattern_idx:
                is_boundary = True
                boundary_fuzziness = boundary.get("fuzziness", 0.0)
                break
        
        # Get community information
        community = self.navigator.get_pattern_community(pattern_idx)
        
        # Get eigenspace coordinates
        eigenspace_coords = None
        if "eigenspace_coordinates" in self.navigator.field_analysis:
            if pattern_idx < len(self.navigator.field_analysis["eigenspace_coordinates"]):
                eigenspace_coords = self.navigator.field_analysis["eigenspace_coordinates"][pattern_idx]
        
        # Get pattern projections
        projections = None
        if "pattern_projections" in self.navigator.field_analysis:
            if pattern_idx < len(self.navigator.field_analysis["pattern_projections"]):
                projections = self.navigator.field_analysis["pattern_projections"][pattern_idx]
        
        # Compile exploration results
        exploration_results = {
            "pattern": pattern_info,
            "resonant_patterns": resonant_patterns,
            "is_boundary": is_boundary,
            "boundary_fuzziness": boundary_fuzziness,
            "community": community,
            "eigenspace_coordinates": eigenspace_coords,
            "projections": projections
        }
        
        # Add Neo4j query results if available
        if self.query_executor and "id" in pattern_info:
            pattern_id = pattern_info["id"]
            try:
                # Get resonance patterns from Neo4j
                neo4j_resonance = self.query_executor.find_pattern_resonance_patterns(pattern_id)
                exploration_results["neo4j_resonance_patterns"] = neo4j_resonance
                
                # Get communities from Neo4j
                neo4j_communities = self.query_executor.find_pattern_communities(pattern_id)
                exploration_results["neo4j_communities"] = neo4j_communities
                
                # Get dimensional resonance from Neo4j
                neo4j_dimensional_resonance = self.query_executor.find_dimensional_resonance(pattern_id)
                exploration_results["neo4j_dimensional_resonance"] = neo4j_dimensional_resonance
                
                # Get complementary patterns from Neo4j
                neo4j_complementary = self.query_executor.find_complementary_patterns(pattern_id)
                exploration_results["neo4j_complementary_patterns"] = neo4j_complementary
            except Exception as e:
                self.logger.error(f"Failed to get Neo4j query results: {str(e)}")
        
        return exploration_results
    
    def explore_dimension(self, dimension: int) -> Dict[str, Any]:
        """Explore a dimension and patterns with strong projections.
        
        Args:
            dimension: Dimension number to explore
            
        Returns:
            Dictionary with dimension exploration results
        """
        # Get dimension information
        dimension_info = {}
        if "principal_dimensions" in self.navigator.field_analysis:
            principal_dims = self.navigator.field_analysis["principal_dimensions"]
            if dimension < len(principal_dims):
                dimension_info = principal_dims[dimension]
        
        # Get patterns with strong projections on this dimension
        strong_patterns = []
        if "pattern_projections" in self.navigator.field_analysis:
            for i, projections in enumerate(self.navigator.field_analysis["pattern_projections"]):
                dim_key = f"dim_{dimension}"
                if dim_key in projections:
                    proj_value = projections[dim_key]
                    if abs(proj_value) > 0.3:  # Threshold for strong projection
                        strong_patterns.append({
                            "index": i,
                            "id": self.navigator.pattern_metadata[i].get("id", f"pattern_{i}"),
                            "type": self.navigator.pattern_metadata[i].get("type", "unknown"),
                            "projection": proj_value
                        })
        
        # Sort by absolute projection value
        strong_patterns.sort(key=lambda p: abs(p["projection"]), reverse=True)
        
        # Compile exploration results
        exploration_results = {
            "dimension": dimension,
            "dimension_info": dimension_info,
            "strong_patterns": strong_patterns
        }
        
        # Add Neo4j query results if available
        if self.query_executor:
            try:
                # Get patterns with strong projections from Neo4j
                neo4j_patterns = self.query_executor.find_patterns_by_projection(dimension)
                exploration_results["neo4j_patterns"] = neo4j_patterns
                
                # Get resonance patterns based on this dimension
                neo4j_resonance = self.query_executor.find_resonance_patterns_by_type("harmonic")
                neo4j_resonance = [r for r in neo4j_resonance if r.get("primary_dimension") == dimension]
                exploration_results["neo4j_resonance_patterns"] = neo4j_resonance
            except Exception as e:
                self.logger.error(f"Failed to get Neo4j query results: {str(e)}")
        
        return exploration_results
    
    def explore_community(self, community_id: Union[int, str]) -> Dict[str, Any]:
        """Explore a community and its patterns.
        
        Args:
            community_id: ID of the community to explore
            
        Returns:
            Dictionary with community exploration results
        """
        # Convert string community ID to int if needed
        if isinstance(community_id, str) and community_id.isdigit():
            community_id = int(community_id)
        
        # Get community information
        community_info = {}
        community_patterns = []
        
        if "communities" in self.navigator.field_analysis:
            communities = self.navigator.field_analysis["communities"]
            if isinstance(community_id, int) and str(community_id) in communities:
                community_info = {
                    "id": community_id,
                    "size": len(communities[str(community_id)])
                }
                
                # Get patterns in this community
                for pattern_idx in communities[str(community_id)]:
                    if pattern_idx < len(self.navigator.pattern_metadata):
                        pattern_info = self.navigator.get_pattern_info(pattern_idx)
                        community_patterns.append(pattern_info)
        
        # Get boundary patterns for this community
        boundary_info = self.navigator.detect_fuzzy_boundaries()
        community_boundaries = []
        
        for boundary in boundary_info.get("boundaries", []):
            pattern_idx = boundary["pattern_idx"]
            pattern_communities = self.navigator.get_pattern_community(pattern_idx)
            
            if isinstance(pattern_communities, list):
                if community_id in pattern_communities:
                    community_boundaries.append({
                        "pattern_idx": pattern_idx,
                        "pattern_id": self.navigator.pattern_metadata[pattern_idx].get("id", f"pattern_{pattern_idx}"),
                        "fuzziness": boundary.get("fuzziness", 0.0),
                        "communities": pattern_communities
                    })
            elif pattern_communities == community_id:
                community_boundaries.append({
                    "pattern_idx": pattern_idx,
                    "pattern_id": self.navigator.pattern_metadata[pattern_idx].get("id", f"pattern_{pattern_idx}"),
                    "fuzziness": boundary.get("fuzziness", 0.0),
                    "communities": [pattern_communities]
                })
        
        # Compile exploration results
        exploration_results = {
            "community": community_info,
            "patterns": community_patterns,
            "boundaries": community_boundaries
        }
        
        # Add Neo4j query results if available
        if self.query_executor:
            neo4j_community_id = f"community_{community_id}"
            try:
                # Get community members from Neo4j
                neo4j_members = self.query_executor.find_community_members(neo4j_community_id)
                exploration_results["neo4j_members"] = neo4j_members
                
                # Get community boundaries from Neo4j
                neo4j_boundaries = self.query_executor.find_community_boundaries(neo4j_community_id)
                exploration_results["neo4j_boundaries"] = neo4j_boundaries
            except Exception as e:
                self.logger.error(f"Failed to get Neo4j query results: {str(e)}")
        
        return exploration_results
    
    def explore_resonance_pattern(self, pattern_type: str, dimension: Optional[int] = None) -> Dict[str, Any]:
        """Explore resonance patterns of a specific type.
        
        Args:
            pattern_type: Type of resonance pattern to explore (harmonic, complementary, sequential)
            dimension: Optional dimension to filter by
            
        Returns:
            Dictionary with resonance pattern exploration results
        """
        # Get resonance patterns from field analysis
        resonance_patterns = []
        
        if "resonance_patterns" in self.navigator.field_analysis:
            for pattern in self.navigator.field_analysis["resonance_patterns"]:
                if pattern["pattern_type"] == pattern_type:
                    if dimension is None or pattern.get("primary_dimension") == dimension:
                        # Get member patterns
                        members = []
                        for member_idx in pattern.get("members", []):
                            if member_idx < len(self.navigator.pattern_metadata):
                                member_info = self.navigator.get_pattern_info(member_idx)
                                members.append(member_info)
                        
                        resonance_patterns.append({
                            "id": pattern.get("id", ""),
                            "pattern_type": pattern_type,
                            "primary_dimension": pattern.get("primary_dimension"),
                            "strength": pattern.get("strength", 0.0),
                            "members": members
                        })
        
        # Sort by strength
        resonance_patterns.sort(key=lambda p: p["strength"], reverse=True)
        
        # Compile exploration results
        exploration_results = {
            "pattern_type": pattern_type,
            "dimension": dimension,
            "resonance_patterns": resonance_patterns
        }
        
        # Add Neo4j query results if available
        if self.query_executor:
            try:
                # Get resonance patterns from Neo4j
                neo4j_patterns = self.query_executor.find_resonance_patterns_by_type(pattern_type)
                
                # Filter by dimension if specified
                if dimension is not None:
                    neo4j_patterns = [p for p in neo4j_patterns if p.get("primary_dimension") == dimension]
                
                exploration_results["neo4j_patterns"] = neo4j_patterns
            except Exception as e:
                self.logger.error(f"Failed to get Neo4j query results: {str(e)}")
        
        return exploration_results
    
    def explore_transition_zone(self, pattern_idx: int) -> Dict[str, Any]:
        """Explore the transition zone around a boundary pattern.
        
        Args:
            pattern_idx: Index of the boundary pattern to explore
            
        Returns:
            Dictionary with transition zone exploration results
        """
        # Get boundary information
        boundary_info = self.navigator.detect_fuzzy_boundaries()
        is_boundary = False
        
        for boundary in boundary_info.get("boundaries", []):
            if boundary["pattern_idx"] == pattern_idx:
                is_boundary = True
                break
        
        if not is_boundary:
            return {"error": "Pattern is not a boundary pattern"}
        
        # Explore the transition zone
        transition_results = self.navigator.explore_transition_zone(pattern_idx)
        
        # Get pattern information
        pattern_info = self.navigator.get_pattern_info(pattern_idx)
        
        # Get community information
        communities = self.navigator.get_pattern_community(pattern_idx)
        if not isinstance(communities, list):
            communities = [communities]
        
        # Compile exploration results
        exploration_results = {
            "pattern": pattern_info,
            "communities": communities,
            "transition_patterns": transition_results
        }
        
        # Add Neo4j query results if available
        if self.query_executor and "id" in pattern_info:
            pattern_id = pattern_info["id"]
            try:
                # Get neighborhood from Neo4j
                neo4j_neighborhood = self.query_executor.find_pattern_neighbors(pattern_id)
                exploration_results["neo4j_neighborhood"] = neo4j_neighborhood
            except Exception as e:
                self.logger.error(f"Failed to get Neo4j query results: {str(e)}")
        
        return exploration_results
    
    def navigate_between_patterns(self, start_idx: int, end_idx: int, 
                                 method: str = "eigenspace") -> Dict[str, Any]:
        """Navigate between two patterns using different methods.
        
        Args:
            start_idx: Index of the start pattern
            end_idx: Index of the end pattern
            method: Navigation method (eigenspace, fuzzy_boundary, direct)
            
        Returns:
            Dictionary with navigation results
        """
        # Get pattern information
        start_info = self.navigator.get_pattern_info(start_idx)
        end_info = self.navigator.get_pattern_info(end_idx)
        
        # Navigate using the specified method
        path = []
        
        if method == "eigenspace":
            path = self.navigator.navigate_eigenspace(start_idx, end_idx)
        elif method == "fuzzy_boundary":
            path_indices = self.navigator._find_fuzzy_boundary_path(start_idx, end_idx)
            path = [self.navigator.get_pattern_info(idx) for idx in path_indices]
        elif method == "direct":
            # Direct path with just start and end
            path = [start_info, end_info]
        else:
            return {"error": f"Unknown navigation method: {method}"}
        
        # Get dimensional resonance between start and end
        resonance = self.navigator._detect_dimensional_resonance(start_idx, end_idx)
        
        # Compile navigation results
        navigation_results = {
            "start": start_info,
            "end": end_info,
            "method": method,
            "path": path,
            "path_length": len(path),
            "dimensional_resonance": resonance
        }
        
        # Add Neo4j query results if available
        if self.query_executor and "id" in start_info and "id" in end_info:
            start_id = start_info["id"]
            end_id = end_info["id"]
            try:
                # Get path from Neo4j
                if method == "eigenspace":
                    neo4j_path = self.query_executor.find_eigenspace_path(start_id, end_id)
                else:
                    neo4j_path = self.query_executor.find_shortest_path(start_id, end_id)
                
                navigation_results["neo4j_path"] = neo4j_path
            except Exception as e:
                self.logger.error(f"Failed to get Neo4j query results: {str(e)}")
        
        return navigation_results
    
    def export_visualization_data(self, output_path: str) -> str:
        """Export visualization data for the field.
        
        Args:
            output_path: Path to save the visualization data
            
        Returns:
            Path to the saved visualization data file
        """
        # Prepare visualization data
        viz_data = {
            "patterns": [],
            "dimensions": [],
            "communities": [],
            "resonance_patterns": [],
            "boundaries": []
        }
        
        # Add patterns
        for i, metadata in enumerate(self.navigator.pattern_metadata):
            pattern_data = {
                "index": i,
                "id": metadata.get("id", f"pattern_{i}"),
                "type": metadata.get("type", "unknown"),
                "metadata": metadata
            }
            
            # Add eigenspace coordinates if available
            if "eigenspace_coordinates" in self.navigator.field_analysis:
                if i < len(self.navigator.field_analysis["eigenspace_coordinates"]):
                    pattern_data["coordinates"] = self.navigator.field_analysis["eigenspace_coordinates"][i]
            
            # Add projections if available
            if "pattern_projections" in self.navigator.field_analysis:
                if i < len(self.navigator.field_analysis["pattern_projections"]):
                    pattern_data["projections"] = self.navigator.field_analysis["pattern_projections"][i]
            
            # Add community information
            pattern_data["community"] = self.navigator.get_pattern_community(i)
            
            viz_data["patterns"].append(pattern_data)
        
        # Add dimensions
        if "principal_dimensions" in self.navigator.field_analysis:
            viz_data["dimensions"] = self.navigator.field_analysis["principal_dimensions"]
        
        # Add communities
        if "communities" in self.navigator.field_analysis:
            for community_id, members in self.navigator.field_analysis["communities"].items():
                community_data = {
                    "id": community_id,
                    "members": members,
                    "size": len(members)
                }
                viz_data["communities"].append(community_data)
        
        # Add resonance patterns
        if "resonance_patterns" in self.navigator.field_analysis:
            viz_data["resonance_patterns"] = self.navigator.field_analysis["resonance_patterns"]
        
        # Add boundary information
        boundary_info = self.navigator.detect_fuzzy_boundaries()
        if "boundaries" in boundary_info:
            viz_data["boundaries"] = boundary_info["boundaries"]
        
        # Save visualization data
        output_file = Path(output_path)
        with open(output_file, "w") as f:
            json.dump(viz_data, f, indent=2)
        
        return str(output_file)
    
    def close(self):
        """Close Neo4j connections."""
        if self.neo4j_schema:
            self.neo4j_schema.close()
        
        if self.query_executor:
            self.query_executor.close()
