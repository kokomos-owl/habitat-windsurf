# src/habitat_evolution/field/field_navigator.py
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

class FieldNavigator:
    """Interface for navigating the topological field.
    
    The FieldNavigator provides methods for navigating and interacting with the
    topological field, enabling the creation of a navigable field representation
    that can be used as a metric for the system's state space.
    
    This component is central to establishing an IO space that allows for
    assessment, prediction, and API interactions with the field topology.
    """
    
    def __init__(self, field_analyzer):
        """Initialize the FieldNavigator.
        
        Args:
            field_analyzer: An instance of TopologicalFieldAnalyzer for analyzing fields
        """
        self.field_analyzer = field_analyzer
        self.current_field = None
        self.pattern_metadata = []
        self.navigation_history = []  # Track navigation history for path analysis
        
    def set_field(self, resonance_matrix: np.ndarray, pattern_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Set and analyze the current field."""
        self.current_field = self.field_analyzer.analyze_field(resonance_matrix, pattern_metadata)
        self.pattern_metadata = pattern_metadata
        
        # Add a 'graph' alias to 'graph_metrics' for backward compatibility
        if "graph_metrics" in self.current_field and "graph" not in self.current_field:
            self.current_field["graph"] = self.current_field["graph_metrics"]
            
        return self.current_field
        
    def get_navigation_coordinates(self, pattern_idx: int, dimensions: int = 2) -> List[float]:
        """Get the position coordinates for a pattern in the field's dimensional space."""
        if not self.current_field:
            return [0.0] * dimensions
            
        projections = self.current_field["topology"]["pattern_projections"]
        if pattern_idx >= len(projections):
            return [0.0] * dimensions
            
        coordinates = []
        for d in range(dimensions):
            dim_key = f"dim_{d}"
            coordinates.append(projections[pattern_idx].get(dim_key, 0.0))
            
        return coordinates
        
    def find_nearest_density_center(self, pattern_idx: int) -> Optional[Dict[str, Any]]:
        """Find the nearest density center to a pattern."""
        if not self.current_field:
            return None
            
        centers = self.current_field["density"]["density_centers"]
        if not centers:
            return None
            
        # Get dimensional coordinates
        pattern_coords = self.get_navigation_coordinates(pattern_idx, dimensions=3)
        
        # Find nearest center
        min_distance = float('inf')
        nearest_center = None
        
        for center in centers:
            center_idx = center["index"]
            center_coords = self.get_navigation_coordinates(center_idx, dimensions=3)
            
            # Calculate distance
            distance = np.sqrt(sum((a - b)**2 for a, b in zip(pattern_coords, center_coords)))
            
            if distance < min_distance:
                min_distance = distance
                nearest_center = center
                
        return nearest_center
        
    def find_paths(self, start_idx: int, end_idx: int, path_type: str = "eigenvector") -> List[int]:
        """Find a path between two patterns in the field."""
        if not self.current_field:
            return []
            
        if path_type == "eigenvector":
            # Follow path along eigenvector projection
            return self._find_eigenvector_path(start_idx, end_idx)
        elif path_type == "gradient":
            # Follow gradient of the field
            return self._find_gradient_path(start_idx, end_idx)
        elif path_type == "graph":
            # Use graph shortest path
            return self._find_graph_path(start_idx, end_idx)
        else:
            return []
            
    def _find_eigenvector_path(self, start_idx: int, end_idx: int) -> List[int]:
        """Find path following eigenvector projections."""
        # Simplified implementation - in practice would use more sophisticated path finding
        if not self.current_field:
            return []
            
        # Get coordinates in top 2 dimensions
        start_coords = self.get_navigation_coordinates(start_idx, dimensions=2)
        end_coords = self.get_navigation_coordinates(end_idx, dimensions=2)
        
        # Create simple path with 5 steps
        path = [start_idx]
        for step in range(1, 6):
            # Interpolate along straight line
            t = step / 5.0
            interp_coords = [
                start_coords[d] * (1 - t) + end_coords[d] * t
                for d in range(2)
            ]
            
            # Find nearest pattern to these coordinates
            nearest_idx = start_idx  # Default
            min_distance = float('inf')
            
            for i in range(len(self.current_field["topology"]["pattern_projections"])):
                if i == start_idx or i == end_idx:
                    continue
                    
                coords = self.get_navigation_coordinates(i, dimensions=2)
                distance = np.sqrt(sum((a - b)**2 for a, b in zip(interp_coords, coords)))
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_idx = i
                    
            if nearest_idx not in path:
                path.append(nearest_idx)
                
        if end_idx not in path:
            path.append(end_idx)
            
        return path
        
    def _find_gradient_path(self, start_idx: int, end_idx: int) -> List[int]:
        """Find path following field gradient."""
        # In practice, would implement gradient descent/ascent
        return self._find_eigenvector_path(start_idx, end_idx)  # Placeholder
        
    def _find_graph_path(self, start_idx: int, end_idx: int) -> List[int]:
        """Find path using graph structure."""
        # In practice, would use nx.shortest_path
        return self._find_eigenvector_path(start_idx, end_idx)  # Placeholder
        
    def suggest_exploration_points(self, current_idx: int, count: int = 3) -> List[Dict[str, Any]]:
        """Suggest interesting points to explore from current position."""
        if not self.current_field:
            return []
            
        # Various exploration strategies
        
        # 1. Density gradient direction
        density_candidates = self._explore_density_gradient(current_idx)
        
        # 2. Cross-community bridges
        community_candidates = self._explore_community_bridges(current_idx)
        
        # 3. Unexplored dimensions
        dimension_candidates = self._explore_dimensions(current_idx)
        
        # Combine and rank candidates
        all_candidates = density_candidates + community_candidates + dimension_candidates
        
        # De-duplicate
        seen_indices = set()
        unique_candidates = []
        for candidate in all_candidates:
            idx = candidate["index"]
            if idx not in seen_indices and idx != current_idx:
                seen_indices.add(idx)
                unique_candidates.append(candidate)
                
        # Return top candidates
        return unique_candidates[:count]
        
    def _explore_density_gradient(self, current_idx: int) -> List[Dict[str, Any]]:
        """Find exploration candidates along density gradient.
        
        Args:
            current_idx: Index of the current pattern
            
        Returns:
            List of candidate patterns along density gradients
        """
        if not self.current_field or "density" not in self.current_field:
            return []
            
        candidates = []
        density_centers = self.current_field["density"]["density_centers"]
        
        if not density_centers:
            return []
            
        # Get current pattern coordinates
        current_coords = self.get_navigation_coordinates(current_idx, dimensions=3)
        
        # For each density center, find patterns in the direction of increasing density
        for center in density_centers:
            center_idx = center["index"]
            center_coords = self.get_navigation_coordinates(center_idx, dimensions=3)
            
            # Calculate direction vector from current to center
            direction = [c - p for c, p in zip(center_coords, current_coords)]
            direction_magnitude = np.sqrt(sum(d**2 for d in direction))
            
            if direction_magnitude < 0.001:  # Too close to center
                continue
                
            # Normalize direction vector
            direction = [d / direction_magnitude for d in direction]
            
            # Find patterns in this direction
            for i in range(len(self.pattern_metadata)):
                if i == current_idx or i == center_idx:
                    continue
                    
                pattern_coords = self.get_navigation_coordinates(i, dimensions=3)
                
                # Calculate vector from current to pattern
                pattern_vector = [p - c for p, c in zip(pattern_coords, current_coords)]
                pattern_magnitude = np.sqrt(sum(v**2 for v in pattern_vector))
                
                if pattern_magnitude < 0.001:  # Too close to current
                    continue
                    
                # Normalize pattern vector
                pattern_vector = [v / pattern_magnitude for v in pattern_vector]
                
                # Calculate dot product (cosine similarity)
                dot_product = sum(d * v for d, v in zip(direction, pattern_vector))
                
                # If pattern is in the direction of the center (dot product > 0)
                # and not too far away
                if dot_product > 0.7 and pattern_magnitude < center["influence_radius"]:
                    candidates.append({
                        "index": i,
                        "similarity": dot_product,
                        "distance": pattern_magnitude,
                        "center_distance": np.sqrt(sum((p - c)**2 for p, c in zip(pattern_coords, center_coords))),
                        "center": center
                    })
        
        # Sort by similarity (higher is better)
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates
        
    def _explore_community_bridges(self, current_idx: int) -> List[Dict[str, Any]]:
        """Find exploration candidates that bridge communities.
        
        Args:
            current_idx: Index of the current pattern
            
        Returns:
            List of candidate patterns that bridge communities
        """
        if not self.current_field or "graph_metrics" not in self.current_field:
            return []
            
        candidates = []
        graph_data = self.current_field["graph_metrics"]
        
        if "communities" not in graph_data or "bridges" not in graph_data:
            return []
            
        communities = graph_data["communities"]
        bridges = graph_data["bridges"]
        
        # Find current pattern's community
        current_community = None
        for comm_idx, comm_members in enumerate(communities):
            if current_idx in comm_members:
                current_community = comm_idx
                break
                
        if current_community is None:
            return []
            
        # Find bridges connected to current community
        for bridge in bridges:
            source_comm = bridge["source_community"]
            target_comm = bridge["target_community"]
            bridge_strength = bridge["strength"]
            
            # If bridge connects to current community
            if source_comm == current_community or target_comm == current_community:
                # Get the other community
                other_comm = target_comm if source_comm == current_community else source_comm
                
                # Find patterns in the other community
                for pattern_idx in communities[other_comm]:
                    # Skip if it's the current pattern
                    if pattern_idx == current_idx:
                        continue
                        
                    # Add as candidate
                    candidates.append({
                        "index": pattern_idx,
                        "similarity": bridge_strength,
                        "source_community": source_comm,
                        "target_community": target_comm,
                        "bridge": bridge
                    })
        
        # Sort by bridge strength (higher is better)
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates
        
    def _explore_dimensions(self, current_idx: int) -> List[Dict[str, Any]]:
        """Find exploration candidates in unexplored dimensions.
        
        Args:
            current_idx: Index of the current pattern
            
        Returns:
            List of candidate patterns in unexplored dimensions
        """
        if not self.current_field or "topology" not in self.current_field:
            return []
            
        candidates = []
        topology = self.current_field["topology"]
        
        if "principal_dimensions" not in topology or "pattern_projections" not in topology:
            return []
            
        dimensions = topology["principal_dimensions"]
        projections = topology["pattern_projections"]
        
        if current_idx >= len(projections):
            return []
            
        # Get current pattern's projection
        current_proj = projections[current_idx]
        
        # For each pattern, calculate its projection similarity to current pattern
        for i, proj in enumerate(projections):
            if i == current_idx:
                continue
                
            # Calculate similarity in primary dimensions (first 2)
            primary_similarity = 0
            for d in range(min(2, len(dimensions))):
                dim_key = f"dim_{d}"
                if dim_key in current_proj and dim_key in proj:
                    primary_similarity += (current_proj[dim_key] - proj[dim_key])**2
                    
            primary_similarity = np.sqrt(primary_similarity)
            
            # Calculate similarity in secondary dimensions (3+)
            secondary_similarity = 0
            secondary_count = 0
            for d in range(2, len(dimensions)):
                dim_key = f"dim_{d}"
                if dim_key in current_proj and dim_key in proj:
                    secondary_similarity += (current_proj[dim_key] - proj[dim_key])**2
                    secondary_count += 1
                    
            if secondary_count > 0:
                secondary_similarity = np.sqrt(secondary_similarity / secondary_count)
            
            # We want patterns that are close in primary dimensions but different in secondary
            if primary_similarity < 0.5 and secondary_similarity > 0.3:
                candidates.append({
                    "index": i,
                    "primary_similarity": 1.0 - primary_similarity,  # Higher is more similar
                    "secondary_difference": secondary_similarity,  # Higher is more different
                    "exploration_value": (1.0 - primary_similarity) * secondary_similarity  # Combined metric
                })
        
        # Sort by exploration value (higher is better)
        candidates.sort(key=lambda x: x["exploration_value"], reverse=True)
        return candidates
        
    def get_field_state_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics describing the current field state.
        
        Returns:
            Dictionary of field state metrics or None if no field is set
        """
        if not self.current_field:
            return None
            
        metrics = {
            "dimensionality": {
                "effective": self.current_field["topology"].get("effective_dimensionality", 0),
                "principal_count": len(self.current_field["topology"].get("principal_dimensions", [])),
                "explained_variance": sum(dim.get("explained_variance", 0) 
                                     for dim in self.current_field["topology"].get("principal_dimensions", []))
            },
            "density": {
                "center_count": len(self.current_field["density"].get("density_centers", [])),
                "max_density": max([center.get("density", 0) 
                                 for center in self.current_field["density"].get("density_centers", []) or [0]],
                                default=0),
                "average_density": np.mean([center.get("density", 0) 
                                        for center in self.current_field["density"].get("density_centers", []) or [0]])
                                       if self.current_field["density"].get("density_centers", []) else 0
            },
            "flow": {
                "gradient_strength": self.current_field["flow"].get("gradient_strength", 0),
                "flow_coherence": self.current_field["flow"].get("flow_coherence", 0),
                "attractor_count": len(self.current_field["flow"].get("attractors", []))
            },
            "stability": {
                "eigenvalue_stability": self.current_field["topology"].get("eigenvalue_stability", 0),
                "community_modularity": self.current_field["graph_metrics"].get("modularity", 0) if "graph_metrics" in self.current_field else 0,
                "pattern_stability": np.mean([meta.get("metrics", {}).get("stability", 0) 
                                          for meta in self.pattern_metadata])
                                         if self.pattern_metadata else 0
            },
            "coherence": {
                "average_coherence": np.mean([meta.get("metrics", {}).get("coherence", 0) 
                                          for meta in self.pattern_metadata])
                                         if self.pattern_metadata else 0,
                "field_coherence": self.current_field["potential"].get("field_coherence", 0) if "potential" in self.current_field else 0,
                "relationship_validity": self.current_field["potential"].get("relationship_validity", 0) if "potential" in self.current_field else 0
            }
        }
        
        return metrics
        
    def get_navigable_field_representation(self) -> Optional[Dict[str, Any]]:
        """Get a navigable representation of the field for external use.
        
        This creates an IO space representation that can be used by external
        components for assessment, prediction, and API interactions.
        
        Returns:
            Dictionary with navigable field representation or None if no field is set
        """
        if not self.current_field:
            return None
            
        # Get 3D coordinates for all patterns
        coordinates = []
        for i in range(len(self.pattern_metadata)):
            coords = self.get_navigation_coordinates(i, dimensions=3)
            coordinates.append({
                "index": i,
                "id": self.pattern_metadata[i].get("id", f"pattern_{i}"),
                "coordinates": coords,
                "type": self.pattern_metadata[i].get("type", "unknown")
            })
            
        # Get connections between patterns
        connections = []
        if "graph_metrics" in self.current_field and "edges" in self.current_field["graph_metrics"]:
            for edge in self.current_field["graph_metrics"]["edges"]:
                connections.append({
                    "source": edge["source"],
                    "target": edge["target"],
                    "weight": edge["weight"],
                    "type": edge.get("type", "resonance")
                })
                
        # Get density centers
        centers = []
        if "density" in self.current_field and "density_centers" in self.current_field["density"]:
            for center in self.current_field["density"]["density_centers"]:
                center_coords = self.get_navigation_coordinates(center["index"], dimensions=3)
                centers.append({
                    "index": center["index"],
                    "coordinates": center_coords,
                    "density": center["density"],
                    "influence_radius": center["influence_radius"]
                })
                
        # Get field metrics
        metrics = self.get_field_state_metrics()
        
        return {
            "coordinates": coordinates,
            "connections": connections,
            "centers": centers,
            "metrics": metrics
        }
        
    def get_pattern_context(self, pattern_idx: int) -> Optional[Dict[str, Any]]:
        """Get contextual information for a specific pattern.
        
        Args:
            pattern_idx: Index of the pattern
            
        Returns:
            Dictionary with pattern context or None if pattern not found
        """
        if not self.current_field or pattern_idx >= len(self.pattern_metadata):
            return None
            
        # Get pattern metadata
        pattern = self.pattern_metadata[pattern_idx]
        
        # Get pattern coordinates
        position = self.get_navigation_coordinates(pattern_idx, dimensions=3)
        
        # Find nearest density center
        center = self.find_nearest_density_center(pattern_idx)
        
        # Find neighbors
        neighbors = []
        if "graph_metrics" in self.current_field and "edges" in self.current_field["graph_metrics"]:
            for edge in self.current_field["graph_metrics"]["edges"]:
                if edge["source"] == pattern_idx:
                    target_idx = edge["target"]
                    if target_idx < len(self.pattern_metadata):
                        neighbors.append({
                            "index": target_idx,
                            "id": self.pattern_metadata[target_idx].get("id", f"pattern_{target_idx}"),
                            "resonance": edge["weight"],
                            "type": self.pattern_metadata[target_idx].get("type", "unknown")
                        })
                elif edge["target"] == pattern_idx:
                    source_idx = edge["source"]
                    if source_idx < len(self.pattern_metadata):
                        neighbors.append({
                            "index": source_idx,
                            "id": self.pattern_metadata[source_idx].get("id", f"pattern_{source_idx}"),
                            "resonance": edge["weight"],
                            "type": self.pattern_metadata[source_idx].get("type", "unknown")
                        })
        
        # Sort neighbors by resonance strength
        neighbors.sort(key=lambda x: x["resonance"], reverse=True)
        
        return {
            "pattern": pattern,
            "position": position,
            "center": center,
            "neighbors": neighbors
        }