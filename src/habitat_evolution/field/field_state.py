"""
TonicHarmonicFieldState module.

This module provides the TonicHarmonicFieldState class, which is responsible for
maintaining state across field operations and providing context for state changes.
It serves as the bridge between the mathematical field analysis and the semantic
understanding in the adaptive_id system.
"""

import uuid
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from copy import deepcopy


class TonicHarmonicFieldState:
    """
    Represents the state of a tonic-harmonic field with versioning and context tracking.
    
    This class is responsible for:
    1. Maintaining state across field operations
    2. Providing context for state changes
    3. Enabling bidirectional updates between field analysis and adaptive_id
    4. Supporting serialization and persistence of field states
    """
    
    def __init__(self, field_analysis: Dict[str, Any]):
        """
        Initialize a TonicHarmonicFieldState from field analysis results.
        
        Args:
            field_analysis: Results from TopologicalFieldAnalyzer
        """
        # Core identifiers
        self.id = str(uuid.uuid4())
        self.version_id = str(uuid.uuid4())
        self.created_at = datetime.now().isoformat()
        self.last_modified = self.created_at
        
        # Extract topology information
        self.effective_dimensionality = field_analysis["topology"]["effective_dimensionality"]
        self.principal_dimensions = field_analysis["topology"]["principal_dimensions"]
        self.eigenvalues = field_analysis["topology"]["eigenvalues"].tolist() if isinstance(field_analysis["topology"]["eigenvalues"], np.ndarray) else field_analysis["topology"]["eigenvalues"]
        
        # Store eigenvectors as list for serialization
        if isinstance(field_analysis["topology"]["eigenvectors"], np.ndarray):
            self.eigenvectors = field_analysis["topology"]["eigenvectors"].tolist()
        else:
            self.eigenvectors = field_analysis["topology"]["eigenvectors"]
            
        # Track resonance relationships between patterns
        self.resonance_relationships = field_analysis.get("resonance_relationships", {})
        
        # Extract density information
        self.density_centers = field_analysis["density"]["density_centers"]
        
        # Convert density map to list for serialization
        if isinstance(field_analysis["density"]["density_map"], np.ndarray):
            self.density_map = field_analysis["density"]["density_map"].tolist()
        else:
            self.density_map = field_analysis["density"]["density_map"]
        
        # Extract field properties
        self.coherence = field_analysis["field_properties"]["coherence"]
        self.navigability_score = field_analysis["field_properties"]["navigability_score"]
        self.stability = field_analysis["field_properties"]["stability"]
        
        # Extract patterns
        self.patterns = field_analysis["patterns"]
        
        # Initialize pattern eigenspace properties tracking
        self.pattern_eigenspace_properties = {}
        self._initialize_pattern_eigenspace_properties(field_analysis)
        
        # Initialize context tracking
        self.temporal_context = {}
        self.spatial_context = {}
        
        # Initialize version history
        self.versions = {
            self.version_id: {
                "timestamp": self.created_at,
                "state": self._get_current_state()
            }
        }
    
    def _initialize_pattern_eigenspace_properties(self, field_analysis: Dict[str, Any]) -> None:
        """
        Initialize eigenspace properties for each pattern.
        
        Args:
            field_analysis: Results from TopologicalFieldAnalyzer
        """
        # Extract pattern projections if available
        pattern_projections = field_analysis.get("topology", {}).get("pattern_projections", {})
        dimensional_resonance = field_analysis.get("topology", {}).get("dimensional_resonance", [])
        
        # Initialize eigenspace properties for each pattern
        for pattern_id, pattern in self.patterns.items():
            self.pattern_eigenspace_properties[pattern_id] = {
                "projections": pattern_projections.get(pattern_id, {}),
                "primary_dimensions": [],
                "resonance_groups": [],
                "eigenspace_position": {}
            }
            
            # Extract primary dimensions (dimensions with strong projections)
            if pattern_id in pattern_projections:
                projections = pattern_projections[pattern_id]
                # Find dimensions with strong projections (absolute value > 0.5)
                primary_dims = [dim for dim, proj in projections.items() 
                               if abs(float(proj)) > 0.5]
                self.pattern_eigenspace_properties[pattern_id]["primary_dimensions"] = primary_dims
                
                # Calculate eigenspace position (for visualization)
                top_dims = sorted(projections.items(), 
                                  key=lambda x: abs(float(x[1])), 
                                  reverse=True)[:3]
                position = {dim: float(proj) for dim, proj in top_dims}
                self.pattern_eigenspace_properties[pattern_id]["eigenspace_position"] = position
        
        # Track resonance group membership
        for resonance_group in dimensional_resonance:
            group_id = resonance_group.get("id")
            members = resonance_group.get("members", [])
            for pattern_id in members:
                if pattern_id in self.pattern_eigenspace_properties:
                    self.pattern_eigenspace_properties[pattern_id]["resonance_groups"].append(group_id)
    
    def _get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state as a dictionary.
        
        Returns:
            Dictionary representation of the current state
        """
        return {
            "effective_dimensionality": self.effective_dimensionality,
            "principal_dimensions": self.principal_dimensions,
            "eigenvalues": self.eigenvalues,
            "eigenvectors": self.eigenvectors,
            "density_centers": self.density_centers,
            "coherence": self.coherence,
            "navigability_score": self.navigability_score,
            "stability": self.stability,
            "patterns": deepcopy(self.patterns),
            "pattern_eigenspace_properties": deepcopy(self.pattern_eigenspace_properties),
            "resonance_relationships": deepcopy(self.resonance_relationships),
            "temporal_context": deepcopy(self.temporal_context),
            "spatial_context": deepcopy(self.spatial_context)
        }
    
    def create_snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of the current state.
        
        Returns:
            Dictionary containing the current state
        """
        snapshot = {
            "id": self.id,
            "version_id": self.version_id,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            **self._get_current_state()
        }
        return snapshot
    
    def restore_from_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Restore the state from a snapshot.
        
        Args:
            snapshot: Dictionary containing the state to restore
        """
        # Restore core identifiers
        self.id = snapshot["id"]
        self.version_id = snapshot["version_id"]
        self.created_at = snapshot["created_at"]
        self.last_modified = snapshot["last_modified"]
        
        # Restore topology information
        self.effective_dimensionality = snapshot["effective_dimensionality"]
        self.principal_dimensions = snapshot["principal_dimensions"]
        self.eigenvalues = snapshot["eigenvalues"]
        self.eigenvectors = snapshot["eigenvectors"]
        
        # Restore density information
        self.density_centers = snapshot["density_centers"]
        
        # Restore field properties
        self.coherence = snapshot["coherence"]
        self.navigability_score = snapshot["navigability_score"]
        self.stability = snapshot["stability"]
        
        # Restore patterns
        self.patterns = snapshot["patterns"]
        
        # Restore context tracking
        self.temporal_context = snapshot["temporal_context"]
        self.spatial_context = snapshot["spatial_context"]
    
    def update_temporal_context(self, key: str, value: Any, origin: str) -> None:
        """
        Update the temporal context with a key-value pair.
        
        Args:
            key: Context key
            value: Context value
            origin: Origin of the context update
        """
        if key not in self.temporal_context:
            self.temporal_context[key] = {}
        
        # Use the value as the key in the temporal context
        self.temporal_context[key][value] = {
            "value": value,
            "origin": origin
        }
        self.last_modified = datetime.now().isoformat()
    
    def update_spatial_context(self, key: str, value: Any, origin: str) -> None:
        """
        Update the spatial context with a key-value pair.
        
        Args:
            key: Context key
            value: Context value
            origin: Origin of the context update
        """
        self.spatial_context[key] = {
            "value": value,
            "origin": origin
        }
        self.last_modified = datetime.now().isoformat()
    
    def create_new_version(self) -> str:
        """
        Create a new version of the field state.
        
        Returns:
            New version ID
        """
        # Create a new version ID
        new_version_id = str(uuid.uuid4())
        self.version_id = new_version_id
        
        # Update the last modified timestamp
        self.last_modified = datetime.now().isoformat()
        
        # Store the current state in the version history
        self.versions[new_version_id] = {
            "timestamp": self.last_modified,
            "state": self._get_current_state()
        }
        
        return new_version_id
    
    def get_state_at_version(self, version_id: str) -> Dict[str, Any]:
        """
        Get the state at a specific version.
        
        Args:
            version_id: Version ID to retrieve
            
        Returns:
            Dictionary containing the state at the specified version
            
        Raises:
            KeyError: If the version ID is not found
        """
        if version_id not in self.versions:
            raise KeyError(f"Version ID {version_id} not found")
        
        version = self.versions[version_id]
        return {
            "id": self.id,
            "version_id": version_id,
            "created_at": self.created_at,
            "last_modified": version["timestamp"],
            **version["state"]
        }
    
    def compare_versions(self, version_id_1: str, version_id_2: str) -> Dict[str, Dict[str, Any]]:
        """
        Compare two versions to identify changes.
        
        Args:
            version_id_1: First version ID
            version_id_2: Second version ID
            
        Returns:
            Dictionary containing the changes between the versions
            
        Raises:
            KeyError: If either version ID is not found
        """
        if version_id_1 not in self.versions:
            raise KeyError(f"Version ID {version_id_1} not found")
        if version_id_2 not in self.versions:
            raise KeyError(f"Version ID {version_id_2} not found")
        
        state_1 = self.versions[version_id_1]["state"]
        state_2 = self.versions[version_id_2]["state"]
        
        changes = {}
        
        # Compare scalar properties
        for key in ["effective_dimensionality", "coherence", "navigability_score", "stability"]:
            if state_1[key] != state_2[key]:
                changes[key] = {
                    "from": state_1[key],
                    "to": state_2[key]
                }
        
        # Compare density centers
        if len(state_1["density_centers"]) != len(state_2["density_centers"]):
            changes["density_centers"] = {
                "from": len(state_1["density_centers"]),
                "to": len(state_2["density_centers"])
            }
        
        # Compare patterns
        if state_1["patterns"] != state_2["patterns"]:
            # Count patterns that changed position
            changed_patterns = []
            for pattern_id in set(state_1["patterns"].keys()) & set(state_2["patterns"].keys()):
                if state_1["patterns"][pattern_id] != state_2["patterns"][pattern_id]:
                    changed_patterns.append(pattern_id)
            
            # Count patterns that were added or removed
            added_patterns = set(state_2["patterns"].keys()) - set(state_1["patterns"].keys())
            removed_patterns = set(state_1["patterns"].keys()) - set(state_2["patterns"].keys())
            
            changes["patterns"] = {
                "changed": changed_patterns,
                "added": list(added_patterns),
                "removed": list(removed_patterns)
            }
        
        return changes
    
    def get_eigenspace_properties(self) -> Dict[str, Any]:
        """
        Get eigenspace properties for analysis and visualization.
        
        Returns:
            Dictionary containing eigenspace properties
        """
        return {
            "effective_dimensions": self.effective_dimensionality,
            "eigenvalues": self.eigenvalues,
            "principal_dimensions": self.principal_dimensions,
            "pattern_eigenspace_properties": self.pattern_eigenspace_properties,
            "resonance_relationships": self.resonance_relationships
        }
    
    def get_pattern_resonance_groups(self) -> Dict[str, List[str]]:
        """
        Get patterns grouped by resonance relationships.
        
        Returns:
            Dictionary mapping resonance group IDs to lists of pattern IDs
        """
        resonance_groups = {}
        
        # Group patterns by resonance group
        for pattern_id, properties in self.pattern_eigenspace_properties.items():
            for group_id in properties.get("resonance_groups", []):
                if group_id not in resonance_groups:
                    resonance_groups[group_id] = []
                resonance_groups[group_id].append(pattern_id)
        
        return resonance_groups
    
    def to_adaptive_id_context(self) -> Dict[str, Any]:
        """
        Convert the field state to AdaptiveID context.
        
        Returns:
            Dictionary containing the field state as AdaptiveID context
        """
        # Get pattern eigenspace properties in a format suitable for ID context
        pattern_eigenspace_context = {}
        for pattern_id, properties in self.pattern_eigenspace_properties.items():
            # Include full projections to ensure dimensional resonance is captured
            projections = {str(k): float(v) for k, v in properties.get("projections", {}).items()}
            
            pattern_eigenspace_context[pattern_id] = {
                "primary_dimensions": properties.get("primary_dimensions", []),
                "resonance_groups": properties.get("resonance_groups", []),
                "eigenspace_position": properties.get("eigenspace_position", {}),
                "projections": projections,
                "dimensional_resonance": self._get_dimensional_resonance_for_pattern(pattern_id)
            }
        
        # Get resonance relationships in a format suitable for ID context
        resonance_context = {}
        for group_id, patterns in self.get_pattern_resonance_groups().items():
            # Extract resonance type and dimension from group_id
            parts = group_id.split("_")
            resonance_type = "unknown"
            dimension = 0
            
            if len(parts) >= 3:
                resonance_type = parts[0] + "_" + parts[1]  # e.g., dim_pos
                dimension = int(parts[2]) if parts[2].isdigit() else 0
            
            # Calculate resonance strength based on eigenvalues
            strength = 0.5
            if hasattr(self, "eigenvalues") and self.eigenvalues and dimension < len(self.eigenvalues):
                strength = float(self.eigenvalues[dimension] / sum(self.eigenvalues))
            
            resonance_context[group_id] = {
                "patterns": patterns,
                "type": resonance_type,
                "dimension": dimension,
                "strength": strength,
                "complementary": resonance_type == "dim_comp"
            }
        
        # Prepare tonic-harmonic specific context
        tonic_harmonic_context = {
            "eigenvalues": [float(v) for v in self.eigenvalues] if hasattr(self, "eigenvalues") and self.eigenvalues else [],
            "effective_dimensionality": self.effective_dimensionality,
            "principal_dimensions": self.principal_dimensions,
            "dimensional_resonance_groups": self._get_dimensional_resonance_groups(),
            "fuzzy_boundaries": self._get_fuzzy_boundary_data() if hasattr(self, "_get_fuzzy_boundary_data") else {}
        }
        
        return {
            "field_state_id": self.id,
            "field_version_id": self.version_id,
            "field_coherence": self.coherence,
            "field_navigability": self.navigability_score,
            "field_stability": self.stability,
            "field_dimensionality": self.effective_dimensionality,
            "field_density_centers": len(self.density_centers),
            "field_patterns": len(self.patterns),
            "field_temporal_context": json.dumps(self.temporal_context),
            "field_spatial_context": json.dumps(self.spatial_context),
            "field_last_modified": self.last_modified,
            "field_eigenspace": {
                "effective_dimensions": self.effective_dimensionality,
                "principal_dimensions": self.principal_dimensions,
                "pattern_eigenspace": pattern_eigenspace_context,
                "resonance_relationships": resonance_context,
                "tonic_harmonic_properties": tonic_harmonic_context
            }
        }
    
    def _get_dimensional_resonance_for_pattern(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get dimensional resonance information for a specific pattern.
        
        Args:
            pattern_id: ID of the pattern
            
        Returns:
            Dictionary containing dimensional resonance information
        """
        resonance_info = {}
        
        # Skip if pattern doesn't have eigenspace properties
        if pattern_id not in self.pattern_eigenspace_properties:
            return resonance_info
            
        properties = self.pattern_eigenspace_properties[pattern_id]
        projections = properties.get("projections", {})
        
        # Find dimensions with strong projections
        for dim, projection in projections.items():
            if isinstance(dim, str):
                dim = int(dim) if dim.isdigit() else 0
                
            # Check if projection is strong (positive or negative)
            if abs(projection) > 0.7:  # Threshold for strong projection
                direction = "positive" if projection > 0 else "negative"
                resonance_info[f"dimension_{dim}"] = {
                    "projection": float(projection),
                    "direction": direction,
                    "strength": abs(float(projection))
                }
                
        return resonance_info
        
    def _get_fuzzy_boundary_data(self) -> Dict[str, Any]:
        """
        Calculate fuzzy boundaries between resonance groups.
        
        This method identifies transition zones between different resonance groups
        and calculates the fuzziness of boundaries between them.
        
        Returns:
            Dictionary containing boundary fuzziness and transition zones
        """
        # Skip if no pattern eigenspace properties
        if not self.pattern_eigenspace_properties:
            return {"boundary_fuzziness": {}, "transition_zones": {}}
            
        # Get resonance groups
        resonance_groups = self.get_pattern_resonance_groups()
        if not resonance_groups:
            return {"boundary_fuzziness": {}, "transition_zones": {}}
            
        # Create a mapping of patterns to their communities (resonance groups)
        community_assignment = {}
        for group_id, patterns in resonance_groups.items():
            for pattern_id in patterns:
                if pattern_id not in community_assignment:
                    community_assignment[pattern_id] = []
                community_assignment[pattern_id].append(group_id)
                
        # Identify boundaries between communities
        communities = set()
        for groups in community_assignment.values():
            communities.update(groups)
            
        boundaries = []
        for c1 in communities:
            for c2 in communities:
                if c1 < c2:  # Avoid duplicates
                    boundaries.append((c1, c2))
                    
        # Calculate boundary fuzziness and identify transition zones
        boundary_fuzziness = {}
        transition_zones = {}
        transition_threshold = 0.5  # Threshold for transition zone membership
        
        for c1, c2 in boundaries:
            # Get patterns in each community
            community1 = [p for p, groups in community_assignment.items() if c1 in groups]
            community2 = [p for p, groups in community_assignment.items() if c2 in groups]
            
            # Skip if either community is empty
            if not community1 or not community2:
                continue
                
            # Calculate distances between patterns in eigenspace
            distances = []
            transition_patterns = []
            
            for p1 in community1:
                for p2 in community2:
                    # Calculate distance in eigenspace
                    dist = self._calculate_eigenspace_distance(p1, p2)
                    distances.append(dist)
                    
                    # Identify patterns in transition zone
                    if dist < transition_threshold:
                        transition_patterns.extend([p1, p2])
            
            # Store results
            if distances:
                boundary_key = f"{c1}__{c2}"
                boundary_fuzziness[boundary_key] = float(sum(distances) / len(distances)) if distances else 1.0
                transition_zones[boundary_key] = list(set(transition_patterns))
        
        return {
            "boundary_fuzziness": boundary_fuzziness,
            "transition_zones": transition_zones
        }
        
    def _calculate_eigenspace_distance(self, pattern1_id: str, pattern2_id: str) -> float:
        """
        Calculate the distance between two patterns in eigenspace.
        
        Args:
            pattern1_id: ID of the first pattern
            pattern2_id: ID of the second pattern
            
        Returns:
            Distance between the patterns in eigenspace
        """
        # Skip if either pattern doesn't have eigenspace properties
        if pattern1_id not in self.pattern_eigenspace_properties or pattern2_id not in self.pattern_eigenspace_properties:
            return 1.0  # Maximum distance
            
        # Get eigenspace positions
        pos1 = self.pattern_eigenspace_properties[pattern1_id].get("eigenspace_position", {})
        pos2 = self.pattern_eigenspace_properties[pattern2_id].get("eigenspace_position", {})
        
        # Skip if either position is empty
        if not pos1 or not pos2:
            # Try using projections instead
            proj1 = self.pattern_eigenspace_properties[pattern1_id].get("projections", {})
            proj2 = self.pattern_eigenspace_properties[pattern2_id].get("projections", {})
            
            if not proj1 or not proj2:
                return 1.0  # Maximum distance
                
            # Calculate distance using projections
            # Use only dimensions that exist in both projections
            common_dims = set(proj1.keys()) & set(proj2.keys())
            if not common_dims:
                return 1.0  # Maximum distance
                
            # Calculate Euclidean distance in projection space
            squared_diff_sum = 0.0
            for dim in common_dims:
                val1 = float(proj1[dim])
                val2 = float(proj2[dim])
                squared_diff_sum += (val1 - val2) ** 2
                
            return min(1.0, (squared_diff_sum / len(common_dims)) ** 0.5)
        
        # Calculate Euclidean distance in eigenspace
        # Use only dimensions that exist in both positions
        common_dims = set(pos1.keys()) & set(pos2.keys())
        if not common_dims:
            return 1.0  # Maximum distance
            
        squared_diff_sum = 0.0
        for dim in common_dims:
            val1 = float(pos1[dim])
            val2 = float(pos2[dim])
            squared_diff_sum += (val1 - val2) ** 2
            
        return min(1.0, (squared_diff_sum / len(common_dims)) ** 0.5)
    
    def _get_dimensional_resonance_groups(self) -> Dict[str, Any]:
        """
        Get all dimensional resonance groups in the field.
        
        Returns:
            Dictionary mapping dimension IDs to resonance information
        """
        resonance_groups = {}
        
        # Skip if no eigenvalues
        if not hasattr(self, "eigenvalues") or not self.eigenvalues:
            return resonance_groups
            
        # For each significant dimension
        for dim in range(min(5, len(self.eigenvalues))):
            # Skip less significant dimensions
            if self.eigenvalues[dim] / sum(self.eigenvalues) < 0.1:
                continue
                
            # Find patterns with strong projections on this dimension
            strong_positive = []
            strong_negative = []
            
            for pattern_id, properties in self.pattern_eigenspace_properties.items():
                projections = properties.get("projections", {})
                projection = projections.get(dim, 0)
                if isinstance(projection, str):
                    projection = float(projection)
                    
                if projection > 0.7:
                    strong_positive.append(pattern_id)
                elif projection < -0.7:
                    strong_negative.append(pattern_id)
            
            # Add resonance groups
            if len(strong_positive) >= 2:
                resonance_groups[f"dim_pos_{dim}"] = {
                    "members": strong_positive,
                    "type": "dimensional_resonance",
                    "direction": "positive",
                    "dimension": dim,
                    "strength": float(self.eigenvalues[dim] / sum(self.eigenvalues))
                }
                
            if len(strong_negative) >= 2:
                resonance_groups[f"dim_neg_{dim}"] = {
                    "members": strong_negative,
                    "type": "dimensional_resonance",
                    "direction": "negative",
                    "dimension": dim,
                    "strength": float(self.eigenvalues[dim] / sum(self.eigenvalues))
                }
                
            # Complementary patterns (opposite projections)
            if len(strong_positive) >= 1 and len(strong_negative) >= 1:
                resonance_groups[f"dim_comp_{dim}"] = {
                    "members": strong_positive + strong_negative,
                    "type": "complementary",
                    "dimension": dim,
                    "strength": float(self.eigenvalues[dim] / sum(self.eigenvalues))
                }
                
        return resonance_groups
    
    def update_from_adaptive_id_context(self, context: Dict[str, Any]) -> None:
        """
        Update the field state from AdaptiveID context.
        
        Args:
            context: Dictionary containing AdaptiveID context
        """
        # Verify that this is the correct field state
        if context.get("field_state_id") != self.id:
            raise ValueError(f"Field state ID mismatch: {context.get('field_state_id')} != {self.id}")
        
        # Create a new version if the context has a different version ID
        if context.get("field_version_id") != self.version_id:
            self.create_new_version()
        
        # Update field properties
        if "field_coherence" in context:
            self.coherence = context["field_coherence"]
        if "field_navigability" in context:
            self.navigability_score = context["field_navigability"]
        if "field_stability" in context:
            self.stability = context["field_stability"]
        if "field_dimensionality" in context:
            self.effective_dimensionality = context["field_dimensionality"]
        
        # Update pattern positions if provided
        if "pattern_positions" in context:
            for pattern_id, position in context["pattern_positions"].items():
                if pattern_id in self.patterns:
                    self.patterns[pattern_id]["position"] = position
        
        # Update pattern eigenspace properties if provided
        if "field_eigenspace" in context and "pattern_eigenspace" in context["field_eigenspace"]:
            pattern_eigenspace = context["field_eigenspace"]["pattern_eigenspace"]
            for pattern_id, properties in pattern_eigenspace.items():
                if pattern_id in self.pattern_eigenspace_properties:
                    # Update eigenspace position if provided
                    if "eigenspace_position" in properties:
                        self.pattern_eigenspace_properties[pattern_id]["eigenspace_position"] = properties["eigenspace_position"]
                    
                    # Update resonance groups if provided
                    if "resonance_groups" in properties:
                        self.pattern_eigenspace_properties[pattern_id]["resonance_groups"] = properties["resonance_groups"]
                        
                    # Update projections if provided
                    if "projections" in properties:
                        self.pattern_eigenspace_properties[pattern_id]["projections"] = properties["projections"]
                        
                    # Update dimensional resonance if provided
                    if "dimensional_resonance" in properties:
                        for dim_id, resonance_data in properties["dimensional_resonance"].items():
                            # Store dimensional resonance data in pattern properties
                            if "dimensional_resonance" not in self.pattern_eigenspace_properties[pattern_id]:
                                self.pattern_eigenspace_properties[pattern_id]["dimensional_resonance"] = {}
                            self.pattern_eigenspace_properties[pattern_id]["dimensional_resonance"][dim_id] = resonance_data
        
        # Update resonance relationships if provided
        if "field_eigenspace" in context and "resonance_relationships" in context["field_eigenspace"]:
            resonance_relationships = context["field_eigenspace"]["resonance_relationships"]
            for group_id, relationship in resonance_relationships.items():
                if group_id not in self.resonance_relationships and "patterns" in relationship:
                    # Add new resonance relationship
                    self.resonance_relationships[group_id] = {
                        "type": relationship.get("type", "unknown"),
                        "dimension": relationship.get("dimension", 0),
                        "patterns": relationship.get("patterns", [])
                    }
                    
                    # Update pattern eigenspace properties to include this group
                    for pattern_id in relationship.get("patterns", []):
                        if pattern_id in self.pattern_eigenspace_properties:
                            if "resonance_groups" not in self.pattern_eigenspace_properties[pattern_id]:
                                self.pattern_eigenspace_properties[pattern_id]["resonance_groups"] = []
                            
                            if group_id not in self.pattern_eigenspace_properties[pattern_id]["resonance_groups"]:
                                self.pattern_eigenspace_properties[pattern_id]["resonance_groups"].append(group_id)
        
        # Update tonic-harmonic specific properties if provided
        if "field_eigenspace" in context and "tonic_harmonic_properties" in context["field_eigenspace"]:
            tonic_harmonic = context["field_eigenspace"]["tonic_harmonic_properties"]
            
            # Update eigenvalues if provided
            if "eigenvalues" in tonic_harmonic and hasattr(self, "eigenvalues"):
                self.eigenvalues = tonic_harmonic["eigenvalues"]
                
            # Update principal dimensions if provided
            if "principal_dimensions" in tonic_harmonic:
                self.principal_dimensions = tonic_harmonic["principal_dimensions"]
                
            # Update effective dimensionality if provided
            if "effective_dimensionality" in tonic_harmonic:
                self.effective_dimensionality = tonic_harmonic["effective_dimensionality"]
                
            # Update resonance relationships from dimensional resonance groups if provided
            if "dimensional_resonance_groups" in tonic_harmonic:
                for group_id, group_info in tonic_harmonic["dimensional_resonance_groups"].items():
                    if group_id not in self.resonance_relationships and "members" in group_info:
                        # Add new resonance relationship
                        self.resonance_relationships[group_id] = {
                            "type": group_info.get("type", "dimensional_resonance"),
                            "dimension": group_info.get("dimension", 0),
                            "patterns": group_info.get("members", [])
                        }
                        
                        # Update pattern eigenspace properties
                        for pattern_id in group_info.get("members", []):
                            if pattern_id in self.pattern_eigenspace_properties:
                                if "resonance_groups" not in self.pattern_eigenspace_properties[pattern_id]:
                                    self.pattern_eigenspace_properties[pattern_id]["resonance_groups"] = []
                                
                                if group_id not in self.pattern_eigenspace_properties[pattern_id]["resonance_groups"]:
                                    self.pattern_eigenspace_properties[pattern_id]["resonance_groups"].append(group_id)
                                    
            # Update fuzzy boundary data if provided
            if "fuzzy_boundaries" in tonic_harmonic and hasattr(self, "boundary_fuzziness"):
                self.boundary_fuzziness = tonic_harmonic["fuzzy_boundaries"].get("boundary_fuzziness", {})
                self.transition_zones = tonic_harmonic["fuzzy_boundaries"].get("transition_zones", {})
        
        # Update the last modified timestamp
        self.last_modified = datetime.now().isoformat()
    
    def to_neo4j(self) -> Dict[str, Any]:
        """
        Serialize the field state for Neo4j storage.
        
        Returns:
            Dictionary containing the serialized field state
        """
        return {
            "id": self.id,
            "version_id": self.version_id,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "effective_dimensionality": self.effective_dimensionality,
            "coherence": self.coherence,
            "navigability_score": self.navigability_score,
            "stability": self.stability,
            "temporal_context": json.dumps(self.temporal_context),
            "spatial_context": json.dumps(self.spatial_context),
            "patterns": json.dumps(self.patterns),
            "pattern_eigenspace_properties": json.dumps(self.pattern_eigenspace_properties),
            "resonance_relationships": json.dumps(self.resonance_relationships),
            "density_centers": json.dumps(self.density_centers),
            "principal_dimensions": json.dumps(self.principal_dimensions),
            "eigenvalues": json.dumps(self.eigenvalues),
            # Eigenvectors can be large, so we'll store them separately if needed
            "eigenvectors_summary": json.dumps({
                "shape": [len(self.eigenvectors), len(self.eigenvectors[0]) if self.eigenvectors else 0],
                "first_vector": self.eigenvectors[0] if self.eigenvectors else []
            })
        }
    
    @classmethod
    def generate_neo4j_resonance_relationships(self) -> List[Dict[str, Any]]:
        """
        Generate Neo4j relationship data for pattern resonance.
        
        Returns:
            List of dictionaries containing relationship data for Neo4j
        """
        relationships = []
        
        # Process resonance groups
        resonance_groups = self.get_pattern_resonance_groups()
        for group_id, patterns in resonance_groups.items():
            # Extract dimension and type from group_id (e.g., dim_pos_1)
            parts = group_id.split("_")
            if len(parts) >= 3:
                rel_type = parts[0] + "_" + parts[1]  # e.g., dim_pos
                dimension = int(parts[2]) if parts[2].isdigit() else 0
                
                # Create relationships between all patterns in the group
                for i, pattern1 in enumerate(patterns):
                    for pattern2 in patterns[i+1:]:
                        relationships.append({
                            "source_id": pattern1,
                            "target_id": pattern2,
                            "type": "RESONATES_WITH",
                            "properties": {
                                "group_id": group_id,
                                "resonance_type": rel_type,
                                "dimension": dimension,
                                "strength": self._calculate_resonance_strength(pattern1, pattern2, dimension)
                            }
                        })
        
        return relationships
    
    def _calculate_resonance_strength(self, pattern1_id: str, pattern2_id: str, dimension: int) -> float:
        """
        Calculate the resonance strength between two patterns in a specific dimension.
        
        Args:
            pattern1_id: ID of the first pattern
            pattern2_id: ID of the second pattern
            dimension: Dimension to calculate resonance in
            
        Returns:
            Resonance strength as a float between 0 and 1
        """
        # Get projections for both patterns
        p1_props = self.pattern_eigenspace_properties.get(pattern1_id, {})
        p2_props = self.pattern_eigenspace_properties.get(pattern2_id, {})
        
        p1_proj = p1_props.get("projections", {}).get(str(dimension), 0)
        p2_proj = p2_props.get("projections", {}).get(str(dimension), 0)
        
        # Calculate dot product of projections
        # For resonance, we want the absolute value of the product
        # This captures both aligned and anti-aligned patterns
        return abs(float(p1_proj) * float(p2_proj))
    
    @classmethod
    def from_neo4j(cls, neo4j_data: Dict[str, Any]) -> 'TonicHarmonicFieldState':
        """
        Create a TonicHarmonicFieldState from Neo4j data.
        
        Args:
            neo4j_data: Dictionary containing Neo4j data
            
        Returns:
            TonicHarmonicFieldState instance
        """
        # Create a minimal field analysis to initialize the instance
        field_analysis = {
            "topology": {
                "effective_dimensionality": neo4j_data["effective_dimensionality"],
                "principal_dimensions": json.loads(neo4j_data["principal_dimensions"]),
                "eigenvalues": json.loads(neo4j_data["eigenvalues"]),
                "eigenvectors": [[0.0] * 10] * 10  # Placeholder
            },
            "density": {
                "density_centers": json.loads(neo4j_data["density_centers"]),
                "density_map": [[0.0] * 10] * 10  # Placeholder
            },
            "field_properties": {
                "coherence": neo4j_data["coherence"],
                "navigability_score": neo4j_data["navigability_score"],
                "stability": neo4j_data["stability"]
            },
            "patterns": json.loads(neo4j_data["patterns"]),
            "resonance_relationships": json.loads(neo4j_data.get("resonance_relationships", "{}"))
        }
        
        # Add pattern eigenspace properties if available
        if "pattern_eigenspace_properties" in neo4j_data:
            field_analysis["topology"]["pattern_eigenspace_properties"] = json.loads(neo4j_data["pattern_eigenspace_properties"])
        
        # Create the instance
        instance = cls(field_analysis)
        
        # Override the generated IDs and timestamps
        instance.id = neo4j_data["id"]
        instance.version_id = neo4j_data["version_id"]
        instance.created_at = neo4j_data["created_at"]
        instance.last_modified = neo4j_data["last_modified"]
        
        # Restore context
        instance.temporal_context = json.loads(neo4j_data["temporal_context"])
        instance.spatial_context = json.loads(neo4j_data["spatial_context"])
        
        # Restore pattern eigenspace properties if available
        if "pattern_eigenspace_properties" in neo4j_data:
            instance.pattern_eigenspace_properties = json.loads(neo4j_data["pattern_eigenspace_properties"])
        
        # Restore resonance relationships if available
        if "resonance_relationships" in neo4j_data:
            instance.resonance_relationships = json.loads(neo4j_data["resonance_relationships"])
        
        # Restore versions (if available)
        if "versions" in neo4j_data:
            instance.versions = json.loads(neo4j_data["versions"])
        else:
            # Create a minimal version history
            instance.versions = {
                instance.version_id: {
                    "timestamp": instance.last_modified,
                    "state": instance._get_current_state()
                }
            }
        
        return instance
    
    def detect_state_changes(self, new_field_analysis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Detect changes between the current state and a new field analysis.
        
        Args:
            new_field_analysis: New field analysis results
            
        Returns:
            Dictionary containing the detected changes
        """
        changes = {}
        
        # Compare scalar properties
        for key, new_key in [
            ("effective_dimensionality", "topology.effective_dimensionality"),
            ("coherence", "field_properties.coherence"),
            ("navigability_score", "field_properties.navigability_score"),
            ("stability", "field_properties.stability")
        ]:
            # Extract the nested value using the key path
            parts = new_key.split(".")
            value = new_field_analysis
            for part in parts:
                value = value[part]
            
            # Compare with the current value
            current_value = getattr(self, key)
            if current_value != value:
                changes[key] = {
                    "from": current_value,
                    "to": value
                }
        
        # Compare density centers
        new_density_centers = new_field_analysis["density"]["density_centers"]
        if len(self.density_centers) != len(new_density_centers):
            changes["density_centers"] = {
                "from": len(self.density_centers),
                "to": len(new_density_centers)
            }
        
        # Compare patterns
        new_patterns = new_field_analysis["patterns"]
        if self.patterns != new_patterns:
            # Count patterns that changed position
            changed_patterns = []
            for pattern_id in set(self.patterns.keys()) & set(new_patterns.keys()):
                if self.patterns[pattern_id] != new_patterns[pattern_id]:
                    changed_patterns.append(pattern_id)
            
            # Count patterns that were added or removed
            added_patterns = set(new_patterns.keys()) - set(self.patterns.keys())
            removed_patterns = set(self.patterns.keys()) - set(new_patterns.keys())
            
            changes["patterns"] = {
                "changed": changed_patterns,
                "added": list(added_patterns),
                "removed": list(removed_patterns)
            }
        
        return changes
    
    def update_from_field_analysis(self, new_field_analysis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Update the field state from a new field analysis.
        
        Args:
            new_field_analysis: New field analysis results
            
        Returns:
            Dictionary containing the detected changes
        """
        # Detect changes
        changes = self.detect_state_changes(new_field_analysis)
        
        # If there are changes, create a new version
        if changes:
            self.create_new_version()
            
            # Update topology information
            self.effective_dimensionality = new_field_analysis["topology"]["effective_dimensionality"]
            self.principal_dimensions = new_field_analysis["topology"]["principal_dimensions"]
            
            # Update eigenvalues and eigenvectors
            if isinstance(new_field_analysis["topology"]["eigenvalues"], np.ndarray):
                self.eigenvalues = new_field_analysis["topology"]["eigenvalues"].tolist()
            else:
                self.eigenvalues = new_field_analysis["topology"]["eigenvalues"]
            
            if isinstance(new_field_analysis["topology"]["eigenvectors"], np.ndarray):
                self.eigenvectors = new_field_analysis["topology"]["eigenvectors"].tolist()
            else:
                self.eigenvectors = new_field_analysis["topology"]["eigenvectors"]
            
            # Update density information
            self.density_centers = new_field_analysis["density"]["density_centers"]
            
            # Update density map
            if isinstance(new_field_analysis["density"]["density_map"], np.ndarray):
                self.density_map = new_field_analysis["density"]["density_map"].tolist()
            else:
                self.density_map = new_field_analysis["density"]["density_map"]
            
            # Update field properties
            self.coherence = new_field_analysis["field_properties"]["coherence"]
            self.navigability_score = new_field_analysis["field_properties"]["navigability_score"]
            self.stability = new_field_analysis["field_properties"]["stability"]
            
            # Update patterns
            self.patterns = new_field_analysis["patterns"]
            
            # Update the last modified timestamp
            self.last_modified = datetime.now().isoformat()
        
        return changes
