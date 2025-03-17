"""
FieldAdaptiveIDBridge module.

This module provides the FieldAdaptiveIDBridge class, which is responsible for
facilitating bidirectional updates between TonicHarmonicFieldState and AdaptiveID components.
"""

from typing import Any, Dict, Optional
import json
from datetime import datetime

# Import the necessary components
from src.habitat_evolution.field.field_state import TonicHarmonicFieldState


class FieldAdaptiveIDBridge:
    """
    Bridge between TonicHarmonicFieldState and AdaptiveID components.
    
    This class is responsible for:
    1. Propagating field state changes to AdaptiveID
    2. Propagating AdaptiveID changes to field state
    3. Maintaining context consistency between components
    4. Tracking state change history for bidirectional learning
    """
    
    def __init__(self, field_state: TonicHarmonicFieldState, adaptive_id: Any):
        """
        Initialize a FieldAdaptiveIDBridge.
        
        Args:
            field_state: TonicHarmonicFieldState instance
            adaptive_id: AdaptiveID instance
        """
        self.field_state = field_state
        self.adaptive_id = adaptive_id
        self.change_history = []
        self.last_sync_timestamp = datetime.now().isoformat()
    
    def propagate_field_state_changes(self) -> Dict[str, Any]:
        """
        Propagate field state changes to AdaptiveID.
        
        Returns:
            Dictionary containing the propagation results
        """
        # Convert field state to AdaptiveID context
        context = self.field_state.to_adaptive_id_context()
        
        # Update AdaptiveID with field state context
        self.adaptive_id.update_temporal_context(
            key="field_state",
            value=context,
            origin="field_state"
        )
        
        # Update specific field properties in AdaptiveID
        for key, value in {
            "coherence": self.field_state.coherence,
            "navigability": self.field_state.navigability_score,
            "stability": self.field_state.stability,
            "dimensionality": self.field_state.effective_dimensionality
        }.items():
            self.adaptive_id.update_spatial_context(
                key=f"field_{key}",
                value=value,
                origin="field_state"
            )
            
        # Propagate tonic-harmonic resonance patterns and relationships
        self._propagate_resonance_patterns()
        
        # Propagate eigenspace properties for each pattern
        self._propagate_eigenspace_properties()
        
        # Record the change in history
        change_record = {
            "timestamp": datetime.now().isoformat(),
            "direction": "field_to_adaptive_id",
            "context": context
        }
        self.change_history.append(change_record)
        self.last_sync_timestamp = change_record["timestamp"]
        
        return {
            "success": True,
            "timestamp": self.last_sync_timestamp,
            "context_keys": list(context.keys())
        }
    
    def propagate_adaptive_id_changes(self) -> Dict[str, Any]:
        """
        Propagate AdaptiveID changes to field state.
        
        Returns:
            Dictionary containing the propagation results
        """
        # Get relevant context from AdaptiveID
        adaptive_id_context = self._extract_adaptive_id_context()
        
        # Update field state with AdaptiveID context
        self.field_state.update_from_adaptive_id_context(adaptive_id_context)
        
        # Record the change in history
        change_record = {
            "timestamp": datetime.now().isoformat(),
            "direction": "adaptive_id_to_field",
            "context": adaptive_id_context
        }
        self.change_history.append(change_record)
        self.last_sync_timestamp = change_record["timestamp"]
        
        return {
            "success": True,
            "timestamp": self.last_sync_timestamp,
            "context_keys": list(adaptive_id_context.keys())
        }
    
    def _extract_adaptive_id_context(self) -> Dict[str, Any]:
        """
        Extract relevant context from AdaptiveID for field state updates.
        
        Returns:
            Dictionary containing AdaptiveID context
        """
        # This method would extract the necessary context from AdaptiveID
        # For now, we'll use a placeholder implementation
        context = {
            "field_state_id": self.field_state.id,
            "field_version_id": self.field_state.version_id
        }
        
        # Get field properties from AdaptiveID if available
        for field_property in ["coherence", "navigability", "stability", "dimensionality"]:
            key = f"field_{field_property}"
            if hasattr(self.adaptive_id, "get_spatial_context"):
                value = self.adaptive_id.get_spatial_context(key)
                if value is not None:
                    context[key] = value
        
        # Get pattern positions if available
        if hasattr(self.adaptive_id, "get_pattern_positions"):
            pattern_positions = self.adaptive_id.get_pattern_positions()
            if pattern_positions:
                context["pattern_positions"] = pattern_positions
                
        # Extract resonance relationships from AdaptiveID if available
        if hasattr(self.adaptive_id, "get_relationships"):
            relationships = self.adaptive_id.get_relationships(relationship_type="RESONATES_WITH")
            if relationships:
                context["resonance_relationships"] = relationships
                
        # Extract eigenspace properties from AdaptiveID if available
        if hasattr(self.adaptive_id, "get_property"):
            eigenspace_properties = {}
            for pattern_id in self.field_state.patterns.keys():
                eigenspace = self.adaptive_id.get_property(entity_id=pattern_id, property_name="eigenspace")
                if eigenspace:
                    eigenspace_properties[pattern_id] = eigenspace
            
            if eigenspace_properties:
                context["pattern_eigenspace_properties"] = eigenspace_properties
        
        return context
    
    def get_change_history(self) -> Dict[str, Any]:
        """
        Get the change history between field state and AdaptiveID.
        
        Returns:
            Dictionary containing the change history
        """
        return {
            "changes": self.change_history,
            "last_sync": self.last_sync_timestamp,
            "count": len(self.change_history)
        }
    
    def _propagate_resonance_patterns(self) -> None:
        """
        Propagate tonic-harmonic resonance patterns to AdaptiveID.
        
        This method ensures that resonance relationships detected in the field
        are properly represented in the ID system.
        """
        # Skip if the field state doesn't have the necessary attributes
        if not hasattr(self.field_state, 'get_pattern_resonance_groups'):
            return
            
        # Get resonance groups from field state
        resonance_groups = self.field_state.get_pattern_resonance_groups()
        if not resonance_groups:
            return
        
        # Create resonance relationships in AdaptiveID
        for group_id, patterns in resonance_groups.items():
            if len(patterns) < 2:
                continue  # Skip groups with less than 2 patterns
                
            # Extract resonance type and dimension from group_id
            parts = group_id.split("_")
            if len(parts) >= 3:
                resonance_type = parts[0] + "_" + parts[1]  # e.g., dim_pos
                dimension = int(parts[2]) if parts[2].isdigit() else 0
                
                # Create relationships between all patterns in the group
                for i in range(len(patterns)):
                    for j in range(i+1, len(patterns)):
                        # Create bidirectional RESONATES_WITH relationship
                        if hasattr(self.adaptive_id, "create_relationship"):
                            self.adaptive_id.create_relationship(
                                source_id=patterns[i],
                                target_id=patterns[j],
                                relationship_type="RESONATES_WITH",
                                properties={
                                    "resonance_type": resonance_type,
                                    "dimension": dimension,
                                    "group_id": group_id,
                                    "strength": self.field_state.eigenvalues[dimension] / sum(self.field_state.eigenvalues) if hasattr(self.field_state, "eigenvalues") and self.field_state.eigenvalues else 0.5
                                }
                            )
    
    def _propagate_eigenspace_properties(self) -> None:
        """
        Propagate eigenspace properties for each pattern to AdaptiveID.
        
        This method ensures that the tonic-harmonic properties of patterns
        are properly represented in the ID system.
        """
        # Skip if pattern_eigenspace_properties is not available
        if not hasattr(self.field_state, "pattern_eigenspace_properties") or not self.field_state.pattern_eigenspace_properties:
            return
            
        # Update eigenspace properties for each pattern in AdaptiveID
        for pattern_id, properties in self.field_state.pattern_eigenspace_properties.items():
            if hasattr(self.adaptive_id, "update_property"):
                # Create a clean representation of eigenspace properties
                eigenspace_data = {
                    "primary_dimensions": properties.get("primary_dimensions", []),
                    "resonance_groups": properties.get("resonance_groups", []),
                    "eigenspace_position": properties.get("eigenspace_position", {}),
                    "projections": {str(k): v for k, v in properties.get("projections", {}).items()}
                }
                
                # Update the pattern's eigenspace properties in AdaptiveID
                self.adaptive_id.update_property(
                    entity_id=pattern_id,
                    property_name="eigenspace",
                    property_value=eigenspace_data
                )
                
                # Also update dimensional properties directly
                for dim_idx, projection in properties.get("projections", {}).items():
                    if isinstance(dim_idx, (int, str)) and isinstance(projection, (int, float)):
                        dim_name = f"dimension_{dim_idx}"
                        self.adaptive_id.update_property(
                            entity_id=pattern_id,
                            property_name=dim_name,
                            property_value=projection
                        )
                        
    def _verify_resonance_consistency(self) -> None:
        """
        Verify that resonance relationships are consistent between field state and AdaptiveID.
        
        This method checks for any inconsistencies in resonance relationships and
        resolves them to ensure both systems have the same understanding of resonance.
        """
        # Skip if the field state doesn't have the necessary attributes
        if not hasattr(self.field_state, 'get_pattern_resonance_groups'):
            return
            
        # Get resonance relationships from field state
        field_resonance = self.field_state.get_pattern_resonance_groups()
        if not field_resonance:
            return
        
        # Get resonance relationships from AdaptiveID
        id_resonance = {}
        if hasattr(self.adaptive_id, "get_relationships"):
            relationships = self.adaptive_id.get_relationships(relationship_type="RESONATES_WITH")
            
            # Group relationships by group_id
            for rel in relationships:
                group_id = rel.get("properties", {}).get("group_id")
                if group_id:
                    if group_id not in id_resonance:
                        id_resonance[group_id] = set()
                    id_resonance[group_id].add(rel.get("source_id"))
                    id_resonance[group_id].add(rel.get("target_id"))
        
        # Check for inconsistencies and resolve them
        for group_id, patterns in field_resonance.items():
            if group_id not in id_resonance:
                # Resonance group exists in field but not in ID system - propagate it
                for i in range(len(patterns)):
                    for j in range(i+1, len(patterns)):
                        if hasattr(self.adaptive_id, "create_relationship"):
                            # Extract resonance type and dimension from group_id
                            parts = group_id.split("_")
                            if len(parts) >= 3:
                                resonance_type = parts[0] + "_" + parts[1]  # e.g., dim_pos
                                dimension = int(parts[2]) if parts[2].isdigit() else 0
                                
                                self.adaptive_id.create_relationship(
                                    source_id=patterns[i],
                                    target_id=patterns[j],
                                    relationship_type="RESONATES_WITH",
                                    properties={
                                        "resonance_type": resonance_type,
                                        "dimension": dimension,
                                        "group_id": group_id,
                                        "strength": self.field_state.eigenvalues[dimension] / sum(self.field_state.eigenvalues) if hasattr(self.field_state, "eigenvalues") and self.field_state.eigenvalues else 0.5
                                    }
                                )
    
    def sync_components(self) -> Dict[str, Any]:
        """
        Synchronize field state and AdaptiveID components.
        
        This method performs a bidirectional sync to ensure both components
        have the most up-to-date information.
        
        Returns:
            Dictionary containing the sync results
        """
        # First propagate field state changes to AdaptiveID
        field_to_adaptive_result = self.propagate_field_state_changes()
        
        # Then propagate AdaptiveID changes to field state
        adaptive_to_field_result = self.propagate_adaptive_id_changes()
        
        # Ensure resonance relationships are consistent
        self._verify_resonance_consistency()
        
        return {
            "success": field_to_adaptive_result["success"] and adaptive_to_field_result["success"],
            "timestamp": datetime.now().isoformat(),
            "field_to_adaptive": field_to_adaptive_result,
            "adaptive_to_field": adaptive_to_field_result
        }
