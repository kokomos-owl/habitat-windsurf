"""Pattern-Aware RAG Integration Module

This module provides integration points between the field topology analysis components
and the Pattern-Aware RAG system, allowing for field topology analysis and navigation
within the pattern-aware retrieval augmented generation system.

The integration enables enhanced pattern discovery, exploration of potentially interesting
regions, and navigation through semantic fields in ways that transcend traditional
vector-based similarity.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ..pattern_aware_rag.pattern_aware_rag import PatternAwareRAG
from .topological_field_analyzer import TopologicalFieldAnalyzer
from .field_navigator import FieldNavigator


# ------------------------------------------------
# Integration functions for PatternAwareRAG class
# ------------------------------------------------

def extend_pattern_aware_rag(rag_instance: PatternAwareRAG) -> None:
    """Extend a PatternAwareRAG instance with field topology functionality.
    
    This function monkey-patches the provided PatternAwareRAG instance with
    methods for field topology analysis and navigation.
    
    Args:
        rag_instance: The PatternAwareRAG instance to extend
    """
    # Initialize field topology components
    rag_instance.field_analyzer = TopologicalFieldAnalyzer()
    rag_instance.field_navigator = FieldNavigator(rag_instance.field_analyzer)
    rag_instance.current_field_analysis = None
    
    # Add methods to the instance
    rag_instance.analyze_pattern_field = analyze_pattern_field.__get__(rag_instance)
    rag_instance.get_pattern_coordinates = get_pattern_coordinates.__get__(rag_instance)
    rag_instance.find_pattern_path = find_pattern_path.__get__(rag_instance)
    rag_instance.find_exploration_regions = find_exploration_regions.__get__(rag_instance)
    rag_instance._get_related_patterns = _get_related_patterns.__get__(rag_instance)
    
    # Store original method reference for later enhancement
    original_process = rag_instance.process_with_patterns
    
    # Enhance the process_with_patterns method
    async def enhanced_process_with_patterns(query: str, context: Optional[Dict[str, Any]] = None):
        # Call the original method
        response = await original_process(query, context)
        
        # Enhance with field topology if patterns exist
        if response.get("patterns"):
            all_patterns = []
            if "query_patterns" in response:
                all_patterns.extend(response["query_patterns"])
            if "retrieval_patterns" in response:
                all_patterns.extend(response["retrieval_patterns"])
            if "augmentation_patterns" in response:
                all_patterns.extend(response["augmentation_patterns"])
            
            # Only analyze if we have patterns
            if all_patterns:
                field_analysis = await rag_instance.analyze_pattern_field(all_patterns)
                
                # Add field navigability data to the response
                response["field_topology"] = {
                    "dimensionality": field_analysis["topology"]["effective_dimensionality"],
                    "density_centers": len(field_analysis["density"]["density_centers"]),
                    "primary_dimensions": [
                        dim["explained_variance"] 
                        for dim in field_analysis["topology"]["principal_dimensions"]
                    ],
                    "field_coherence": field_analysis["field_properties"]["coherence"],
                    "field_navigability": field_analysis["field_properties"]["navigability_score"]
                }
                
                # Calculate pattern positions within the field
                pattern_positions = {}
                for pattern in all_patterns:
                    position = rag_instance.get_pattern_coordinates(pattern)
                    if position["pattern_index"] != -1:
                        pattern_positions[pattern] = position["coordinates"]
                
                response["pattern_positions"] = pattern_positions
        
        return response
    
    # Bind the enhanced method to the instance
    rag_instance.process_with_patterns = enhanced_process_with_patterns.__get__(rag_instance)


async def analyze_pattern_field(self, patterns: List[str]) -> Dict[str, Any]:
    """Analyze pattern field topology and create navigable space.
    
    Args:
        patterns: List of pattern strings to analyze
        
    Returns:
        A dictionary containing the field analysis results
    """
    # Calculate resonance matrix
    n = len(patterns)
    resonance_matrix = np.zeros((n, n))
    
    # First gather metadata for each pattern
    pattern_metadata = []
    for i, pattern in enumerate(patterns):
        # Get additional pattern information
        pattern_info = await self.pattern_evolution.get_pattern_info(pattern)
        pattern_metadata.append({
            "pattern": pattern,
            "domain": pattern_info.get("domain", "unknown"),
            "keywords": pattern_info.get("keywords", []),
            "strength": pattern_info.get("strength", 0.5)
        })
    
    # Calculate resonance for all pattern pairs
    for i in range(n):
        for j in range(i, n):
            # Calculate tonic-harmonic resonance
            resonance = self.harmonic_calculator.calculate_resonance(
                patterns[i], patterns[j]
            )
            resonance_matrix[i, j] = resonance
            resonance_matrix[j, i] = resonance  # Symmetric
    
    # Analyze field topology
    self.current_field_analysis = self.field_navigator.set_field(
        resonance_matrix, pattern_metadata
    )
    
    # Return field analysis
    return self.current_field_analysis


def get_pattern_coordinates(self, pattern: str) -> Dict[str, Any]:
    """Get coordinates and position in the navigable field for a pattern.
    
    Args:
        pattern: The pattern string to get coordinates for
        
    Returns:
        A dictionary containing coordinates and nearest center information
    """
    if not self.current_field_analysis:
        return {"coordinates": [0.0, 0.0, 0.0], "nearest_center": None, "pattern_index": -1}
    
    # Find pattern index
    pattern_index = -1
    for i, meta in enumerate(self.field_navigator.pattern_metadata):
        if meta["pattern"] == pattern:
            pattern_index = i
            break
    
    if pattern_index == -1:
        return {"coordinates": [0.0, 0.0, 0.0], "nearest_center": None, "pattern_index": -1}
    
    # Get coordinates in 3D space
    coordinates = self.field_navigator.get_navigation_coordinates(
        pattern_index, dimensions=3
    )
    
    # Find nearest density center
    nearest_center = self.field_navigator.find_nearest_density_center(pattern_index)
    
    return {
        "coordinates": coordinates,
        "nearest_center": nearest_center,
        "pattern_index": pattern_index
    }


async def find_pattern_path(self, start_pattern: str, end_pattern: str) -> List[str]:
    """Find a path between two patterns in the navigable field.
    
    Args:
        start_pattern: The starting pattern
        end_pattern: The destination pattern
        
    Returns:
        A list of pattern strings representing the path
    """
    if not self.current_field_analysis:
        return []
    
    # Get pattern indices
    start_idx = -1
    end_idx = -1
    
    for i, meta in enumerate(self.field_navigator.pattern_metadata):
        if meta["pattern"] == start_pattern:
            start_idx = i
        if meta["pattern"] == end_pattern:
            end_idx = i
            
    if start_idx == -1 or end_idx == -1:
        return []
    
    # Find path indices
    path_indices = self.field_navigator.find_paths(start_idx, end_idx)
    
    # Convert to pattern strings
    path_patterns = []
    for idx in path_indices:
        if 0 <= idx < len(self.field_navigator.pattern_metadata):
            path_patterns.append(self.field_navigator.pattern_metadata[idx]["pattern"])
    
    return path_patterns


def find_exploration_regions(self) -> List[Dict[str, Any]]:
    """Identify promising regions for exploration in the pattern field.
    
    Returns:
        A list of dictionaries describing interesting regions to explore
    """
    if not self.current_field_analysis:
        return []
    
    regions = []
    
    # 1. High-density regions with unexplored potential
    density_centers = self.current_field_analysis["density"]["density_centers"]
    for center in density_centers[:3]:  # Top 3 density centers
        metadata = self.field_navigator.pattern_metadata[center["index"]]
        regions.append({
            "type": "density_center",
            "center_pattern": metadata["pattern"],
            "importance": center["density"],
            "coordinates": self.field_navigator.get_navigation_coordinates(center["index"], 3),
            "related_patterns": self._get_related_patterns(center["index"], 3)
        })
    
    # 2. Boundary regions (high gradient areas)
    flow = self.current_field_analysis["flow"]
    if flow["flow_channels"]:
        for channel in flow["flow_channels"][:2]:  # Top 2 flow channels
            start_idx = channel[0][0]  # First point in channel
            end_idx = channel[-1][0]   # Last point in channel
            
            if start_idx < len(self.field_navigator.pattern_metadata) and end_idx < len(self.field_navigator.pattern_metadata):
                regions.append({
                    "type": "boundary",
                    "start_pattern": self.field_navigator.pattern_metadata[start_idx]["pattern"],
                    "end_pattern": self.field_navigator.pattern_metadata[end_idx]["pattern"],
                    "gradient_strength": flow["avg_gradient"],
                    "path": self.field_navigator.find_paths(start_idx, end_idx, "gradient")
                })
    
    # 3. Dimensional extremes (patterns that strongly represent a dimension)
    topology = self.current_field_analysis["topology"]
    for dim_idx, dimension in enumerate(topology["principal_dimensions"][:2]):  # Top 2 dimensions
        eigenvector = dimension["eigenvector"]
        
        # Find patterns at extremes of this dimension
        max_val = -float('inf')
        min_val = float('inf')
        max_idx = 0
        min_idx = 0
        
        for i, val in enumerate(eigenvector):
            if val > max_val:
                max_val = val
                max_idx = i
            if val < min_val:
                min_val = val
                min_idx = i
        
        if max_idx < len(self.field_navigator.pattern_metadata) and min_idx < len(self.field_navigator.pattern_metadata):
            regions.append({
                "type": "dimension",
                "dimension": dim_idx,
                "explained_variance": dimension["explained_variance"],
                "positive_extreme": self.field_navigator.pattern_metadata[max_idx]["pattern"],
                "negative_extreme": self.field_navigator.pattern_metadata[min_idx]["pattern"],
                "span_path": self.field_navigator.find_paths(min_idx, max_idx, "eigenvector")
            })
    
    return regions


def _get_related_patterns(self, center_idx: int, count: int) -> List[str]:
    """Get patterns related to a center pattern.
    
    Args:
        center_idx: Index of the center pattern
        count: Number of related patterns to retrieve
        
    Returns:
        A list of related pattern strings
    """
    if not self.current_field_analysis or center_idx >= len(self.field_navigator.pattern_metadata):
        return []
    
    # Simple implementation - find patterns with highest resonance
    matrix = np.array(self.current_field_analysis["topology"]["dimension_strengths"])
    if matrix.size == 0:
        return []
    
    related_indices = np.argsort(matrix[center_idx])[-count-1:-1]
    
    return [
        self.field_navigator.pattern_metadata[idx]["pattern"]
        for idx in related_indices
        if idx < len(self.field_navigator.pattern_metadata) and idx != center_idx
    ]