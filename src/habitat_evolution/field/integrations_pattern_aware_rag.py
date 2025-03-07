# In pattern_aware_rag.py

# Add to constructor
def __init__(self, pattern_evolution_service, field_state_service, ...):
    # ... existing initialization
    
    # Initialize field topology components
    self.field_analyzer = TopologicalFieldAnalyzer()
    self.field_navigator = FieldNavigator(self.field_analyzer)
    self.current_field_analysis = None
    
# Add or modify method for field analysis
async def analyze_pattern_field(self, patterns: List[str]) -> Dict[str, Any]:
    """Analyze pattern field topology and create navigable space."""
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

# Add method to get navigable coordinates for patterns
def get_pattern_coordinates(self, pattern: str) -> Dict[str, Any]:
    """Get coordinates and position in the navigable field for a pattern."""
    if not self.current_field_analysis:
        return {"coordinates": [0.0, 0.0, 0.0], "nearest_center": None}
    
    # Find pattern index
    pattern_index = -1
    for i, meta in enumerate(self.field_navigator.pattern_metadata):
        if meta["pattern"] == pattern:
            pattern_index = i
            break
    
    if pattern_index == -1:
        return {"coordinates": [0.0, 0.0, 0.0], "nearest_center": None}
    
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

# Continuing in pattern_aware_rag.py

# Add method to find paths between patterns
async def find_pattern_path(self, start_pattern: str, end_pattern: str) -> List[str]:
    """Find a path between two patterns in the navigable field."""
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

# Update existing process_with_patterns method to include field topology
async def process_with_patterns(self, query: str, context: Optional[Dict[str, Any]] = None):
    # Existing method code...
    
    # After extracting patterns and before generating response
    # Create and analyze the pattern field
    all_patterns = query_patterns + retrieval_patterns + augmentation_patterns
    field_analysis = await self.analyze_pattern_field(all_patterns)
    
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
        position = self.get_pattern_coordinates(pattern)
        if position["pattern_index"] != -1:
            pattern_positions[pattern] = position["coordinates"]
    
    response["pattern_positions"] = pattern_positions
    
    # Return updated response
    return response

# Add method to find exploration regions
def find_exploration_regions(self) -> List[Dict[str, Any]]:
    """Identify promising regions for exploration in the pattern field."""
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
    """Get patterns related to a center pattern."""
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