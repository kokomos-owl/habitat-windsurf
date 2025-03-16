"""FlowDynamicsAnalyzer for tracking energy flow and emergent patterns in tonic_harmonic fields.

This module provides functionality to analyze the dynamic flow of energy through
the tonic_harmonic field, identify density centers, and predict emergent patterns.
It integrates with the SemanticBoundaryDetector to analyze flow dynamics at transition zones.
"""

import numpy as np
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from scipy import stats
import networkx as nx
from copy import deepcopy

from habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer
from habitat_evolution.field.field_navigator import FieldNavigator
from habitat_evolution.field.semantic_boundary_detector import SemanticBoundaryDetector


class FlowDynamicsAnalyzer:
    """Analyzes flow dynamics in tonic_harmonic fields.
    
    This class tracks energy flow through the field, identifies density centers where
    patterns tend to accumulate, and predicts emergent patterns based on flow dynamics.
    It also integrates with the SemanticBoundaryDetector to analyze flow at transition zones.
    """
    
    def __init__(
        self,
        field_analyzer: TopologicalFieldAnalyzer,
        field_navigator: FieldNavigator,
        boundary_detector: Optional[SemanticBoundaryDetector] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the FlowDynamicsAnalyzer with dependencies and configuration.
        
        Args:
            field_analyzer: TopologicalFieldAnalyzer instance for field analysis
            field_navigator: FieldNavigator instance for field navigation
            boundary_detector: Optional SemanticBoundaryDetector for boundary analysis
            config: Configuration dictionary with the following optional parameters:
                - flow_threshold: Minimum flow magnitude to consider (default: 0.5)
                - density_threshold: Minimum density score for centers (default: 0.6)
                - emergence_probability_threshold: Minimum probability for emergent patterns (default: 0.7)
                - temporal_window_size: Size of temporal window for flow analysis (default: 5)
                - max_density_centers: Maximum number of density centers to identify (default: 10)
        """
        self.field_analyzer = field_analyzer
        self.field_navigator = field_navigator
        self.boundary_detector = boundary_detector
        
        # Default configuration
        default_config = {
            "flow_threshold": 0.5,
            "density_threshold": 0.6,
            "emergence_probability_threshold": 0.7,
            "temporal_window_size": 5,
            "max_density_centers": 10
        }
        
        # Apply user configuration over defaults
        self.config = default_config
        if config:
            self.config.update(config)
            
        # Initialize state
        self.flow_history = []
        self.current_field_data = None
        self.current_vectors = None
        self.current_metadata = None
        
    def analyze_flow_dynamics(
        self, 
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze flow dynamics in the field.
        
        Args:
            vectors: Pattern vectors to analyze
            metadata: Metadata for each pattern
            
        Returns:
            Dict containing energy flow data, density centers, emergent patterns, and flow metrics
        """
        # Store current data
        self.current_vectors = vectors
        self.current_metadata = metadata
        
        # Analyze field topology
        self.current_field_data = self.field_analyzer.analyze_field(vectors, metadata)
        self.field_navigator.set_field(vectors, metadata)
        
        # Calculate energy flow
        if "resonance_matrix" not in self.current_field_data:
            # If resonance_matrix is not in the field data, create it
            # Compute cosine similarity matrix
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            normalized_vectors = vectors / norms
            resonance_matrix = np.dot(normalized_vectors, normalized_vectors.T)
            self.current_field_data["resonance_matrix"] = resonance_matrix
            
        energy_flow = self._calculate_energy_flow(
            self.current_field_data["resonance_matrix"],
            metadata
        )
        
        # Identify density centers
        density_centers = self._identify_density_centers(
            energy_flow,
            self.current_field_data
        )
        
        # Predict emergent patterns
        emergent_patterns = self._predict_emergent_patterns(
            energy_flow,
            density_centers,
            vectors,
            metadata
        )
        
        # Calculate flow metrics
        flow_metrics = self._calculate_flow_metrics(
            energy_flow,
            density_centers,
            self.current_field_data
        )
        
        # Compile results
        flow_dynamics = {
            "energy_flow": energy_flow,
            "density_centers": density_centers,
            "emergent_patterns": emergent_patterns,
            "flow_metrics": flow_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update history
        self.flow_history.append(flow_dynamics)
        
        return flow_dynamics
    
    def analyze_boundary_flow_dynamics(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze flow dynamics specifically at semantic boundaries.
        
        Args:
            vectors: Pattern vectors to analyze
            metadata: Metadata for each pattern
            
        Returns:
            Dict containing boundary flow data, boundary density centers, and boundary emergent patterns
        """
        if not self.boundary_detector:
            raise ValueError("SemanticBoundaryDetector is required for boundary flow analysis")
        
        # Get standard flow dynamics
        flow_dynamics = self.analyze_flow_dynamics(vectors, metadata)
        
        # Get transition patterns from boundary detector
        transition_patterns = self.boundary_detector.detect_transition_patterns(vectors, metadata)
        
        # Extract transition pattern indices
        transition_indices = [p["pattern_idx"] for p in transition_patterns]
        
        # Filter flow data to boundary regions
        boundary_flow_vectors = {}
        for idx in transition_indices:
            if idx < len(flow_dynamics["energy_flow"]["flow_vectors"]):
                boundary_flow_vectors[idx] = flow_dynamics["energy_flow"]["flow_vectors"][idx]
        
        # Filter density centers near boundaries
        boundary_density_centers = []
        for center in flow_dynamics["density_centers"]:
            contributing_patterns = center["contributing_patterns"]
            if any(idx in transition_indices for idx in contributing_patterns):
                boundary_density_centers.append(center)
        
        # Filter emergent patterns near boundaries
        boundary_emergent_patterns = []
        for pattern in flow_dynamics["emergent_patterns"]:
            contributing_centers = pattern["contributing_centers"]
            center_indices = [c["center_idx"] for c in contributing_centers]
            if any(idx in transition_indices for idx in center_indices):
                boundary_emergent_patterns.append(pattern)
        
        # Compile boundary flow results
        boundary_flow = {
            "boundary_flow_vectors": boundary_flow_vectors,
            "boundary_density_centers": boundary_density_centers,
            "boundary_emergent_patterns": boundary_emergent_patterns,
            "transition_patterns": transition_patterns,
            "timestamp": datetime.now().isoformat()
        }
        
        return boundary_flow
    
    def _calculate_energy_flow(
        self,
        resonance_matrix: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate energy flow through the field based on resonance relationships.
        
        Args:
            resonance_matrix: Matrix of resonance relationships between patterns
            metadata: Metadata for each pattern
            
        Returns:
            Dict containing flow vectors, flow magnitude, source patterns, and sink patterns
        """
        n_patterns = resonance_matrix.shape[0]
        
        # Calculate flow vectors (direction and magnitude of energy flow)
        flow_vectors = np.zeros((n_patterns, n_patterns))
        for i in range(n_patterns):
            for j in range(n_patterns):
                if i != j:
                    # Flow from i to j is proportional to resonance and inversely proportional to distance
                    flow_vectors[i, j] = resonance_matrix[i, j]
        
        # Normalize flow vectors
        row_sums = flow_vectors.sum(axis=1, keepdims=True)
        flow_vectors = np.divide(flow_vectors, row_sums, out=np.zeros_like(flow_vectors), where=row_sums!=0)
        
        # Calculate net flow magnitude for each pattern
        outflow = flow_vectors.sum(axis=1)
        inflow = flow_vectors.sum(axis=0)
        flow_magnitude = inflow - outflow
        
        # Identify source patterns (net outflow)
        source_patterns = [i for i in range(n_patterns) if flow_magnitude[i] < -self.config["flow_threshold"]]
        
        # Identify sink patterns (net inflow)
        sink_patterns = [i for i in range(n_patterns) if flow_magnitude[i] > self.config["flow_threshold"]]
        
        # Compile flow data
        energy_flow = {
            "flow_vectors": flow_vectors,
            "flow_magnitude": flow_magnitude,
            "source_patterns": source_patterns,
            "sink_patterns": sink_patterns
        }
        
        return energy_flow
    
    def _identify_density_centers(
        self,
        energy_flow: Dict[str, Any],
        field_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify density centers where patterns tend to accumulate.
        
        Args:
            energy_flow: Energy flow data from _calculate_energy_flow
            field_data: Field topology data from TopologicalFieldAnalyzer
            
        Returns:
            List of density centers with their properties
        """
        flow_magnitude = energy_flow["flow_magnitude"]
        sink_patterns = energy_flow["sink_patterns"]
        
        # Start with sink patterns as potential density centers
        potential_centers = sink_patterns.copy()
        
        # Add community centers from field data if available
        if "community_centers" in field_data:
            for community, center_idx in field_data["community_centers"].items():
                if center_idx not in potential_centers:
                    potential_centers.append(center_idx)
        
        # Calculate density score for each potential center
        density_centers = []
        for center_idx in potential_centers:
            # Ensure center_idx is within bounds
            if center_idx >= len(flow_magnitude):
                continue
                
            # Density score is based on flow magnitude and community membership
            density_score = 0.5 + 0.5 * (
                flow_magnitude[center_idx] / max(abs(flow_magnitude.max()), abs(flow_magnitude.min()))
            )
            
            # Find patterns that contribute to this center
            contributing_patterns = []
            flow_vectors = energy_flow["flow_vectors"]
            for i in range(len(flow_magnitude)):
                # Check if indices are within bounds
                if i < flow_vectors.shape[0] and center_idx < flow_vectors.shape[1]:
                    if flow_vectors[i, center_idx] > self.config["flow_threshold"]:
                        contributing_patterns.append(i)
            
            # Calculate influence radius based on contributing patterns
            influence_radius = 0.3  # Default
            if contributing_patterns:
                # Use standard deviation of resonance as influence radius
                if "resonance_matrix" in field_data:
                    resonance_values = []
                    for p in contributing_patterns:
                        resonance_values.append(field_data["resonance_matrix"][center_idx, p])
                    if resonance_values:
                        influence_radius = max(0.1, np.std(resonance_values))
            
            # Only include centers above threshold
            if density_score >= self.config["density_threshold"]:
                density_centers.append({
                    "center_idx": center_idx,
                    "density_score": density_score,
                    "influence_radius": influence_radius,
                    "contributing_patterns": contributing_patterns
                })
        
        # Sort by density score and limit to max centers
        density_centers.sort(key=lambda x: x["density_score"], reverse=True)
        return density_centers[:self.config["max_density_centers"]]
    
    def _predict_emergent_patterns(
        self,
        energy_flow: Dict[str, Any],
        density_centers: List[Dict[str, Any]],
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Predict emergent patterns based on flow dynamics and density centers.
        
        Args:
            energy_flow: Energy flow data from _calculate_energy_flow
            density_centers: Density centers from _identify_density_centers
            vectors: Pattern vectors
            metadata: Metadata for each pattern
            
        Returns:
            List of predicted emergent patterns with their properties
        """
        emergent_patterns = []
        
        # No predictions if no density centers
        if not density_centers:
            return emergent_patterns
        
        # For each pair of density centers, predict potential emergent patterns
        for i, center1 in enumerate(density_centers):
            for j, center2 in enumerate(density_centers):
                if i >= j:  # Skip self-pairs and duplicates
                    continue
                
                # Calculate midpoint between centers as potential emergence point
                center1_idx = center1["center_idx"]
                center2_idx = center2["center_idx"]
                
                # Skip if centers are too far apart
                if self.current_field_data is not None and "resonance_matrix" in self.current_field_data:
                    # Check if indices are within bounds
                    resonance_matrix = self.current_field_data["resonance_matrix"]
                    if (center1_idx < resonance_matrix.shape[0] and 
                        center2_idx < resonance_matrix.shape[1]):
                        resonance = resonance_matrix[center1_idx, center2_idx]
                        if resonance < 0.3:  # Skip if resonance is too low
                            continue
                
                # Calculate emergence probability based on center properties
                probability = (center1["density_score"] + center2["density_score"]) / 2
                
                # Only include predictions above threshold
                if probability >= self.config["emergence_probability_threshold"]:
                    # Estimate vector for emergent pattern (weighted average of contributing patterns)
                    contributing_patterns = list(set(center1["contributing_patterns"] + center2["contributing_patterns"]))
                    
                    if contributing_patterns:
                        pattern_vectors = vectors[contributing_patterns]
                        weights = np.array([
                            energy_flow["flow_magnitude"][idx] for idx in contributing_patterns
                        ])
                        weights = np.abs(weights)  # Use absolute magnitude
                        weights = weights / weights.sum() if weights.sum() != 0 else np.ones_like(weights) / len(weights)
                        
                        estimated_vector = np.average(pattern_vectors, axis=0, weights=weights)
                    else:
                        # If no contributing patterns, use average of center vectors
                        estimated_vector = (vectors[center1_idx] + vectors[center2_idx]) / 2
                    
                    emergent_patterns.append({
                        "emergence_point": f"between_{center1_idx}_{center2_idx}",
                        "probability": probability,
                        "contributing_centers": [center1, center2],
                        "estimated_vector": estimated_vector
                    })
        
        # Sort by probability
        emergent_patterns.sort(key=lambda x: x["probability"], reverse=True)
        
        return emergent_patterns
    
    def _calculate_flow_metrics(
        self,
        energy_flow: Dict[str, Any],
        density_centers: List[Dict[str, Any]],
        field_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate metrics that characterize the flow dynamics.
        
        Args:
            energy_flow: Energy flow data from _calculate_energy_flow
            density_centers: Density centers from _identify_density_centers
            field_data: Field topology data from TopologicalFieldAnalyzer
            
        Returns:
            Dict containing flow coherence, stability, density distribution, and emergence potential
        """
        # Calculate flow coherence (how aligned the flow is)
        flow_vectors = energy_flow["flow_vectors"]
        flow_coherence = 0.0
        
        if flow_vectors.size > 0:
            # Coherence is measured by consistency of flow directions
            flow_directions = flow_vectors / np.linalg.norm(flow_vectors, axis=1, keepdims=True)
            flow_directions = np.nan_to_num(flow_directions)  # Handle division by zero
            
            # Calculate average cosine similarity between flow directions
            n_patterns = flow_directions.shape[0]
            total_similarity = 0.0
            count = 0
            
            for i in range(n_patterns):
                for j in range(i+1, n_patterns):
                    similarity = np.dot(flow_directions[i], flow_directions[j])
                    total_similarity += similarity
                    count += 1
            
            flow_coherence = total_similarity / count if count > 0 else 0.0
            # Normalize to [0, 1]
            flow_coherence = (flow_coherence + 1) / 2
        
        # Calculate flow stability (how consistent the flow is over time)
        flow_stability = 0.8  # Default value
        if len(self.flow_history) > 1:
            # Compare current flow with previous flow
            prev_flow = self.flow_history[-1]["energy_flow"]["flow_vectors"]
            curr_flow = flow_vectors
            
            if prev_flow.shape == curr_flow.shape:
                # Stability is measured by correlation between consecutive flow patterns
                prev_flat = prev_flow.flatten()
                curr_flat = curr_flow.flatten()
                
                correlation = np.corrcoef(prev_flat, curr_flat)[0, 1]
                flow_stability = (correlation + 1) / 2  # Normalize to [0, 1]
        
        # Calculate density distribution (how evenly distributed the density centers are)
        density_distribution = 0.5  # Default value
        if density_centers:
            # Get density scores
            density_scores = [center["density_score"] for center in density_centers]
            
            # Calculate Gini coefficient (measure of inequality)
            density_scores = np.sort(density_scores)
            n = len(density_scores)
            index = np.arange(1, n+1)
            gini = (np.sum((2 * index - n - 1) * density_scores)) / (n * np.sum(density_scores))
            
            # Invert so higher values mean more even distribution
            density_distribution = 1 - gini
        
        # Calculate emergence potential (likelihood of new patterns emerging)
        emergence_potential = 0.0
        if "eigenvalues" in field_data:
            # Use eigenvalue distribution as indicator of emergence potential
            eigenvalues = field_data["eigenvalues"]
            if len(eigenvalues) > 0:
                # Convert to numpy array if it's a list
                if isinstance(eigenvalues, list):
                    eigenvalues = np.array(eigenvalues)
                
                # Normalize eigenvalues
                eigenvalues = eigenvalues / eigenvalues.sum() if eigenvalues.sum() != 0 else eigenvalues
                
                # Calculate entropy of eigenvalue distribution
                entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
                max_entropy = np.log2(len(eigenvalues))
                
                # Normalize entropy to [0, 1]
                emergence_potential = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Compile metrics
        flow_metrics = {
            "flow_coherence": flow_coherence,
            "flow_stability": flow_stability,
            "density_distribution": density_distribution,
            "emergence_potential": emergence_potential
        }
        
        return flow_metrics
