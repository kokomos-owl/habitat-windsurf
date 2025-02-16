"""
Vector-based attention monitoring system for pattern evolution.

Combines attention mechanisms with vector space analysis for:
- Edge detection
- Stability analysis
- Density patterns
- Turbulence detection
- Drift monitoring

Includes rich logging for metrics and agentic navigation.
"""

import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
from habitat_evolution.core.pattern.attention import AttentionFilter
from .metrics_logger import MetricsLogger

@dataclass
class VectorSpaceMetrics:
    """Metrics for vector space analysis."""
    edge_strength: float
    stability_score: float
    local_density: float
    turbulence_level: float
    drift_velocity: np.ndarray
    attention_weight: float
    timestamp: datetime

class VectorAttentionMonitor:
    """Monitors vector space dynamics with attention-weighted analysis."""
    
    def __init__(
        self,
        attention_filter: AttentionFilter,
        window_size: int = 10,
        edge_threshold: float = 0.3,
        stability_threshold: float = 0.7,
        density_radius: float = 0.1,
        logger: Optional[logging.Logger] = None
    ):
        self.attention_filter = attention_filter
        self.window_size = window_size
        self.edge_threshold = edge_threshold
        self.stability_threshold = stability_threshold
        self.density_radius = density_radius
        
        # Initialize logging
        self.logger = logger or logging.getLogger(__name__)
        self.metrics_logger = MetricsLogger("vector_attention")
        
        # History buffers
        self.vector_history: List[np.ndarray] = []
        self.metrics_history: List[VectorSpaceMetrics] = []
    
    def process_vector(self, vector: np.ndarray, context: Dict) -> VectorSpaceMetrics:
        """Process a new vector with attention-weighted analysis."""
        # Apply attention filter
        attention_weight = self.attention_filter.evaluate(context)
        
        # Calculate metrics before updating history
        metrics = self._calculate_metrics(vector, attention_weight)
        
        # Update history after metric calculation
        self.vector_history.append(vector)
        if len(self.vector_history) > self.window_size:
            self.vector_history.pop(0)
        self.metrics_history.append(metrics)
        
        # Log metrics for monitoring
        self._log_metrics(metrics)
        
        return metrics
    
    def _calculate_metrics(self, current_vector: np.ndarray, attention_weight: float) -> VectorSpaceMetrics:
        """Calculate comprehensive vector space metrics."""
        # Edge detection
        edge_strength = self._detect_edge(current_vector)
        
        # Stability analysis
        stability_score = self._analyze_stability()
        
        # Density calculation
        local_density = self._calculate_density(current_vector)
        
        # Turbulence detection
        turbulence_level = self._detect_turbulence()
        
        # Drift analysis
        drift_velocity = self._analyze_drift()
        
        return VectorSpaceMetrics(
            edge_strength=edge_strength,
            stability_score=stability_score,
            local_density=local_density,
            turbulence_level=turbulence_level,
            drift_velocity=drift_velocity,
            attention_weight=attention_weight,
            timestamp=datetime.now()
        )
    
    def _detect_edge(self, vector: np.ndarray) -> float:
        """Detect semantic boundaries between patterns."""
        if not self.vector_history:  # First vector
            return 0.0
        
        # Simple cosine distance is sufficient for detecting semantic shifts
        prev_vector = self.vector_history[-1]
        cosine_sim = np.dot(vector, prev_vector) / (
            np.linalg.norm(vector) * np.linalg.norm(prev_vector)
        )
        return 1.0 - cosine_sim  # Higher value = stronger edge
    
    def _compute_local_structure(self, vector: np.ndarray, neighborhood: np.ndarray, k: int = 3) -> np.ndarray:
        """Compute local topological structure around a vector.
        
        Uses k-nearest neighbors to establish local relationships that should
        be preserved as the space evolves.
        
        Args:
            vector: The central vector to analyze
            neighborhood: Array of vectors in the local neighborhood
            k: Number of nearest neighbors to consider
            
        Returns:
            Array of relative positions to k-nearest neighbors
        """
        if len(neighborhood) < 2:
            # Not enough neighbors for topology
            return np.array([np.zeros_like(vector)])
            
        # Adjust k if we don't have enough neighbors
        effective_k = min(k, len(neighborhood) - 1)
        
        # Calculate distances to all neighbors
        distances = np.array([np.linalg.norm(vector - v) for v in neighborhood])
        # Get k nearest indices (excluding self)
        nearest_idx = np.argpartition(distances, effective_k)[:effective_k]
        nearest_idx = nearest_idx[nearest_idx != np.where((neighborhood == vector).all(axis=1))[0][0]]
        # Return relative positions to preserve
        return np.array([neighborhood[i] - vector for i in nearest_idx])
    
    def _measure_topology_preservation(self, local_structures: np.ndarray) -> float:
        """Measure how well local topological relationships are preserved.
        
        Compares consecutive local structures to detect changes in neighborhood
        relationships that might indicate meaning-structure breakdown.
        
        Args:
            local_structures: Array of local neighborhood structures
            
        Returns:
            Float between 0 and 1 indicating topology preservation
        """
        if len(local_structures) < 2:
            return 1.0
            
        # Compare consecutive local structures
        preservation_scores = []
        for s1, s2 in zip(local_structures[:-1], local_structures[1:]):
            # Calculate cosine similarity between corresponding vectors
            similarities = [np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                          for v1, v2 in zip(s1, s2)]
            # Convert to angle changes (1 = identical, 0 = orthogonal)
            preservation_scores.append(np.mean(np.abs(similarities)))
            
        return np.mean(preservation_scores) if preservation_scores else 1.0
    
    def _compute_basis_drift(self, vectors: np.ndarray) -> float:
        """Compute rate of change in the effective basis vectors.
        
        Tracks how quickly the principal components of the vector space
        are evolving, which indicates potential meaning-structure drift.
        
        Args:
            vectors: Array of vectors to analyze
            
        Returns:
            Float between 0 and 1 indicating basis drift rate
        """
        if len(vectors) < 3:
            return 0.0
            
        # For small sets, use simpler direction change metric
        if len(vectors) < 5:
            deltas = np.diff(vectors, axis=0)
            angles = np.array([np.abs(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))
                             for d1, d2 in zip(deltas[:-1], deltas[1:])])
            return 1.0 - np.mean(angles)
            
        # For larger sets, use PCA-based approach
        window_size = min(5, len(vectors) // 2)
        basis_vectors = []
        
        for i in range(len(vectors) - window_size):
            window = vectors[i:i+window_size]
            # Ensure we have enough samples and dimensions
            if window.shape[0] <= 1 or window.shape[1] <= 1:
                continue
                
            # Center the data
            centered = window - np.mean(window, axis=0)
            
            try:
                # Get principal components
                _, v = np.linalg.eigh(np.cov(centered.T))
                basis_vectors.append(v[:, -1])  # Most significant component
            except np.linalg.LinAlgError:
                # Fall back to direction of maximum variance
                direction = np.argmax(np.var(centered, axis=0))
                basis = np.zeros(centered.shape[1])
                basis[direction] = 1.0
                basis_vectors.append(basis)
            
        # Calculate average change in basis vectors
        if len(basis_vectors) < 2:
            return 0.0
            
        basis_changes = [np.abs(1 - np.abs(np.dot(b1, b2))) 
                        for b1, b2 in zip(basis_vectors[:-1], basis_vectors[1:])]
        return np.mean(basis_changes)

    def _analyze_stability(self) -> float:
        """Analyze stability using adaptive, window-size-dependent measures.
        
        Uses two different strategies based on window size:
        
        Small Windows (< 4 vectors):
        - Maximum magnitude detection with 2x sensitivity
        - Direction change analysis
        - 70/30 weight between magnitude and angle stability
        - Squared final score for non-linear response
        
        Large Windows (≥ 4 vectors):
        1. Local topology preservation (50%): How well neighborhood relationships persist
        2. Basis stability (30%): How quickly the underlying basis vectors evolve
        3. Local coherence (20%): Pattern consistency with 10x variance sensitivity
        
        Returns:
            Float between 0 and 1 indicating overall stability. Lower scores indicate
            potential semantic shifts or meaning-structure breakdown.
        """
        if len(self.vector_history) < 2:
            return 1.0
            
        vectors = np.array(self.vector_history)
        
        # For very small sets, use simpler metrics
        if len(vectors) < 4:
            # Measure vector changes with emphasis on large shifts
            deltas = np.diff(vectors, axis=0)
            magnitudes = np.linalg.norm(deltas, axis=1)
            max_magnitude = np.max(magnitudes)
            
            # Check for orthogonal shifts
            angles = np.array([np.abs(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))
                             if np.linalg.norm(d1) * np.linalg.norm(d2) > 0 else 0.0
                             for d1, d2 in zip(deltas[:-1], deltas[1:])])
            
            # Combine magnitude and direction changes
            # More sensitive to large magnitudes
            magnitude_stability = 1.0 - np.clip(2.0 * max_magnitude, 0, 1)
            angle_stability = np.mean(angles) if len(angles) > 0 else 1.0
            
            # Emphasize large changes
            stability = 0.7 * magnitude_stability + 0.3 * angle_stability
            return np.power(stability, 2)  # Square to emphasize instability
        
        # 1. Local Topology Preservation
        local_structures = np.array([self._compute_local_structure(v, vectors) 
                                   for v in vectors])
        topology_score = self._measure_topology_preservation(local_structures)
        
        # 2. Basis Evolution Rate
        basis_drift = self._compute_basis_drift(vectors)
        basis_stability = 1.0 - basis_drift
        
        # 3. Local Pattern Coherence (using original variance measure)
        variance = np.var(vectors, axis=0).mean()
        local_coherence = 1.0 / (1.0 + 10.0 * variance)  # Even more sensitive to variance
        
        # Weighted combination with emphasis on topology preservation
        stability = (
            0.5 * topology_score +    # Structure preservation
            0.3 * basis_stability +   # Basis stability
            0.2 * local_coherence    # Local coherence
        )
        
        # Square the result to emphasize instability
        return np.power(stability, 2)
        
        # Log detailed stability components
        self.logger.debug(
            f"Stability Analysis | Topology: {topology_score:.3f} | "
            f"Basis: {basis_stability:.3f} | Coherence: {local_coherence:.3f}"
        )
        
        return stability
    
    def _calculate_density(self, vector: np.ndarray) -> float:
        """Calculate density for pattern identification using adaptive radius.
        
        Uses three key components:
        1. Adaptive radius based on local statistics
        2. Attention-weighted neighbor contributions
        3. Temporal decay for recent relevance
        
        Returns:
            Float between 0 and 1 indicating local pattern density
        """
        if len(self.vector_history) < 2:
            return 0.0
        
        vectors = np.array(self.vector_history)
        distances = np.linalg.norm(vectors - vector, axis=1)
        
        # Adaptive radius based on local statistics
        local_std = np.std(distances)
        adaptive_radius = max(self.density_radius, 0.5 * local_std)
        
        # Get attention weights and temporal weights
        attention_weights = np.array([m.attention_weight for m in self.metrics_history])
        time_deltas = [(datetime.now() - m.timestamp).total_seconds() 
                      for m in self.metrics_history]
        temporal_weights = np.exp(-np.array(time_deltas) / 3600)  # 1-hour half-life
        
        # Combined weights
        weights = attention_weights * temporal_weights
        weights = weights / np.sum(weights)  # Normalize
        
        # Weighted density calculation
        density_contributions = np.exp(-distances / adaptive_radius)
        weighted_density = np.sum(density_contributions * weights)
        
        return weighted_density
    
    def _detect_turbulence(self) -> float:
        """Detect turbulence for back pressure adjustment."""
        if len(self.vector_history) < 3:
            return 0.0
        
        # Rate of change in vector directions
        vectors = np.array(self.vector_history)
        deltas = np.diff(vectors, axis=0)
        return float(np.std(np.linalg.norm(deltas, axis=1)))
    
    def _analyze_drift(self) -> np.ndarray:
        """Track pattern evolution direction."""
        if len(self.vector_history) < 2:
            # For first vector, return zero vector of same dimension
            return np.zeros_like(self.vector_history[0]) if self.vector_history else np.zeros(3)
        
        # Simple moving average of changes
        vectors = np.array(self.vector_history)
        return np.mean(np.diff(vectors, axis=0), axis=0)
    
    def _log_metrics(self, metrics: VectorSpaceMetrics):
        """Log metrics for monitoring and navigation."""
        # Calculate processing time
        processing_time = (datetime.now() - metrics.timestamp).total_seconds()
        
        log_data = {
            "timestamp": metrics.timestamp.isoformat(),
            "edge_strength": metrics.edge_strength,
            "stability_score": metrics.stability_score,
            "local_density": metrics.local_density,
            "turbulence_level": metrics.turbulence_level,
            "drift_magnitude": float(np.linalg.norm(metrics.drift_velocity)),
            "attention_weight": metrics.attention_weight,
            "processing_time": processing_time,
            "history_size": len(self.vector_history)
        }
        
        # Log to metrics system
        self.metrics_logger.log_metrics("vector_space", log_data)
        
        # Log system state
        if processing_time > 0.1:  # More than 100ms
            self.logger.warning(
                f"Slow processing detected: {processing_time:.3f}s | "
                f"History: {len(self.vector_history)} vectors | "
                f"Stability: {metrics.stability_score:.3f}"
            )
        
        # Log pattern events
        if metrics.edge_strength > self.edge_threshold:
            self.logger.info(
                f"Pattern boundary detected | Edge: {metrics.edge_strength:.3f} | "
                f"Stability: {metrics.stability_score:.3f} | "
                f"Density: {metrics.local_density:.3f}"
            )
        
        # Log system stress indicators
        if metrics.stability_score < self.stability_threshold:
            self.logger.warning(
                f"System stress - Low stability: {metrics.stability_score:.3f} | "
                f"Turbulence: {metrics.turbulence_level:.3f} | "
                f"Processing: {processing_time:.3f}s"
            )
        
        if metrics.turbulence_level > 0.5:
            self.logger.warning(
                f"System stress - High turbulence: {metrics.turbulence_level:.3f} | "
                f"Stability: {metrics.stability_score:.3f} | "
                f"Edge: {metrics.edge_strength:.3f}"
            )
