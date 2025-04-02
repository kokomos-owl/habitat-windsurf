"""
Topology detector for identifying topological features in the semantic landscape.

This module provides functionality to analyze pattern evolution histories and
learning window interactions to detect emergent topological features such as
frequency domains, boundaries, resonance points, and field dynamics.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from datetime import datetime, timedelta
import logging
from scipy import signal, fft
from sklearn.cluster import DBSCAN

from src.habitat_evolution.pattern_aware_rag.topology.models import (
    FrequencyDomain, Boundary, ResonancePoint, FieldMetrics, TopologyState
)

logger = logging.getLogger(__name__)


class TopologyDetector:
    """Detects topological features in the semantic landscape."""
    
    def __init__(self, 
                 min_samples: int = 5, 
                 frequency_resolution: float = 0.01,
                 boundary_threshold: float = 0.2,
                 resonance_threshold: float = 0.7):
        """
        Initialize the topology detector.
        
        Args:
            min_samples: Minimum samples required for frequency analysis
            frequency_resolution: Resolution for frequency detection
            boundary_threshold: Threshold for boundary detection
            resonance_threshold: Threshold for resonance point detection
        """
        self.min_samples = min_samples
        self.frequency_resolution = frequency_resolution
        self.boundary_threshold = boundary_threshold
        self.resonance_threshold = resonance_threshold
    
    def detect_frequency_domains(self, pattern_histories: Dict[str, List[Dict[str, Any]]]) -> Dict[str, FrequencyDomain]:
        """
        Detect frequency domains from pattern evolution histories.
        
        Args:
            pattern_histories: Dictionary mapping pattern IDs to their evolution histories
            
        Returns:
            Dictionary of detected frequency domains
        """
        logger.info(f"Detecting frequency domains from {len(pattern_histories)} patterns")
        
        # Extract time series of harmonic values for each pattern
        harmonic_series = {}
        timestamps = {}
        
        for pattern_id, history in pattern_histories.items():
            if len(history) < self.min_samples:
                logger.debug(f"Skipping pattern {pattern_id} with insufficient samples")
                continue
                
            # Extract harmonic values and timestamps
            harmonic_values = []
            time_points = []
            
            for event in sorted(history, key=lambda x: x.get('timestamp', '')):
                if 'harmonic_value' in event and 'timestamp' in event:
                    harmonic_values.append(event['harmonic_value'])
                    time_points.append(event['timestamp'])
            
            if len(harmonic_values) >= self.min_samples:
                harmonic_series[pattern_id] = harmonic_values
                timestamps[pattern_id] = time_points
        
        # Perform frequency analysis on each pattern
        pattern_frequencies = {}
        
        for pattern_id, values in harmonic_series.items():
            # Convert to numpy array
            signal_values = np.array(values)
            
            # Perform FFT
            fft_result = np.abs(fft.fft(signal_values))
            freqs = fft.fftfreq(len(signal_values), d=self.frequency_resolution)
            
            # Find dominant frequency (excluding DC component)
            idx = np.argmax(fft_result[1:]) + 1
            dominant_freq = freqs[idx]
            
            # Calculate bandwidth (standard deviation of frequency components)
            power_spectrum = fft_result**2
            total_power = np.sum(power_spectrum)
            normalized_power = power_spectrum / total_power
            bandwidth = np.sqrt(np.sum(normalized_power * (freqs - dominant_freq)**2))
            
            pattern_frequencies[pattern_id] = {
                'dominant_frequency': abs(dominant_freq),
                'bandwidth': bandwidth,
                'power': fft_result[idx]
            }
        
        # Cluster patterns by frequency characteristics
        if len(pattern_frequencies) < 2:
            logger.warning("Insufficient patterns for clustering")
            return {}
            
        # Extract features for clustering
        features = []
        pattern_ids = []
        
        for pattern_id, freq_data in pattern_frequencies.items():
            features.append([
                freq_data['dominant_frequency'],
                freq_data['bandwidth']
            ])
            pattern_ids.append(pattern_id)
            
        features = np.array(features)
        
        # Normalize features
        features_normalized = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(features_normalized)
        labels = clustering.labels_
        
        # Create frequency domains from clusters
        domains = {}
        
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise points
                continue
                
            # Get patterns in this cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_pattern_ids = [pattern_ids[i] for i in cluster_indices]
            
            # Calculate cluster properties
            cluster_features = features[cluster_indices]
            center = np.mean(cluster_features, axis=0)
            
            # Calculate phase coherence
            phase_coherence = self._calculate_phase_coherence(
                [harmonic_series[pid] for pid in cluster_pattern_ids]
            )
            
            # Create domain
            domain = FrequencyDomain(
                dominant_frequency=float(center[0]),
                bandwidth=float(center[1]),
                phase_coherence=phase_coherence,
                center_coordinates=tuple(center),
                pattern_ids=set(cluster_pattern_ids)
            )
            
            domains[domain.id] = domain
            
        logger.info(f"Detected {len(domains)} frequency domains")
        return domains
    
    def _calculate_phase_coherence(self, signals: List[List[float]]) -> float:
        """Calculate phase coherence between multiple signals."""
        if len(signals) < 2:
            return 0.0
            
        # Ensure all signals have the same length
        min_length = min(len(s) for s in signals)
        signals = [s[:min_length] for s in signals]
        
        # Calculate phase differences
        phase_diffs = []
        
        for i in range(len(signals) - 1):
            for j in range(i + 1, len(signals)):
                # Calculate cross-correlation
                corr = signal.correlate(signals[i], signals[j], mode='full')
                # Find the index of maximum correlation
                max_idx = np.argmax(corr)
                # Convert to phase difference (normalized to [0, 1])
                phase_diff = abs(max_idx - (len(corr) // 2)) / (len(corr) // 2)
                phase_diffs.append(1.0 - phase_diff)  # Higher value = more coherent
        
        # Average phase coherence
        return sum(phase_diffs) / len(phase_diffs) if phase_diffs else 0.0
    
    def calculate_harmonic_landscape(self, patterns, learning_windows) -> np.ndarray:
        """
        Calculate the harmonic landscape from patterns and learning windows.
        
        This creates an N-dimensional representation of harmonic values across
        the semantic space, which can be used to detect boundaries and resonance points.
        
        Args:
            patterns: List of pattern objects
            learning_windows: List of learning window objects
            
        Returns:
            N-dimensional array representing the harmonic landscape
        """
        # For the POC, we'll create a simplified 2D landscape
        # In a full implementation, this would be N-dimensional based on semantic embeddings
        
        # Create a 50x50 grid for demonstration
        landscape = np.zeros((50, 50))
        
        # For each pattern, add its influence to the landscape
        for pattern in patterns:
            if not hasattr(pattern, 'evolution_history') or not pattern.evolution_history:
                continue
                
            # Calculate average harmonic value
            harmonic_values = [
                event.get('harmonic_value', 0) 
                for event in pattern.evolution_history 
                if 'harmonic_value' in event
            ]
            
            if not harmonic_values:
                continue
                
            avg_harmonic = sum(harmonic_values) / len(harmonic_values)
            
            # Assign a random position for the POC
            # In a real implementation, this would use semantic coordinates
            x = hash(pattern.pattern_id) % 50
            y = (hash(pattern.pattern_id) // 50) % 50
            
            # Add a Gaussian bump centered at this position
            for i in range(50):
                for j in range(50):
                    distance = np.sqrt((i - x)**2 + (j - y)**2)
                    landscape[i, j] += avg_harmonic * np.exp(-0.1 * distance**2)
        
        # Normalize the landscape
        if np.max(landscape) > 0:
            landscape = landscape / np.max(landscape)
            
        return landscape
    
    def detect_boundaries(self, harmonic_landscape: np.ndarray) -> Dict[str, Boundary]:
        """
        Detect boundaries in the harmonic landscape.
        
        Boundaries are regions of rapid change in harmonic values.
        
        Args:
            harmonic_landscape: N-dimensional array of harmonic values
            
        Returns:
            Dictionary of detected boundaries
        """
        # Calculate gradient magnitude
        gradient_y, gradient_x = np.gradient(harmonic_landscape)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Threshold to find boundary points
        boundary_points = gradient_magnitude > self.boundary_threshold
        
        # Group boundary points into distinct boundaries
        labeled_boundaries, num_boundaries = self._label_connected_components(boundary_points)
        
        # Create boundary objects
        boundaries = {}
        
        for i in range(1, num_boundaries + 1):
            # Get points in this boundary
            boundary_indices = np.where(labeled_boundaries == i)
            points = list(zip(boundary_indices[0], boundary_indices[1]))
            
            if not points:
                continue
                
            # Calculate boundary properties
            avg_gradient = np.mean(gradient_magnitude[boundary_indices])
            
            # Create boundary object
            boundary = Boundary(
                sharpness=float(avg_gradient),
                permeability=float(1.0 - avg_gradient),  # Inverse relationship
                stability=float(0.5),  # Default for now
                dimensionality=1,  # 1D boundary in 2D space
                coordinates=[(float(x), float(y)) for x, y in points[:100]]  # Limit points for efficiency
            )
            
            boundaries[boundary.id] = boundary
            
        return boundaries
    
    def _label_connected_components(self, binary_image: np.ndarray) -> Tuple[np.ndarray, int]:
        """Label connected components in a binary image."""
        # Simple implementation for POC
        # In production, use scipy.ndimage.label
        
        height, width = binary_image.shape
        labeled = np.zeros_like(binary_image, dtype=int)
        current_label = 0
        
        # Simple 4-connected component labeling
        for i in range(height):
            for j in range(width):
                if binary_image[i, j] and labeled[i, j] == 0:
                    current_label += 1
                    self._flood_fill(binary_image, labeled, i, j, current_label)
        
        return labeled, current_label
    
    def _flood_fill(self, binary_image: np.ndarray, labeled: np.ndarray, 
                   i: int, j: int, label: int):
        """Flood fill algorithm for connected component labeling."""
        height, width = binary_image.shape
        stack = [(i, j)]
        
        while stack:
            x, y = stack.pop()
            
            if (x < 0 or x >= height or y < 0 or y >= width or 
                not binary_image[x, y] or labeled[x, y] != 0):
                continue
                
            labeled[x, y] = label
            
            # Add neighbors
            stack.append((x + 1, y))
            stack.append((x - 1, y))
            stack.append((x, y + 1))
            stack.append((x, y - 1))
    
    def detect_resonance_points(self, harmonic_landscape: np.ndarray) -> Dict[str, ResonancePoint]:
        """
        Detect resonance points in the harmonic landscape.
        
        Resonance points are local maxima in the harmonic landscape.
        
        Args:
            harmonic_landscape: N-dimensional array of harmonic values
            
        Returns:
            Dictionary of detected resonance points
        """
        # Find local maxima
        from scipy.ndimage import maximum_filter
        from scipy.ndimage import generate_binary_structure, binary_erosion
        
        # Define neighborhood structure
        neighborhood = generate_binary_structure(2, 2)
        
        # Apply maximum filter
        local_max = maximum_filter(harmonic_landscape, footprint=neighborhood) == harmonic_landscape
        
        # Remove background
        background = (harmonic_landscape < self.resonance_threshold)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
        
        # Get resonance points
        detected_maxima = local_max & ~eroded_background
        
        # Get coordinates and values
        maxima_coords = np.where(detected_maxima)
        maxima_values = harmonic_landscape[maxima_coords]
        
        # Create resonance point objects
        resonance_points = {}
        
        for i in range(len(maxima_values)):
            x, y = maxima_coords[0][i], maxima_coords[1][i]
            strength = float(maxima_values[i])
            
            # Calculate attractor radius based on local curvature
            attractor_radius = self._calculate_attractor_radius(harmonic_landscape, x, y)
            
            # Create resonance point
            point = ResonancePoint(
                coordinates=(float(x), float(y)),
                strength=strength,
                stability=float(0.5),  # Default for now
                attractor_radius=attractor_radius
            )
            
            resonance_points[point.id] = point
            
        return resonance_points
    
    def _calculate_attractor_radius(self, landscape: np.ndarray, x: int, y: int) -> float:
        """Calculate the attractor radius based on local curvature."""
        # Simple estimate based on second derivatives
        height, width = landscape.shape
        
        if x <= 0 or x >= height - 1 or y <= 0 or y >= width - 1:
            return 1.0
            
        # Calculate second derivatives
        d2x = landscape[x+1, y] - 2*landscape[x, y] + landscape[x-1, y]
        d2y = landscape[x, y+1] - 2*landscape[x, y] + landscape[x, y-1]
        
        # Average curvature
        curvature = abs(d2x) + abs(d2y)
        
        # Inverse relationship: higher curvature = smaller radius
        if curvature > 0:
            return float(1.0 / curvature)
        else:
            return 5.0  # Default radius
    
    def analyze_field_dynamics(self, pattern_histories: Dict[str, List[Dict[str, Any]]], 
                              time_period: Dict[str, Any]) -> FieldMetrics:
        """
        Analyze field dynamics from pattern histories.
        
        Args:
            pattern_histories: Dictionary mapping pattern IDs to their evolution histories
            time_period: Dictionary with 'start' and 'end' keys for the analysis period
            
        Returns:
            FieldMetrics object with calculated metrics
        """
        # Extract all harmonic values in the time period
        all_harmonics = []
        
        for pattern_id, history in pattern_histories.items():
            for event in history:
                if 'timestamp' in event and 'harmonic_value' in event:
                    timestamp = event['timestamp']
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp)
                        
                    start_time = time_period.get('start')
                    end_time = time_period.get('end')
                    
                    if start_time and end_time:
                        if isinstance(start_time, str):
                            start_time = datetime.fromisoformat(start_time)
                        if isinstance(end_time, str):
                            end_time = datetime.fromisoformat(end_time)
                            
                        if start_time <= timestamp <= end_time:
                            all_harmonics.append(event['harmonic_value'])
                    else:
                        all_harmonics.append(event['harmonic_value'])
        
        if not all_harmonics:
            logger.warning("No harmonic values found for field dynamics analysis")
            return FieldMetrics()
            
        # Calculate field coherence (normalized standard deviation)
        coherence = 1.0 - min(1.0, np.std(all_harmonics) / max(np.mean(all_harmonics), 0.001))
        
        # Calculate entropy
        hist, _ = np.histogram(all_harmonics, bins=10, range=(0, 1), density=True)
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        normalized_entropy = entropy / np.log2(10)  # Normalize by max entropy
        
        # Create field metrics
        metrics = FieldMetrics(
            coherence=float(coherence),
            entropy=float(normalized_entropy),
            # Other metrics would be calculated in a full implementation
            adaptation_rate=0.5,
            homeostasis_index=0.5
        )
        
        return metrics
        
    def analyze_topology(self, patterns, learning_windows, time_period) -> TopologyState:
        """
        Perform complete topology analysis.
        
        Args:
            patterns: List of pattern objects
            learning_windows: List of learning window objects
            time_period: Dictionary with 'start' and 'end' keys for the analysis period
            
        Returns:
            TopologyState object with all detected features
        """
        logger.info(f"Performing topology analysis for {len(patterns)} patterns")
        
        # Extract pattern histories
        pattern_histories = {p.pattern_id: p.evolution_history for p in patterns}
        
        # Detect frequency domains
        frequency_domains = self.detect_frequency_domains(pattern_histories)
        
        # Calculate harmonic landscape
        harmonic_landscape = self.calculate_harmonic_landscape(patterns, learning_windows)
        
        # Detect boundaries
        boundaries = self.detect_boundaries(harmonic_landscape)
        
        # Detect resonance points
        resonance_points = self.detect_resonance_points(harmonic_landscape)
        
        # Analyze field dynamics
        field_metrics = self.analyze_field_dynamics(pattern_histories, time_period)
        
        # Create topology state
        state = TopologyState(
            frequency_domains=frequency_domains,
            boundaries=boundaries,
            resonance_points=resonance_points,
            field_metrics=field_metrics
        )
        
        logger.info(f"Topology analysis complete: {len(frequency_domains)} domains, "
                   f"{len(boundaries)} boundaries, {len(resonance_points)} resonance points")
        
        return state
