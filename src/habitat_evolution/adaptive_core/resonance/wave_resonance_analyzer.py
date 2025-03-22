"""
Wave Resonance Analyzer for detecting tonic-harmonic relationships.

This module provides functionality to analyze semantic relationships as wave-like
phenomena, detecting harmonic resonance, phase coherence, and resonance cascades
between domains and actants.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from datetime import datetime
import math

logger = logging.getLogger(__name__)

class WaveResonanceAnalyzer:
    """
    Analyzes semantic relationships as wave-like phenomena to detect tonic-harmonic resonance.
    
    This class observes (rather than constructs) wave-like properties in semantic relationships,
    detecting natural harmonic resonance between domains and tracking how resonance propagates
    through the semantic network.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the WaveResonanceAnalyzer with configuration parameters.
        
        Args:
            config: Configuration dictionary with the following optional parameters:
                - harmonic_tolerance: Tolerance for harmonic relationship detection (default: 0.15)
                - phase_coherence_threshold: Threshold for phase coherence (default: 0.7)
                - cascade_depth: Maximum depth for resonance cascades (default: 3)
                - frequency_resolution: Resolution for frequency analysis (default: 0.05)
                - min_amplitude: Minimum amplitude to consider (default: 0.2)
                - visualization_resolution: Number of points for visualization (default: 20)
        """
        default_config = {
            "harmonic_tolerance": 0.15,        # Tolerance for harmonic relationship detection
            "phase_coherence_threshold": 0.7,  # Threshold for phase coherence
            "cascade_depth": 3,                # Maximum depth for resonance cascades
            "frequency_resolution": 0.05,      # Resolution for frequency analysis
            "min_amplitude": 0.2,              # Minimum amplitude to consider
            "visualization_resolution": 20     # Number of points for visualization
        }
        
        self.config = default_config
        if config:
            self.config.update(config)
    
    def calculate_harmonic_resonance(self, domain_waves: List[Dict[str, Any]]) -> np.ndarray:
        """
        Calculate harmonic resonance between domains based on their wave properties.
        
        Args:
            domain_waves: List of domain dictionaries with wave properties
                (frequency, amplitude, phase)
            
        Returns:
            Matrix of resonance values between domains
        """
        n_domains = len(domain_waves)
        resonance_matrix = np.zeros((n_domains, n_domains))
        
        # Set diagonal to 1.0 (self-resonance)
        np.fill_diagonal(resonance_matrix, 1.0)
        
        # Calculate resonance between each pair of domains
        for i in range(n_domains):
            for j in range(i+1, n_domains):
                domain1 = domain_waves[i]
                domain2 = domain_waves[j]
                
                # Calculate frequency ratio (ensure smaller frequency is denominator)
                if domain1["frequency"] <= domain2["frequency"]:
                    freq_ratio = domain2["frequency"] / domain1["frequency"]
                else:
                    freq_ratio = domain1["frequency"] / domain2["frequency"]
                
                # Calculate how close the ratio is to a simple fraction (harmonic relationship)
                # Simple fractions are 1:1, 1:2, 2:3, 3:4, etc.
                harmonic_score = self._calculate_harmonic_score(freq_ratio)
                
                # Calculate phase alignment
                phase_diff = abs(domain1["phase"] - domain2["phase"]) % (2 * np.pi)
                phase_alignment = 1.0 - min(phase_diff, 2 * np.pi - phase_diff) / np.pi
                
                # Calculate amplitude interaction
                amplitude_interaction = min(domain1["amplitude"], domain2["amplitude"]) / max(domain1["amplitude"], domain2["amplitude"])
                
                # Combine factors into overall resonance
                # Harmonic score is most important, followed by phase alignment and amplitude
                resonance = 0.6 * harmonic_score + 0.3 * phase_alignment + 0.1 * amplitude_interaction
                
                # Set in matrix (symmetric)
                resonance_matrix[i, j] = resonance_matrix[j, i] = resonance
        
        return resonance_matrix
    
    def _calculate_harmonic_score(self, ratio: float) -> float:
        """
        Calculate how closely a frequency ratio matches a simple harmonic fraction.
        
        Args:
            ratio: Frequency ratio (always >= 1.0)
            
        Returns:
            Score between 0.0 and 1.0, where 1.0 means perfect harmonic match
        """
        # Perfect unison (1:1)
        if abs(ratio - 1.0) < self.config["harmonic_tolerance"]:
            return 1.0
        
        # Perfect octave (1:2)
        if abs(ratio - 2.0) < self.config["harmonic_tolerance"]:
            return 0.95
        
        # Perfect fifth (2:3)
        if abs(ratio - 1.5) < self.config["harmonic_tolerance"]:
            return 0.9
        
        # Perfect fourth (3:4)
        if abs(ratio - 1.33) < self.config["harmonic_tolerance"]:
            return 0.85
        
        # Major third (4:5)
        if abs(ratio - 1.25) < self.config["harmonic_tolerance"]:
            return 0.8
        
        # Minor third (5:6)
        if abs(ratio - 1.2) < self.config["harmonic_tolerance"]:
            return 0.75
        
        # Check for other simple ratios with small integers (up to 8)
        for n in range(1, 9):
            for m in range(n+1, 9):
                if math.gcd(n, m) == 1:  # Only consider coprime pairs
                    simple_ratio = m / n
                    if abs(ratio - simple_ratio) < self.config["harmonic_tolerance"]:
                        # Score inversely proportional to the sum of numerator and denominator
                        return max(0.7, 1.0 - 0.05 * (n + m))
        
        # For non-harmonic ratios, calculate a continuous score
        # that decreases as the ratio becomes more complex
        # Find the closest simple fraction and calculate distance
        best_distance = float('inf')
        for n in range(1, 12):
            for m in range(n+1, 12):
                if math.gcd(n, m) == 1:
                    simple_ratio = m / n
                    distance = abs(ratio - simple_ratio)
                    if distance < best_distance:
                        best_distance = distance
        
        # Score based on distance to closest simple fraction
        return max(0.0, 0.6 - best_distance * 2.0)
    
    def calculate_phase_coherence(self, domain_waves: List[Dict[str, Any]]) -> np.ndarray:
        """
        Calculate phase coherence between domains.
        
        Args:
            domain_waves: List of domain dictionaries with wave properties
            
        Returns:
            Matrix of phase coherence values between domains
        """
        n_domains = len(domain_waves)
        coherence_matrix = np.zeros((n_domains, n_domains))
        
        # Set diagonal to 1.0 (self-coherence)
        np.fill_diagonal(coherence_matrix, 1.0)
        
        # Calculate coherence between each pair of domains
        for i in range(n_domains):
            for j in range(i+1, n_domains):
                domain1 = domain_waves[i]
                domain2 = domain_waves[j]
                
                # Calculate phase difference
                phase_diff = abs(domain1["phase"] - domain2["phase"]) % (2 * np.pi)
                
                # Convert to coherence (1.0 = perfectly in phase, 0.0 = perfectly out of phase)
                coherence = 1.0 - min(phase_diff, 2 * np.pi - phase_diff) / np.pi
                
                # Set in matrix (symmetric)
                coherence_matrix[i, j] = coherence_matrix[j, i] = coherence
        
        return coherence_matrix
    
    def detect_resonance_cascades(self, domain_waves: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect resonance cascades through multiple domains.
        
        Args:
            domain_waves: List of domain dictionaries with wave properties
            
        Returns:
            List of detected cascades with their properties
        """
        # Calculate resonance matrix
        resonance_matrix = self.calculate_harmonic_resonance(domain_waves)
        
        # Create a graph representation
        n_domains = len(domain_waves)
        graph = {}
        for i in range(n_domains):
            graph[i] = []
            for j in range(n_domains):
                if i != j and resonance_matrix[i, j] >= self.config["phase_coherence_threshold"]:
                    graph[i].append((j, resonance_matrix[i, j]))
        
        # Find all paths up to max_depth
        cascades = []
        for start in range(n_domains):
            self._find_cascades(graph, start, [start], [], cascades, domain_waves)
        
        return cascades
    
    def _find_cascades(self, graph: Dict[int, List[Tuple[int, float]]], 
                      current: int, path: List[int], strengths: List[float],
                      cascades: List[Dict[str, Any]], domain_waves: List[Dict[str, Any]],
                      depth: int = 0):
        """
        Recursively find resonance cascades starting from a given domain.
        
        Args:
            graph: Graph representation of resonance connections
            current: Current domain index
            path: Current path of domain indices
            strengths: Resonance strengths along the path
            cascades: List to collect found cascades
            domain_waves: Original domain wave data
            depth: Current recursion depth
        """
        # If path has at least 3 domains, record it as a cascade
        if len(path) >= 3:
            cascade = {
                "path": [domain_waves[i]["id"] for i in path],
                "domain_indices": path.copy(),
                "strength": np.mean(strengths),
                "length": len(path)
            }
            cascades.append(cascade)
        
        # Stop recursion if max depth reached
        if depth >= self.config["cascade_depth"]:
            return
        
        # Explore neighbors
        for neighbor, strength in graph[current]:
            if neighbor not in path:  # Avoid cycles
                path.append(neighbor)
                strengths.append(strength)
                self._find_cascades(graph, neighbor, path, strengths, cascades, domain_waves, depth + 1)
                path.pop()
                strengths.pop()
    
    def analyze_actant_journey_waves(self, actant_journeys: List[Dict[str, Any]], 
                                    domain_waves: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze actant journeys as wave phenomena.
        
        Args:
            actant_journeys: List of actant journey dictionaries
            domain_waves: List of domain dictionaries with wave properties
            
        Returns:
            List of journey analyses with wave properties
        """
        # Create a mapping from domain IDs to indices
        domain_id_to_index = {domain["id"]: i for i, domain in enumerate(domain_waves)}
        
        journey_analyses = []
        for journey in actant_journeys:
            actant_id = journey["actant_id"]
            path = journey["journey"]
            
            # Skip journeys that are too short
            if len(path) < 2:
                continue
            
            # Extract domain indices and timestamps
            domain_indices = []
            timestamps = []
            predicate_types = []
            
            for step in path:
                domain_id = step["domain_id"]
                if domain_id in domain_id_to_index:
                    domain_indices.append(domain_id_to_index[domain_id])
                    timestamps.append(step["timestamp"])
                    predicate_types.append(step["predicate_type"])
            
            # Skip if not enough valid domains
            if len(domain_indices) < 2:
                continue
            
            # Calculate resonance along the journey
            resonance_strengths = []
            for i in range(len(domain_indices) - 1):
                idx1 = domain_indices[i]
                idx2 = domain_indices[i + 1]
                resonance_matrix = self.calculate_harmonic_resonance(domain_waves)
                resonance_strengths.append(resonance_matrix[idx1, idx2])
            
            # Calculate average resonance
            avg_resonance = np.mean(resonance_strengths) if resonance_strengths else 0.0
            
            # Calculate wave properties of the journey
            frequencies = [domain_waves[idx]["frequency"] for idx in domain_indices]
            amplitudes = [domain_waves[idx]["amplitude"] for idx in domain_indices]
            phases = [domain_waves[idx]["phase"] for idx in domain_indices]
            
            # Calculate frequency progression (is it harmonic?)
            freq_ratios = [frequencies[i+1] / frequencies[i] for i in range(len(frequencies) - 1)]
            harmonic_progression = np.mean([self._calculate_harmonic_score(ratio) for ratio in freq_ratios]) if freq_ratios else 0.0
            
            # Calculate phase coherence along the journey
            phase_diffs = [abs(phases[i+1] - phases[i]) % (2 * np.pi) for i in range(len(phases) - 1)]
            phase_coherence = np.mean([1.0 - min(diff, 2 * np.pi - diff) / np.pi for diff in phase_diffs]) if phase_diffs else 0.0
            
            # Analyze the journey
            analysis = {
                "actant_id": actant_id,
                "journey_length": len(domain_indices),
                "domain_path": [domain_waves[idx]["id"] for idx in domain_indices],
                "resonance_strength": avg_resonance,
                "wave_properties": {
                    "harmonic_progression": harmonic_progression,
                    "phase_coherence": phase_coherence,
                    "amplitude_trend": np.polyfit(range(len(amplitudes)), amplitudes, 1)[0] if len(amplitudes) > 1 else 0.0,
                    "frequency_trend": np.polyfit(range(len(frequencies)), frequencies, 1)[0] if len(frequencies) > 1 else 0.0
                },
                "predicate_evolution": predicate_types
            }
            
            journey_analyses.append(analysis)
        
        return journey_analyses
    
    def generate_wave_visualization(self, domain_waves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate data for visualizing domains as waves.
        
        Args:
            domain_waves: List of domain dictionaries with wave properties
            
        Returns:
            Visualization data structure
        """
        # Generate time points
        n_points = self.config["visualization_resolution"]
        time_points = np.linspace(0, 2 * np.pi, n_points)
        
        # Generate wave values for each domain
        domain_visualizations = []
        for domain in domain_waves:
            frequency = domain["frequency"]
            amplitude = domain["amplitude"]
            phase = domain["phase"]
            
            # Calculate wave values
            values = amplitude * np.sin(frequency * time_points + phase)
            
            domain_visualizations.append({
                "domain_id": domain["id"],
                "frequency": frequency,
                "amplitude": amplitude,
                "phase": phase,
                "values": values.tolist()
            })
        
        # Calculate interference pattern
        interference = np.zeros(n_points)
        for domain_vis in domain_visualizations:
            interference += np.array(domain_vis["values"])
        
        return {
            "time_points": time_points.tolist(),
            "domain_waves": domain_visualizations,
            "interference": {
                "values": interference.tolist(),
                "max": float(np.max(interference)),
                "min": float(np.min(interference))
            }
        }
    
    def calculate_comparative_metrics(self, domain_waves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comparative metrics between tonic-harmonic and traditional approaches.
        
        Args:
            domain_waves: List of domain dictionaries with wave properties
            
        Returns:
            Dictionary of comparative metrics
        """
        # Calculate tonic-harmonic resonance
        resonance_matrix = self.calculate_harmonic_resonance(domain_waves)
        
        # Calculate traditional similarity (cosine similarity of content vectors if available)
        # If not available, use a simple baseline
        traditional_matrix = np.zeros_like(resonance_matrix)
        np.fill_diagonal(traditional_matrix, 1.0)
        
        for i in range(len(domain_waves)):
            for j in range(i+1, len(domain_waves)):
                # Simple baseline: inverse of frequency difference
                freq_diff = abs(domain_waves[i]["frequency"] - domain_waves[j]["frequency"])
                traditional_sim = 1.0 / (1.0 + freq_diff)
                traditional_matrix[i, j] = traditional_matrix[j, i] = traditional_sim
        
        # Calculate average scores
        tonic_harmonic_score = float(np.mean(resonance_matrix[np.triu_indices_from(resonance_matrix, k=1)]))
        traditional_score = float(np.mean(traditional_matrix[np.triu_indices_from(traditional_matrix, k=1)]))
        
        # Calculate improvement percentage
        improvement = (tonic_harmonic_score - traditional_score) / traditional_score * 100 if traditional_score > 0 else 0.0
        
        # Detailed comparison for each domain pair
        detailed_comparison = []
        for i in range(len(domain_waves)):
            for j in range(i+1, len(domain_waves)):
                detailed_comparison.append({
                    "domain_pair": [domain_waves[i]["id"], domain_waves[j]["id"]],
                    "tonic_harmonic": float(resonance_matrix[i, j]),
                    "traditional": float(traditional_matrix[i, j]),
                    "improvement": float((resonance_matrix[i, j] - traditional_matrix[i, j]) / traditional_matrix[i, j] * 100) 
                                  if traditional_matrix[i, j] > 0 else 0.0
                })
        
        return {
            "tonic_harmonic_score": tonic_harmonic_score,
            "traditional_similarity_score": traditional_score,
            "improvement_percentage": improvement,
            "detailed_comparison": detailed_comparison
        }
