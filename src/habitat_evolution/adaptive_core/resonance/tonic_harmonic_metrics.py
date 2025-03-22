"""
Tonic-Harmonic Metrics for measuring resonance relationships.

This module provides metrics that measure tonic-harmonic relationships
between domains, allowing for quantitative analysis of resonance patterns.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

class TonicHarmonicMetrics:
    """
    Provides metrics for measuring tonic-harmonic relationships between domains.
    
    This class calculates various metrics that quantify the harmonic coherence,
    phase alignment, and resonance stability of domains, as well as comparative
    metrics against traditional similarity approaches.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TonicHarmonicMetrics with configuration parameters.
        
        Args:
            config: Configuration dictionary with the following optional parameters:
                - harmonic_coherence_weight: Weight for harmonic coherence in overall score (default: 0.4)
                - phase_alignment_weight: Weight for phase alignment in overall score (default: 0.3)
                - resonance_stability_weight: Weight for resonance stability in overall score (default: 0.3)
                - frequency_bands: Frequency bands for analysis
        """
        default_config = {
            "harmonic_coherence_weight": 0.4,  # Weight for harmonic coherence in overall score
            "phase_alignment_weight": 0.3,     # Weight for phase alignment in overall score
            "resonance_stability_weight": 0.3,  # Weight for resonance stability in overall score
            "frequency_bands": [               # Frequency bands for analysis
                {"name": "low", "min": 0.0, "max": 0.5},
                {"name": "medium", "min": 0.5, "max": 1.0},
                {"name": "high", "min": 1.0, "max": 2.0}
            ]
        }
        
        self.config = default_config
        if config:
            self.config.update(config)
    
    def calculate_harmonic_coherence(self, domain_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate harmonic coherence metrics for domain pairs.
        
        Args:
            domain_data: List of domain dictionaries with frequency, amplitude, and phase
            
        Returns:
            List of coherence metrics for each domain pair
        """
        coherence_metrics = []
        
        # Calculate coherence for each domain pair
        for i in range(len(domain_data)):
            for j in range(i+1, len(domain_data)):
                domain1 = domain_data[i]
                domain2 = domain_data[j]
                
                # Calculate frequency ratio (ensure smaller frequency is denominator)
                if domain1["frequency"] <= domain2["frequency"]:
                    freq_ratio = domain2["frequency"] / domain1["frequency"]
                else:
                    freq_ratio = domain1["frequency"] / domain2["frequency"]
                
                # Calculate how close the ratio is to a simple fraction (harmonic relationship)
                harmonic_score = self._calculate_harmonic_score(freq_ratio)
                
                # Calculate amplitude interaction
                amplitude_product = domain1["amplitude"] * domain2["amplitude"]
                amplitude_ratio = min(domain1["amplitude"], domain2["amplitude"]) / max(domain1["amplitude"], domain2["amplitude"])
                
                # Calculate overall coherence
                coherence = 0.7 * harmonic_score + 0.3 * amplitude_ratio
                
                coherence_metrics.append({
                    "domain1_id": domain1["id"],
                    "domain2_id": domain2["id"],
                    "frequency_ratio": freq_ratio,
                    "harmonic_score": harmonic_score,
                    "amplitude_product": amplitude_product,
                    "amplitude_ratio": amplitude_ratio,
                    "coherence": coherence
                })
        
        # Sort by coherence (descending)
        coherence_metrics.sort(key=lambda x: x["coherence"], reverse=True)
        
        return coherence_metrics
    
    def _calculate_harmonic_score(self, ratio: float) -> float:
        """
        Calculate how closely a frequency ratio matches a simple harmonic fraction.
        
        Args:
            ratio: Frequency ratio (always >= 1.0)
            
        Returns:
            Score between 0.0 and 1.0, where 1.0 means perfect harmonic match
        """
        # Perfect unison (1:1)
        if abs(ratio - 1.0) < 0.05:
            return 1.0
        
        # Perfect octave (1:2)
        if abs(ratio - 2.0) < 0.05:
            return 0.95
        
        # Perfect fifth (2:3)
        if abs(ratio - 1.5) < 0.05:
            return 0.9
        
        # Perfect fourth (3:4)
        if abs(ratio - 1.33) < 0.05:
            return 0.85
        
        # Major third (4:5)
        if abs(ratio - 1.25) < 0.05:
            return 0.8
        
        # Minor third (5:6)
        if abs(ratio - 1.2) < 0.05:
            return 0.75
        
        # Check for other simple ratios with small integers (up to 8)
        for n in range(1, 9):
            for m in range(n+1, 9):
                if math.gcd(n, m) == 1:  # Only consider coprime pairs
                    simple_ratio = m / n
                    if abs(ratio - simple_ratio) < 0.05:
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
    
    def calculate_phase_alignment(self, domain_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate phase alignment metrics for domain pairs.
        
        Args:
            domain_data: List of domain dictionaries with frequency, amplitude, and phase
            
        Returns:
            List of alignment metrics for each domain pair
        """
        alignment_metrics = []
        
        # Calculate alignment for each domain pair
        for i in range(len(domain_data)):
            for j in range(i+1, len(domain_data)):
                domain1 = domain_data[i]
                domain2 = domain_data[j]
                
                # Calculate phase difference
                phase_diff = abs(domain1["phase"] - domain2["phase"]) % (2 * np.pi)
                
                # Convert to alignment (1.0 = perfectly in phase, 0.0 = perfectly out of phase)
                alignment = 1.0 - min(phase_diff, 2 * np.pi - phase_diff) / np.pi
                
                # Calculate constructive/destructive interference potential
                interference_potential = alignment * 2 - 1  # Range: -1 (destructive) to 1 (constructive)
                
                # Calculate effective amplitude when combined
                combined_amplitude = np.sqrt(domain1["amplitude"]**2 + domain2["amplitude"]**2 + 
                                           2 * domain1["amplitude"] * domain2["amplitude"] * np.cos(phase_diff))
                
                alignment_metrics.append({
                    "domain1_id": domain1["id"],
                    "domain2_id": domain2["id"],
                    "phase_difference": float(phase_diff),
                    "alignment": float(alignment),
                    "interference_potential": float(interference_potential),
                    "combined_amplitude": float(combined_amplitude)
                })
        
        # Sort by alignment (descending)
        alignment_metrics.sort(key=lambda x: x["alignment"], reverse=True)
        
        return alignment_metrics
    
    def calculate_resonance_stability(self, domain_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate resonance stability metrics for domains.
        
        Args:
            domain_data: List of domain dictionaries with stability information
            
        Returns:
            List of stability metrics for each domain
        """
        stability_metrics = []
        
        for domain in domain_data:
            # Use provided stability if available, otherwise calculate from other properties
            if "stability" in domain:
                stability = domain["stability"]
            else:
                # Calculate stability based on frequency and amplitude
                # Higher frequency and lower amplitude tend to be less stable
                frequency_factor = 1.0 / (1.0 + domain["frequency"])
                amplitude_factor = domain["amplitude"]
                stability = 0.6 * frequency_factor + 0.4 * amplitude_factor
            
            # Determine frequency band
            frequency = domain["frequency"]
            frequency_band = "unknown"
            for band in self.config["frequency_bands"]:
                if band["min"] <= frequency < band["max"]:
                    frequency_band = band["name"]
                    break
            
            stability_metrics.append({
                "domain_id": domain["id"],
                "stability": float(stability),
                "frequency": float(domain["frequency"]),
                "amplitude": float(domain["amplitude"]),
                "frequency_band": frequency_band
            })
        
        # Sort by stability (descending)
        stability_metrics.sort(key=lambda x: x["stability"], reverse=True)
        
        return stability_metrics
    
    def calculate_overall_score(self, domain_data: List[Dict[str, Any]], 
                               resonance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall tonic-harmonic score.
        
        Args:
            domain_data: List of domain dictionaries with wave properties
            resonance_data: Dictionary with resonance matrix between domains
            
        Returns:
            Dictionary with overall score and component scores
        """
        # Calculate component metrics
        coherence_metrics = self.calculate_harmonic_coherence(domain_data)
        alignment_metrics = self.calculate_phase_alignment(domain_data)
        stability_metrics = self.calculate_resonance_stability(domain_data)
        
        # Calculate average scores for each component
        avg_coherence = np.mean([m["coherence"] for m in coherence_metrics]) if coherence_metrics else 0.0
        avg_alignment = np.mean([m["alignment"] for m in alignment_metrics]) if alignment_metrics else 0.0
        avg_stability = np.mean([m["stability"] for m in stability_metrics]) if stability_metrics else 0.0
        
        # Calculate overall score as weighted average
        overall_score = (
            self.config["harmonic_coherence_weight"] * avg_coherence +
            self.config["phase_alignment_weight"] * avg_alignment +
            self.config["resonance_stability_weight"] * avg_stability
        )
        
        # Calculate resonance matrix statistics
        if "matrix" in resonance_data:
            matrix = np.array(resonance_data["matrix"])
            matrix_avg = float(np.mean(matrix))
            matrix_max = float(np.max(matrix))
            matrix_min = float(np.min(matrix))
            matrix_std = float(np.std(matrix))
        else:
            matrix_avg = matrix_max = matrix_min = matrix_std = 0.0
        
        return {
            "overall_score": float(overall_score),
            "component_scores": {
                "harmonic_coherence": float(avg_coherence),
                "phase_alignment": float(avg_alignment),
                "resonance_stability": float(avg_stability)
            },
            "matrix_statistics": {
                "average": matrix_avg,
                "maximum": matrix_max,
                "minimum": matrix_min,
                "standard_deviation": matrix_std
            },
            "domain_count": len(domain_data),
            "pair_count": len(coherence_metrics)
        }
    
    def calculate_comparative_metrics(self, domain_data: List[Dict[str, Any]], 
                                     resonance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comparative metrics between tonic-harmonic and traditional approaches.
        
        Args:
            domain_data: List of domain dictionaries with wave properties
            resonance_data: Dictionary with resonance matrix between domains
            
        Returns:
            Dictionary of comparative metrics
        """
        # Calculate tonic-harmonic score
        th_score = self.calculate_overall_score(domain_data, resonance_data)
        
        # Calculate traditional similarity score (using content vectors if available)
        # If not available, use a simple baseline
        traditional_score = 0.0
        detailed_comparison = []
        
        # Create domain ID to index mapping
        domain_id_to_index = {domain["id"]: i for i, domain in enumerate(domain_data)}
        
        # Compare each domain pair
        for i in range(len(domain_data)):
            for j in range(i+1, len(domain_data)):
                domain1 = domain_data[i]
                domain2 = domain_data[j]
                
                # Get tonic-harmonic resonance from matrix
                if "matrix" in resonance_data:
                    th_resonance = resonance_data["matrix"][i][j]
                else:
                    # Calculate from scratch
                    coherence_metrics = self.calculate_harmonic_coherence([domain1, domain2])
                    alignment_metrics = self.calculate_phase_alignment([domain1, domain2])
                    
                    th_resonance = (
                        self.config["harmonic_coherence_weight"] * coherence_metrics[0]["coherence"] +
                        self.config["phase_alignment_weight"] * alignment_metrics[0]["alignment"]
                    ) / (self.config["harmonic_coherence_weight"] + self.config["phase_alignment_weight"])
                
                # Calculate traditional similarity
                # If content vectors are available, use cosine similarity
                if "content_vector" in domain1 and "content_vector" in domain2:
                    vec1 = np.array(domain1["content_vector"])
                    vec2 = np.array(domain2["content_vector"])
                    traditional_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                else:
                    # Simple baseline: inverse of frequency difference
                    freq_diff = abs(domain1["frequency"] - domain2["frequency"])
                    traditional_sim = 1.0 / (1.0 + freq_diff)
                
                # Calculate improvement
                improvement = (th_resonance - traditional_sim) / traditional_sim * 100 if traditional_sim > 0 else 0.0
                
                detailed_comparison.append({
                    "domain_pair": [domain1["id"], domain2["id"]],
                    "tonic_harmonic": float(th_resonance),
                    "traditional": float(traditional_sim),
                    "improvement": float(improvement)
                })
                
                # Accumulate for average
                traditional_score += traditional_sim
        
        # Calculate average traditional score
        pair_count = len(domain_data) * (len(domain_data) - 1) // 2
        traditional_score = traditional_score / pair_count if pair_count > 0 else 0.0
        
        # Calculate overall improvement
        improvement_percentage = (th_score["overall_score"] - traditional_score) / traditional_score * 100 if traditional_score > 0 else 0.0
        
        # Sort detailed comparison by improvement (descending)
        detailed_comparison.sort(key=lambda x: x["improvement"], reverse=True)
        
        return {
            "tonic_harmonic_score": float(th_score["overall_score"]),
            "traditional_similarity_score": float(traditional_score),
            "improvement_percentage": float(improvement_percentage),
            "detailed_comparison": detailed_comparison,
            "component_scores": th_score["component_scores"]
        }
    
    def analyze_frequency_band_distribution(self, domain_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the distribution of domains across frequency bands.
        
        Args:
            domain_data: List of domain dictionaries with frequency information
            
        Returns:
            Dictionary with frequency band distribution analysis
        """
        # Count domains in each frequency band
        band_counts = defaultdict(int)
        band_domains = defaultdict(list)
        
        for domain in domain_data:
            frequency = domain["frequency"]
            for band in self.config["frequency_bands"]:
                if band["min"] <= frequency < band["max"]:
                    band_name = band["name"]
                    band_counts[band_name] += 1
                    band_domains[band_name].append(domain["id"])
                    break
        
        # Calculate percentages
        total_domains = len(domain_data)
        band_percentages = {band: count / total_domains * 100 for band, count in band_counts.items()}
        
        # Calculate average properties per band
        band_properties = {}
        for band_name, domain_ids in band_domains.items():
            band_domains_data = [d for d in domain_data if d["id"] in domain_ids]
            
            if band_domains_data:
                avg_amplitude = np.mean([d["amplitude"] for d in band_domains_data])
                avg_stability = np.mean([d.get("stability", 0.5) for d in band_domains_data])
                
                band_properties[band_name] = {
                    "average_amplitude": float(avg_amplitude),
                    "average_stability": float(avg_stability),
                    "domain_count": len(band_domains_data)
                }
        
        return {
            "band_counts": dict(band_counts),
            "band_percentages": band_percentages,
            "band_properties": band_properties,
            "band_domains": band_domains,
            "total_domains": total_domains
        }
    
    def calculate_resonance_network_metrics(self, domain_data: List[Dict[str, Any]], 
                                          resonance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics for the resonance network.
        
        Args:
            domain_data: List of domain dictionaries with wave properties
            resonance_data: Dictionary with resonance matrix between domains
            
        Returns:
            Dictionary of network metrics
        """
        if "matrix" not in resonance_data:
            return {"error": "Resonance matrix not provided"}
        
        matrix = np.array(resonance_data["matrix"])
        
        # Calculate network density (proportion of strong resonances)
        threshold = 0.7  # Threshold for strong resonance
        strong_resonances = np.sum(matrix >= threshold)
        total_possible = len(domain_data) * (len(domain_data) - 1) // 2  # Excluding self-resonance
        network_density = strong_resonances / total_possible if total_possible > 0 else 0.0
        
        # Calculate clustering metrics
        clusters = []
        visited = set()
        
        for i in range(len(domain_data)):
            if i in visited:
                continue
            
            # Find connected domains with strong resonance
            cluster = self._find_resonance_cluster(matrix, i, threshold)
            
            if len(cluster) > 1:  # Only consider clusters with at least 2 domains
                clusters.append({
                    "domains": [domain_data[idx]["id"] for idx in cluster],
                    "size": len(cluster),
                    "average_resonance": float(np.mean([matrix[i][j] for i in cluster for j in cluster if i != j]))
                })
                
                visited.update(cluster)
        
        # Sort clusters by size (descending)
        clusters.sort(key=lambda x: x["size"], reverse=True)
        
        # Calculate average cluster size
        avg_cluster_size = np.mean([c["size"] for c in clusters]) if clusters else 0.0
        
        return {
            "network_density": float(network_density),
            "strong_resonance_count": int(strong_resonances),
            "cluster_count": len(clusters),
            "average_cluster_size": float(avg_cluster_size),
            "largest_cluster_size": max([c["size"] for c in clusters]) if clusters else 0,
            "clusters": clusters
        }
    
    def _find_resonance_cluster(self, matrix: np.ndarray, start_idx: int, threshold: float) -> Set[int]:
        """
        Find a cluster of domains with strong resonance connections.
        
        Args:
            matrix: Resonance matrix
            start_idx: Starting domain index
            threshold: Threshold for strong resonance
            
        Returns:
            Set of domain indices in the cluster
        """
        cluster = {start_idx}
        queue = [start_idx]
        
        while queue:
            current = queue.pop(0)
            
            for i in range(len(matrix)):
                if i not in cluster and matrix[current][i] >= threshold:
                    cluster.add(i)
                    queue.append(i)
        
        return cluster
