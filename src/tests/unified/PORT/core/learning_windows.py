"""
Learning Windows Interface for Natural Knowledge Integration.

This module provides mechanisms for managing learning windows that form
naturally around areas of high knowledge density and pattern emergence.

Key Components:
    - LearningWindowInterface: Main interface for natural window management
    - DensityAnalysis: Natural density calculation and field analysis
    - DomainAlignments: Natural cross-domain path identification

The system focuses on:
    - Natural window formation
    - Organic density emergence
    - Light pattern observation
    - Unforced domain alignment

Typical usage:
    1. Initialize LearningWindowInterface
    2. Allow windows to form naturally
    3. Observe density patterns
    4. Track natural alignments
    5. Maintain light coherence
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class DensityMetrics:
    """Natural density metrics for learning windows."""
    local: float = 0.0           # Natural local density
    cross_domain: float = 0.0    # Natural cross-domain density
    gradient: float = 0.0        # Natural density gradient
    alignments: List[Dict] = None  # Natural domain alignments
    
    def __post_init__(self):
        self.alignments = [] if self.alignments is None else self.alignments

class LearningWindowInterface:
    """Interface for natural learning window management."""
    
    def __init__(self):
        self.window_registry = {}
        self.pattern_cache = {}
        self.density_cache = {}
        self.coherence_threshold = 0.85  # Natural threshold
        
    def register_window(self, window_data: dict) -> str:
        """Register a naturally forming learning window."""
        window_id = str(len(self.window_registry))
        
        # Calculate natural density metrics
        density_metrics = self._calculate_density_metrics(window_data)
        
        # Track natural pattern evolution
        self._update_pattern_cache(window_data, density_metrics)
        
        # Store with natural metadata
        self.window_registry[window_id] = {
            "state": {
                **window_data,
                "density": density_metrics
            },
            "coherence": {
                "timestamp": datetime.now().isoformat(),
                "structural_integrity": self._calculate_structural_integrity(window_data),
                "semantic_stability": self._calculate_semantic_stability(window_data),
                "propagation_potential": window_data.get("potential", 0)
            }
        }
        
        # Update density cache naturally
        self.density_cache[window_id] = {
            "metrics": density_metrics,
            "gradients": self._calculate_density_gradients(density_metrics)
        }
        
        return window_id
        
    def get_window(self, window_id: str) -> dict:
        """Retrieve natural window state."""
        if window_id not in self.window_registry:
            return None
            
        window = self.window_registry[window_id]
        return {
            **window,
            "patterns": self._get_relevant_patterns(window_id),
            "coherence_status": self._assess_coherence_status(window)
        }
        
    def get_density_analysis(self, window_id: str = None) -> dict:
        """Get natural density analysis."""
        if window_id and window_id in self.density_cache:
            return self.density_cache[window_id]
            
        # Analyze natural field density
        field_density = {
            "global_density": self._calculate_field_density(),
            "density_centers": self._identify_density_centers(),
            "cross_domain_paths": self._analyze_cross_domain_paths()
        }
        
        return field_density
        
    def _calculate_density_metrics(self, window_data: dict) -> dict:
        """Calculate natural density metrics."""
        channels = window_data.get("channels", {})
        structural = channels.get("structural", {})
        semantic = channels.get("semantic", {})
        
        # Calculate natural local density
        local_density = (
            structural.get("strength", 0.5) * 0.4 +  # Natural structural weight
            semantic.get("sustainability", 0.5) * 0.3 +  # Natural semantic weight
            window_data.get("score", 0.5) * 0.3  # Natural overall weight
        )
        
        # Calculate natural cross-domain potential
        cross_domain = (
            semantic.get("strength", 0.5) * 0.4 +  # Natural semantic weight
            window_data.get("potential", 0.5) * 0.4 +  # Natural potential weight
            window_data.get("horizon", 0.5) * 0.2  # Natural horizon weight
        )
        
        # Identify natural domain alignments
        alignments = self._identify_domain_alignments(window_data)
        
        return {
            "local": local_density,
            "cross_domain": cross_domain,
            "gradient": abs(local_density - cross_domain),
            "alignments": alignments
        }
        
    def _calculate_density_gradients(self, density_metrics: dict) -> dict:
        """Calculate natural density gradients."""
        return {
            "local_to_cross": density_metrics["gradient"],
            "alignment_strength": sum(a["strength"] for a in density_metrics["alignments"]) / 
                                len(density_metrics["alignments"]) if density_metrics["alignments"] else 0,
            "field_potential": (density_metrics["local"] + density_metrics["cross_domain"]) / 2
        }
        
    def _identify_domain_alignments(self, window_data: dict) -> list:
        """Identify natural domain alignments."""
        semantic_patterns = window_data.get("semantic_patterns", [])
        alignments = []
        
        if semantic_patterns:
            # Group by natural domains
            domain_strengths = {}
            domain_counts = {}
            for pattern in semantic_patterns:
                domain = pattern.get("domain", "general")
                strength = pattern.get("strength", 0.5)
                domain_strengths[domain] = domain_strengths.get(domain, 0) + strength
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Convert to natural alignment vectors
            alignments = [
                {"domain": domain, "strength": strength / domain_counts[domain]}
                for domain, strength in domain_strengths.items()
            ]
            
            # Sort by natural strength
            alignments.sort(key=lambda x: (-x["strength"], x["domain"]))
        
        return alignments
        
    def _calculate_field_density(self) -> float:
        """Calculate natural field density."""
        if not self.density_cache:
            return 0.0
            
        densities = [
            cache["metrics"]["local"]
            for cache in self.density_cache.values()
        ]
        return sum(densities) / len(densities)
        
    def _identify_density_centers(self) -> list:
        """Identify natural high-density centers."""
        centers = []
        
        # Calculate natural averages
        avg_local = sum(cache["metrics"]["local"] for cache in self.density_cache.values()) / len(self.density_cache) if self.density_cache else 0
        avg_cross = sum(cache["metrics"]["cross_domain"] for cache in self.density_cache.values()) / len(self.density_cache) if self.density_cache else 0
        
        # Identify natural centers
        for window_id, cache in self.density_cache.items():
            metrics = cache["metrics"]
            local = metrics["local"]
            cross = metrics["cross_domain"]
            
            # Calculate natural density
            density = (local * 0.4 + cross * 0.6)  # Natural weighting
            
            # Natural density center formation
            if density > 0.8:  # Natural threshold
                centers.append({
                    "window_id": window_id,
                    "density": density,
                    "alignments": metrics["alignments"],
                    "type": "local" if local > cross else "cross_domain"
                })
            # Natural emergence detection
            elif (local > avg_local * 1.2 and cross > avg_cross * 1.2):
                centers.append({
                    "window_id": window_id,
                    "density": density,
                    "alignments": metrics["alignments"],
                    "type": "emerging"
                })
        
        return sorted(centers, key=lambda x: x["density"], reverse=True)
        
    def _analyze_cross_domain_paths(self) -> list:
        """Analyze natural cross-domain paths."""
        paths = []
        windows = sorted(self.density_cache.items(), key=lambda x: x[0])
        
        for i in range(len(windows) - 1):
            current = windows[i][1]
            next_window = windows[i + 1][1]
            
            # Natural path formation
            if current["metrics"]["cross_domain"] > self.coherence_threshold:
                paths.append({
                    "start": windows[i][0],
                    "end": windows[i + 1][0],
                    "strength": (current["metrics"]["cross_domain"] + 
                                next_window["metrics"]["cross_domain"]) / 2,
                    "alignments": current["metrics"]["alignments"]
                })
        
        return sorted(paths, key=lambda x: x["strength"], reverse=True)
        
    def _update_pattern_cache(self, window_data: dict, density_metrics: dict):
        """Update pattern cache with natural evolution."""
        window_patterns = window_data.get("patterns", [])
        
        for pattern in window_patterns:
            pattern_id = pattern.get("id")
            if pattern_id:
                self.pattern_cache[pattern_id] = {
                    "pattern": pattern,
                    "density": density_metrics,
                    "timestamp": datetime.now().isoformat()
                }
                
    def _get_relevant_patterns(self, window_id: str) -> list:
        """Get naturally relevant patterns."""
        if window_id not in self.window_registry:
            return []
            
        window = self.window_registry[window_id]
        window_patterns = window["state"].get("patterns", [])
        
        return [
            self.pattern_cache.get(pattern.get("id"), {})
            for pattern in window_patterns
            if pattern.get("id") in self.pattern_cache
        ]
        
    def _assess_coherence_status(self, window: dict) -> str:
        """Assess natural coherence status."""
        coherence = window.get("coherence", {})
        
        structural = coherence.get("structural_integrity", 0)
        semantic = coherence.get("semantic_stability", 0)
        potential = coherence.get("propagation_potential", 0)
        
        # Natural coherence assessment
        if all(x > 0.8 for x in [structural, semantic, potential]):
            return "highly_coherent"
        elif all(x > 0.6 for x in [structural, semantic, potential]):
            return "moderately_coherent"
        elif all(x > 0.4 for x in [structural, semantic, potential]):
            return "emerging_coherence"
        else:
            return "forming"
        
    def _calculate_structural_integrity(self, window_data: dict) -> float:
        """Calculate natural structural integrity."""
        channels = window_data.get("channels", {})
        structural = channels.get("structural", {})
        
        return structural.get("integrity", 0.5)
        
    def _calculate_semantic_stability(self, window_data: dict) -> float:
        """Calculate natural semantic stability."""
        channels = window_data.get("channels", {})
        semantic = channels.get("semantic", {})
        
        return semantic.get("stability", 0.5)
