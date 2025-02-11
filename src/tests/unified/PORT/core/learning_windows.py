"""
Learning Windows Interface for Habitat System.

This module provides the interface for managing and analyzing learning windows,
which represent optimal periods for knowledge integration and relationship formation.
"""

class LearningWindowInterface:
    """Interface for managing and analyzing learning windows while maintaining coherence."""
    
    def __init__(self):
        self.window_registry = {}
        self.pattern_cache = {}
        self.density_cache = {}
        self.coherence_threshold = 0.85
        
    def register_window(self, window_data: dict) -> str:
        """Register a learning window with coherence validation and density analysis."""
        window_id = str(len(self.window_registry))
        
        if not self._validate_coherence(window_data):
            return None
            
        # Calculate density metrics
        density_metrics = self._calculate_density_metrics(window_data)
        
        # Track pattern evolution with density
        self._update_pattern_cache(window_data, density_metrics)
        
        # Store with coherence and density metadata
        self.window_registry[window_id] = {
            "state": {
                **window_data,
                "density": density_metrics
            },
            "coherence": {
                "timestamp": "2025-01-01T15:25:18-05:00",
                "structural_integrity": self._calculate_structural_integrity(window_data),
                "semantic_stability": self._calculate_semantic_stability(window_data),
                "propagation_potential": window_data.get("potential", 0)
            }
        }
        
        # Update density cache
        self.density_cache[window_id] = {
            "metrics": density_metrics,
            "gradients": self._calculate_density_gradients(density_metrics)
        }
        
        return window_id
        
    def get_window(self, window_id: str) -> dict:
        """Retrieve window data with coherence context."""
        if window_id not in self.window_registry:
            return None
            
        window = self.window_registry[window_id]
        return {
            **window,
            "patterns": self._get_relevant_patterns(window_id),
            "coherence_status": self._assess_coherence_status(window)
        }
        
    def get_density_analysis(self, window_id: str = None) -> dict:
        """Get density analysis for specific window or entire field."""
        if window_id and window_id in self.density_cache:
            return self.density_cache[window_id]
            
        # Analyze entire field density
        field_density = {
            "global_density": self._calculate_field_density(),
            "density_centers": self._identify_density_centers(),
            "cross_domain_paths": self._analyze_cross_domain_paths()
        }
        
        return field_density
        
    def _calculate_density_metrics(self, window_data: dict) -> dict:
        """Calculate density metrics using existing measures."""
        channels = window_data.get("channels", {})
        structural = channels.get("structural", {})
        semantic = channels.get("semantic", {})
        
        # Calculate local density as weighted average of structural and semantic metrics
        local_density = (
            structural.get("strength", 0.5) * 0.4 +  # 40% weight on structural strength
            semantic.get("sustainability", 0.5) * 0.3 +  # 30% weight on semantic sustainability
            window_data.get("score", 0.5) * 0.3  # 30% weight on overall score
        )
        
        # Calculate cross-domain potential as weighted combination
        cross_domain = (
            semantic.get("strength", 0.5) * 0.4 +  # 40% weight on semantic strength
            window_data.get("potential", 0.5) * 0.4 +  # 40% weight on potential
            window_data.get("horizon", 0.5) * 0.2  # 20% weight on horizon
        )
        
        # Identify domain alignments based on meaning patterns
        alignments = self._identify_domain_alignments(window_data)
        
        return {
            "local": local_density,
            "cross_domain": cross_domain,
            "gradient": abs(local_density - cross_domain),
            "alignments": alignments
        }
        
    def _calculate_density_gradients(self, density_metrics: dict) -> dict:
        """Calculate density gradients across the field."""
        return {
            "local_to_cross": density_metrics["gradient"],
            "alignment_strength": sum(a["strength"] for a in density_metrics["alignments"]) / 
                                len(density_metrics["alignments"]) if density_metrics["alignments"] else 0,
            "field_potential": (density_metrics["local"] + density_metrics["cross_domain"]) / 2
        }
        
    def _identify_domain_alignments(self, window_data: dict) -> list:
        """Identify domain alignments from meaning patterns."""
        semantic_patterns = window_data.get("semantic_patterns", [])
        alignments = []
        
        if semantic_patterns:
            # Group by domain and calculate alignment strength
            domain_strengths = {}
            domain_counts = {}
            for pattern in semantic_patterns:
                domain = pattern.get("domain", "general")
                strength = pattern.get("strength", 0.5)
                domain_strengths[domain] = domain_strengths.get(domain, 0) + strength
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Convert to alignment vectors with average strength per domain
            alignments = [
                {"domain": domain, "strength": strength / domain_counts[domain]}
                for domain, strength in domain_strengths.items()
            ]
            
            # Sort by strength for consistent ordering
            alignments.sort(key=lambda x: (-x["strength"], x["domain"]))
        
        return alignments
        
    def _calculate_field_density(self) -> float:
        """Calculate overall field density."""
        if not self.density_cache:
            return 0.0
            
        densities = [
            cache["metrics"]["local"]
            for cache in self.density_cache.values()
        ]
        return sum(densities) / len(densities)
        
    def _identify_density_centers(self) -> list:
        """Identify high-density centers in the field."""
        centers = []
        
        # Calculate average metrics across windows
        avg_local = sum(cache["metrics"]["local"] for cache in self.density_cache.values()) / len(self.density_cache) if self.density_cache else 0
        avg_cross = sum(cache["metrics"]["cross_domain"] for cache in self.density_cache.values()) / len(self.density_cache) if self.density_cache else 0
        
        # Identify windows with high density in either local or cross-domain metrics
        for window_id, cache in self.density_cache.items():
            metrics = cache["metrics"]
            local = metrics["local"]
            cross = metrics["cross_domain"]
            
            # Calculate combined density score
            density = (local * 0.4 + cross * 0.6)  # Weight cross-domain higher
            
            # Window is a density center if it has high combined density
            if density > 0.8:  # Direct threshold for high density
                centers.append({
                    "window_id": window_id,
                    "density": density,
                    "alignments": metrics["alignments"],
                    "type": "local" if local > cross else "cross_domain"
                })
            # Or if it significantly exceeds averages
            elif (local > avg_local * 1.2 and cross > avg_cross * 1.2):
                centers.append({
                    "window_id": window_id,
                    "density": density,
                    "alignments": metrics["alignments"],
                    "type": "emerging"
                })
        
        return sorted(centers, key=lambda x: x["density"], reverse=True)
        
    def _analyze_cross_domain_paths(self) -> list:
        """Analyze potential paths between domains."""
        paths = []
        windows = sorted(self.density_cache.items(), key=lambda x: x[0])
        
        for i in range(len(windows) - 1):
            current = windows[i][1]
            next_window = windows[i + 1][1]
            
            if current["metrics"]["cross_domain"] > self.coherence_threshold:
                paths.append({
                    "start": windows[i][0],
                    "end": windows[i + 1][0],
                    "strength": (current["metrics"]["cross_domain"] + 
                               next_window["metrics"]["cross_domain"]) / 2,
                    "stability": current["gradients"]["field_potential"]
                })
        
        return sorted(paths, key=lambda x: x["strength"], reverse=True)
        
    def _validate_coherence(self, window_data: dict) -> bool:
        """Validate window maintains minimum coherence requirements."""
        if not window_data:
            return False
            
        required_fields = ["score", "potential", "horizon"]
        if not all(field in window_data for field in required_fields):
            return False
            
        return (window_data["score"] >= self.coherence_threshold and 
                window_data["potential"] > 0 and 
                window_data["horizon"] > 0)
                
    def _calculate_structural_integrity(self, window_data: dict) -> float:
        """Calculate structural integrity score."""
        channels = window_data.get("channels", {})
        structural = channels.get("structural", {})
        
        # Weighted combination of structural metrics
        return (
            structural.get("strength", 0) * 0.6 +  # 60% weight on strength
            structural.get("sustainability", 0) * 0.4  # 40% weight on sustainability
        )
        
    def _calculate_semantic_stability(self, window_data: dict) -> float:
        """Calculate semantic stability score."""
        channels = window_data.get("channels", {})
        semantic = channels.get("semantic", {})
        
        # Weighted combination of semantic metrics
        return (
            semantic.get("strength", 0) * 0.6 +  # 60% weight on strength
            semantic.get("sustainability", 0) * 0.4  # 40% weight on sustainability
        )
        
    def _update_pattern_cache(self, window_data: dict, density_metrics: dict):
        """Update pattern cache with new window data."""
        pattern_key = f"{len(self.window_registry)}"
        
        self.pattern_cache[pattern_key] = {
            "structural": self._calculate_structural_integrity(window_data),
            "semantic": self._calculate_semantic_stability(window_data),
            "density": density_metrics
        }
        
    def _get_relevant_patterns(self, window_id: str) -> list:
        """Get patterns relevant to specific window."""
        window_patterns = []
        for pattern_key, pattern in self.pattern_cache.items():
            if pattern_key == window_id:
                window_patterns.append(pattern)
        return window_patterns
        
    def _assess_coherence_status(self, window: dict) -> dict:
        """Assess current coherence status of window."""
        coherence = window["coherence"]
        structural = coherence["structural_integrity"]
        semantic = coherence["semantic_stability"]
        propagation = coherence["propagation_potential"]
        
        # Window is coherent if either structural or semantic coherence is high
        # and propagation potential is sufficient
        is_coherent = (
            (structural >= self.coherence_threshold or 
             semantic >= self.coherence_threshold) and
            propagation >= self.coherence_threshold * 0.8  # Allow slightly lower propagation threshold
        )
        
        return {
            "is_coherent": is_coherent,
            "structural_status": "stable" if structural >= self.coherence_threshold else "evolving",
            "semantic_status": "stable" if semantic >= self.coherence_threshold else "evolving",
            "propagation_status": "active" if propagation >= self.coherence_threshold else "passive"
        }
