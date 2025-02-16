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
from ..core.pattern.attention import AttentionFilter
from ..metrics import MetricsLogger

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
        
        # Update history
        self.vector_history.append(vector)
        if len(self.vector_history) > self.window_size:
            self.vector_history.pop(0)
        
        # Calculate metrics
        metrics = self._calculate_metrics(vector, attention_weight)
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
        if len(self.vector_history) < 2:
            return 0.0
        
        # Simple cosine distance is sufficient for detecting semantic shifts
        prev_vector = self.vector_history[-1]
        cosine_sim = np.dot(vector, prev_vector) / (
            np.linalg.norm(vector) * np.linalg.norm(prev_vector)
        )
        return 1.0 - cosine_sim  # Higher value = stronger edge
    
    def _analyze_stability(self) -> float:
        """Analyze stability for learning window adjustment."""
        if len(self.vector_history) < self.window_size:
            return 1.0
            
        vectors = np.array(self.vector_history)
        # Simple variance-based stability measure
        variance = np.var(vectors, axis=0).mean()
        return 1.0 / (1.0 + variance)  # Inverse variance as stability
    
    def _calculate_density(self, vector: np.ndarray) -> float:
        """Calculate density for pattern identification."""
        if len(self.vector_history) < 2:
            return 0.0
        
        # Count close neighbors - sufficient for basic clustering
        vectors = np.array(self.vector_history)
        distances = np.linalg.norm(vectors - vector, axis=1)
        return np.mean(distances < self.density_radius)
    
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
            return np.zeros_like(self.vector_history[0])
        
        # Simple moving average of changes
        vectors = np.array(self.vector_history)
        return np.mean(np.diff(vectors, axis=0), axis=0)
    
    def _log_metrics(self, metrics: VectorSpaceMetrics):
        """Log metrics for monitoring and navigation."""
        log_data = {
            "timestamp": metrics.timestamp.isoformat(),
            "edge_strength": metrics.edge_strength,
            "stability_score": metrics.stability_score,
            "local_density": metrics.local_density,
            "turbulence_level": metrics.turbulence_level,
            "drift_magnitude": float(np.linalg.norm(metrics.drift_velocity)),
            "attention_weight": metrics.attention_weight
        }
        
        # Log to metrics system
        self.metrics_logger.log_metrics("vector_space", log_data)
        
        # Log significant events
        if metrics.edge_strength > self.edge_threshold:
            self.logger.info(f"Semantic edge detected: {metrics.edge_strength:.3f}")
        
        if metrics.stability_score < self.stability_threshold:
            self.logger.warning(f"Low stability detected: {metrics.stability_score:.3f}")
        
        if metrics.turbulence_level > 0.5:
            self.logger.warning(f"High turbulence detected: {metrics.turbulence_level:.3f}")
