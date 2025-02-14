"""Social practice formation and stability analysis."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from datetime import datetime
import numpy as np

from .pattern import SocialPattern, PatternMetrics
from .field import SocialField

@dataclass
class PracticeMetrics:
    """Metrics for social practice analysis."""
    # Core metrics
    stability: float
    adoption_level: float
    institutionalization: float
    
    # Network metrics
    network_centrality: float
    influence_score: float
    
    # Resource metrics
    resource_efficiency: float
    knowledge_density: float
    
    # Resilience metrics
    adaptation_capacity: float
    response_diversity: float

@dataclass
class PracticeConfig:
    """Configuration for practice analysis."""
    # Formation thresholds
    min_stability: float = 0.6
    min_adoption: float = 0.4
    min_institutionalization: float = 0.5
    
    # Evolution parameters
    stability_momentum: float = 0.2
    network_growth_rate: float = 0.1
    resource_efficiency_gain: float = 0.15

class SocialPractice:
    """Represents an emerging or established social practice."""
    
    def __init__(self,
                 practice_id: str,
                 source_pattern: SocialPattern,
                 config: PracticeConfig):
        self.id = practice_id
        self.config = config
        self.creation_time = datetime.now()
        self.source_pattern = source_pattern
        self.contributing_patterns: Set[str] = {source_pattern.id}
        
        self.metrics = PracticeMetrics(
            stability=source_pattern.metrics.stability_index,
            adoption_level=source_pattern.metrics.adoption_rate,
            institutionalization=source_pattern.metrics.institutionalization,
            network_centrality=0.0,
            influence_score=0.0,
            resource_efficiency=0.0,
            knowledge_density=0.0,
            adaptation_capacity=0.0,
            response_diversity=0.0
        )
        
        # Track resource and knowledge flows
        self.resource_flows: Dict[str, float] = {}
        self.knowledge_flows: Dict[str, float] = {}
    
    def update(self, dt: float, active_patterns: Dict[str, SocialPattern]):
        """Update practice state and metrics."""
        self._update_core_metrics(active_patterns)
        self._update_network_metrics(active_patterns)
        self._update_resource_metrics(dt)
        self._update_resilience_metrics(active_patterns)
        
        # Identify new contributing patterns
        self._identify_contributors(active_patterns)
    
    def _update_core_metrics(self, active_patterns: Dict[str, SocialPattern]):
        """Update core practice metrics."""
        if not self.contributing_patterns:
            return
            
        # Calculate weighted averages from contributing patterns
        contributing = [
            p for pid, p in active_patterns.items()
            if pid in self.contributing_patterns
        ]
        
        self.metrics.stability = np.mean([
            p.metrics.stability_index for p in contributing
        ])
        
        self.metrics.adoption_level = np.mean([
            p.metrics.adoption_rate for p in contributing
        ])
        
        self.metrics.institutionalization = np.mean([
            p.metrics.institutionalization for p in contributing
        ])
    
    def _update_network_metrics(self, active_patterns: Dict[str, SocialPattern]):
        """Update network-based metrics."""
        if not active_patterns:
            return
            
        # Calculate network centrality based on pattern relationships
        relationship_counts = [
            len(p.relationships) for p in active_patterns.values()
            if p.id in self.contributing_patterns
        ]
        
        if relationship_counts:
            self.metrics.network_centrality = (
                np.mean(relationship_counts) / len(active_patterns)
            )
        
        # Calculate influence based on pattern reach
        influence_scores = [
            p.metrics.influence_reach for p in active_patterns.values()
            if p.id in self.contributing_patterns
        ]
        
        if influence_scores:
            self.metrics.influence_score = np.mean(influence_scores)
    
    def _update_resource_metrics(self, dt: float):
        """Update resource and knowledge flow metrics."""
        if not self.resource_flows:
            return
            
        # Update resource efficiency
        efficiency_delta = (
            self.metrics.stability * 0.4 +
            self.metrics.network_centrality * 0.6
        ) * self.config.resource_efficiency_gain * dt
        
        self.metrics.resource_efficiency = min(
            1.0, self.metrics.resource_efficiency + efficiency_delta
        )
        
        # Update knowledge density based on flow patterns
        total_knowledge = sum(self.knowledge_flows.values())
        active_flows = len([f for f in self.knowledge_flows.values() if f > 0.1])
        
        if active_flows > 0:
            self.metrics.knowledge_density = total_knowledge / active_flows
    
    def _update_resilience_metrics(self, active_patterns: Dict[str, SocialPattern]):
        """Update resilience and adaptation metrics."""
        if not active_patterns:
            return
            
        # Calculate adaptation capacity from stability and diversity
        self.metrics.adaptation_capacity = (
            self.metrics.stability * 0.4 +
            self.metrics.knowledge_density * 0.3 +
            self.metrics.network_centrality * 0.3
        )
        
        # Calculate response diversity from contributing patterns
        pattern_types = {
            p.id: p.metrics.field_coherence
            for p in active_patterns.values()
            if p.id in self.contributing_patterns
        }
        
        if pattern_types:
            self.metrics.response_diversity = np.std(list(pattern_types.values()))
    
    def _identify_contributors(self, active_patterns: Dict[str, SocialPattern]):
        """Identify patterns that contribute to this practice."""
        for pid, pattern in active_patterns.items():
            if pid in self.contributing_patterns:
                continue
                
            # Check if pattern should contribute based on:
            # - High stability
            # - Strong institutionalization
            # - Network connection to existing contributors
            if (pattern.metrics.stability_index > self.config.min_stability and
                pattern.metrics.institutionalization > self.config.min_institutionalization):
                
                # Check for connections to existing contributors
                connected = any(
                    cpid in pattern.relationships
                    for cpid in self.contributing_patterns
                )
                
                if connected:
                    self.contributing_patterns.add(pid)
                    
                    # Initialize flows for new contributor
                    self.resource_flows[pid] = 0.0
                    self.knowledge_flows[pid] = 0.0
