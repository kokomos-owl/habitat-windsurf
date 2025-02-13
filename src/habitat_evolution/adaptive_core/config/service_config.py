"""
Configuration management for Adaptive Core services.
"""

from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class PatternEvolutionConfig:
    """Configuration for pattern evolution service"""
    coherence_threshold: float = 0.7
    update_interval: int = 1000  # milliseconds
    max_patterns: int = 10000
    min_signal_strength: float = 0.3
    max_uncertainty: float = 0.5

@dataclass
class MetricsConfig:
    """Configuration for metrics service"""
    calculation_batch_size: int = 100
    cache_duration: int = 3600  # seconds
    update_frequency: int = 60  # seconds
    precision: int = 4
    storage_retention: int = 30  # days

@dataclass
class StateManagementConfig:
    """Configuration for state management service"""
    max_versions: int = 100
    cleanup_interval: int = 86400  # seconds
    compression_enabled: bool = True
    diff_storage: bool = True
    retention_period: int = 90  # days

@dataclass
class RelationshipConfig:
    """Configuration for relationship service"""
    max_relationships_per_pattern: int = 1000
    strength_threshold: float = 0.3
    phase_lock_threshold: float = 0.8
    bidirectional_threshold: float = 0.7
    cleanup_interval: int = 3600  # seconds

@dataclass
class QualityMetricsConfig:
    """Configuration for quality metrics service"""
    signal_threshold: float = 0.5
    coherence_threshold: float = 0.7
    stability_threshold: float = 0.6
    flow_threshold: float = 0.4
    update_interval: int = 300  # seconds

@dataclass
class EventManagementConfig:
    """Configuration for event management service"""
    max_subscribers: int = 1000
    queue_size: int = 10000
    retention_period: int = 30  # days
    batch_size: int = 100
    processing_interval: int = 1  # seconds

@dataclass
class PersistenceConfig:
    """Configuration for persistence layer"""
    connection_pool_size: int = 10
    retry_attempts: int = 3
    timeout: int = 30  # seconds
    batch_size: int = 100
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds

@dataclass
class ServiceConfig:
    """Master configuration for all services"""
    pattern_evolution: PatternEvolutionConfig = field(default_factory=PatternEvolutionConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    state_management: StateManagementConfig = field(default_factory=StateManagementConfig)
    relationship: RelationshipConfig = field(default_factory=RelationshipConfig)
    quality_metrics: QualityMetricsConfig = field(default_factory=QualityMetricsConfig)
    event_management: EventManagementConfig = field(default_factory=EventManagementConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary representation"""
        return {
            "pattern_evolution": {
                "coherence_threshold": self.pattern_evolution.coherence_threshold,
                "update_interval": self.pattern_evolution.update_interval,
                "max_patterns": self.pattern_evolution.max_patterns,
                "min_signal_strength": self.pattern_evolution.min_signal_strength,
                "max_uncertainty": self.pattern_evolution.max_uncertainty
            },
            "metrics": {
                "calculation_batch_size": self.metrics.calculation_batch_size,
                "cache_duration": self.metrics.cache_duration,
                "update_frequency": self.metrics.update_frequency,
                "precision": self.metrics.precision,
                "storage_retention": self.metrics.storage_retention
            },
            "state_management": {
                "max_versions": self.state_management.max_versions,
                "cleanup_interval": self.state_management.cleanup_interval,
                "compression_enabled": self.state_management.compression_enabled,
                "diff_storage": self.state_management.diff_storage,
                "retention_period": self.state_management.retention_period
            },
            "relationship": {
                "max_relationships_per_pattern": self.relationship.max_relationships_per_pattern,
                "strength_threshold": self.relationship.strength_threshold,
                "phase_lock_threshold": self.relationship.phase_lock_threshold,
                "bidirectional_threshold": self.relationship.bidirectional_threshold,
                "cleanup_interval": self.relationship.cleanup_interval
            },
            "quality_metrics": {
                "signal_threshold": self.quality_metrics.signal_threshold,
                "coherence_threshold": self.quality_metrics.coherence_threshold,
                "stability_threshold": self.quality_metrics.stability_threshold,
                "flow_threshold": self.quality_metrics.flow_threshold,
                "update_interval": self.quality_metrics.update_interval
            },
            "event_management": {
                "max_subscribers": self.event_management.max_subscribers,
                "queue_size": self.event_management.queue_size,
                "retention_period": self.event_management.retention_period,
                "batch_size": self.event_management.batch_size,
                "processing_interval": self.event_management.processing_interval
            },
            "persistence": {
                "connection_pool_size": self.persistence.connection_pool_size,
                "retry_attempts": self.persistence.retry_attempts,
                "timeout": self.persistence.timeout,
                "batch_size": self.persistence.batch_size,
                "cache_enabled": self.persistence.cache_enabled,
                "cache_ttl": self.persistence.cache_ttl
            }
        }
