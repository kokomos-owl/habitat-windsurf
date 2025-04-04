#!/usr/bin/env python3
"""
Standalone test for Elastic Memory RAG integration.

This module provides a self-contained test environment for the elastic
semantic memory approach, implementing the RAG↔Evolution↔Persistence loop
without dependencies on other modules.
"""

import logging
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
import uuid
import random
import threading
import networkx as nx
import matplotlib.pyplot as plt

# Import AdaptiveID if available, otherwise use mock implementation
try:
    from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
except ImportError:
    # We'll use our mock implementation below
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create directories for test outputs
project_root = Path(__file__).parent
test_dir = project_root / "test_results" / "elastic_memory_rag"
test_dir.mkdir(parents=True, exist_ok=True)
persistence_dir = test_dir / "persistence"
persistence_dir.mkdir(exist_ok=True)
visualization_dir = test_dir / "visualizations"
visualization_dir.mkdir(exist_ok=True)

# ====================================================================
# Mock Classes and Data Structures
# ====================================================================

class MockLogManager:
    """Mock logging manager for AdaptiveID."""
    
    def __init__(self):
        self.logs = []
    
    def info(self, message):
        self.logs.append({"level": "INFO", "message": message, "timestamp": datetime.now().isoformat()})
    
    def error(self, message):
        self.logs.append({"level": "ERROR", "message": message, "timestamp": datetime.now().isoformat()})

class MockAdaptiveID:
    """Mock implementation of AdaptiveID for testing.
    
    This class implements the core functionality of AdaptiveID needed for our tests,
    including versioning, context tracking, and state management.
    """
    
    @staticmethod
    def generate() -> str:
        """Generate a new unique ID."""
        return str(uuid.uuid4())
    
    def __init__(self, base_concept: str, creator_id: str = "test_system"):
        """Initialize a mock AdaptiveID."""
        self.id = self.generate()
        self.base_concept = base_concept
        self.creator_id = creator_id
        self.weight = 1.0
        self.confidence = 0.5
        self.uncertainty = 0.5
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Version tracking
        self.versions = {}
        self.current_version = self.generate()
        
        # Context tracking
        self.temporal_context = {}
        self.spatial_context = {}
        
        # Metadata
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "version_count": 0
        }
        
        # Logging
        self.logger = MockLogManager()
        
        # Create initial version
        self._create_initial_version()
    
    def _create_initial_version(self) -> None:
        """Create the initial version."""
        initial_data = {
            "base_concept": self.base_concept,
            "weight": self.weight,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty
        }
        
        with self._lock:
            self.versions[self.current_version] = {
                "version_id": self.current_version,
                "data": initial_data,
                "timestamp": self.metadata["created_at"],
                "origin": "initialization"
            }
            self.metadata["version_count"] = 1
    
    def update_confidence(self, new_confidence: float, origin: str) -> None:
        """Update the confidence score."""
        with self._lock:
            old_confidence = self.confidence
            self.confidence = max(0.0, min(1.0, new_confidence))  # Clamp between 0 and 1
            
            # Create new version
            version_id = self.generate()
            timestamp = datetime.now().isoformat()
            
            self.versions[version_id] = {
                "version_id": version_id,
                "data": {
                    "base_concept": self.base_concept,
                    "weight": self.weight,
                    "confidence": self.confidence,
                    "uncertainty": self.uncertainty
                },
                "timestamp": timestamp,
                "origin": origin
            }
            
            self.current_version = version_id
            self.metadata["last_modified"] = timestamp
            self.metadata["version_count"] += 1
            
            self.logger.info(f"Updated confidence from {old_confidence} to {self.confidence} (origin: {origin})")
    
    def update_temporal_context(self, key: str, value: Any, origin: str) -> None:
        """Update temporal context."""
        with self._lock:
            if key not in self.temporal_context:
                self.temporal_context[key] = {}
            
            timestamp = datetime.now().isoformat()
            self.temporal_context[key][timestamp] = {
                "value": value,
                "origin": origin
            }
            self.metadata["last_modified"] = timestamp
    
    def get_temporal_context(self, key: str) -> Any:
        """Get temporal context value."""
        if key not in self.temporal_context:
            return None
        
        # Return most recent value
        timestamps = sorted(self.temporal_context[key].keys())
        if not timestamps:
            return None
        
        return self.temporal_context[key][timestamps[-1]]["value"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "base_concept": self.base_concept,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "weight": self.weight,
            "current_version": self.current_version,
            "metadata": self.metadata
        }

class Pattern:
    """Pattern class for representing extracted patterns."""
    
    def __init__(self, id: str, text: str, source: str = None, confidence: float = 0.5, metadata: Dict[str, Any] = None):
        """
        Initialize a pattern.
        
        Args:
            id: Pattern ID
            text: Pattern text
            source: Source of the pattern
            confidence: Confidence score for the pattern
            metadata: Optional metadata dictionary
        """
        self.id = id
        self.text = text
        self.source = source
        self.confidence = confidence
        self.metadata = metadata or {}
        self.quality = self.metadata.get("quality_state", "uncertain")
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "state": self.state,
            "quality": self.quality,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }

class Relationship:
    """Mock Relationship class for testing."""
    
    def __init__(self, source: str, target: str, predicate: str, metadata: Dict[str, Any] = None):
        """Initialize a relationship."""
        self.id = str(uuid.uuid4())
        self.source = source
        self.target = target
        self.predicate = predicate
        self.metadata = metadata or {}
        self.quality = metadata.get("quality", "uncertain")
        self.confidence = metadata.get("confidence", 0.5)
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary."""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "predicate": self.predicate,
            "metadata": self.metadata,
            "quality": self.quality,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }

class QualityAwarePatternContext:
    """Mock QualityAwarePatternContext class for testing."""
    
    def __init__(self, coherence_level: float = 0.5):
        """Initialize a quality-aware pattern context."""
        self.coherence_level = coherence_level
        self.pattern_state_distribution = {"ACTIVE": 7, "EMERGING": 3}
        self.patterns = []
        self.entity_quality = {}
        self.predicate_quality = {}
    
    def prioritize_patterns_by_quality(self) -> List[Pattern]:
        """Prioritize patterns by quality."""
        return sorted(self.patterns, key=lambda p: 1.0 if p.quality == "good" else (0.5 if p.quality == "uncertain" else 0.0), reverse=True)
    
    def add_pattern(self, pattern: Pattern):
        """Add a pattern to the context."""
        self.patterns.append(pattern)
    
    def set_entity_quality(self, entity: str, quality: str, confidence: float = None):
        """Set entity quality."""
        self.entity_quality[entity] = {
            "quality": quality,
            "confidence": confidence or (0.8 if quality == "good" else (0.5 if quality == "uncertain" else 0.2))
        }
    
    def set_predicate_quality(self, predicate: str, quality: str, confidence: float = None):
        """Set predicate quality."""
        self.predicate_quality[predicate] = {
            "quality": quality,
            "confidence": confidence or (0.8 if quality == "good" else (0.5 if quality == "uncertain" else 0.2))
        }

class RetrievalResult:
    """Mock RetrievalResult class for testing."""
    
    def __init__(self, patterns: List[Pattern] = None, entity_relationships: List[Dict[str, Any]] = None):
        """Initialize a retrieval result."""
        self.patterns = patterns or []
        self.entity_relationships = entity_relationships or []
        self.quality_distribution = self._calculate_quality_distribution()
        self.confidence = self._calculate_confidence()
    
    def _calculate_quality_distribution(self) -> Dict[str, int]:
        """Calculate quality distribution."""
        distribution = {"good": 0, "uncertain": 0, "poor": 0}
        for pattern in self.patterns:
            quality = pattern.quality
            distribution[quality] = distribution.get(quality, 0) + 1
        return distribution
    
    def _calculate_confidence(self) -> float:
        """Calculate overall confidence."""
        if not self.patterns:
            return 0.0
        
        total_confidence = sum(pattern.confidence for pattern in self.patterns)
        return total_confidence / len(self.patterns)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert retrieval result to dictionary."""
        return {
            "patterns": [pattern.to_dict() for pattern in self.patterns],
            "entity_relationships": self.entity_relationships,
            "quality_distribution": self.quality_distribution,
            "confidence": self.confidence
        }

class MockEventBus:
    """Mock EventBus for testing."""
    
    def __init__(self):
        """Initialize the mock event bus."""
        self.subscribers = {}
    
    def subscribe(self, event_type, callback):
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def publish(self, event):
        """Publish an event."""
        event_type = event.get("event_type")
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                callback(event)

# ====================================================================
# Core Implementation Classes
# ====================================================================

class PredicateQualityTracker:
    """
    Tracks quality states for predicates (relationships between entities).
    
    This class manages the quality states (poor, uncertain, good) for predicates,
    tracks confidence scores, and records transition histories.
    """
    
    def __init__(self, event_bus=None, logger=None):
        """
        Initialize the predicate quality tracker.
        
        Args:
            event_bus: Optional event bus for publishing events
            logger: Optional logger
        """
        self.event_bus = event_bus
        self.logger = logger or logging.getLogger(__name__)
        
        # Quality state tracking
        self.predicate_quality = {}  # predicate -> quality_state
        self.predicate_confidence = {}  # predicate -> confidence_score
        
        # Domain-specific quality tracking
        self.domain_predicate_specialization = {}  # predicate -> {(source_domain, target_domain) -> count}
        
        # Transition history
        self.quality_transition_history = {}  # predicate -> [transition_events]
    
    def get_predicate_quality(self, predicate: str) -> str:
        """
        Get the current quality state for a predicate.
        
        Args:
            predicate: The predicate to get quality for
            
        Returns:
            Quality state (poor, uncertain, good), defaults to uncertain
        """
        return self.predicate_quality.get(predicate, "uncertain")
    
    def get_predicate_confidence(self, predicate: str) -> float:
        """
        Get the current confidence score for a predicate.
        
        Args:
            predicate: The predicate to get confidence for
            
        Returns:
            Confidence score (0.0-1.0), defaults to 0.5
        """
        return self.predicate_confidence.get(predicate, 0.5)
    
    def transition_predicate_quality(self, predicate: str, to_quality: str,
                                    source_domain: Optional[str] = None,
                                    target_domain: Optional[str] = None,
                                    evidence: Optional[str] = None) -> bool:
        """
        Transition a predicate to a new quality state.
        
        Args:
            predicate: The predicate to transition
            to_quality: The target quality state
            source_domain: Optional source domain for domain-specific transitions
            target_domain: Optional target domain for domain-specific transitions
            evidence: Optional evidence supporting this transition
            
        Returns:
            True if transition was successful, False otherwise
        """
        # Get current quality
        from_quality = self.get_predicate_quality(predicate)
        
        # Update quality state
        self.predicate_quality[predicate] = to_quality
        
        # Update confidence based on quality
        quality_confidence = {
            "poor": 0.2,
            "uncertain": 0.5,
            "good": 0.8
        }
        self.predicate_confidence[predicate] = quality_confidence.get(to_quality, 0.5)
        
        # Track domain specialization if domains are provided
        if source_domain and target_domain:
            if predicate not in self.domain_predicate_specialization:
                self.domain_predicate_specialization[predicate] = {}
            
            domain_pair = (source_domain, target_domain)
            self.domain_predicate_specialization[predicate][domain_pair] = self.domain_predicate_specialization[predicate].get(domain_pair, 0) + 1
        
        # Record transition in history
        transition = {
            "timestamp": datetime.now().isoformat(),
            "from_quality": from_quality,
            "to_quality": to_quality,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "evidence": evidence
        }
        
        if predicate not in self.quality_transition_history:
            self.quality_transition_history[predicate] = []
        
        self.quality_transition_history[predicate].append(transition)
        
        # Publish event if event bus is available
        if self.event_bus:
            event_data = {
                "predicate": predicate,
                "from_quality": from_quality,
                "to_quality": to_quality,
                "confidence": self.predicate_confidence[predicate],
                "source_domain": source_domain,
                "target_domain": target_domain,
                "evidence": evidence
            }
            
            event = {
                "event_type": "predicate.quality.transition",
                "source": "predicate_quality_tracker",
                "data": event_data
            }
            
            self.event_bus.publish(event)
        
        self.logger.info(f"Predicate quality transition: {predicate} from {from_quality} to {to_quality}")
        
        return True
    
    def get_domain_specialization(self, predicate: str) -> Dict[Tuple[str, str], int]:
        """
        Get domain specialization for a predicate.
        
        Args:
            predicate: The predicate to get specialization for
            
        Returns:
            Dictionary mapping (source_domain, target_domain) to count
        """
        return self.domain_predicate_specialization.get(predicate, {})
    
    def get_transition_history(self, predicate: str) -> List[Dict[str, Any]]:
        """
        Get transition history for a predicate.
        
        Args:
            predicate: The predicate to get history for
            
        Returns:
            List of transition events
        """
        return self.quality_transition_history.get(predicate, [])
    
    def get_all_predicates(self) -> List[str]:
        """
        Get all tracked predicates.
        
        Returns:
            List of all predicates
        """
        return list(self.predicate_quality.keys())
    
    def get_predicates_by_quality(self, quality: str) -> List[str]:
        """
        Get predicates with a specific quality.
        
        Args:
            quality: Quality state to filter by
            
        Returns:
            List of predicates with the specified quality
        """
        return [pred for pred, qual in self.predicate_quality.items() if qual == quality]
    
    def get_predicates_by_domain_pair(self, source_domain: str, target_domain: str) -> List[str]:
        """
        Get predicates that connect a specific domain pair.
        
        Args:
            source_domain: Source domain
            target_domain: Target domain
            
        Returns:
            List of predicates that connect the specified domains
        """
        result = []
        for predicate, specialization in self.domain_predicate_specialization.items():
            if (source_domain, target_domain) in specialization:
                result.append(predicate)
        return result


class SemanticMemoryPersistence:
    """
    Persistence layer for semantic memory.
    
    This class handles the storage and retrieval of entity and predicate quality states,
    transition histories, and vector field snapshots.
    """
    
    def __init__(self, base_dir: str = None, logger=None):
        """
        Initialize the semantic memory persistence layer.
        
        Args:
            base_dir: Base directory for persistence files
            logger: Optional logger
        """
        self.base_dir = base_dir or os.path.join(os.getcwd(), "persistence")
        self.logger = logger or logging.getLogger(__name__)


        # Create directories
        self.entity_dir = os.path.join(self.base_dir, "entities")
        self.predicate_dir = os.path.join(self.base_dir, "predicates")
        self.network_dir = os.path.join(self.base_dir, "networks")
        self.field_dir = os.path.join(self.base_dir, "vector_field")
        
        os.makedirs(self.entity_dir, exist_ok=True)
        os.makedirs(self.predicate_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)
        os.makedirs(self.field_dir, exist_ok=True)
        
        self.logger.info(f"Initialized SemanticMemoryPersistence with base_dir={self.base_dir}")
    
    def save_entity_quality(self, entity_quality: Dict[str, str], entity_confidence: Dict[str, float],
                          quality_transition_history: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Save entity quality states and transition history.
        
        Args:
            entity_quality: Dictionary mapping entity names to quality states
            entity_confidence: Dictionary mapping entity names to confidence scores
            quality_transition_history: Dictionary mapping entity names to transition histories
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"entity_quality_{timestamp}.json"
        filepath = os.path.join(self.entity_dir, filename)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "entity_quality": entity_quality,
            "entity_confidence": entity_confidence,
            "quality_transition_history": quality_transition_history
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved entity quality data to {filepath}")
        
        return filepath
    
    def save_predicate_quality(self, predicate_quality: Dict[str, str], predicate_confidence: Dict[str, float],
                             domain_predicate_specialization: Dict[str, Dict[Tuple[str, str], int]],
                             quality_transition_history: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Save predicate quality states and transition history.
        
        Args:
            predicate_quality: Dictionary mapping predicate names to quality states
            predicate_confidence: Dictionary mapping predicate names to confidence scores
            domain_predicate_specialization: Dictionary mapping predicates to domain specializations
            quality_transition_history: Dictionary mapping predicate names to transition histories
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predicate_quality_{timestamp}.json"
        filepath = os.path.join(self.predicate_dir, filename)
        
        # Convert tuple keys to strings for JSON serialization
        serializable_specialization = {}
        for predicate, specialization in domain_predicate_specialization.items():
            serializable_specialization[predicate] = {
                f"{source_domain}:{target_domain}": count
                for (source_domain, target_domain), count in specialization.items()
            }
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "predicate_quality": predicate_quality,
            "predicate_confidence": predicate_confidence,
            "domain_predicate_specialization": serializable_specialization,
            "quality_transition_history": quality_transition_history
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved predicate quality data to {filepath}")
        
        return filepath
    
    def save_entity_network(self, entity_network) -> str:
        """
        Save entity network graph.
        
        Args:
            entity_network: NetworkX graph of entity relationships
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"entity_network_{timestamp}.json"
        filepath = os.path.join(self.network_dir, filename)
        
        # Convert NetworkX graph to serializable format
        data = {
            "timestamp": datetime.now().isoformat(),
            "nodes": [
                {"id": node, **entity_network.nodes[node]}
                for node in entity_network.nodes
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    **entity_network.edges[source, target]
                }
                for source, target in entity_network.edges
            ]
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved entity network to {filepath}")
        
        return filepath
    
    def save_vector_field_snapshot(self, field_metrics: Dict[str, float],
                                 field_stability: float,
                                 field_coherence: float,
                                 field_density: Dict[str, float]) -> str:
        """
        Save vector field snapshot.
        
        Args:
            field_metrics: Dictionary of field metrics
            field_stability: Field stability score
            field_coherence: Field coherence score
            field_density: Dictionary of field density metrics
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vector_field_{timestamp}.json"
        filepath = os.path.join(self.field_dir, filename)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "field_metrics": field_metrics,
            "field_stability": field_stability,
            "field_coherence": field_coherence,
            "field_density": field_density
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved vector field snapshot to {filepath}")
        
        return filepath
    
    def save_complete_state(self, entity_quality: Dict[str, str], entity_confidence: Dict[str, float],
                          entity_transition_history: Dict[str, List[Dict[str, Any]]],
                          predicate_quality: Dict[str, str], predicate_confidence: Dict[str, float],
                          domain_predicate_specialization: Dict[str, Dict[Tuple[str, str], int]],
                          predicate_transition_history: Dict[str, List[Dict[str, Any]]],
                          field_metrics: Dict[str, float], field_stability: float,
                          field_coherence: float, field_density: Dict[str, float],
                          entity_network) -> Dict[str, str]:
        """
        Save complete state of the semantic memory.
        
        Args:
            entity_quality: Dictionary mapping entity names to quality states
            entity_confidence: Dictionary mapping entity names to confidence scores
            entity_transition_history: Dictionary mapping entity names to transition histories
            predicate_quality: Dictionary mapping predicate names to quality states
            predicate_confidence: Dictionary mapping predicate names to confidence scores
            domain_predicate_specialization: Dictionary mapping predicates to domain specializations
            predicate_transition_history: Dictionary mapping predicate names to transition histories
            field_metrics: Dictionary of field metrics
            field_stability: Field stability score
            field_coherence: Field coherence score
            field_density: Dictionary of field density metrics
            entity_network: NetworkX graph of entity relationships
            
        Returns:
            Dictionary mapping data type to saved file path
        """
        # Save entity quality
        entity_quality_path = self.save_entity_quality(
            entity_quality=entity_quality,
            entity_confidence=entity_confidence,
            quality_transition_history=entity_transition_history
        )
        
        # Save predicate quality
        predicate_quality_path = self.save_predicate_quality(
            predicate_quality=predicate_quality,
            predicate_confidence=predicate_confidence,
            domain_predicate_specialization=domain_predicate_specialization,
            quality_transition_history=predicate_transition_history
        )
        
        # Save entity network
        entity_network_path = self.save_entity_network(entity_network)
        
        # Save vector field snapshot
        vector_field_path = self.save_vector_field_snapshot(
            field_metrics=field_metrics,
            field_stability=field_stability,
            field_coherence=field_coherence,
            field_density=field_density
        )
        
        # Save metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_filename = f"state_metadata_{timestamp}.json"
        metadata_filepath = os.path.join(self.base_dir, metadata_filename)
        
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "entity_quality_path": entity_quality_path,
            "predicate_quality_path": predicate_quality_path,
            "entity_network_path": entity_network_path,
            "vector_field_path": vector_field_path,
            "entity_count": len(entity_quality),
            "predicate_count": len(predicate_quality),
            "relationship_count": entity_network.number_of_edges() if entity_network else 0
        }
        
        with open(metadata_filepath, "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved complete state metadata to {metadata_filepath}")
        
        return {
            "entity_quality": entity_quality_path,
            "predicate_quality": predicate_quality_path,
            "entity_network": entity_network_path,
            "vector_field": vector_field_path,
            "metadata": metadata_filepath
        }
    
    def get_latest_state(self) -> Dict[str, Any]:
        """
        Get the latest state of the semantic memory.
        
        Returns:
            Dictionary with the latest state data
        """
        # Find the latest metadata file
        metadata_files = [f for f in os.listdir(self.base_dir) if f.startswith("state_metadata_")]
        
        if not metadata_files:
            self.logger.warning("No state metadata files found")
            return {}
        
        latest_metadata_file = sorted(metadata_files)[-1]
        metadata_path = os.path.join(self.base_dir, latest_metadata_file)
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Load entity quality
        entity_quality_data = {}
        if "entity_quality_path" in metadata and os.path.exists(metadata["entity_quality_path"]):
            with open(metadata["entity_quality_path"], "r") as f:
                entity_quality_data = json.load(f)
        
        # Load predicate quality
        predicate_quality_data = {}
        if "predicate_quality_path" in metadata and os.path.exists(metadata["predicate_quality_path"]):
            with open(metadata["predicate_quality_path"], "r") as f:
                predicate_quality_data = json.load(f)
        
        # Load vector field
        vector_field_data = {}
        if "vector_field_path" in metadata and os.path.exists(metadata["vector_field_path"]):
            with open(metadata["vector_field_path"], "r") as f:
                vector_field_data = json.load(f)
        
        return {
            "timestamp": metadata.get("timestamp"),
            "entity_quality": entity_quality_data,
            "predicate_quality": predicate_quality_data,
            "vector_field": vector_field_data,
            "entity_count": metadata.get("entity_count", 0),
            "predicate_count": metadata.get("predicate_count", 0),
            "relationship_count": metadata.get("relationship_count", 0)
        }
    
    def load_entity_network(self):
        """
        Load the latest entity network.
        
        Returns:
            NetworkX graph of entity relationships
        """
        # Find the latest network file
        network_files = [f for f in os.listdir(self.network_dir) if f.startswith("entity_network_")]
        
        if not network_files:
            self.logger.warning("No entity network files found")
            return nx.DiGraph()
        
        latest_network_file = sorted(network_files)[-1]
        network_path = os.path.join(self.network_dir, latest_network_file)
        
        with open(network_path, "r") as f:
            data = json.load(f)
        
        # Create NetworkX graph from serialized data
        G = nx.DiGraph()
        
        # Add nodes
        for node_data in data.get("nodes", []):
            node_id = node_data.pop("id")
            G.add_node(node_id, **node_data)
        
        # Add edges
        for edge_data in data.get("edges", []):
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            G.add_edge(source, target, **edge_data)
        
        self.logger.info(f"Loaded entity network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G


class QualityEnhancedRetrieval:
    """
    Quality-enhanced retrieval component for elastic semantic memory.
    
    This class enhances retrieval capabilities based on entity and predicate quality assessments.
    """
    
    def __init__(self, predicate_quality_tracker=None, persistence_layer=None, event_bus=None):
        """
        Initialize the quality-enhanced retrieval component.
        
        Args:
            predicate_quality_tracker: Predicate quality tracker
            persistence_layer: Semantic memory persistence layer
            event_bus: Event bus for publishing events
        """
        self.predicate_quality_tracker = predicate_quality_tracker
        self.persistence_layer = persistence_layer
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
    
    def retrieve(self, query: str, context: Optional[QualityAwarePatternContext] = None, 
                 max_results: int = 10, use_persistence: bool = True) -> RetrievalResult:
        """
        Retrieve patterns with quality awareness and persistence.
        
        Args:
            query: Query string
            context: Quality-aware pattern context
            max_results: Maximum number of results to return
            use_persistence: Whether to use persistence layer
            
        Returns:
            RetrievalResult with retrieved patterns and persistence info
        """
        # Create mock patterns
        patterns = []
        for i in range(max_results):
            pattern_id = f"pattern_{i}"
            pattern_text = f"Pattern {i} related to {query}"
            entity = f"entity_{i}"
            quality = "good" if i < max_results // 3 else ("uncertain" if i < 2 * max_results // 3 else "poor")
            confidence = 0.8 if quality == "good" else (0.5 if quality == "uncertain" else 0.2)
            
            pattern = Pattern(
                id=pattern_id,
                text=pattern_text,
                confidence=confidence,
                source="mock_retrieval"
            )
            patterns.append(pattern)
        
        # Create mock entity relationships
        entity_relationships = []
        entities = [f"entity_{i}" for i in range(5)]
        predicates = ["affects", "part_of", "contains", "related_to", "causes"]
        
        for i in range(10):
            source = random.choice(entities)
            target = random.choice([e for e in entities if e != source])
            predicate = random.choice(predicates)
            
            relationship = {
                "source": source,
                "target": target,
                "predicate": predicate,
                "confidence": random.uniform(0.5, 0.9)
            }
            
            entity_relationships.append(relationship)
        
        # Create retrieval result
        result = RetrievalResult(
            patterns=patterns,
            entity_relationships=entity_relationships
        )
        
        return result


class MockPatternAwareRAG:
    """Mock implementation of PatternAwareRAG for testing.
    
    This class provides a simplified version of the PatternAwareRAG functionality
    needed for our elastic memory RAG integration test.
    """
    
    def __init__(self, event_bus=None):
        """Initialize the mock PatternAwareRAG."""
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        
        # Pattern tracking
        self.patterns = []
        self.relationships = []
        
        # State tracking
        self.window_metrics = {
            'local_density': 0.0,
            'global_density': 0.0,
            'coherence': 0.5,
            'cross_paths': [],
            'back_pressure': 0.0,
            'flow_stability': 0.5
        }
        
        # Learning window state
        self.current_window_state = "CLOSED"  # Can be CLOSED, OPENING, or OPEN
        
        self.logger.info("Initialized Mock PatternAwareRAG")
    
    async def process_with_patterns(self, query: str, context: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Process query with pattern awareness."""
        # Mock pattern extraction
        query_patterns = self._extract_query_patterns(query)
        
        # Mock retrieval
        retrieval_patterns = [f"retrieval_pattern_{i}" for i in range(3)]
        
        # Mock augmentation
        augmentation_patterns = [f"augmentation_pattern_{i}" for i in range(2)]
        
        # Create result
        result = {
            "content": f"Processed query: {query}",
            "pattern_id": str(uuid.uuid4()),
            "coherence": {
                "flow_state": "stable",
                "patterns": query_patterns + retrieval_patterns,
                "confidence": 0.7,
                "emergence_potential": 0.3
            }
        }
        
        # Create pattern context
        pattern_context = {
            "query_patterns": query_patterns,
            "retrieval_patterns": retrieval_patterns,
            "augmentation_patterns": augmentation_patterns,
            "coherence_level": 0.7
        }
        
        return result, pattern_context
    
    def _extract_query_patterns(self, query: str) -> List[str]:
        """Extract patterns from query."""
        # Simple mock implementation
        words = query.split()
        patterns = []
        
        if len(words) >= 2:
            # Create patterns from adjacent word pairs
            for i in range(len(words) - 1):
                patterns.append(f"{words[i]}_{words[i+1]}")
        
        # Add single word patterns for important words
        for word in words:
            if len(word) > 4:  # Only consider longer words as significant
                patterns.append(word)
        
        return patterns[:5]  # Limit to 5 patterns


class ElasticMemoryRAGIntegration:
    """Integration between elastic semantic memory and pattern-aware RAG.
    
    This class implements the complete RAG↔Evolution↔Persistence loop,
    enabling bidirectional entity-predicate evolution and quality-enhanced
    retrieval based on persisted semantic memory.
    """
    
    def __init__(self, 
                 predicate_quality_tracker: Optional[PredicateQualityTracker] = None,
                 persistence_layer: Optional[SemanticMemoryPersistence] = None,
                 quality_retrieval: Optional[QualityEnhancedRetrieval] = None,
                 pattern_aware_rag = None,
                 event_bus = None,
                 persistence_base_dir: str = None,
                 adaptive_id = None):
        """
        Initialize the elastic memory RAG integration.
        
        Args:
            predicate_quality_tracker: Optional predicate quality tracker
            persistence_layer: Optional semantic memory persistence layer
            quality_retrieval: Optional quality-enhanced retrieval component
            pattern_aware_rag: Optional pattern-aware RAG component
            event_bus: Optional event bus for publishing events
            persistence_base_dir: Base directory for persistence files
            adaptive_id: Optional AdaptiveID component
        """
        self.event_bus = event_bus
        
        # Initialize predicate quality tracker
        self.predicate_quality_tracker = predicate_quality_tracker or PredicateQualityTracker(event_bus, logger)
        
        # Initialize persistence layer
        persistence_dir = persistence_base_dir or os.path.join(os.getcwd(), 'persistence')
        self.persistence_layer = persistence_layer or SemanticMemoryPersistence(persistence_dir, logger)
        
        # Initialize quality-enhanced retrieval
        self.quality_retrieval = quality_retrieval or QualityEnhancedRetrieval(
            predicate_quality_tracker=self.predicate_quality_tracker,
            persistence_layer=self.persistence_layer,
            event_bus=self.event_bus
        )
        
        # Initialize pattern-aware RAG
        self.pattern_aware_rag = pattern_aware_rag or MockPatternAwareRAG(event_bus)
        
        # Initialize AdaptiveID
        self.adaptive_id = adaptive_id
        
        # Entity network for tracking relationships
        self.entity_network = nx.DiGraph()
        
        # Track entity and predicate quality states
        self.entity_quality = {}
        self.entity_confidence = {}
        self.entity_transition_history = {}
        
        # Entity ID tracking with AdaptiveID
        self.entity_ids = {}
        
        # Entity categories based on the enhanced relationship model
        self.entity_categories = {
            'CLIMATE_HAZARD': [],
            'ECOSYSTEM': [],
            'INFRASTRUCTURE': [],
            'ADAPTATION_STRATEGY': [],
            'ASSESSMENT_COMPONENT': []
        }
        
        # Relationship categories
        self.relationship_categories = {
            'structural': ['part_of', 'contains', 'component_of'],
            'causal': ['causes', 'affects', 'damages', 'mitigates'],
            'functional': ['protects_against', 'analyzes', 'evaluates'],
            'temporal': ['precedes', 'concurrent_with']
        }
        
        # Field metrics
        self.field_metrics = {
            'local_density': 0.0,
            'global_density': 0.0,
            'stability': 0.5,
            'coherence': 0.5
        }
        
        logger.info("Initialized Elastic Memory RAG Integration")
    
    def retrieve_with_quality(self, query: str, context: Optional[QualityAwarePatternContext] = None, max_results: int = 10) -> RetrievalResult:
        """
        Retrieve patterns with quality awareness.
        
        Args:
            query: The query to retrieve patterns for
            context: Optional quality-aware pattern context
            max_results: Maximum number of results to return
            
        Returns:
            RetrievalResult object containing patterns and entity relationships
        """
        logger.info(f"Retrieving with quality for query: {query}")
        
        # Use quality-enhanced retrieval
        result = self.quality_retrieval.retrieve(
            query=query,
            context=context,
            max_results=max_results
        )
        
        # Extract entity relationships from result
        entity_relationships = result.entity_relationships
        
        # Update entity network with relationships
        for relationship in entity_relationships:
            source = relationship.get("source")
            target = relationship.get("target")
            predicate = relationship.get("predicate")
            
            if source and target and predicate:
                # Add nodes if they don't exist
                if source not in self.entity_network:
                    self.entity_network.add_node(source)
                    self.entity_quality[source] = "uncertain"
                    self.entity_confidence[source] = 0.5
                
                if target not in self.entity_network:
                    self.entity_network.add_node(target)
                    self.entity_quality[target] = "uncertain"
                    self.entity_confidence[target] = 0.5
                
                # Add edge with predicate
                self.entity_network.add_edge(source, target, predicate=predicate)
        
        # Update field metrics based on retrieval
        self._update_field_metrics(result)
        
        return result
    
    def _update_field_metrics(self, result: RetrievalResult):
        """
        Update field metrics based on retrieval result.
        
        Args:
            result: RetrievalResult from quality-enhanced retrieval
        """
        # Update local density based on pattern count
        pattern_count = len(result.patterns) if hasattr(result, 'patterns') else 0
        self.field_metrics['local_density'] = min(1.0, pattern_count / 20)
        
        # Update global density based on entity network size
        entity_count = self.entity_network.number_of_nodes()
        relationship_count = self.entity_network.number_of_edges()
        
        if entity_count > 0:
            # Calculate potential edges in a complete graph: n(n-1)/2
            potential_edges = (entity_count * (entity_count - 1)) / 2
            # Calculate global density as ratio of actual to potential edges
            global_density = relationship_count / potential_edges if potential_edges > 0 else 0
            self.field_metrics['global_density'] = global_density
        
        # Update coherence based on quality states
        good_entities = sum(1 for quality in self.entity_quality.values() if quality == 'good')
        good_predicates = sum(1 for predicate in self.predicate_quality_tracker.predicate_quality 
                             if self.predicate_quality_tracker.get_predicate_quality(predicate) == 'good')
        
        total_entities = len(self.entity_quality)
        total_predicates = len(self.predicate_quality_tracker.predicate_quality)
        
        if total_entities > 0 and total_predicates > 0:
            entity_coherence = good_entities / total_entities
            predicate_coherence = good_predicates / total_predicates
            self.field_metrics['coherence'] = (entity_coherence + predicate_coherence) / 2
        
        # Update stability based on transition history
        transition_count = sum(len(history) for history in self.entity_transition_history.values())
        stability = 1.0 - min(1.0, transition_count / (total_entities * 2) if total_entities > 0 else 0)
        self.field_metrics['stability'] = stability
        
    def _update_field_metrics_after_reinforcement(self):
        """
        Update field metrics after contextual reinforcement.
        """
        # Update coherence based on quality states after reinforcement
        good_entities = sum(1 for quality in self.entity_quality.values() if quality == 'good')
        good_predicates = sum(1 for predicate in self.predicate_quality_tracker.predicate_quality 
                             if self.predicate_quality_tracker.get_predicate_quality(predicate) == 'good')
        
        total_entities = len(self.entity_quality)
        total_predicates = len(self.predicate_quality_tracker.predicate_quality)
        
        if total_entities > 0 and total_predicates > 0:
            entity_coherence = good_entities / total_entities
            predicate_coherence = good_predicates / total_predicates
            # Increase coherence after reinforcement
            self.field_metrics['coherence'] = min(1.0, ((entity_coherence + predicate_coherence) / 2) + 0.05)
        
        # Increase stability after reinforcement
        self.field_metrics['stability'] = min(1.0, self.field_metrics['stability'] + 0.02)
        
        # Update density metrics based on new relationships
        entity_count = self.entity_network.number_of_nodes()
        relationship_count = self.entity_network.number_of_edges()
        
        if entity_count > 0:
            # Calculate potential edges in a complete graph: n(n-1)/2
            potential_edges = (entity_count * (entity_count - 1)) / 2
            # Calculate global density as ratio of actual to potential edges
            global_density = relationship_count / potential_edges if potential_edges > 0 else 0
            self.field_metrics['global_density'] = global_density
            
            # Increase local density slightly after reinforcement
            self.field_metrics['local_density'] = min(1.0, self.field_metrics['local_density'] + 0.03)
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save current state to persistence layer.
        
        Returns:
            Dictionary with persistence information
        """
        persistence_paths = self.persistence_layer.save_complete_state(
            entity_quality=self.entity_quality,
            entity_confidence=self.entity_confidence,
            entity_transition_history=self.entity_transition_history,
            predicate_quality={pred: self.predicate_quality_tracker.get_predicate_quality(pred) 
                              for pred in self.predicate_quality_tracker.predicate_quality},
            predicate_confidence={pred: self.predicate_quality_tracker.get_predicate_confidence(pred) 
                                for pred in self.predicate_quality_tracker.predicate_quality},
            domain_predicate_specialization=self.predicate_quality_tracker.domain_predicate_specialization,
            predicate_transition_history=self.predicate_quality_tracker.quality_transition_history,
            field_metrics=self.field_metrics,
            field_stability=self.field_metrics.get('stability', 0.5),
            field_coherence=self.field_metrics.get('coherence', 0.5),
            field_density={'global': self.field_metrics.get('global_density', 0.0),
                          'local': self.field_metrics.get('local_density', 0.0)},
            entity_network=self.entity_network
        )
        
        logger.info(f"Saved elastic memory state with {len(self.entity_quality)} entities and {self.entity_network.number_of_edges()} relationships")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'persistence_paths': persistence_paths,
            'entity_count': len(self.entity_quality),
            'predicate_count': len(self.predicate_quality_tracker.predicate_quality),
            'relationship_count': self.entity_network.number_of_edges()
        }
    
    def load_state(self) -> Dict[str, Any]:
        """
        Load state from persistence layer.
        
        Returns:
            Dictionary with loaded state information
        """
        # Load latest state
        latest_state = self.persistence_layer.get_latest_state()
        
        # Update entity quality
        entity_quality_data = latest_state.get('entity_quality', {})
        if 'entity_quality' in entity_quality_data:
            self.entity_quality = entity_quality_data['entity_quality']
        
        if 'entity_confidence' in entity_quality_data:
            self.entity_confidence = entity_quality_data['entity_confidence']
        
        if 'quality_transition_history' in entity_quality_data:
            self.entity_transition_history = entity_quality_data['quality_transition_history']
        
        # Load entity network if available
        entity_network = self.persistence_layer.load_entity_network()
        if entity_network and entity_network.number_of_nodes() > 0:
            self.entity_network = entity_network
        
        # Load predicate quality tracker state if available
        predicate_quality_data = latest_state.get('predicate_quality', {})
        if predicate_quality_data:
            # Create a new predicate quality tracker with loaded data
            new_tracker = PredicateQualityTracker(self.event_bus, logger)
            
            if 'predicate_quality' in predicate_quality_data:
                new_tracker.predicate_quality = predicate_quality_data['predicate_quality']
            
            if 'predicate_confidence' in predicate_quality_data:
                new_tracker.predicate_confidence = predicate_quality_data['predicate_confidence']
            
            if 'quality_transition_history' in predicate_quality_data:
                new_tracker.quality_transition_history = predicate_quality_data['quality_transition_history']
            
            # Replace current tracker
            self.predicate_quality_tracker = new_tracker
        
        # Load field metrics if available
        vector_field_data = latest_state.get('vector_field', {})
        if 'field_metrics' in vector_field_data:
            self.field_metrics = vector_field_data['field_metrics']
        
        logger.info(f"Loaded elastic memory state with {len(self.entity_quality)} entities and {self.entity_network.number_of_edges()} relationships")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'loaded_from': latest_state.get('timestamp', 'unknown'),
            'entity_count': len(self.entity_quality),
            'predicate_count': len(self.predicate_quality_tracker.predicate_quality),
            'relationship_count': self.entity_network.number_of_edges()
        }
    
    def transition_entity_quality(self, entity: str, from_quality: str, to_quality: str, 
                                 evidence: Optional[str] = None) -> bool:
        """
        Transition an entity to a new quality state.
        
        Args:
            entity: The entity to transition
            from_quality: The current quality state
            to_quality: The target quality state
            evidence: Optional evidence supporting this transition
            
        Returns:
            True if transition was successful, False otherwise
        """
        # Update entity quality
        self.entity_quality[entity] = to_quality
        
        # Update confidence based on quality
        quality_confidence = {
            'poor': 0.2,
            'uncertain': 0.5,
            'good': 0.8
        }
        self.entity_confidence[entity] = quality_confidence.get(to_quality, 0.5)
        
        # Record transition in history
        transition = {
            'timestamp': datetime.now().isoformat(),
            'from_quality': from_quality,
            'to_quality': to_quality,
            'evidence': evidence
        }
        
        if entity not in self.entity_transition_history:
            self.entity_transition_history[entity] = []
        
        self.entity_transition_history[entity].append(transition)
        
        # Update entity in network
        if entity in self.entity_network:
            self.entity_network.nodes[entity]['quality'] = to_quality
            self.entity_network.nodes[entity]['confidence'] = self.entity_confidence[entity]
        
        # Publish event if event bus is available
        if self.event_bus:
            event_data = {
                'entity': entity,
                'from_quality': from_quality,
                'to_quality': to_quality,
                'confidence': self.entity_confidence[entity],
                'evidence': evidence
            }
            
            event = {
                'event_type': 'entity.quality.transition',
                'source': 'elastic_memory_rag_integration',
                'data': event_data
            }
            
            self.event_bus.publish(event)
        
        logger.info(f"Entity quality transition: {entity} from {from_quality} to {to_quality}")
        
        return True
    
    def transition_predicate_quality(self, predicate: str, to_quality: str,
                                    source_domain: Optional[str] = None,
                                    target_domain: Optional[str] = None,
                                    evidence: Optional[str] = None) -> bool:
        """
        Transition a predicate to a new quality state.
        
        Args:
            predicate: The predicate to transition
            to_quality: The target quality state
            source_domain: Optional source domain for domain-specific transitions
            target_domain: Optional target domain for domain-specific transitions
            evidence: Optional evidence supporting this transition
            
        Returns:
            True if transition was successful, False otherwise
        """
        return self.predicate_quality_tracker.transition_predicate_quality(
            predicate=predicate,
            to_quality=to_quality,
            source_domain=source_domain,
            target_domain=target_domain,
            evidence=evidence
        )
    
    def apply_contextual_reinforcement(self, entities: List[str], 
                                      relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply contextual reinforcement to improve entity and relationship quality.
        
        Args:
            entities: List of entities to reinforce
            relationships: List of relationships to reinforce
            
        Returns:
            Dictionary with reinforcement results
        """
        reinforced_entities = []
        reinforced_predicates = []
        
        # Reinforce entities
        for entity in entities:
            current_quality = self.entity_quality.get(entity, 'uncertain')
            
            if current_quality == 'uncertain':
                # Count relationships involving this entity
                entity_relationships = [
                    rel for rel in relationships 
                    if rel.get('source') == entity or rel.get('target') == entity
                ]
                
                # If entity has multiple relationships, improve its quality
                if len(entity_relationships) >= 2:
                    success = self.transition_entity_quality(
                        entity=entity,
                        from_quality='uncertain',
                        to_quality='good',
                        evidence=f"Contextual reinforcement: entity involved in {len(entity_relationships)} relationships"
                    )
                    
                    if success:
                        reinforced_entities.append(entity)
        
        # Reinforce predicates
        predicate_counts = {}
        for rel in relationships:
            predicate = rel.get('predicate')
            if predicate:
                predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1
        
        for predicate, count in predicate_counts.items():
            current_quality = self.predicate_quality_tracker.get_predicate_quality(predicate)
            
            if current_quality == 'uncertain' and count >= 3:
                # If predicate occurs frequently, improve its quality
                success = self.transition_predicate_quality(
                    predicate=predicate,
                    to_quality='good',
                    evidence=f"Contextual reinforcement: predicate used in {count} relationships"
                )
                
                if success:
                    reinforced_predicates.append(predicate)
        
        logger.info(f"Applied contextual reinforcement: improved {len(reinforced_entities)} entities and {len(reinforced_predicates)} predicates")
        
        return {
            'reinforced_entities': reinforced_entities,
            'reinforced_predicates': reinforced_predicates,
            'entity_count': len(entities),
            'relationship_count': len(relationships)
        }
    
    def _update_entity_network(self, relationships: List[Dict[str, Any]]):
        """
        Update entity network with relationships.
        
        Args:
            relationships: List of relationship dictionaries
        """
        for rel in relationships:
            source = rel.get('source')
            target = rel.get('target')
            predicate = rel.get('predicate')
            quality = rel.get('quality', 'uncertain')
            confidence = rel.get('confidence', 0.5)
            
            if source and target and predicate:
                # Add nodes if they don't exist
                if source not in self.entity_network:
                    self.entity_network.add_node(
                        source, 
                        quality=self.entity_quality.get(source, 'uncertain'),
                        confidence=self.entity_confidence.get(source, 0.5)
                    )
                
                if target not in self.entity_network:
                    self.entity_network.add_node(
                        target, 
                        quality=self.entity_quality.get(target, 'uncertain'),
                        confidence=self.entity_confidence.get(target, 0.5)
                    )
                
                # Add or update edge
                self.entity_network.add_edge(
                    source, target, 
                    predicate=predicate, 
                    quality=quality,
                    confidence=confidence,
                    timestamp=datetime.now().isoformat()
                )
    
    def _update_entity_quality_from_patterns(self, patterns: List[Pattern]):
        """
        Update entity quality states from patterns.
        
        Args:
            patterns: List of patterns
        """
        for pattern in patterns:
            entity = pattern.metadata.get('entity')
            quality = pattern.metadata.get('quality_state', 'uncertain')
            confidence = pattern.metadata.get('confidence', 0.5)
            
            if entity:
                # Update entity quality if it's an improvement
                current_quality = self.entity_quality.get(entity, 'uncertain')
                
                if quality == 'good' and current_quality != 'good':
                    self.transition_entity_quality(
                        entity=entity,
                        from_quality=current_quality,
                        to_quality=quality,
                        evidence=f"Quality improvement from pattern: {pattern.text[:50]}..."
                    )
                else:
                    # Just update the quality without a formal transition
                    self.entity_quality[entity] = quality
                    self.entity_confidence[entity] = confidence
    
    def _update_predicate_quality(self, relationships: List[Dict[str, Any]]):
        """
        Update predicate quality based on relationships.
        
        Args:
            relationships: List of relationship dictionaries
        """
        # Group relationships by predicate
        predicate_groups = {}
        for rel in relationships:
            predicate = rel.get('predicate')
            if predicate:
                if predicate not in predicate_groups:
                    predicate_groups[predicate] = []
                predicate_groups[predicate].append(rel)
        
        # Analyze each predicate group
        for predicate, rels in predicate_groups.items():
            current_quality = self.predicate_quality_tracker.get_predicate_quality(predicate)
            
            # Check for domain-specific patterns
            domain_pairs = set()
            for rel in rels:
                source = rel.get('source')
                target = rel.get('target')
                source_domain = self._get_entity_domain(source)
                target_domain = self._get_entity_domain(target)
                
                if source_domain and target_domain:
                    domain_pairs.add((source_domain, target_domain))
            
            # If predicate consistently connects specific domains, reinforce it
            if len(rels) >= 3 and len(domain_pairs) <= 2:
                # Predicate has consistent domain usage
                if current_quality == 'uncertain':
                    # Improve predicate quality
                    self.predicate_quality_tracker.transition_predicate_quality(
                        predicate=predicate,
                        to_quality='good',
                        source_domain=list(domain_pairs)[0][0] if domain_pairs else None,
                        target_domain=list(domain_pairs)[0][1] if domain_pairs else None,
                        evidence=f"Consistent domain usage across {len(rels)} relationships"
                    )
    
    def _get_entity_domain(self, entity: str) -> Optional[str]:
        """
        Get the domain of an entity based on naming patterns.
        
        Args:
            entity: Entity name
            
        Returns:
            Domain name or None if unknown
        """
        # Simple domain detection based on entity name patterns
        entity_lower = entity.lower() if entity else ""
        
        if any(term in entity_lower for term in ['sea level', 'flood', 'storm', 'erosion', 'precipitation']):
            return 'CLIMATE_HAZARD'
        elif any(term in entity_lower for term in ['marsh', 'wetland', 'beach', 'estuary', 'ecosystem']):
            return 'ECOSYSTEM'
        elif any(term in entity_lower for term in ['culvert', 'stormwater', 'wastewater', 'infrastructure']):
            return 'INFRASTRUCTURE'
        elif any(term in entity_lower for term in ['retreat', 'shoreline', 'adaptation', 'resilience']):
            return 'ADAPTATION_STRATEGY'
        elif any(term in entity_lower for term in ['assessment', 'vulnerability', 'metric', 'evaluation']):
            return 'ASSESSMENT_COMPONENT'
        
        return None


def run_elastic_memory_rag_integration_test():
    """
    Run a test of the elastic memory RAG integration.
    
    Returns:
        Dictionary with test results
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logger.info("Starting Elastic Memory RAG Integration Test")
    
    # Create mock event bus
    event_bus = MockEventBus()
    
    # Create test directory for persistence
    test_persistence_dir = os.path.join(test_dir, 'test_run')
    os.makedirs(test_persistence_dir, exist_ok=True)
    
    # Initialize integration
    integration = ElasticMemoryRAGIntegration(
        event_bus=event_bus,
        persistence_base_dir=test_persistence_dir
    )
    
    # Create mock context
    context = QualityAwarePatternContext(coherence_level=0.7)
    
    # Test query
    query = "How do salt marshes protect coastal communities from sea level rise?"
    
    # Create AdaptiveIDs for key entities
    logger.info("Creating AdaptiveIDs for key entities")
    entity_adaptive_ids = {}
    for entity, category in [
        ("salt marshes", "ECOSYSTEM"),
        ("coastal communities", "INFRASTRUCTURE"),
        ("sea level rise", "CLIMATE_HAZARD"),
        ("living shorelines", "ADAPTATION_STRATEGY"),
        ("vulnerability assessment", "ASSESSMENT_COMPONENT")
    ]:
        # Create AdaptiveID
        adaptive_id = MockAdaptiveID(entity, "test_system")
        
        # Add to entity categories
        if category in integration.entity_categories:
            integration.entity_categories[category].append(entity)
        
        # Store in entity_ids dictionary
        integration.entity_ids[entity] = adaptive_id.id
        entity_adaptive_ids[entity] = adaptive_id
        
        # Add temporal context
        adaptive_id.update_temporal_context("category", category, "initialization")
        adaptive_id.update_temporal_context("creation_time", datetime.now().isoformat(), "initialization")
    
    # Process query with pattern-aware RAG
    logger.info(f"Processing query with pattern-aware RAG: {query}")
    try:
        # Note: In a real implementation, we would use asyncio.run() to run the async function
        # For this test, we'll mock the result
        pattern_result = {
            "content": "Salt marshes protect coastal communities from sea level rise by absorbing wave energy, "
                      "reducing erosion, and providing a buffer against storm surges. They act as natural "
                      "infrastructure that can adapt to changing sea levels by accumulating sediment over time.",
            "pattern_id": str(uuid.uuid4()),
            "coherence": {
                "flow_state": "stable",
                "patterns": ["salt_marshes", "coastal_communities", "sea_level_rise", "protection", "adaptation"],
                "confidence": 0.75,
                "emergence_potential": 0.4
            }
        }
        
        pattern_context = {
            "query_patterns": ["salt_marshes", "coastal_communities", "sea_level_rise", "protection"],
            "retrieval_patterns": ["wave_energy", "erosion", "storm_surge", "natural_infrastructure", "sediment_accumulation"],
            "augmentation_patterns": ["adaptation", "buffer", "natural_solution"],
            "coherence_level": 0.75
        }
    except Exception as e:
        logger.error(f"Error processing with pattern-aware RAG: {e}")
        pattern_result = {"error": str(e)}
        pattern_context = {}
    
    # Retrieve patterns with quality awareness
    logger.info(f"Retrieving patterns for query: {query}")
    result = integration.retrieve_with_quality(query, context)
    
    # Extract entities and relationships from pattern result
    entities = ["salt marshes", "coastal communities", "sea level rise", "wave energy", "erosion", "storm surge"]
    
    # Create relationships with categories based on our enhanced relationship model
    relationships = [
        # Structural relationships
        {"source": "salt marshes", "target": "coastal communities", "predicate": "part_of", 
         "source_category": "ECOSYSTEM", "target_category": "INFRASTRUCTURE"},
        
        # Causal relationships
        {"source": "salt marshes", "target": "coastal communities", "predicate": "protects", 
         "source_category": "ECOSYSTEM", "target_category": "INFRASTRUCTURE"},
        {"source": "sea level rise", "target": "coastal communities", "predicate": "threatens", 
         "source_category": "CLIMATE_HAZARD", "target_category": "INFRASTRUCTURE"},
        {"source": "salt marshes", "target": "sea level rise", "predicate": "mitigates", 
         "source_category": "ECOSYSTEM", "target_category": "CLIMATE_HAZARD"},
        {"source": "salt marshes", "target": "wave energy", "predicate": "absorbs", 
         "source_category": "ECOSYSTEM", "target_category": "CLIMATE_HAZARD"},
        {"source": "salt marshes", "target": "erosion", "predicate": "reduces", 
         "source_category": "ECOSYSTEM", "target_category": "CLIMATE_HAZARD"},
        
        # Functional relationships
        {"source": "salt marshes", "target": "storm surge", "predicate": "protects_against", 
         "source_category": "ECOSYSTEM", "target_category": "CLIMATE_HAZARD"},
        
        # Temporal relationships
        {"source": "sea level rise", "target": "erosion", "predicate": "precedes", 
         "source_category": "CLIMATE_HAZARD", "target_category": "CLIMATE_HAZARD"}
    ]
    
    # Apply contextual reinforcement
    logger.info("Applying contextual reinforcement")
    reinforcement_result = integration.apply_contextual_reinforcement(entities, relationships)
    
    # Update AdaptiveID confidence based on reinforcement
    for entity, adaptive_id in entity_adaptive_ids.items():
        if entity in reinforcement_result["reinforced_entities"]:
            # Increase confidence for reinforced entities
            new_confidence = min(1.0, adaptive_id.confidence + 0.2)
            adaptive_id.update_confidence(new_confidence, "contextual_reinforcement")
            
            # Add temporal context about the reinforcement
            adaptive_id.update_temporal_context(
                "reinforcement", 
                {
                    "timestamp": datetime.now().isoformat(),
                    "source": "elastic_memory_rag_test",
                    "confidence_change": "+0.2"
                },
                "contextual_reinforcement"
            )
    
    # Save state
    logger.info("Saving state to persistence layer")
    save_result = integration.save_state()
    
    # Load state
    logger.info("Loading state from persistence layer")
    load_result = integration.load_state()
    
    # Visualize entity network
    logger.info("Visualizing entity network")
    viz_path = os.path.join(visualization_dir, f"entity_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(integration.entity_network)
    
    # Draw nodes with different colors based on quality and category
    node_colors = []
    node_sizes = []
    for node in integration.entity_network.nodes():
        quality = integration.entity_quality.get(node, "uncertain")
        
        # Determine color based on quality
        if quality == "good":
            base_color = "green"
        elif quality == "poor":
            base_color = "red"
        else:  # uncertain
            base_color = "orange"
        
        # Adjust size based on category
        category = None
        for cat, entities in integration.entity_categories.items():
            if node in entities:
                category = cat
                break
        
        if category == "CLIMATE_HAZARD":
            node_sizes.append(600)
        elif category == "ECOSYSTEM":
            node_sizes.append(500)
        elif category == "INFRASTRUCTURE":
            node_sizes.append(550)
        elif category == "ADAPTATION_STRATEGY":
            node_sizes.append(450)
        elif category == "ASSESSMENT_COMPONENT":
            node_sizes.append(400)
        else:
            node_sizes.append(350)
            
        node_colors.append(base_color)
    
    nx.draw_networkx_nodes(integration.entity_network, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Draw edges with different colors based on predicate quality and category
    edge_colors = []
    edge_widths = []
    for source, target, data in integration.entity_network.edges(data=True):
        predicate = data.get("predicate", "")
        quality = integration.predicate_quality_tracker.get_predicate_quality(predicate)
        
        # Determine color based on quality
        if quality == "good":
            edge_colors.append("green")
        elif quality == "poor":
            edge_colors.append("red")
        else:  # uncertain
            edge_colors.append("orange")
        
        # Determine width based on relationship category
        category = None
        for cat, predicates in integration.relationship_categories.items():
            if predicate in predicates:
                category = cat
                break
        
        if category == "causal":
            edge_widths.append(3.0)
        elif category == "structural":
            edge_widths.append(2.5)
        elif category == "functional":
            edge_widths.append(2.0)
        elif category == "temporal":
            edge_widths.append(1.5)
        else:
            edge_widths.append(1.0)
    
    nx.draw_networkx_edges(integration.entity_network, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7)
    
    # Draw labels
    nx.draw_networkx_labels(integration.entity_network, pos, font_size=10, font_weight="bold")
    edge_labels = {(source, target): data.get("predicate", "") for source, target, data in integration.entity_network.edges(data=True)}
    nx.draw_networkx_edge_labels(integration.entity_network, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Entity-Predicate Network with Quality States and Categories")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(viz_path)
    plt.close()
    
    logger.info(f"Saved network visualization to {viz_path}")
    
    # Return test results
    return {
        'retrieval_result': {
            'pattern_count': len(result.patterns),
            'quality_distribution': result.quality_distribution,
            'confidence': result.confidence
        },
        'reinforcement_result': reinforcement_result,
        'persistence_result': {
            'save': save_result,
            'load': load_result
        },
        'entity_count': len(integration.entity_quality),
        'relationship_count': integration.entity_network.number_of_edges(),
        'predicate_count': len(integration.predicate_quality_tracker.predicate_quality),
        'visualization_path': viz_path
    }


if __name__ == "__main__":
    # Run the test
    results = run_elastic_memory_rag_integration_test()
    
    # Print summary
    print("\nElastic Memory RAG Integration Test Summary:")
    print("-------------------------------------")
    print(f"Entity count: {results['entity_count']}")
    print(f"Relationship count: {results['relationship_count']}")
    print(f"Predicate count: {results['predicate_count']}")
    print("\nRetrieval Results:")
    print(f"  Pattern count: {results['retrieval_result']['pattern_count']}")
    print(f"  Quality distribution: {results['retrieval_result']['quality_distribution']}")
    print(f"  Confidence: {results['retrieval_result']['confidence']}")
    print("\nReinforcement Results:")
    print(f"  Reinforced entities: {len(results['reinforcement_result']['reinforced_entities'])}")
    print(f"  Reinforced predicates: {len(results['reinforcement_result']['reinforced_predicates'])}")
    print("\nPersistence Results:")
    print(f"  Save timestamp: {results['persistence_result']['save']['timestamp']}")
    print(f"  Load timestamp: {results['persistence_result']['load']['timestamp']}")
    print(f"\nVisualization saved to: {results['visualization_path']}")
    print("-------------------------------------")
    print("Test completed successfully!")
