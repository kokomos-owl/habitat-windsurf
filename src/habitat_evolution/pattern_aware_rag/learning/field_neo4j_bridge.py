"""
Bridge between field-aware components and Neo4j persistence.

This module provides integration between pattern-aware RAG input and Neo4j persistence
while maintaining provenance through AdaptiveID. It ensures that incoming prompt-generated 
content is properly aligned with the current habitat/Neo4j semantic state before integration.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

from ...adaptive_core.id.adaptive_id import AdaptiveID
from ...visualization.pattern_id import PatternAdaptiveID
from ..learning.learning_health_integration import HealthFieldObserver, FieldObserver

class FieldStateNeo4jBridge:
    """
    Bridge between pattern-aware RAG input and Neo4j persistence.
    
    This bridge ensures field state integrity by:
    1. Checking alignment of incoming prompt content with existing Neo4j state
    2. Maintaining provenance through AdaptiveID across both operating modes
    3. Applying field state metrics to ensure coherence in the semantic graph
    4. Supporting dual-mode operation (Neo4j persistence and Direct LLM)
    """
    
    def __init__(
        self, 
        field_observer: Optional[FieldObserver] = None,
        persistence_mode: str = "neo4j",
        pattern_db = None,
        provenance_tracker = None
    ):
        """
        Initialize the bridge.
        
        Args:
            field_observer: Observer instance that tracks field metrics
            persistence_mode: 'neo4j' or 'direct' for persistence strategy
            pattern_db: Optional Neo4j pattern database connection
            provenance_tracker: Optional provenance tracking component
        """
        self.field_observer = field_observer
        self.persistence_mode = persistence_mode
        self.pattern_db = pattern_db
        self.provenance_tracker = provenance_tracker
        self.logger = logging.getLogger("field_bridge")
        
        # Cache for current state
        self.current_field_state = {}
        self.current_neo4j_state = {}
        
    def align_incoming_pattern(self, 
                              pattern_data: Dict[str, Any], 
                              user_id: str) -> Dict[str, Any]:
        """
        Align incoming pattern data with existing Neo4j semantic state.
        
        This is the core function that applies field state awareness to incoming content.
        It ensures that new patterns align with the existing habitat state.
        
        Args:
            pattern_data: Raw pattern data from prompt-generated content
            user_id: ID of the user who generated this content
            
        Returns:
            Aligned pattern data with field state metrics
        """
        # Create AdaptiveID for provenance tracking
        pattern_id = pattern_data.get("id", f"pattern_{datetime.now().isoformat()}")
        
        # Create adaptive ID if not exists
        if not isinstance(pattern_data.get("adaptive_id"), AdaptiveID):
            pattern_data["adaptive_id"] = AdaptiveID(
                base_concept=pattern_id,
                creator_id=user_id
            )
        
        # Apply field state
        if self.field_observer:
            # Get current field metrics
            field_metrics = self.field_observer.get_field_metrics()
            
            # Get tonic-harmonic analysis if available
            if len(self.field_observer.wave_history) >= 3 and len(self.field_observer.tonic_history) >= 3:
                harmonic_analysis = self.field_observer.perform_harmonic_analysis(
                    self.field_observer.wave_history, 
                    self.field_observer.tonic_history
                )
                
                # Apply field state metrics
                pattern_data["field_state"] = {
                    "stability": self.field_observer.wave_history[-1] if self.field_observer.wave_history else 0.8,
                    "tonic": self.field_observer.tonic_history[-1] if self.field_observer.tonic_history else 0.5,
                    "harmonic": harmonic_analysis.get("harmonic", [])[-1] if harmonic_analysis.get("harmonic") else 0.4,
                    "coherence": field_metrics.get("coherence", 0.7),
                    "boundaries": harmonic_analysis.get("boundaries", []),
                    "observed_at": datetime.now().isoformat()
                }
                
                # Update adaptive ID with field state for provenance
                if isinstance(pattern_data["adaptive_id"], AdaptiveID):
                    pattern_data["adaptive_id"].add_version(
                        {
                            "field_state": pattern_data["field_state"],
                            "confidence": pattern_data.get("confidence", 0.8)
                        },
                        origin="field_bridge"
                    )
            
        return pattern_data
    
    def check_state_alignment(self, 
                             pattern_data: Dict[str, Any],
                             threshold: float = 0.7) -> Tuple[bool, Dict[str, float]]:
        """
        Check if the incoming pattern aligns with current Neo4j state.
        
        Args:
            pattern_data: Pattern data to check
            threshold: Minimum alignment threshold (0.0-1.0)
            
        Returns:
            Tuple of (is_aligned, alignment_metrics)
        """
        if not self.pattern_db or self.persistence_mode != "neo4j":
            # In direct mode, always consider aligned
            return True, {"direct_mode": 1.0}
            
        # Get field state from pattern
        pattern_field_state = pattern_data.get("field_state", {})
        
        # Get current Neo4j state - this would call the pattern DB interface
        try:
            neo4j_state = self.pattern_db.get_current_state()
            
            # Calculate alignment metrics
            alignment = {}
            
            # Field state alignment
            if "stability" in pattern_field_state and "avg_stability" in neo4j_state:
                stability_diff = abs(pattern_field_state["stability"] - neo4j_state["avg_stability"])
                alignment["stability"] = max(0, 1.0 - stability_diff)
                
            # Pattern type alignment
            if "pattern_type" in pattern_data and "pattern_types" in neo4j_state:
                alignment["pattern_type"] = 1.0 if pattern_data["pattern_type"] in neo4j_state["pattern_types"] else 0.5
                
            # Calculate overall alignment
            if alignment:
                overall_alignment = sum(alignment.values()) / len(alignment)
                is_aligned = overall_alignment >= threshold
                
                return is_aligned, alignment
            
        except Exception as e:
            self.logger.error(f"Failed to check state alignment: {e}")
            
        # Default to aligned if check fails
        return True, {"default": 1.0}
    
    def prepare_for_neo4j(self, 
                         pattern_data: Dict[str, Any]) -> Optional[PatternAdaptiveID]:
        """
        Convert aligned pattern data to Neo4j compatible format.
        
        Args:
            pattern_data: Aligned pattern data
            
        Returns:
            Neo4j compatible pattern or None if conversion fails
        """
        if self.persistence_mode != "neo4j":
            return None
            
        try:
            # Create PatternAdaptiveID for Neo4j visualization
            pattern_type = pattern_data.get("pattern_type", "unknown")
            hazard_type = pattern_data.get("hazard_type", "unknown")
            
            pattern_visualization = PatternAdaptiveID(
                pattern_type=pattern_type,
                hazard_type=hazard_type,
                creator_id=pattern_data["adaptive_id"].creator_id if "adaptive_id" in pattern_data else "pattern_bridge",
                confidence=pattern_data.get("confidence", 0.8)
            )
            
            # Update metrics
            field_state = pattern_data.get("field_state", {})
            pattern_visualization.update_metrics(
                position=(0, 0),  # Default position
                field_state=field_state.get("stability", 0.8) * field_state.get("tonic", 0.5),
                coherence=field_state.get("coherence", 0.7),
                energy_state=field_state.get("stability", 0.8)
            )
            
            # Update temporal context if available
            if "temporal_context" in pattern_data:
                pattern_visualization.temporal_context = (
                    pattern_data["temporal_context"] 
                    if isinstance(pattern_data["temporal_context"], str) 
                    else json.dumps(pattern_data["temporal_context"])
                )
                
            return pattern_visualization
            
        except Exception as e:
            self.logger.error(f"Failed to prepare pattern for Neo4j: {e}")
            return None
            
    def process_prompt_generated_content(self, 
                                       content: Union[Dict[str, Any], List[Dict[str, Any]]], 
                                       user_id: str) -> Dict[str, Any]:
        """
        Process prompt-generated content to prepare for Neo4j integration.
        
        Args:
            content: Raw content from pattern-aware RAG
            user_id: ID of the user who generated this content
            
        Returns:
            Processed content with field state metrics and alignment info
        """
        # Handle single pattern or list of patterns
        patterns = content if isinstance(content, list) else [content]
        processed_patterns = []
        
        for pattern in patterns:
            # Align with field state
            aligned_pattern = self.align_incoming_pattern(pattern, user_id)
            
            # Check alignment with Neo4j state
            is_aligned, alignment_metrics = self.check_state_alignment(aligned_pattern)
            aligned_pattern["neo4j_alignment"] = {
                "is_aligned": is_aligned,
                "metrics": alignment_metrics
            }
            
            # Prepare for Neo4j if in Neo4j mode
            if self.persistence_mode == "neo4j":
                neo4j_pattern = self.prepare_for_neo4j(aligned_pattern)
                if neo4j_pattern:
                    aligned_pattern["neo4j_pattern"] = neo4j_pattern.to_dict()
                
            processed_patterns.append(aligned_pattern)
            
        # Return as same type as input
        if isinstance(content, list):
            return {
                "patterns": processed_patterns,
                "field_state": self._get_current_field_state(),
                "processed_at": datetime.now().isoformat()
            }
        else:
            return processed_patterns[0] if processed_patterns else {}
            
    def _get_current_field_state(self) -> Dict[str, Any]:
        """Get current field state metrics."""
        if not self.field_observer:
            return {}
            
        field_metrics = self.field_observer.get_field_metrics()
        
        # Calculate harmonic state if possible
        if len(self.field_observer.wave_history) >= 3 and len(self.field_observer.tonic_history) >= 3:
            harmonic_analysis = self.field_observer.perform_harmonic_analysis(
                self.field_observer.wave_history, 
                self.field_observer.tonic_history
            )
            
            return {
                "stability": self.field_observer.wave_history[-1] if self.field_observer.wave_history else 0.8,
                "tonic": self.field_observer.tonic_history[-1] if self.field_observer.tonic_history else 0.5,
                "harmonic": harmonic_analysis.get("harmonic", [])[-1] if harmonic_analysis.get("harmonic") else 0.4,
                "coherence": field_metrics.get("coherence", 0.7),
                "boundaries": harmonic_analysis.get("boundaries", []),
                "observed_at": datetime.now().isoformat()
            }
        
        return field_metrics
        
    def integrate_with_neo4j(self, pattern_data: Dict[str, Any]) -> bool:
        """
        Integrate aligned pattern with Neo4j.
        
        Args:
            pattern_data: Aligned pattern data with Neo4j format
            
        Returns:
            Success status
        """
        if self.persistence_mode != "neo4j" or not self.pattern_db:
            return False
            
        try:
            # Extract Neo4j pattern if available
            neo4j_pattern = pattern_data.get("neo4j_pattern")
            if not neo4j_pattern:
                neo4j_pattern = self.prepare_for_neo4j(pattern_data)
                if isinstance(neo4j_pattern, PatternAdaptiveID):
                    neo4j_pattern = neo4j_pattern.to_dict()
                    
            if not neo4j_pattern:
                return False
                
            # Create graph data
            graph_data = {
                "nodes": [neo4j_pattern],
                "relationships": []
            }
            
            # Add relationships if available
            if "relationships" in pattern_data:
                for rel in pattern_data["relationships"]:
                    graph_data["relationships"].append({
                        "source": neo4j_pattern["id"],
                        "target": rel["target_id"],
                        "type": rel["type"],
                        "properties": {
                            "strength": rel.get("strength", 1.0),
                            "created_at": datetime.now().isoformat()
                        }
                    })
            
            # Send to Neo4j
            return self.pattern_db.create_graph(graph_data)
                
        except Exception as e:
            self.logger.error(f"Failed to integrate pattern with Neo4j: {e}")
            return False


# Mock Pattern DB for testing
class MockPatternDB:
    """Mock pattern database for testing."""
    
    def __init__(self):
        self.data = {"nodes": [], "relationships": []}
        
    def create_graph(self, graph_data):
        """Store graph data."""
        self.data = graph_data
        return True
        
    def get_graph(self):
        """Get stored graph data."""
        return self.data
        
    def get_current_state(self):
        """Get current state metrics."""
        # Calculate average metrics
        stability_values = [
            node.get("field_state", 0.5) 
            for node in self.data.get("nodes", [])
            if "field_state" in node
        ]
        
        pattern_types = list(set(
            node.get("pattern_type", "unknown") 
            for node in self.data.get("nodes", [])
        ))
        
        return {
            "avg_stability": sum(stability_values) / len(stability_values) if stability_values else 0.5,
            "pattern_types": pattern_types,
            "node_count": len(self.data.get("nodes", [])),
            "relationship_count": len(self.data.get("relationships", []))
        }
