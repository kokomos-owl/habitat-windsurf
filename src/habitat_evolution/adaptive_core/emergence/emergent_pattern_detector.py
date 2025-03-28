"""
Emergent Pattern Detector

This module implements the EmergentPatternDetector class, which detects patterns
that emerge naturally from semantic observations without imposing predefined categories.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import uuid
import logging
import math
from collections import defaultdict

from ..id.adaptive_id import AdaptiveID
from .semantic_current_observer import SemanticCurrentObserver


class EmergentPatternDetector:
    """
    Detects patterns that emerge naturally from semantic observations.
    
    This class identifies recurring patterns based on frequency and consistency,
    without categorizing them in advance. It allows patterns to function as
    first-class entities that can influence the system's behavior.
    """
    
    def __init__(self, semantic_observer: SemanticCurrentObserver, threshold: int = 3):
        """
        Initialize an emergent pattern detector.
        
        Args:
            semantic_observer: Observer for semantic currents
            threshold: Minimum frequency threshold for pattern detection
        """
        self.semantic_observer = semantic_observer
        self.threshold = threshold
        
        # Separate threshold for meta-pattern detection (naturally lower than pattern threshold)
        self.meta_pattern_threshold = max(1, threshold // 2)  # Default to half the pattern threshold
        
        # Threshold for approaching patterns/meta-patterns (percentage of full threshold)
        self.approaching_threshold_factor = 0.7  # 70% of threshold to be considered "approaching"
        
        # Field state influence parameters
        self.field_coherence = 0.5  # Default coherence
        self.field_stability = 0.5  # Default stability
        
        # Storage for patterns and their evolution
        self.potential_patterns = []
        self.approaching_patterns = []  # Patterns that are approaching threshold
        self.pattern_history = []
        self.pattern_evolution = {}
        
        # Storage for meta-patterns
        self.approaching_meta_patterns = []  # Meta-patterns approaching threshold
        
        # Create an AdaptiveID for this detector
        self.adaptive_id = AdaptiveID(
            base_concept="emergent_pattern_detector",
            creator_id="system"
        )
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
    
    def detect_patterns(self) -> List[Dict[str, Any]]:
        """
        Detect potential patterns based on observed frequencies.
        
        Returns:
            List of detected patterns
        """
        # Reset pattern collections for this detection cycle
        self.potential_patterns = []
        self.approaching_patterns = []
        
        # Get all observations from the semantic observer
        observations = self.semantic_observer.observed_relationships
        frequencies = self.semantic_observer.relationship_frequency
        
        # Calculate field-adjusted threshold
        field_adjusted_threshold = self._get_field_adjusted_threshold(self.threshold)
        approaching_threshold = field_adjusted_threshold * self.approaching_threshold_factor
        
        # Find relationships that exceed or approach the threshold
        for rel_key, count in frequencies.items():
            rel_data = observations[rel_key]
            # Extract source, predicate, target from the relationship data
            source = rel_data["source"]
            predicate = rel_data["predicate"]
            target = rel_data["target"]
            
            # Create pattern ID
            pattern_id = f"pattern_{len(self.potential_patterns) + len(self.approaching_patterns)}_{source}_{predicate}_{target}"
            
            # Calculate confidence based on count and field state
            base_confidence = min(1.0, count / 10.0)
            field_influence = (self.field_coherence * 0.3) + (self.field_stability * 0.2)
            confidence = min(1.0, base_confidence + field_influence)
            
            # Create pattern object
            pattern = {
                "id": pattern_id,
                "type": "relationship_pattern",
                "relationship": {
                    "source": source,
                    "predicate": predicate,
                    "target": target,
                    "context": rel_data.get("context", {})
                },
                "frequency": count,
                "confidence": confidence,
                "detection_timestamp": datetime.now().isoformat(),
                "threshold_status": "unknown"  # Will be set below
            }
            
            # Determine if pattern exceeds threshold, approaches it, or neither
            if count >= field_adjusted_threshold:
                pattern["threshold_status"] = "exceeded"
                self.potential_patterns.append(pattern)
            elif count >= approaching_threshold:
                pattern["threshold_status"] = "approaching"
                self.approaching_patterns.append(pattern)
                
                # Log approaching patterns
                logging.info(f"Pattern approaching threshold: {pattern_id}")
                logging.info(f"  Frequency: {count} (Threshold: {field_adjusted_threshold}, Approaching: {approaching_threshold})")
                logging.info(f"  Confidence: {confidence}")
                evolved_from = self._check_pattern_evolution(pattern)
                if evolved_from:
                    pattern["evolved_from"] = evolved_from
                
                # Update the AdaptiveID with this pattern
                pattern_key = f"pattern_detected_{rel_key}"
                pattern_data = {
                    "pattern_id": pattern["id"],
                    "source": pattern["source"],
                    "predicate": pattern["predicate"],
                    "target": pattern["target"],
                    "frequency": pattern["frequency"],
                    "confidence": pattern["confidence"],
                    "detection_timestamp": pattern["detection_timestamp"]
                }
                self.adaptive_id.update_temporal_context(pattern_key, pattern_data, "pattern_detection")
                
                self.potential_patterns.append(pattern)
                
                # Add to pattern history
                self.pattern_history.append(pattern)
        
        # Detect meta-patterns (patterns of patterns)
        self._detect_meta_patterns()
        
        return self.potential_patterns
    
    def _calculate_confidence(self, rel_data: Dict[str, Any]) -> float:
        """
        Calculate confidence score for a pattern.
        
        Args:
            rel_data: Relationship data
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence on frequency
        frequency = rel_data["frequency"]
        frequency_factor = min(1.0, frequency / 10.0)  # Cap at 1.0
        
        # Factor in recency
        last_observed = datetime.fromisoformat(rel_data["last_observed"])
        now = datetime.now()
        hours_since = (now - last_observed).total_seconds() / 3600
        recency_factor = math.exp(-hours_since / 24)  # Decay over 24 hours
        
        # Factor in consistency of contexts
        context_similarity = self._calculate_context_similarity(rel_data["contexts"])
        
        # Combine factors
        confidence = 0.4 * frequency_factor + 0.3 * recency_factor + 0.3 * context_similarity
        
        return min(1.0, confidence)
    
    def _calculate_context_similarity(self, contexts: List[Dict[str, Any]]) -> float:
        """
        Calculate similarity between contexts.
        
        Args:
            contexts: List of context dictionaries
            
        Returns:
            Similarity score between 0 and 1
        """
        if len(contexts) <= 1:
            return 0.5  # Neutral score for insufficient data
        
        # Extract context keys
        all_keys = set()
        for context_data in contexts:
            if "context" in context_data:
                all_keys.update(context_data["context"].keys())
        
        if not all_keys:
            return 0.5  # Neutral score for empty contexts
        
        # Count key occurrences
        key_counts = defaultdict(int)
        for context_data in contexts:
            if "context" in context_data:
                for key in context_data["context"].keys():
                    key_counts[key] += 1
        
        # Calculate average consistency
        consistency_sum = sum(count / len(contexts) for count in key_counts.values())
        avg_consistency = consistency_sum / len(key_counts) if key_counts else 0.5
        
        return avg_consistency
    
    def _check_pattern_evolution(self, pattern: Dict[str, Any]) -> Optional[str]:
        """
        Check if this pattern is an evolution of an existing pattern.
        
        Args:
            pattern: The pattern to check
            
        Returns:
            ID of the pattern this evolved from, or None
        """
        # Look for similar patterns in history
        for hist_pattern in reversed(self.pattern_history):  # Check most recent first
            # Skip if this is the same pattern
            if hist_pattern["source"] == pattern["source"] and \
               hist_pattern["predicate"] == pattern["predicate"] and \
               hist_pattern["target"] == pattern["target"]:
                continue
            
            # Check for evolution relationships
            if hist_pattern["source"] == pattern["source"] and \
               hist_pattern["target"] == pattern["target"]:
                # Predicate evolution
                evolved_id = hist_pattern["id"]
                
                # Record evolution
                if evolved_id not in self.pattern_evolution:
                    self.pattern_evolution[evolved_id] = []
                
                self.pattern_evolution[evolved_id].append({
                    "evolution_type": "predicate_evolution",
                    "from_pattern": evolved_id,
                    "to_pattern": pattern["id"],
                    "from_predicate": hist_pattern["predicate"],
                    "to_predicate": pattern["predicate"],
                    "timestamp": datetime.now().isoformat()
                })
                
                return evolved_id
            
            elif hist_pattern["predicate"] == pattern["predicate"] and \
                 hist_pattern["target"] == pattern["target"]:
                # Subject evolution
                evolved_id = hist_pattern["id"]
                
                # Record evolution
                if evolved_id not in self.pattern_evolution:
                    self.pattern_evolution[evolved_id] = []
                
                self.pattern_evolution[evolved_id].append({
                    "evolution_type": "subject_evolution",
                    "from_pattern": evolved_id,
                    "to_pattern": pattern["id"],
                    "from_subject": hist_pattern["source"],
                    "to_subject": pattern["source"],
                    "timestamp": datetime.now().isoformat()
                })
                
                return evolved_id
            
            elif hist_pattern["source"] == pattern["source"] and \
                 hist_pattern["predicate"] == pattern["predicate"]:
                # Object evolution
                evolved_id = hist_pattern["id"]
                
                # Record evolution
                if evolved_id not in self.pattern_evolution:
                    self.pattern_evolution[evolved_id] = []
                
                self.pattern_evolution[evolved_id].append({
                    "evolution_type": "object_evolution",
                    "from_pattern": evolved_id,
                    "to_pattern": pattern["id"],
                    "from_object": hist_pattern["target"],
                    "to_object": pattern["target"],
                    "timestamp": datetime.now().isoformat()
                })
                
                return evolved_id
        
        return None
    
    def _detect_meta_patterns(self) -> List[Dict[str, Any]]:
        """
        Detect meta-patterns using field topology and pattern relationships.
        
        Instead of counting discrete evolution types, this approach looks for
        topological features in the pattern space that indicate meta-patterns.
        
        Returns:
            List of meta-patterns
        """
        meta_patterns = []
        self.approaching_meta_patterns = []
        
        # Need at least a few patterns to detect meta-patterns
        if len(self.pattern_history) < 5:
            return meta_patterns
        
        # Create a pattern relationship graph
        pattern_graph = self._create_pattern_relationship_graph()
        
        # Analyze graph topology to find meta-pattern candidates
        meta_pattern_candidates = self._analyze_pattern_topology(pattern_graph)
        
        # Process each meta-pattern candidate
        for i, candidate in enumerate(meta_pattern_candidates):
            # Calculate confidence based on topological properties and field state
            base_confidence = min(1.0, candidate["strength"] / 3.0)
            field_influence = (self.field_coherence * 0.4) + (self.field_stability * 0.3)
            confidence = min(1.0, base_confidence + field_influence)
            
            # Create meta-pattern object
            meta_pattern = {
                "id": f"meta_pattern_{len(meta_patterns) + len(self.approaching_meta_patterns)}_{candidate['type']}",
                "type": "topological_meta_pattern",
                "pattern_type": candidate["type"],
                "frequency": candidate["strength"],
                "confidence": confidence,
                "detection_timestamp": datetime.now().isoformat(),
                "patterns": candidate["patterns"],
                "topology": candidate["topology"],
                "threshold_status": "unknown"  # Will be set below
            }
            
            # Calculate field-adjusted meta-pattern threshold
            field_adjusted_meta_threshold = self._get_field_adjusted_meta_threshold()
            approaching_meta_threshold = field_adjusted_meta_threshold * self.approaching_threshold_factor
            
            # Determine if meta-pattern exceeds threshold, approaches it, or neither
            if candidate["strength"] >= field_adjusted_meta_threshold:
                meta_pattern["threshold_status"] = "exceeded"
                
                # Log detailed meta-pattern information
                logging.info(f"Detected topological meta-pattern: {meta_pattern['id']}")
                logging.info(f"  Pattern type: {candidate['type']}")
                logging.info(f"  Strength: {candidate['strength']} (Threshold: {field_adjusted_meta_threshold})")
                logging.info(f"  Confidence: {meta_pattern['confidence']}")
                logging.info(f"  Topology: {candidate['topology']}")
                
                # Update the AdaptiveID with this meta-pattern
                timestamp = datetime.now().isoformat()
                self.adaptive_id.update_temporal_context(
                    "meta_pattern_detected", 
                    {
                        "type": candidate["type"],
                        "data": meta_pattern,
                        "timestamp": timestamp
                    },
                    "pattern_detection"
                )
                
                # Publish event for meta-pattern detection
                if hasattr(self, 'pattern_publisher') and self.pattern_publisher is not None:
                    self.pattern_publisher.publish_event(
                        "pattern.meta.detected",
                        meta_pattern
                    )
                    logging.info(f"Published meta-pattern detection event: {meta_pattern['id']}")
                
                meta_patterns.append(meta_pattern)
            elif candidate["strength"] >= approaching_meta_threshold or confidence >= 0.7:
                meta_pattern["threshold_status"] = "approaching"
                
                # Log approaching meta-pattern
                logging.info(f"Meta-pattern approaching threshold: {meta_pattern['id']}")
                logging.info(f"  Pattern type: {candidate['type']}")
                logging.info(f"  Strength: {candidate['strength']} (Threshold: {field_adjusted_meta_threshold}, Approaching: {approaching_meta_threshold})")
                logging.info(f"  Confidence: {meta_pattern['confidence']}")
                logging.info(f"  Topology: {candidate['topology']}")
                
                # Update the AdaptiveID with this approaching meta-pattern
                timestamp = datetime.now().isoformat()
                self.adaptive_id.update_temporal_context(
                    "meta_pattern_approaching", 
                    {
                        "type": candidate["type"],
                        "data": meta_pattern,
                        "timestamp": timestamp
                    },
                    "pattern_detection"
                )
                
                # Publish event for approaching meta-pattern
                if hasattr(self, 'pattern_publisher') and self.pattern_publisher is not None:
                    self.pattern_publisher.publish_event(
                        "meta_pattern_approaching",
                        meta_pattern
                    )
                    logging.info(f"Published approaching meta-pattern event: {meta_pattern['id']}")
                
                self.approaching_meta_patterns.append(meta_pattern)
        

        
        # Add meta-patterns to potential patterns
        self.potential_patterns.extend(meta_patterns)
        
        # Return both exceeded and approaching meta-patterns
        all_meta_patterns = meta_patterns + self.approaching_meta_patterns
        return all_meta_patterns
    
    def _create_pattern_relationship_graph(self) -> Dict[str, Any]:
        """
        Create a graph representation of pattern relationships.
        
        This looks at how patterns relate to each other in the semantic space,
        rather than just counting discrete evolution types.
        
        Returns:
            Graph representation of pattern relationships
        """
        # Create graph structure
        graph = {
            "nodes": {},
            "edges": [],
            "clusters": [],
            "field_metrics": {
                "coherence": self.field_coherence,
                "stability": self.field_stability,
                "density": 0.0,  # Will be calculated
                "connectivity": 0.0  # Will be calculated
            }
        }
        
        # Add patterns as nodes
        for pattern in self.pattern_history:
            node_id = pattern["id"]
            graph["nodes"][node_id] = {
                "id": node_id,
                "type": "pattern",
                "source": pattern.get("source", ""),
                "predicate": pattern.get("predicate", ""),
                "target": pattern.get("target", ""),
                "frequency": pattern.get("frequency", 1),
                "confidence": pattern.get("confidence", 0.5),
                "timestamp": pattern.get("detection_timestamp", "")
            }
        
        # Add evolution relationships as edges
        for pattern_id, evolutions in self.pattern_evolution.items():
            for evolution in evolutions:
                to_pattern = evolution.get("to_pattern")
                if to_pattern and pattern_id in graph["nodes"] and to_pattern in graph["nodes"]:
                    edge = {
                        "source": pattern_id,
                        "target": to_pattern,
                        "type": evolution.get("evolution_type", "unknown"),
                        "weight": 1.0,  # Base weight
                        "timestamp": evolution.get("timestamp", "")
                    }
                    graph["edges"].append(edge)
        
        # Add semantic relationships between patterns
        for i, pattern1 in enumerate(self.pattern_history):
            for pattern2 in self.pattern_history[i+1:]:
                # Skip if already connected through evolution
                if any(e["source"] == pattern1["id"] and e["target"] == pattern2["id"] for e in graph["edges"]) or \
                   any(e["source"] == pattern2["id"] and e["target"] == pattern1["id"] for e in graph["edges"]):
                    continue
                
                # Check for semantic relationships
                relationship_strength = self._calculate_semantic_relationship(pattern1, pattern2)
                if relationship_strength > 0.3:  # Threshold for meaningful relationships
                    edge = {
                        "source": pattern1["id"],
                        "target": pattern2["id"],
                        "type": "semantic_relationship",
                        "weight": relationship_strength,
                        "timestamp": datetime.now().isoformat()
                    }
                    graph["edges"].append(edge)
        
        # Calculate graph metrics
        if graph["edges"]:
            graph["field_metrics"]["density"] = len(graph["edges"]) / (len(graph["nodes"]) * (len(graph["nodes"]) - 1) / 2) if len(graph["nodes"]) > 1 else 0
            graph["field_metrics"]["connectivity"] = sum(e["weight"] for e in graph["edges"]) / len(graph["edges"])
        
        return graph
    
    def _calculate_semantic_relationship(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """
        Calculate the semantic relationship strength between two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Relationship strength (0.0-1.0)
        """
        # Start with base relationship strength
        strength = 0.0
        
        # Check for shared elements
        if pattern1.get("source") == pattern2.get("source"):
            strength += 0.3
        if pattern1.get("predicate") == pattern2.get("predicate"):
            strength += 0.3
        if pattern1.get("target") == pattern2.get("target"):
            strength += 0.3
        
        # Check for semantic similarity in context if available
        if "context" in pattern1 and "context" in pattern2:
            context_similarity = self._calculate_context_consistency([pattern1["context"], pattern2["context"]])
            strength += context_similarity * 0.2
        
        # Apply field influence
        field_factor = (self.field_coherence * 0.1) + (self.field_stability * 0.1)
        strength = min(1.0, strength + field_factor)
        
        return strength
    
    def _analyze_pattern_topology(self, pattern_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze pattern graph topology to identify meta-pattern candidates.
        
        This looks for topological features like clusters, paths, and hubs
        that indicate meta-patterns in the continuous pattern space.
        
        Args:
            pattern_graph: Graph representation of pattern relationships
            
        Returns:
            List of meta-pattern candidates
        """
        candidates = []
        
        # Skip if graph is too small
        if len(pattern_graph["nodes"]) < 3 or not pattern_graph["edges"]:
            return candidates
        
        # 1. Identify clusters (densely connected patterns)
        clusters = self._identify_pattern_clusters(pattern_graph)
        for i, cluster in enumerate(clusters):
            if len(cluster) >= 3:  # Minimum size for a meaningful cluster
                candidates.append({
                    "type": "pattern_cluster",
                    "patterns": cluster,
                    "strength": len(cluster) * pattern_graph["field_metrics"]["connectivity"],
                    "topology": {
                        "type": "cluster",
                        "density": pattern_graph["field_metrics"]["density"],
                        "size": len(cluster)
                    }
                })
        
        # 2. Identify chains (sequential pattern relationships)
        chains = self._identify_pattern_chains(pattern_graph)
        for i, chain in enumerate(chains):
            if len(chain) >= 3:  # Minimum length for a meaningful chain
                candidates.append({
                    "type": "pattern_sequence",
                    "patterns": chain,
                    "strength": len(chain) * 0.8,  # Chains are slightly less significant than clusters
                    "topology": {
                        "type": "chain",
                        "length": len(chain),
                        "linearity": 0.8  # Measure of how linear the chain is
                    }
                })
        
        # 3. Identify hubs (patterns with many connections)
        hubs = self._identify_pattern_hubs(pattern_graph)
        for hub_id, connections in hubs.items():
            if len(connections) >= 3:  # Minimum connections for a hub
                candidates.append({
                    "type": "pattern_hub",
                    "patterns": [hub_id] + connections,
                    "strength": len(connections) * 0.7,  # Hubs are slightly less significant than chains
                    "topology": {
                        "type": "hub",
                        "center": hub_id,
                        "connections": len(connections)
                    }
                })
        
        return candidates
    
    def _identify_pattern_clusters(self, pattern_graph: Dict[str, Any]) -> List[List[str]]:
        """
        Identify clusters of densely connected patterns.
        
        Args:
            pattern_graph: Graph representation of pattern relationships
            
        Returns:
            List of pattern clusters (each cluster is a list of pattern IDs)
        """
        # Simple clustering based on edge density
        clusters = []
        visited = set()
        
        for node_id in pattern_graph["nodes"]:
            if node_id in visited:
                continue
                
            # Find connected component
            cluster = [node_id]
            visited.add(node_id)
            queue = [node_id]
            
            while queue:
                current = queue.pop(0)
                
                # Find neighbors
                neighbors = []
                for edge in pattern_graph["edges"]:
                    if edge["source"] == current and edge["target"] not in visited:
                        neighbors.append(edge["target"])
                    elif edge["target"] == current and edge["source"] not in visited:
                        neighbors.append(edge["source"])
                
                # Add neighbors to cluster
                for neighbor in neighbors:
                    cluster.append(neighbor)
                    visited.add(neighbor)
                    queue.append(neighbor)
            
            if len(cluster) > 1:  # Only consider clusters with multiple patterns
                clusters.append(cluster)
        
        return clusters
    
    def _identify_pattern_chains(self, pattern_graph: Dict[str, Any]) -> List[List[str]]:
        """
        Identify chains of sequentially related patterns.
        
        Args:
            pattern_graph: Graph representation of pattern relationships
            
        Returns:
            List of pattern chains (each chain is a list of pattern IDs)
        """
        # Find paths in the graph
        chains = []
        visited_edges = set()
        
        # Find all starting points (nodes with only outgoing edges or fewer incoming than outgoing)
        start_nodes = []
        for node_id in pattern_graph["nodes"]:
            outgoing = sum(1 for e in pattern_graph["edges"] if e["source"] == node_id)
            incoming = sum(1 for e in pattern_graph["edges"] if e["target"] == node_id)
            if outgoing > 0 and (incoming == 0 or outgoing > incoming):
                start_nodes.append(node_id)
        
        # If no clear starting points, use any node with outgoing edges
        if not start_nodes:
            start_nodes = [e["source"] for e in pattern_graph["edges"]]
            start_nodes = list(set(start_nodes))  # Remove duplicates
        
        # Find chains from each starting point
        for start in start_nodes:
            self._find_chains_from_node(start, [], pattern_graph, visited_edges, chains)
        
        return chains
    
    def _find_chains_from_node(self, node: str, current_chain: List[str], 
                              pattern_graph: Dict[str, Any], visited_edges: set,
                              chains: List[List[str]]):
        """
        Recursively find chains starting from a node.
        
        Args:
            node: Current node ID
            current_chain: Chain built so far
            pattern_graph: Graph representation
            visited_edges: Set of visited edges
            chains: List to store found chains
        """
        # Add current node to chain
        current_chain.append(node)
        
        # Find outgoing edges
        outgoing_edges = []
        for edge in pattern_graph["edges"]:
            edge_id = f"{edge['source']}-{edge['target']}" 
            if edge["source"] == node and edge_id not in visited_edges:
                outgoing_edges.append((edge, edge["target"]))
                visited_edges.add(edge_id)
        
        # If no outgoing edges, we've reached the end of a chain
        if not outgoing_edges:
            if len(current_chain) >= 3:  # Only consider chains with at least 3 patterns
                chains.append(current_chain.copy())
            current_chain.pop()  # Remove current node before returning
            return
        
        # Follow each outgoing edge
        for edge, target in outgoing_edges:
            # Avoid cycles
            if target not in current_chain:
                self._find_chains_from_node(target, current_chain, pattern_graph, visited_edges, chains)
        
        # Remove current node before returning
        current_chain.pop()
    
    def _identify_pattern_hubs(self, pattern_graph: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Identify hub patterns with many connections.
        
        Args:
            pattern_graph: Graph representation of pattern relationships
            
        Returns:
            Dictionary mapping hub pattern IDs to lists of connected pattern IDs
        """
        # Count connections for each node
        connections = {}
        for node_id in pattern_graph["nodes"]:
            connected_nodes = []
            for edge in pattern_graph["edges"]:
                if edge["source"] == node_id and edge["target"] not in connected_nodes:
                    connected_nodes.append(edge["target"])
                elif edge["target"] == node_id and edge["source"] not in connected_nodes:
                    connected_nodes.append(edge["source"])
            
            if len(connected_nodes) >= 3:  # Only consider nodes with at least 3 connections
                connections[node_id] = connected_nodes
        
        return connections
        
    def _get_field_adjusted_threshold(self, base_threshold):
        """
        Calculate field-adjusted threshold for pattern detection.
        
        Higher coherence and stability lead to lower thresholds (more sensitive detection).
        
        Args:
            base_threshold: Base threshold value to adjust
            
        Returns:
            Field-adjusted threshold
        """
        # Calculate field influence factor (0.0-0.5 range)
        field_factor = (self.field_coherence * 0.3) + (self.field_stability * 0.2)
        
        # Apply field influence (can reduce threshold by up to 50%)
        adjusted_threshold = base_threshold * (1.0 - field_factor)
        
        # Ensure threshold doesn't go below 1
        return max(1.0, adjusted_threshold)
    
    def _get_field_adjusted_meta_threshold(self):
        """
        Calculate field-adjusted threshold for meta-pattern detection.
        
        Meta-patterns are more sensitive to field state than regular patterns.
        
        Returns:
            Field-adjusted meta-pattern threshold
        """
        # Start with base meta-pattern threshold
        base_threshold = self.meta_pattern_threshold
        
        # Calculate field influence factor (0.0-0.7 range) - stronger for meta-patterns
        field_factor = (self.field_coherence * 0.4) + (self.field_stability * 0.3)
        
        # Apply field influence (can reduce threshold by up to 70%)
        adjusted_threshold = base_threshold * (1.0 - field_factor)
        
        # Ensure threshold doesn't go below 1
        return max(1.0, adjusted_threshold)
    
    def update_field_state(self, coherence: float, stability: float):
        """
        Update field state parameters that influence threshold adjustments.
        
        Args:
            coherence: Field coherence value (0.0-1.0)
            stability: Field stability value (0.0-1.0)
        """
        self.field_coherence = max(0.0, min(1.0, coherence))
        self.field_stability = max(0.0, min(1.0, stability))
        
        logging.info(f"Updated field state: coherence={self.field_coherence:.2f}, stability={self.field_stability:.2f}")
        logging.info(f"Adjusted thresholds: pattern={self._get_field_adjusted_threshold(self.threshold):.2f}, meta-pattern={self._get_field_adjusted_meta_threshold():.2f}")
    
    def get_pattern_evolution(self, pattern_id: str) -> List[Dict[str, Any]]:
        """
        Get the evolution history for a pattern.
        
        Args:
            pattern_id: ID of the pattern
            
        Returns:
            List of evolution events
        """
        if pattern_id in self.pattern_evolution:
            return self.pattern_evolution[pattern_id]
        
        # Check if this pattern evolved from another
        for pattern in self.pattern_history:
            if pattern["id"] == pattern_id and "evolved_from" in pattern:
                return self.get_pattern_evolution(pattern["evolved_from"])
        
        return []
    
    def register_with_field_observer(self, field_observer) -> None:
        """
        Register this detector with a field observer.
        
        Args:
            field_observer: The field observer to register with
        """
        self.adaptive_id.register_with_field_observer(field_observer)
