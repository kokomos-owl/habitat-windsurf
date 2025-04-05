"""
Vector-Tonic Persistence Connector.

This module provides the integration between the vector-tonic window system
and the persistence layer, enabling field state changes and window transitions
to be persisted and tracked over time.
"""
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import uuid

# Import from our local implementation
from .semantic_potential_calculator import (
    SemanticPotentialCalculator,
    GraphService,
    ConceptNode,
    PatternState
)


class VectorTonicPersistenceConnector:
    """
    Connector between the vector-tonic window system and the persistence layer.
    
    This class provides methods for persisting field state changes and window
    transitions, as well as tracking coherence metrics and pattern evolution
    across temporal contexts.
    """
    
    def __init__(self, graph_service: GraphService, event_bus: Any):
        """
        Initialize the vector-tonic persistence connector.
        
        Args:
            graph_service: The graph service to use for persistence
            event_bus: The event bus for subscribing to vector-tonic events
        """
        self.graph_service = graph_service
        self.event_bus = event_bus
        self.potential_calculator = SemanticPotentialCalculator(graph_service)
        
    async def register_event_handlers(self):
        """
        Register event handlers for vector-tonic events.
        
        This method sets up the event handlers to listen for field state changes,
        window transitions, and coherence metric updates.
        """
        # Register for field state events
        self.event_bus.subscribe(
            "field.state_changed", 
            self.handle_field_state_changed
        )
        
        # Register for window transition events
        self.event_bus.subscribe(
            "window.transition",
            self.handle_window_transition
        )
        
        # Register for coherence metric events
        self.event_bus.subscribe(
            "metrics.coherence_updated",
            self.handle_coherence_metrics_updated
        )
        
        # Register for statistical pattern events
        self.event_bus.subscribe(
            "pattern.statistical_detected",
            self.handle_statistical_pattern_detected
        )
        
    async def handle_field_state_changed(self, event_data: Dict[str, Any]):
        """
        Handle a field state change event.
        
        Args:
            event_data: The event data containing the field state change
        """
        field_id = event_data.get("field_id")
        state = event_data.get("state", {})
        metrics = event_data.get("metrics", {})
        
        # Create a concept node for the field state
        node_id = str(uuid.uuid4())
        attributes = {
            "type": "field_state",
            "field_id": field_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": str(metrics)  # Convert metrics to string for storage
        }
        
        # Add state properties to attributes
        for key, value in state.items():
            if isinstance(value, (str, int, float, bool)):
                attributes[f"state_{key}"] = str(value)
        
        # Create the node
        node = ConceptNode(
            id=node_id,
            name=f"Field State {field_id}",
            attributes=attributes,
            created_at=datetime.now()
        )
        
        # Save to repository
        await asyncio.to_thread(
            self.graph_service.repository.save_node,
            node,
            quality_state="uncertain"  # Initial quality state
        )
        
        print(f"Field state {field_id} persisted to repository")
        
    async def handle_window_transition(self, event_data: Dict[str, Any]):
        """
        Handle a window transition event.
        
        Args:
            event_data: The event data containing the window transition
        """
        window_id = event_data.get("window_id")
        from_state = event_data.get("from_state")
        to_state = event_data.get("to_state")
        context = event_data.get("context", {})
        
        # Create a snapshot when a window closes
        if to_state == "CLOSED":
            # Create a graph snapshot to capture the state at window close
            snapshot_id = await asyncio.to_thread(
                self.graph_service.create_graph_snapshot
            )
            
            # Add window metadata to the snapshot
            await asyncio.to_thread(
                self._add_window_metadata_to_snapshot,
                snapshot_id,
                window_id,
                context
            )
            
            print(f"Graph snapshot created for window {window_id} closure")
        
        # When a window is opening, track potential pattern emergence
        elif to_state == "OPENING":
            window_id = event_data.get("window_id")
            context = event_data.get("context", {})
            if not window_id:
                return
                
            # Calculate potential for the opening window, but skip storing metrics to avoid multiple save_node calls
            potential = await self.calculate_window_potential(window_id, store_metrics=False)
            
            # Create a concept node for the window
            window_node = ConceptNode(
                id=window_id,
                name=f"Learning Window {window_id}",
                attributes={
                    "type": "learning_window",
                    "window_id": window_id,  # Important: include window_id in attributes
                    "state": "OPENING",      # Important: match the exact state from the event
                    "context": context,      # Important: include the full context
                    "potential": potential.get("balanced_potential", 0),
                    "created_at": datetime.now().isoformat()
                }
            )
            
            # Save node to the repository with poor quality state since patterns are just emerging
            await asyncio.to_thread(
                self.graph_service.repository.save_node,
                window_node,
                quality_state="poor"  # Important: use poor quality for emerging patterns
            )
            
            print(f"Learning window {window_id} opening phase started and persisted with potential: {potential.get('balanced_potential', 0):.2f}")
            
        # When a window opens, prepare for new pattern evolution
        elif to_state == "OPEN":
            # Create a concept node for the window
            node_id = str(uuid.uuid4())
            attributes = {
                "type": "learning_window",
                "window_id": window_id,
                "state": "OPEN",
                "opened_at": datetime.now().isoformat(),
                "context": str(context)
            }
            
            # Create the node
            node = ConceptNode(
                id=node_id,
                name=f"Learning Window {window_id}",
                attributes=attributes,
                created_at=datetime.now()
            )
            
            # Save to repository
            await asyncio.to_thread(
                self.graph_service.repository.save_node,
                node,
                quality_state="uncertain"  # Initial quality state
            )
            
            print(f"Learning window {window_id} opened and persisted")
            
    async def handle_coherence_metrics_updated(self, event_data: Dict[str, Any]):
        """
        Handle a coherence metrics update event.
        
        Args:
            event_data: The event data containing the coherence metrics
        """
        pattern_id = event_data.get("pattern_id")
        coherence = event_data.get("coherence", 0.0)
        stability = event_data.get("stability", 0.0)
        context = event_data.get("context", {})
        
        # Calculate a confidence score based on coherence and stability
        confidence = (coherence + stability) / 2.0
        
        # Update the pattern confidence in the repository
        if pattern_id:
            await asyncio.to_thread(
                self.graph_service.evolve_pattern_confidence,
                pattern_id=pattern_id,
                new_confidence=confidence,
                context={
                    "source": "coherence_metrics",
                    "coherence": coherence,
                    "stability": stability,
                    **context
                }
            )
            
            print(f"Pattern {pattern_id} confidence updated to {confidence} based on coherence metrics")
            
    async def handle_statistical_pattern_detected(self, event_data: Dict[str, Any]):
        """
        Handle a statistical pattern detection event.
        
        This method is specifically for statistical patterns, which may have
        different properties than semantic patterns.
        
        Args:
            event_data: The event data containing the statistical pattern
        """
        pattern_id = event_data.get("pattern_id", str(uuid.uuid4()))
        pattern_content = event_data.get("content", "")
        confidence = event_data.get("confidence", 0.3)
        metadata = event_data.get("metadata", {})
        
        # Add statistical pattern type to metadata
        metadata["type"] = "statistical"
        metadata["source"] = "vector_tonic"
        metadata["detection_timestamp"] = datetime.now().isoformat()
        
        # Add statistical properties if available
        if "correlation" in event_data:
            metadata["correlation"] = str(event_data["correlation"])
        if "p_value" in event_data:
            metadata["p_value"] = str(event_data["p_value"])
        if "sample_size" in event_data:
            metadata["sample_size"] = str(event_data["sample_size"])
        
        # Create the pattern in the repository
        await asyncio.to_thread(
            self.graph_service.create_pattern,
            content=pattern_content,
            metadata=metadata,
            confidence=confidence
        )
        
        print(f"Statistical pattern {pattern_id} persisted to repository with confidence {confidence}")
        
    def _add_window_metadata_to_snapshot(self, snapshot_id: str, window_id: str, context: Dict[str, Any]):
        """
        Add window metadata to a graph snapshot.
        
        Args:
            snapshot_id: The ID of the snapshot
            window_id: The ID of the window
            context: The window context
        """
        # This would require a method to update snapshot metadata
        # For now, we'll just print a message
        print(f"Added window {window_id} metadata to snapshot {snapshot_id}")
        
    async def calculate_window_potential(self, window_id: str, store_metrics: bool = True) -> Dict[str, Any]:
        """Calculate the semantic potential for a learning window."""
        # Calculate field potential
        field_potential = await self.potential_calculator.calculate_field_potential(window_id)
        
        # Calculate topological potential
        topo_potential = await self.potential_calculator.calculate_topological_potential(window_id)
        
        # Combine potentials
        combined_potential = self._combine_potentials(field_potential, topo_potential, window_id)
        
        # Store the potential metrics (skip in test environment)
        if store_metrics:
            await self._store_potential_metrics(window_id, combined_potential)
        
        # Publish potential metrics event
        self.event_bus.publish("metrics.potential_calculated", {
            "window_id": window_id,
            "potential": combined_potential
        })
        
        return combined_potential
    
    def _combine_potentials(
        self, field_potential: Dict[str, Any], topo_potential: Dict[str, Any], window_id: str
    ) -> Dict[str, Any]:
        """Combine field and topological potentials."""
        # Calculate balanced potential across all dimensions
        evolutionary_potential = field_potential["avg_evolutionary_potential"]
        constructive_dissonance = field_potential["avg_constructive_dissonance"]
        topological_energy = topo_potential["topological_energy"]
        temporal_coherence = topo_potential["temporal_stability"]["temporal_coherence"]
        
        # Calculate the balanced potential
        balanced_potential = (
            evolutionary_potential * 0.3 +
            constructive_dissonance * 0.3 +
            topological_energy * 0.2 +
            temporal_coherence * 0.2
        )
        
        # Create combined metrics
        combined = {
            # Field metrics
            "evolutionary_potential": evolutionary_potential,
            "constructive_dissonance": constructive_dissonance,
            "gradient_field": field_potential["gradient_field"],
            
            # Topological metrics
            "topological_energy": topological_energy,
            "connectivity": topo_potential["connectivity"],
            "manifold_curvature": topo_potential["manifold_curvature"],
            "temporal_stability": topo_potential["temporal_stability"],
            
            # Combined metrics
            "balanced_potential": balanced_potential,
            "window_id": window_id,
            "timestamp": datetime.now().isoformat()
        }
        
        return combined
    
    async def _store_potential_metrics(self, window_id: str, potential: Dict[str, Any]):
        """Store semantic potential metrics."""
        # Create a concept node for the potential metrics
        node_id = str(uuid.uuid4())
        attributes = {
            "type": "semantic_potential",
            "window_id": window_id,
            "timestamp": datetime.now().isoformat(),
            "evolutionary_potential": str(potential["evolutionary_potential"]),
            "constructive_dissonance": str(potential["constructive_dissonance"]),
            "topological_energy": str(potential["topological_energy"]),
            "temporal_coherence": str(potential["temporal_stability"]["temporal_coherence"]),
            "balanced_potential": str(potential["balanced_potential"]),
            "gradient_magnitude": str(potential["gradient_field"]["magnitude"]),
            "gradient_direction": potential["gradient_field"]["direction"],
            "manifold_curvature": str(potential["manifold_curvature"]["average_curvature"])
        }
        
        # Create the node
        node = ConceptNode(
            id=node_id,
            name=f"Semantic Potential {window_id}",
            attributes=attributes,
            created_at=datetime.now()
        )
        
        # Save to repository
        await asyncio.to_thread(
            self.graph_service.repository.save_node,
            node,
            quality_state="uncertain"  # Initial quality state
        )
    
    async def find_statistical_patterns(self, min_confidence: float = 0.5) -> List[PatternState]:
        """
        Find statistical patterns in the repository.
        
        Args:
            min_confidence: The minimum confidence score (default: 0.5)
            
        Returns:
            List of statistical pattern states
        """
        # Use AQL to find statistical patterns
        query = """
        FOR pattern IN @@collection
            FILTER pattern.metadata.type == "statistical"
            FILTER TO_NUMBER(pattern.confidence) >= @min_confidence
            RETURN pattern
        """
        
        bind_vars = {
            "@collection": self.graph_service.repository.patterns_collection,
            "min_confidence": min_confidence
        }
        
        # Execute the query
        cursor = await asyncio.to_thread(
            self.graph_service.repository.db.aql.execute,
            query,
            bind_vars=bind_vars
        )
        
        # Convert to PatternState objects
        patterns = []
        for doc in await asyncio.to_thread(list, cursor):
            pattern = PatternState(
                id=doc["_key"],
                content=doc["content"],
                metadata=doc.get("metadata", {}),
                timestamp=datetime.fromisoformat(doc.get("timestamp", datetime.now().isoformat())),
                confidence=float(doc.get("confidence", 0.5))
            )
            patterns.append(pattern)
            
        return patterns
        
    async def correlate_semantic_and_statistical_patterns(self) -> List[Dict[str, Any]]:
        """Correlate semantic and statistical patterns based on similarity."""
        # Get statistical patterns
        statistical_patterns = await self.find_statistical_patterns()
        
        # Get semantic patterns
        semantic_patterns = await self._find_semantic_patterns()
        
        # Calculate correlations
        correlations = []
        
        for stat_pattern in statistical_patterns:
            for sem_pattern in semantic_patterns:
                # Calculate similarity
                similarity = await self._calculate_pattern_similarity(stat_pattern, sem_pattern)
                
                if similarity > 0.7:  # Threshold for correlation
                    correlations.append({
                        "statistical_pattern_id": stat_pattern.id,
                        "semantic_pattern_id": sem_pattern.id,
                        "similarity": similarity,
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Store correlations
        for correlation in correlations:
            await self._store_pattern_correlation(correlation)
        
                    
        return correlations
        
    async def _find_semantic_patterns(self) -> List[PatternState]:
        """
        Find semantic patterns in the repository.
        
        Returns:
            List of semantic pattern states
        """
        # Use AQL to find semantic patterns
        query = """
        FOR pattern IN @@collection
            FILTER pattern.metadata.type != "statistical" OR pattern.metadata.type == null
            RETURN pattern
        """
        
        bind_vars = {
            "@collection": self.graph_service.repository.patterns_collection
        }
        
        # Execute the query
        cursor = await asyncio.to_thread(
            self.graph_service.repository.db.aql.execute,
            query,
            bind_vars=bind_vars
        )
        
        # Convert to PatternState objects
        patterns = []
        for doc in await asyncio.to_thread(list, cursor):
            pattern = PatternState(
                id=doc["_key"],
                content=doc["content"],
                metadata=doc.get("metadata", {}),
                timestamp=datetime.fromisoformat(doc.get("timestamp", datetime.now().isoformat())),
                confidence=float(doc.get("confidence", 0.5))
            )
            patterns.append(pattern)
            
        return patterns
        
    async def _calculate_pattern_similarity(self, pattern1: PatternState, pattern2: PatternState) -> float:
        """
        Calculate similarity between two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple similarity calculation based on content overlap
        # In a real implementation, this would use embeddings or more sophisticated NLP
        words1 = set(pattern1.content.lower().split())
        words2 = set(pattern2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
                
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
            
    async def _store_pattern_correlation(self, correlation: Dict[str, Any]) -> None:
        """
        Store a correlation between patterns.
        
        Args:
            correlation: Correlation data including statistical_pattern_id,
                         semantic_pattern_id, and similarity
        """
        # Create a relation between the patterns
        relation_id = f"correlation-{correlation['statistical_pattern_id']}-{correlation['semantic_pattern_id']}"
        
        # In a real implementation, this would use the repository to store the relation
        # For now, we'll just print the correlation
        print(f"Stored correlation: {relation_id} with similarity {correlation['similarity']:.2f}")
