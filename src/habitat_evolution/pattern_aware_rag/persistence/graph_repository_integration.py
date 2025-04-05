"""
Integration between PatternAwareRAG and GraphStateRepository.

This module provides the integration layer between the pattern-aware RAG system
and the graph state repository, enabling pattern quality evolution to be persisted
and retrieved from the database.
"""
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime

from src.habitat_evolution.adaptive_core.persistence.services.graph_service import GraphService
# Create a mock PatternAwareRAG class for now to avoid import errors
class PatternAwareRAG:
    """Mock PatternAwareRAG class for integration testing."""
    
    def __init__(self):
        self.event_bus = MockEventBus()
        
    async def register_pattern(self, pattern_id, pattern_content, confidence=None, metadata=None):
        """Register a pattern with the RAG system."""
        pass
        
    async def update_pattern_confidence(self, pattern_id, confidence):
        """Update a pattern's confidence score."""
        pass
        
    async def get_all_patterns(self):
        """Get all patterns from the RAG system."""
        return []


class MockEventBus:
    """Mock event bus for testing."""
    
    def __init__(self):
        self.subscribers = {}
        
    def subscribe(self, event_type, callback):
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def publish(self, event_type, event_data):
        """Publish an event."""
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                callback(event_data)
from src.tests.adaptive_core.persistence.arangodb.test_state_models import PatternState


class PatternRepositoryConnector:
    """
    Connector between PatternAwareRAG and the graph repository.
    
    This class provides methods for persisting patterns detected by the
    PatternAwareRAG system and retrieving them from the repository.
    """
    
    def __init__(self, graph_service: GraphService, pattern_rag: PatternAwareRAG):
        """
        Initialize the pattern repository connector.
        
        Args:
            graph_service: The graph service to use for persistence
            pattern_rag: The pattern-aware RAG system
        """
        self.graph_service = graph_service
        self.pattern_rag = pattern_rag
        
    async def register_event_handlers(self):
        """
        Register event handlers for pattern events.
        
        This method sets up the event handlers to listen for pattern
        detection and quality evolution events from the PatternAwareRAG system.
        """
        # Register for pattern detection events
        self.pattern_rag.event_bus.subscribe(
            "pattern.detected", 
            self.handle_pattern_detected
        )
        
        # Register for pattern quality evolution events
        self.pattern_rag.event_bus.subscribe(
            "pattern.quality_evolved",
            self.handle_pattern_quality_evolved
        )
        
        # Register for learning window events
        self.pattern_rag.event_bus.subscribe(
            "window.closed",
            self.handle_window_closed
        )
        
    async def handle_pattern_detected(self, event_data: Dict[str, Any]):
        """
        Handle a pattern detection event.
        
        Args:
            event_data: The event data containing the detected pattern
        """
        pattern_id = event_data.get("pattern_id")
        pattern_content = event_data.get("content")
        confidence = event_data.get("confidence", 0.3)
        metadata = event_data.get("metadata", {})
        
        # Add source information to metadata
        metadata["source"] = "pattern_aware_rag"
        metadata["detection_timestamp"] = datetime.now().isoformat()
        
        # Create the pattern in the repository
        await asyncio.to_thread(
            self.graph_service.create_pattern,
            content=pattern_content,
            metadata=metadata,
            confidence=confidence
        )
        
        print(f"Pattern {pattern_id} persisted to repository with confidence {confidence}")
        
    async def handle_pattern_quality_evolved(self, event_data: Dict[str, Any]):
        """
        Handle a pattern quality evolution event.
        
        Args:
            event_data: The event data containing the quality evolution
        """
        pattern_id = event_data.get("pattern_id")
        new_confidence = event_data.get("confidence")
        context = event_data.get("context", {})
        
        # Add source information to context
        context["source"] = "pattern_aware_rag"
        context["evolution_timestamp"] = datetime.now().isoformat()
        
        # Update the pattern confidence in the repository
        await asyncio.to_thread(
            self.graph_service.evolve_pattern_confidence,
            pattern_id=pattern_id,
            new_confidence=new_confidence,
            context=context
        )
        
        print(f"Pattern {pattern_id} evolved to confidence {new_confidence}")
        
    async def handle_window_closed(self, event_data: Dict[str, Any]):
        """
        Handle a learning window closed event.
        
        Args:
            event_data: The event data containing the window information
        """
        window_id = event_data.get("window_id")
        
        # Create a graph snapshot to capture the state at window close
        await asyncio.to_thread(
            self.graph_service.create_graph_snapshot
        )
        
        print(f"Graph snapshot created for window {window_id}")
        
    async def load_patterns_into_rag(self, quality_threshold: str = "good"):
        """
        Load patterns from the repository into the RAG system.
        
        Args:
            quality_threshold: The minimum quality state to load (default: good)
        """
        # Get patterns by quality
        patterns = await asyncio.to_thread(
            self.graph_service.repository.find_nodes_by_quality,
            self.graph_service.repository.patterns_collection,
            quality_threshold
        )
        
        # Convert to PatternState objects
        pattern_states = []
        for pattern_doc in patterns:
            pattern = PatternState(
                id=pattern_doc["_key"],
                content=pattern_doc["content"],
                metadata=pattern_doc.get("metadata", {}),
                timestamp=datetime.fromisoformat(pattern_doc.get("timestamp")),
                confidence=float(pattern_doc.get("confidence", 0.5))
            )
            pattern_states.append(pattern)
            
        # Load patterns into RAG
        for pattern in pattern_states:
            await self.pattern_rag.register_pattern(
                pattern_id=pattern.id,
                pattern_content=pattern.content,
                confidence=pattern.confidence,
                metadata=pattern.metadata
            )
            
        print(f"Loaded {len(pattern_states)} patterns into RAG system")
        
    async def synchronize_patterns(self):
        """
        Synchronize patterns between the RAG system and the repository.
        
        This method ensures that patterns in the RAG system and the repository
        are consistent, updating confidence scores and quality states as needed.
        """
        # Get all patterns from RAG
        rag_patterns = await self.pattern_rag.get_all_patterns()
        
        # Get all patterns from repository
        repo_patterns = await asyncio.to_thread(
            self._get_all_patterns_from_repo
        )
        
        # Create a mapping of pattern IDs to patterns
        rag_pattern_map = {p["id"]: p for p in rag_patterns}
        repo_pattern_map = {p.id: p for p in repo_patterns}
        
        # Find patterns in RAG but not in repo
        for pattern_id, pattern in rag_pattern_map.items():
            if pattern_id not in repo_pattern_map:
                # Create pattern in repo
                await asyncio.to_thread(
                    self.graph_service.create_pattern,
                    content=pattern["content"],
                    metadata=pattern.get("metadata", {}),
                    confidence=pattern.get("confidence", 0.3)
                )
                
        # Find patterns in repo but not in RAG
        for pattern_id, pattern in repo_pattern_map.items():
            if pattern_id not in rag_pattern_map:
                # Load pattern into RAG
                await self.pattern_rag.register_pattern(
                    pattern_id=pattern.id,
                    pattern_content=pattern.content,
                    confidence=pattern.confidence,
                    metadata=pattern.metadata
                )
                
        # Update confidence scores for patterns in both
        for pattern_id in set(rag_pattern_map.keys()) & set(repo_pattern_map.keys()):
            rag_pattern = rag_pattern_map[pattern_id]
            repo_pattern = repo_pattern_map[pattern_id]
            
            # If confidence scores differ, update the lower one to match the higher one
            if abs(rag_pattern.get("confidence", 0) - repo_pattern.confidence) > 0.01:
                if rag_pattern.get("confidence", 0) > repo_pattern.confidence:
                    # Update repo pattern
                    await asyncio.to_thread(
                        self.graph_service.evolve_pattern_confidence,
                        pattern_id=pattern_id,
                        new_confidence=rag_pattern.get("confidence", 0),
                        context={"source": "synchronization"}
                    )
                else:
                    # Update RAG pattern
                    await self.pattern_rag.update_pattern_confidence(
                        pattern_id=pattern_id,
                        confidence=repo_pattern.confidence
                    )
                    
        print(f"Synchronized {len(rag_pattern_map)} RAG patterns and {len(repo_pattern_map)} repository patterns")
        
    def _get_all_patterns_from_repo(self) -> List[PatternState]:
        """
        Get all patterns from the repository.
        
        Returns:
            List of pattern state objects
        """
        # This would need a method to get all patterns from the repository
        # For now, we'll return an empty list
        return []
