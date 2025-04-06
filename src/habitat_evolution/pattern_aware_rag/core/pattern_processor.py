"""
Pattern processor for sequential foundation of pattern-aware RAG.
"""
from typing import Dict, Optional, List, Any
from datetime import datetime, timezone
from dataclasses import dataclass
from ..state.test_states import PatternState, GraphStateSnapshot, ConceptNode, ConceptRelation
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from .exceptions import InvalidStateError, StateValidationError

@dataclass
class EventCoordinator:
    """Coordinates pattern-aware events and state transitions."""
    events: List[Any] = None
    
    def __post_init__(self):
        self.events = [] if self.events is None else self.events
    
    def register_event(self, event: Any) -> None:
        """Register a new event."""
        self.events.append(event)
    
    def get_events(self) -> List[Any]:
        """Get all registered events."""
        return self.events

class PatternProcessor:
    """Processes patterns through the sequential foundation stages."""
    
    async def extract_pattern(self, document: Dict) -> PatternState:
        """Extract pattern from document with provenance tracking."""
        pattern = PatternState(
            id="temp_" + str(hash(document["content"])),
            content=document["content"],
            metadata=document["metadata"],
            timestamp=datetime.fromisoformat(document["metadata"]["timestamp"])
        )
        return pattern
    
    async def assign_adaptive_id(self, pattern: PatternState) -> AdaptiveID:
        """Assign Adaptive ID to pattern."""
        # Create deterministic base concept from pattern content
        base_concept = pattern.content[:10].lower().replace(" ", "_")
        # Create an AdaptiveID with the pattern's metadata
        return AdaptiveID(
            base_concept=base_concept,
            creator_id=pattern.metadata.get("source", "system"),
            weight=1.0,
            confidence=1.0,
            uncertainty=0.0
        )
    
    async def form_prompt(self, state: GraphStateSnapshot, context: Dict[str, Any]) -> str:
        """Form a prompt based on graph state and context."""
        # Basic template with state and context integration
        template = (
            "Given the following state and context:\n"
            "State ID: {state_id}\n"
            "Number of nodes: {node_count}\n"
            "Number of patterns: {pattern_count}\n"
            "Context type: {context_type}\n"
            "\nPatterns:\n{pattern_content}\n"
            "\nContext:\n{context_details}\n"
            "\nAnalyze the patterns and their relationships."
        )
        
        # Format pattern content
        pattern_content = "\n".join(
            f"- {p.content} (confidence: {p.confidence:.2f})"
            for p in state.patterns
        )
        
        # Format context details
        context_details = "\n".join(
            f"- {k}: {v}"
            for k, v in context.items()
        )
        
        return template.format(
            state_id=state.id,
            node_count=len(state.nodes),
            pattern_count=len(state.patterns),
            context_type=context.get("type", "default"),
            pattern_content=pattern_content,
            context_details=context_details
        )

    async def construct_prompt(self, state: GraphStateSnapshot, template_key: str) -> str:
        """Construct a prompt using the current state and template.
        
        Args:
            state: The current graph state snapshot
            template_key: Key to identify which template to use
            
        Returns:
            str: The constructed prompt with variables substituted
            
        Raises:
            ValueError: If template_key is invalid or required variables are missing
        """
        # Template definitions
        templates = {
            "basic_template": (
                "Pattern: {pattern_content}\n"
                "Node: {node_name}\n"
                "Analysis required."
            ),
            "nested_template": (
                "Pattern Type: {pattern_type}\n"
                "Node Type: {node_type}\n"
                "Detailed analysis needed."
            )
        }
        
        # Validate template key
        if template_key not in templates:
            raise ValueError(f"Invalid template key: {template_key}")
            
        # Get the template
        template = templates[template_key]
        
        try:
            # Get pattern and node for both templates
            if not state.patterns or not state.nodes:
                raise ValueError("Missing required pattern or node")
                
            pattern = state.patterns[0]
            node = state.nodes[0]
            
            # For basic template
            if template_key == "basic_template":
                if not pattern.content or not node.name:
                    raise ValueError("Missing required variable: content or name")
                if not pattern.metadata.get("type") or not node.attributes.get("type"):
                    raise ValueError("Missing required variable: type in pattern metadata or node attributes")
                return template.format(
                    pattern_content=pattern.content,
                    node_name=node.name
                )
            
            # For nested template
            elif template_key == "nested_template":
                if not pattern.metadata.get("type") or not node.attributes.get("type"):
                    raise ValueError("Missing required variable: type in pattern metadata or node attributes")
                return template.format(
                    pattern_type=pattern.metadata["type"],
                    node_type=node.attributes["type"]
                )
        
        except (KeyError, IndexError) as e:
            raise ValueError(f"Missing required variable: {str(e)}")
            
        except Exception as e:
            raise ValueError(f"Error constructing prompt: {str(e)}")
        # Basic template with state and context integration
        template = (
            "Given the following state and context:\n"
            "State ID: {state_id}\n"
            "Number of nodes: {node_count}\n"
            "Number of patterns: {pattern_count}\n"
            "Context type: {context_type}\n"
            "\nPatterns:\n{pattern_content}\n"
            "\nContext:\n{context_details}\n"
            "\nAnalyze the patterns and their relationships."
        )
        
        # Format pattern content
        pattern_content = "\n".join(
            f"- {p.content} (confidence: {p.confidence:.2f})"
            for p in state.patterns
        )
        
        # Format context details
        context_details = "\n".join(
            f"- {k}: {v}"
            for k, v in context.items()
        )
        
        return template.format(
            state_id=state.id,
            node_count=len(state.nodes),
            pattern_count=len(state.patterns),
            context_type=context.get("type", "default"),
            pattern_content=pattern_content,
            context_details=context_details
        )
    
    async def reach_consensus(self, state: Optional[GraphStateSnapshot]) -> Dict[str, Any]:
        """Reach consensus on the current state.
        
        Args:
            state: State to reach consensus on
            
        Returns:
            Dict containing consensus result with keys:
                - achieved: bool, True if consensus reached
                - reason: str, reason for failure if not achieved
                - confidence: float, confidence level of consensus
                
        Raises:
            ValueError: If state is None
        """
        if state is None:
            raise ValueError("State cannot be None")
            
        result = {
            "achieved": False,
            "reason": "",
            "confidence": 0.0
        }
        
        try:
            # Check pattern confidence
            if not state.patterns:
                result["reason"] = "No patterns in state"
                return result
                
            # Calculate confidence metrics
            confidences = [p.confidence for p in state.patterns]
            min_confidence = min(confidences)
            avg_confidence = sum(confidences) / len(confidences)
            
            # Apply temporal decay (10% per update)
            decay = 0.9
            decayed_confidence = avg_confidence * decay
            
            # Apply stability-based smoothing
            stability = 0.7  # Favor new confidence values while maintaining some stability
            smoothed_confidence = stability * decayed_confidence + (1 - stability) * avg_confidence
            
            # Check minimum confidence threshold
            if min_confidence < 0.5:
                result["reason"] = "Pattern confidence too low"
                result["confidence"] = min_confidence
                return result
            
            # Validate relations
            state.validate_relations()
            
            # All checks passed
            result["achieved"] = True
            result["confidence"] = max(0.0, min(1.0, smoothed_confidence))  # Clip to [0,1]
            return result
            
        except Exception as e:
            result["reason"] = str(e)
            return result
    
    async def synchronize_states(self, state1: Optional[GraphStateSnapshot], state2: Optional[GraphStateSnapshot]) -> GraphStateSnapshot:
        """Synchronize two states into a new consistent state."""
        if state1 is None or state2 is None:
            raise ValueError("States cannot be None")
            
        # Ensure timestamps are timezone-aware
        if state1.timestamp.tzinfo is None:
            state1.timestamp = state1.timestamp.astimezone()
        if state2.timestamp.tzinfo is None:
            state2.timestamp = state2.timestamp.astimezone()
            
        # Take the higher version number
        new_version = max(state1.version, state2.version)
        
        # Merge nodes and patterns, preferring items from the newer state
        if state1.timestamp > state2.timestamp:
            primary, secondary = state1, state2
        else:
            primary, secondary = state2, state1
        
        # Create new state with merged data
        return GraphStateSnapshot(
            id=f"merged_{primary.id}_{secondary.id}",
            nodes=primary.nodes,
            relations=primary.relations,
            patterns=primary.patterns,
            timestamp=datetime.now(timezone.utc),
            version=new_version + 1
        )
    
    async def resolve_conflicts(self, states: List[GraphStateSnapshot]) -> GraphStateSnapshot:
        """Resolve conflicts between multiple states."""
        if not states:
            raise ValueError("No states provided")
            
        # Ensure all states have timezone-aware timestamps
        for state in states:
            if state.timestamp.tzinfo is None:
                state.timestamp = state.timestamp.astimezone()
        
        # Sort states by version and timestamp
        sorted_states = sorted(
            states,
            key=lambda s: (s.version, s.timestamp),
            reverse=True
        )
        
        # Start with the newest state
        result = sorted_states[0]
        
        # Merge with other states
        for state in sorted_states[1:]:
            result = await self.synchronize_states(result, state)
            
        return result
        
    async def prepare_graph_state(self, pattern: PatternState, adaptive_id: Optional[AdaptiveID]) -> GraphStateSnapshot:
        """Prepare graph-ready state from pattern and adaptive ID.
        
        Args:
            pattern: Pattern to prepare state for
            adaptive_id: Assigned adaptive ID
            
        Returns:
            Graph-ready state
            
        Raises:
            ValueError: If adaptive_id is None
        """
        if adaptive_id is None:
            raise ValueError("Adaptive ID is required")
            
        # Create concept node from pattern
        concept = ConceptNode(
            id=str(adaptive_id),
            name=pattern.content[:50],  # Use first 50 chars as name
            attributes={
                "source": pattern.metadata.get("source", ""),
                "timestamp": pattern.metadata.get("timestamp", "")
            },
            created_at=pattern.timestamp
        )
        
        # Create relation between pattern and concept
        relation = ConceptRelation(
            source_id=concept.id,
            target_id=pattern.id,
            relation_type="CONTAINS",
            weight=1.0
        )
        
        # Create initial state
        return GraphStateSnapshot(
            id=f"state_{pattern.id}",
            nodes=[concept],
            relations=[relation],
            patterns=[pattern],
            timestamp=pattern.timestamp,
            version=1
        )
    


