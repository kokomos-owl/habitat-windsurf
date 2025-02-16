"""
Claude State Handler for Pattern-Aware RAG.

Manages the transformation of graph states into Claude-optimized prompts,
ensuring proper concept relation representation and context preservation.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.schema import Document

@dataclass
class ConceptRelation:
    """Represents a relationship between concepts in the graph."""
    source_id: str
    target_id: str
    relation_type: str
    strength: float
    context: Dict[str, str]
    timestamp: datetime

@dataclass
class GraphStateContext:
    """Represents the full context of a graph state for Claude."""
    state_id: str
    timestamp: datetime
    concepts: Dict[str, Dict[str, any]]
    relations: List[ConceptRelation]
    coherence_metrics: Dict[str, float]
    temporal_context: Dict[str, any]

    def to_prompt_context(self) -> Dict[str, any]:
        """Convert state to Claude-optimized context."""
        return {
            "state_summary": self._build_state_summary(),
            "concept_network": self._build_concept_network(),
            "coherence_metrics": self.coherence_metrics,
            "temporal_context": self._build_temporal_context()
        }
    
    def _build_state_summary(self) -> str:
        """Build a concise summary of the current state."""
        return f"""Current State ({self.state_id}):
- Timestamp: {self.timestamp}
- Concepts: {len(self.concepts)}
- Relations: {len(self.relations)}
- Coherence: {self.coherence_metrics.get('overall', 0.0):.2f}"""

    def _build_concept_network(self) -> List[str]:
        """Build a structured representation of concept relationships."""
        network = []
        for relation in self.relations:
            source = self.concepts.get(relation.source_id, {}).get('name', 'unknown')
            target = self.concepts.get(relation.target_id, {}).get('name', 'unknown')
            network.append(
                f"{source} --[{relation.relation_type}:{relation.strength:.2f}]--> {target}"
            )
        return network

    def _build_temporal_context(self) -> Dict[str, any]:
        """Build temporal context with evolution markers."""
        return {
            "current_timestamp": self.timestamp,
            "evolution_markers": self.temporal_context.get('evolution_markers', []),
            "stability_index": self.temporal_context.get('stability_index', 0.0)
        }

class ClaudeStateHandler:
    """Handles graph state transformation for Claude interaction."""
    
    def __init__(self):
        self._state_template = PromptTemplate(
            input_variables=["state_summary", "concept_network", 
                           "coherence_metrics", "temporal_context"],
            template='''
Current Graph State Context:
{state_summary}

Concept Network:
{concept_network}

Coherence Metrics:
{coherence_metrics}

Temporal Context:
{temporal_context}
'''
        )
    
    def prepare_state_for_claude(
        self, 
        state: GraphStateContext
    ) -> Document:
        """Prepare graph state for Claude consumption."""
        context = state.to_prompt_context()
        formatted_prompt = self._state_template.format(**context)
        
        # Create a Document with metadata for LangChain
        return Document(
            page_content=formatted_prompt,
            metadata={
                "state_id": state.state_id,
                "timestamp": state.timestamp.isoformat(),
                "coherence_score": state.coherence_metrics.get('overall', 0.0),
                "concept_count": len(state.concepts),
                "relation_count": len(state.relations)
            }
        )
    
    def validate_claude_requirements(
        self, 
        state: GraphStateContext
    ) -> Tuple[bool, Optional[str]]:
        """Validate if state meets Claude's requirements."""
        if not state.concepts:
            return False, "No concepts present in state"
        
        if not state.relations:
            return False, "No relations present in state"
        
        if state.coherence_metrics.get('overall', 0.0) < 0.3:
            return False, "Coherence score too low for reliable processing"
        
        return True, None
