"""
LangChain Configuration for Pattern-Aware RAG.

Configures LangChain components for Claude integration, ensuring proper
state handling and prompt optimization.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from langchain.prompts import PromptTemplate

from .test_states import GraphStateSnapshot
from langchain.schema import Document
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore

@dataclass
class StatePromptConfig:
    """Configuration for state-aware prompting."""
    
    # Minimum coherence requirements
    min_coherence_score: float = 0.3
    min_relationship_strength: float = 0.4
    min_concept_confidence: float = 0.6
    
    # Context window settings
    max_concepts_per_prompt: int = 15
    max_relations_per_prompt: int = 25
    
    # Temporal settings
    max_history_window: int = 5  # Number of previous states to include
    
    def validate_state_metrics(self, metrics: Dict[str, float]) -> bool:
        """Validate state metrics meet minimum requirements."""
        return (
            metrics.get('coherence', 0.0) >= self.min_coherence_score and
            metrics.get('relationship_strength', 0.0) >= self.min_relationship_strength and
            metrics.get('concept_confidence', 0.0) >= self.min_concept_confidence
        )

class ClaudeLangChainIntegration:
    """Manages LangChain integration for Claude interaction."""
    
    def __init__(
        self,
        config: StatePromptConfig,
        persist_dir: str = "./.habitat/embeddings"
    ):
        self.config = config
        self._store = LocalFileStore(persist_dir)
        self._setup_embeddings()
        self._setup_prompts()
    
    def _setup_embeddings(self):
        """Setup embedding configuration."""
        underlying_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=self._get_claude_embeddings(),
            document_embedding_cache=self._store,
            namespace="claude_state_embeddings"
        )
        
        self.vector_store = Chroma(
            collection_name="state_embeddings",
            embedding_function=underlying_embeddings
        )
    
    def _setup_prompts(self):
        """Setup prompt templates optimized for Claude."""
        self.state_template = PromptTemplate(
            input_variables=[
                "graph_state",
                "concept_relations",
                "temporal_context",
                "coherence_metrics"
            ],
            template="""
System: You are working with a graph state that represents conceptual relationships. 
The state has the following characteristics:

Graph State Overview:
{graph_state}

Concept Relationships (ordered by strength):
{concept_relations}

Temporal Context:
{temporal_context}

Coherence Metrics:
{coherence_metrics}

Human: Based on this graph state context, please process the following query while maintaining coherence and respecting relationship strengths.

Query: {query}

Assistant: I'll analyze the query within the given graph state context, ensuring that:
1. All responses align with existing concept relationships
2. Coherence metrics are maintained or improved
3. Temporal context is respected
4. Pattern evolution follows established pathways

Analysis:"""
        )

    def _get_claude_embeddings(self):
        """Get Claude-specific embeddings configuration."""
        from langchain.embeddings import CohereEmbeddings
        return CohereEmbeddings(
            model="embed-english-v3.0",  # Compatible with Claude's understanding
            input_type="search_document"
        )

    def prepare_state_context(
        self,
        graph_state: GraphStateSnapshot,
        max_concepts: Optional[int] = None,
        max_relations: Optional[int] = None
    ) -> Dict[str, str]:
        """Prepare graph state context for prompt template."""
        from .state_handler import GraphStateHandler
        
        handler = GraphStateHandler()
        context = handler.prepare_prompt_context(
            state=graph_state,
            max_concepts=max_concepts or self.config.max_concepts_per_prompt,
            max_relations=max_relations or self.config.max_relations_per_prompt
        )
        
        # Validate state coherence
        metrics = handler.validate_state_coherence(graph_state)
        if not self.config.validate_state_metrics({
            'coherence': metrics.overall_coherence,
            'relationship_strength': metrics.relationship_strength,
            'concept_confidence': metrics.concept_confidence
        }):
            raise ValueError(
                "Graph state does not meet minimum coherence requirements"
            )
        
        return context
    
    def store_state_embedding(
        self,
        state: GraphStateSnapshot,
        previous_state: Optional[GraphStateSnapshot] = None,
        metadata: Optional[Dict[str, any]] = None
    ) -> None:
        """Store state embedding in vector store with evolution tracking."""
        from .state_evolution import StateEvolutionTracker
        
        # Initialize evolution tracker if needed
        if not hasattr(self, '_evolution_tracker'):
            self._evolution_tracker = StateEvolutionTracker()
        
        # Create transaction if we have a previous state
        if previous_state:
            transaction = self._evolution_tracker.create_transaction(
                from_state=previous_state,
                to_state=state,
                changes=metadata or {},
                emergence_type="NATURAL"  # Default to natural emergence
            )
            # Add transaction to metadata
            metadata = metadata or {}
            metadata.update({
                'transaction_id': transaction.transaction_id,
                'coherence_delta': {
                    'concept_confidence': transaction.coherence_delta.concept_confidence,
                    'relationship_strength': transaction.coherence_delta.relationship_strength,
                    'pattern_stability': transaction.coherence_delta.pattern_stability,
                    'overall_coherence': transaction.coherence_delta.overall_coherence
                }
            })
        
        # Prepare and store embedding
        context = self.prepare_state_context(state)
        doc = Document(
            page_content=self.state_template.format(
                **context,
                query="STATE_EMBEDDING"
            ),
            metadata=metadata or {}
        )
        self.vector_store.add_documents([doc])
    
    def find_similar_states(
        self,
        query_state: GraphStateSnapshot,
        k: int = 5,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Document]:
        """Find similar states from vector store with temporal filtering."""
        # Get evolution history if needed
        if hasattr(self, '_evolution_tracker') and (start_time or end_time):
            history = self._evolution_tracker.get_state_history(start_time, end_time)
            if history:
                # Use historical context for search
                context = self.prepare_state_context(query_state)
                context['temporal_context'] = f"{context['temporal_context']}\n\nEvolution History:\n" + \
                    '\n'.join(f"- {tx.timestamp}: {tx.emergence_type}" for tx in history[-5:])
                
                query_doc = self.state_template.format(
                    **context,
                    query="SIMILARITY_SEARCH"
                )
                return self.vector_store.similarity_search(query_doc, k=k)
        
        # Default search without history
        context = self.prepare_state_context(query_state)
        query_doc = self.state_template.format(
            **context,
            query="SIMILARITY_SEARCH"
        )
        return self.vector_store.similarity_search(query_doc, k=k)
