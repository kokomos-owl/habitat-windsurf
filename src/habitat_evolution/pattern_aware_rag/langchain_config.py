"""
LangChain Configuration for Pattern-Aware RAG.

Configures LangChain components for Claude integration, ensuring proper
state handling and prompt optimization.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from langchain.prompts import PromptTemplate
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
        graph_state: Dict[str, any],
        max_concepts: Optional[int] = None,
        max_relations: Optional[int] = None
    ) -> Dict[str, str]:
        """Prepare graph state context for prompt template."""
        max_concepts = max_concepts or self.config.max_concepts_per_prompt
        max_relations = max_relations or self.config.max_relations_per_prompt
        
        # Format graph state overview
        state_overview = (
            f"ID: {graph_state['id']}\n"
            f"Timestamp: {graph_state['timestamp']}\n"
            f"Total Concepts: {len(graph_state['concepts'])}\n"
            f"Total Relations: {len(graph_state['relations'])}\n"
            f"Overall Coherence: {graph_state['metrics']['coherence']:.2f}"
        )
        
        # Format concept relations (sorted by strength)
        relations = sorted(
            graph_state['relations'],
            key=lambda x: x['strength'],
            reverse=True
        )[:max_relations]
        
        relation_str = "\n".join(
            f"- {r['source']} --[{r['type']}: {r['strength']:.2f}]--> {r['target']}"
            for r in relations
        )
        
        # Format temporal context
        temporal = (
            f"Evolution Stage: {graph_state['temporal']['stage']}\n"
            f"Stability Index: {graph_state['temporal']['stability']:.2f}\n"
            f"Recent Changes: {', '.join(graph_state['temporal']['recent_changes'])}"
        )
        
        return {
            "graph_state": state_overview,
            "concept_relations": relation_str,
            "temporal_context": temporal,
            "coherence_metrics": str(graph_state['metrics'])
        }
    
    def store_state_embedding(
        self,
        state: Dict[str, any],
        metadata: Optional[Dict[str, any]] = None
    ) -> None:
        """Store state embedding in vector store."""
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
        query_state: Dict[str, any],
        k: int = 5
    ) -> List[Document]:
        """Find similar states from vector store."""
        context = self.prepare_state_context(query_state)
        query_doc = self.state_template.format(
            **context,
            query="SIMILARITY_SEARCH"
        )
        return self.vector_store.similarity_search(query_doc, k=k)
