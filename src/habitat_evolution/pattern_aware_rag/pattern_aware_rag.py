"""Pattern-aware RAG controller with integrated service management."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory

from habitat_evolution.core.pattern import (
    FieldDrivenPatternManager,
    PatternQualityAnalyzer,
    SignalMetrics,
    FlowMetrics,
    PatternState
)
from habitat_evolution.core.services.field.interfaces import (
    FieldStateService,
    GradientService,
    FlowDynamicsService
)
from habitat_evolution.adaptive_core.persistence.interfaces import (
    MetricsRepository,
    EventRepository
)
from habitat_evolution.core.services.event_bus import LocalEventBus
from habitat_evolution.adaptive_core.services.interfaces import (
    PatternEvolutionService,
    MetricsService,
    QualityMetricsService,
    EventManagementService
)
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from habitat_evolution.adaptive_core.models import Pattern, Relationship

from .interfaces.pattern_emergence import PatternEmergenceInterface as EmergenceFlow
from .learning.learning_control import WindowState as StateSpaceCondition
from .interfaces.pattern_emergence import PatternMetrics as EvolutionMetrics
from .monitoring.vector_attention_monitor import VectorAttentionMonitor
from .core.coherence_interface import CoherenceInterface as FlowDynamics, StateAlignment as FlowState
from .state.test_states import GraphStateSnapshot as PatternGraphService
from .superceeded.coherence_embeddings import EmbeddingContext, CoherenceEmbeddings

logger = logging.getLogger(__name__)

@dataclass
class RAGPatternContext:
    """Context for pattern-aware RAG operations."""
    query_patterns: List[str]
    retrieval_patterns: List[str]
    augmentation_patterns: List[str]
    coherence_level: float
    temporal_context: Optional[Dict[str, Any]] = None
    state_space: Optional[StateSpaceCondition] = None
    evolution_metrics: Optional[EvolutionMetrics] = None
    density_centers: Optional[List[Dict[str, Any]]] = None
    cross_domain_paths: Optional[List[Dict[str, Any]]] = None
    global_density: Optional[float] = None

class LearningWindowState(Enum):
    """Learning window states with OPENING for potential emergence."""
    CLOSED = "CLOSED"
    OPENING = "OPENING"
    OPEN = "OPEN"

@dataclass
class WindowMetrics:
    """Metrics for learning window state determination."""
    local_density: float
    global_density: float
    coherence: float
    cross_paths: List[str]
    back_pressure: float
    flow_stability: float

@dataclass
class PatternMetrics:
    """Core pattern metrics."""
    coherence: float
    emergence_rate: float
    cross_pattern_flow: float
    energy_state: float
    adaptation_rate: float
    stability: float

class PatternAwareRAG:
    """RAG controller with integrated service management and pattern awareness."""
    
    def __init__(
        self,
        pattern_evolution_service: PatternEvolutionService,
        field_state_service: FieldStateService,
        gradient_service: GradientService,
        flow_dynamics_service: FlowDynamicsService,
        metrics_service: MetricsService,
        quality_metrics_service: QualityMetricsService,
        event_service: EventManagementService,
        coherence_analyzer: Any,
        emergence_flow: EmergenceFlow,
        settings: Any,
        graph_service: Optional[PatternGraphService] = None
    ):
        # Core services
        self.pattern_evolution = pattern_evolution_service
        self.field_state = field_state_service
        self.gradient = gradient_service
        self.flow_dynamics = flow_dynamics_service
        self.metrics = metrics_service
        self.quality = quality_metrics_service
        self.events = event_service
        
        # RAG components
        self.coherence_analyzer = coherence_analyzer
        self.emergence_flow = emergence_flow
        self.settings = settings
        
        # Initialize coherence-aware embeddings
        self.embeddings = CoherenceEmbeddings()
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=settings.VECTOR_STORE_DIR
        )
        
        # Pattern management
        self.pattern_manager = FieldDrivenPatternManager(
            pattern_store=self.pattern_evolution.pattern_store,
            relationship_store=self.pattern_evolution.relationship_store,
            event_bus=self.events,
            quality_analyzer=PatternQualityAnalyzer()
        )
        
        # State tracking
        self.current_window_state = LearningWindowState.CLOSED
        self.window_metrics = WindowMetrics(
            local_density=0.0,
            global_density=0.0,
            coherence=0.0,
            cross_paths=[],
            back_pressure=0.0,
            flow_stability=0.0
        )
        
        # Graph integration
        self.graph = graph_service or PatternGraphService()
        
        # Pattern-specific prompts
        self._initialize_pattern_prompts()
        
        # Subscribe to events
        self._setup_event_subscriptions()
        
    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for pattern and state changes."""
        self.events.subscribe(
            "pattern.evolution",
            self._handle_pattern_evolution
        )
        self.events.subscribe(
            "pattern.quality",
            self._handle_quality_update
        )
        self.events.subscribe(
            "field.state.updated",
            self._handle_field_state_update
        )

    async def process_with_patterns(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], RAGPatternContext]:
        """Process query with pattern awareness and service integration."""
        # Get field state and window metrics
        field_state = await self._get_current_field_state(context)
        window_metrics = await self._calculate_window_metrics(field_state)
        window_state = await self._determine_window_state(window_metrics)
        
        # Let emergence flow guide pattern extraction
        state_space = self.emergence_flow.context.state_space
        query_patterns = await self._extract_query_patterns(query)
        
        # Create embedding context with window awareness
        embedding_context = EmbeddingContext(
            flow_state=self.emergence_flow.get_flow_state(),
            evolution_metrics=EvolutionMetrics(),
            pattern_context={
                "patterns": query_patterns,
                "window_state": window_state.value,
                "window_metrics": window_metrics.__dict__
            }
        )
        
        # Retrieve with coherence-aware embeddings
        docs, retrieval_patterns = await self._retrieve_with_patterns(
            query,
            query_patterns,
            embedding_context
        )
        
        # Update evolution metrics based on retrieval
        evolution_metrics = self._calculate_evolution_metrics(
            docs,
            query_patterns,
            retrieval_patterns
        )
        
        # Update embedding context with new metrics
        embedding_context.evolution_metrics = evolution_metrics
        
        # Augment with updated context
        result, augmentation_patterns = await self._augment_with_patterns(
            query,
            docs,
            query_patterns,
            retrieval_patterns,
            embedding_context
        )
        
        # Update pattern evolution
        pattern_id = await self._update_pattern_evolution(
            query=query,
            rag_output=result,
            window_metrics=window_metrics
        )
        
        # Update result with pattern ID
        result["pattern_id"] = pattern_id
        
        # Create pattern context
        pattern_context = RAGPatternContext(
            query_patterns=query_patterns,
            retrieval_patterns=retrieval_patterns,
            augmentation_patterns=augmentation_patterns,
            coherence_level=state_space.coherence,
            temporal_context=context.get("temporal") if context else None,
            state_space=state_space,
            evolution_metrics=evolution_metrics
        )
        
        # Let coherence emerge naturally
        coherence_insight = await self.coherence_analyzer.analyze_coherence(
            pattern_context,
            result["content"]
        )
        
        # Update result with emergence insights
        result.update({
            "coherence": {
                "flow_state": coherence_insight.flow_state,
                "patterns": coherence_insight.patterns,
                "confidence": coherence_insight.confidence,
                "emergence_potential": coherence_insight.emergence_potential,
                "state_space": state_space,
                "evolution_metrics": evolution_metrics
            }
        })
        
        # Let emergence flow track patterns
        await self.emergence_flow.observe_emergence(
            {"rag_patterns": pattern_context},
            {"rag_state": coherence_insight.flow_state}
        )
        
        return result, pattern_context
        
    def _initialize_pattern_prompts(self) -> None:
        """Initialize pattern-specific prompt templates."""
        self.pattern_prompts = {
            "extraction": PromptTemplate(
                input_variables=["query"],
                template="""
                Analyze the query for potential patterns:
                1. Measurement patterns (quantities, metrics)
                2. Impact patterns (effects, consequences)
                3. Evolution patterns (changes over time)
                4. Relationship patterns (connections, dependencies)
                
                Query: {query}
                
                Patterns:"""
            ),
            "retrieval": PromptTemplate(
                input_variables=["query", "patterns"],
                template="""
                Enhance retrieval using identified patterns:
                - Measurement context: {patterns.measurements}
                - Impact context: {patterns.impacts}
                - Evolution context: {patterns.evolution}
                - Relationship context: {patterns.relationships}
                
                Query: {query}
                
                Enhanced query:"""
            )
        }
        
    async def _get_current_field_state(self, context: Optional[Dict[str, Any]]) -> Any:
        """Get current field state with context."""
        if not context or "field_id" not in context:
            # Create new field state
            field_id = AdaptiveID.generate()
            return await self.field_state.create_field_state(field_id)
        
        return await self.field_state.get_field_state(context["field_id"])

    async def _calculate_window_metrics(self, field_state: Any) -> WindowMetrics:
        """Calculate comprehensive window metrics."""
        # Get local density
        local_density = await self.field_state.calculate_local_density(
            field_state.id,
            field_state.position
        )
        
        # Get global density
        global_density = await self.metrics.calculate_global_density()
        
        # Calculate coherence
        coherence = await self.quality.calculate_coherence(field_state.id)
        
        # Get cross paths
        cross_paths = await self.pattern_evolution.get_cross_pattern_paths(
            field_state.id
        )
        
        # Calculate back pressure
        back_pressure = await self.flow_dynamics.calculate_back_pressure(
            field_state.id
        )
        
        # Get flow stability
        flow_stability = await self.flow_dynamics.calculate_flow_stability(
            field_state.id
        )
        
        return WindowMetrics(
            local_density=local_density,
            global_density=global_density,
            coherence=coherence,
            cross_paths=cross_paths,
            back_pressure=back_pressure,
            flow_stability=flow_stability
        )

    async def _determine_window_state(self, metrics: WindowMetrics) -> LearningWindowState:
        """Determine learning window state based on metrics."""
        # Check if window should be CLOSED
        if metrics.local_density < self.config["thresholds"]["density"]:
            return LearningWindowState.CLOSED
            
        # Check for OPENING state
        if metrics.coherence < self.config["thresholds"]["coherence"] and \
           metrics.back_pressure < self.config["thresholds"]["back_pressure"] and \
           len(metrics.cross_paths) >= self.config["thresholds"]["cross_paths"]:
            return LearningWindowState.OPENING
            
        # Default to OPEN if conditions are met
        if metrics.coherence >= self.config["thresholds"]["coherence"]:
            return LearningWindowState.OPEN
            
        return LearningWindowState.CLOSED

    async def _update_pattern_evolution(self, query: str, rag_output: Dict[str, Any], window_metrics: WindowMetrics) -> str:
        """Update pattern evolution system and sync with graph."""
        # Create pattern from RAG output
        pattern = Pattern(
            type="query_pattern",
            content={
                "query": query,
                "output": rag_output
            },
            metrics=PatternMetrics(
                coherence=window_metrics.coherence,
                emergence_rate=0.0,
                cross_pattern_flow=len(window_metrics.cross_paths),
                energy_state=window_metrics.local_density,
                adaptation_rate=0.0,
                stability=window_metrics.flow_stability
            )
        )
        
        # Register pattern
        pattern_id = await self.pattern_evolution.register_pattern(
            pattern_type="query_pattern",
            content=pattern.content,
            context={
                "window_state": self.current_window_state.value,
                "metrics": window_metrics.__dict__
            }
        )
        
        # Sync with graph
        await self._sync_with_graph(pattern_id, pattern)
        
        return pattern_id
        
    async def _sync_with_graph(self, pattern_id: str, pattern: Pattern) -> None:
        """Bidirectional sync with graph database."""
        # Store pattern in graph
        await self.graph.store_pattern(pattern)
        
        # Track relationships
        if pattern.metrics.cross_pattern_flow > 0:
            related_patterns = await self.pattern_evolution.get_related_patterns(pattern_id)
            await self.graph.track_relationships(
                pattern_id,
                [p.id for p in related_patterns]
            )
            
        # Map density centers if coherence is high
        if pattern.metrics.coherence >= self.config["thresholds"]["coherence"]:
            centers = await self.graph.map_density_centers(
                coherence_threshold=self.config["thresholds"]["coherence"]
            )
            
            # Update pattern evolution with density insights
            if centers:
                await self.pattern_evolution.update_pattern_context(
                    pattern_id,
                    {"density_centers": centers}
                )

    async def _handle_pattern_evolution(self, event: Any) -> None:
        """Handle pattern evolution events."""
        pattern_id = event.get("pattern_id")
        if not pattern_id:
            return
            
        # Update pattern metrics
        metrics = await self.pattern_evolution.get_pattern_metrics(pattern_id)
        
        # Update window metrics if needed
        if metrics.coherence > self.window_metrics.coherence:
            self.window_metrics.coherence = metrics.coherence
            
        # Emit metric update event
        await self.events.emit("pattern.metrics.updated", {
            "pattern_id": pattern_id,
            "metrics": metrics.__dict__,
            "timestamp": datetime.now().isoformat()
        })

    async def _handle_quality_update(self, event: Any) -> None:
        """Handle quality metric updates."""
        quality_metrics = event.get("metrics")
        if not quality_metrics:
            return
            
        # Update window metrics
        self.window_metrics.coherence = quality_metrics.get(
            "coherence",
            self.window_metrics.coherence
        )
        
        # Check for state transition
        new_state = await self._determine_window_state(self.window_metrics)
        if new_state != self.current_window_state:
            self.current_window_state = new_state
            await self.events.emit("window.state.changed", {
                "old_state": self.current_window_state.value,
                "new_state": new_state.value,
                "metrics": self.window_metrics.__dict__,
                "timestamp": datetime.now().isoformat()
            })

    async def _handle_field_state_update(self, event: Any) -> None:
        """Handle field state updates."""
        field_id = event.get("field_id")
        if not field_id:
            return
            
        # Update window metrics
        field_state = await self.field_state.get_field_state(field_id)
        if field_state:
            self.window_metrics = await self._calculate_window_metrics(field_state)
            
        # Check for state transition
        new_state = await self._determine_window_state(self.window_metrics)
        if new_state != self.current_window_state:
            self.current_window_state = new_state
            await self.events.emit("window.state.changed", {
                "field_id": field_id,
                "old_state": self.current_window_state.value,
                "new_state": new_state.value,
                "metrics": self.window_metrics.__dict__,
                "timestamp": datetime.now().isoformat()
            })

    async def process_document(self, doc: str, context: RAGPatternContext) -> Dict[str, Any]:
        """Process document with density-aware pattern evolution."""
        # Extract initial patterns
        patterns = await self.coherence_analyzer.extract_patterns(doc)
        
        # Get density analysis from window metrics
        density = {
            "density_centers": context.density_centers or [],
            "cross_domain_paths": context.cross_domain_paths or [],
            "global_density": context.global_density or 0
        }
        
        # Update context with density insights
        context.density_centers = density["density_centers"]
        context.cross_domain_paths = density["cross_domain_paths"]
        context.global_density = density["global_density"]
        
        # Process through pattern evolution
        pattern_id = await self._update_pattern_evolution(
            doc,
            {"content": doc},
            self.window_metrics
        )
        
        # Get evolution state
        evolution_state = await self.pattern_evolution.get_pattern_state(pattern_id)
        
        # Register evolution in learning window
        window_data = {
            "score": evolution_state.metrics.evolution_score,
            "potential": evolution_state.dynamics.emergence_readiness,
            "horizon": evolution_state.dynamics.direction,
            "channels": {
                "structural": {
                    "strength": evolution_state.metrics.structure_alignment,
                    "sustainability": evolution_state.metrics.stability
                },
                "semantic": {
                    "strength": evolution_state.metrics.meaning_alignment,
                    "sustainability": evolution_state.metrics.coherence
                }
            },
            "semantic_patterns": [
                {"domain": d, "strength": s} 
                for d, s in patterns["domain_strengths"].items()
            ]
        }
        self.evolution_manager.learning_window_interface.register_window(window_data)
        
        return {
            "density_score": context.global_density,
            "cross_domain_strength": max(
                (p["strength"] for p in context.cross_domain_paths),
                default=0
            ),
            "enhanced_patterns": [
                {
                    "domain": center["domain"],
                    "density": center["density"],
                    "alignments": center.get("alignments", [])
                }
                for center in context.density_centers
            ]
        }

    async def enhance_patterns(self, doc: str, context: RAGPatternContext) -> Dict[str, Any]:
        """Enhance patterns with density awareness."""
        # Get current density state
        density = self.evolution_manager.learning_window_interface.get_density_analysis()
        
        # Extract domains from density centers
        enhanced_domains = {
            center["domain"]
            for center in density.get("density_centers", [])
            if center["density"] > self.settings.DENSITY_THRESHOLD
        }
        
        # Enhance patterns in high-density regions
        enhanced_patterns = []
        for domain in enhanced_domains:
            # Get domain-specific patterns
            domain_patterns = await self.coherence_analyzer.extract_domain_patterns(
                doc, domain
            )
            
            # Find related patterns through density paths
            related_patterns = []
            for path in density.get("cross_domain_paths", []):
                if path["source"] == domain and path["strength"] > self.settings.PATH_THRESHOLD:
                    target_patterns = await self.coherence_analyzer.extract_domain_patterns(
                        doc, path["target"]
                    )
                    related_patterns.extend(target_patterns)
            
            # Combine and enhance patterns
            combined_patterns = domain_patterns + related_patterns
            enhanced = await self.pattern_enhancer.enhance_patterns(
                combined_patterns,
                context.coherence_level
            )
            
            enhanced_patterns.append({
                "domain": domain,
                "patterns": enhanced,
                "density_score": next(
                    (c["density"] for c in density.get("density_centers", [])
                     if c["domain"] == domain),
                    0
                ),
                "stability": next(
                    (p["stability"] for p in density.get("cross_domain_paths", [])
                     if p["source"] == domain),
                    0
                )
            })
        
        # Update learning windows with enhancement results
        window_data = {
            "score": context.coherence_level,
            "enhanced_domains": list(enhanced_domains),
            "enhancement_patterns": enhanced_patterns
        }
        self.evolution_manager.learning_window_interface.register_window(window_data)
        
        return {
            "enhancement_score": context.coherence_level,
            "enhanced_domains": list(enhanced_domains),
            "coherence_level": context.coherence_level,
            "patterns": enhanced_patterns
        }

    def get_evolution_state(self) -> Dict[str, Any]:
        """Get current evolution state with density metrics."""
        density = self.evolution_manager.learning_window_interface.get_density_analysis()
        return {
            "density_score": density.get("global_density", 0),
            "cross_domain_strength": max(
                (p["strength"] for p in density.get("cross_domain_paths", [])),
                default=0
            ),
            "enhanced_patterns": [
                {
                    "domain": center["domain"],
                    "density": center["density"],
                    "alignments": center.get("alignments", [])
                }
                for center in density.get("density_centers", [])
            ]
        }

    def get_enhancement_state(self) -> Dict[str, Any]:
        """Get current enhancement state with density metrics."""
        density = self.evolution_manager.learning_window_interface.get_density_analysis()
        windows = self.evolution_manager.learning_window_interface.get_learning_opportunities()
        
        latest_window = windows[-1] if windows else {}
        return {
            "enhancement_score": latest_window.get("score", 0),
            "enhanced_domains": latest_window.get("enhanced_domains", []),
            "coherence_level": latest_window.get("score", 0),
            "patterns": latest_window.get("enhancement_patterns", [])
        }

    async def _extract_query_patterns(
        self,
        query: str
    ) -> List[str]:
        """Extract patterns from query."""
        chain = LLMChain(
            llm=self.rag.llm,
            prompt=self.pattern_prompts["extraction"]
        )
        result = await chain.arun(query=query)
        return self._parse_patterns(result)
        
    async def _retrieve_with_patterns(
        self,
        query: str,
        patterns: List[str],
        embedding_context: EmbeddingContext
    ) -> Tuple[List[Any], List[str]]:
        """Retrieve documents with pattern awareness."""
        # Enhance query with patterns
        chain = LLMChain(
            llm=self.rag.llm,
            prompt=self.pattern_prompts["retrieval"]
        )
        enhanced_query = await chain.arun(
            query=query,
            patterns=self._format_patterns(patterns)
        )
        
        # Retrieve with enhanced query and embedding context
        docs = await self.vector_store.asimilarity_search(
            enhanced_query,
            k=self.settings.RETRIEVAL_K,
            embedding_context=embedding_context
        )
        
        # Extract patterns from retrieved docs
        retrieval_patterns = await self._extract_doc_patterns(docs)
        
        return docs, retrieval_patterns
        
    async def _augment_with_patterns(
        self,
        query: str,
        docs: List[Any],
        query_patterns: List[str],
        retrieval_patterns: List[str],
        embedding_context: EmbeddingContext
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Augment results with pattern awareness."""
        # Combine patterns for context
        all_patterns = list(set(query_patterns + retrieval_patterns))
        
        # Augment with pattern context
        result = await self.rag.augment_response(
            query,
            docs,
            additional_context={
                "patterns": all_patterns,
                "pattern_types": self._categorize_patterns(all_patterns)
            },
            embedding_context=embedding_context
        )
        
        # Extract patterns from augmented result
        augmentation_patterns = await self._extract_augmentation_patterns(
            result
        )
        
        return result, augmentation_patterns
        
    def _assess_coherence_level(
        self,
        query_patterns: List[str],
        retrieval_patterns: List[str],
        augmentation_patterns: List[str]
    ) -> float:
        """Assess coherence level of pattern integration."""
        # Calculate pattern overlap
        pattern_sets = [
            set(query_patterns),
            set(retrieval_patterns),
            set(augmentation_patterns)
        ]
        overlap = len(set.intersection(*pattern_sets))
        total = len(set.union(*pattern_sets))
        
        coherence_score = overlap / total if total > 0 else 0.0
        
        return coherence_score
            
    def _track_patterns(
        self,
        pattern_context: RAGPatternContext
    ) -> None:
        """Track patterns through evolution manager."""
        for pattern_type, patterns in self._categorize_patterns(
            pattern_context.query_patterns +
            pattern_context.retrieval_patterns +
            pattern_context.augmentation_patterns
        ).items():
            for pattern in patterns:
                print(pattern)
                
    def _parse_patterns(self, raw_patterns: str) -> List[str]:
        """Parse patterns from raw text."""
        # Implementation depends on pattern format
        return []  # Placeholder
        
    def _format_patterns(self, patterns: List[str]) -> Dict[str, List[str]]:
        """Format patterns for prompt template."""
        # Implementation depends on pattern categorization
        return {}  # Placeholder
        
    async def _extract_doc_patterns(self, docs: List[Any]) -> List[str]:
        """Extract patterns from retrieved documents."""
        # Implementation depends on document structure
        return []  # Placeholder
        
    async def _extract_augmentation_patterns(
        self,
        result: Dict[str, Any]
    ) -> List[str]:
        """Extract patterns from augmented result."""
        # Implementation depends on result structure
        return []  # Placeholder
        
    def _categorize_patterns(
        self,
        patterns: List[str]
    ) -> Dict[str, List[str]]:
        """Categorize patterns by type."""
        # Implementation depends on pattern types
        return {}  # Placeholder

    def _calculate_evolution_metrics(
        self,
        docs: List[Any],
        query_patterns: List[str],
        retrieval_patterns: List[str]
    ) -> EvolutionMetrics:
        """Calculate evolution metrics based on retrieval patterns and document coherence."""
        metrics = EvolutionMetrics()
        
        # Track pattern similarity scores
        pattern_scores = []
        for doc in docs:
            doc_patterns = self._extract_doc_patterns(doc)
            similarity = self._calculate_pattern_similarity(
                query_patterns + retrieval_patterns,
                doc_patterns
            )
            pattern_scores.append(similarity)
        
        # Calculate concrete evolution metrics
        metrics.pattern_coherence = sum(pattern_scores) / len(pattern_scores) if pattern_scores else 0.0
        metrics.pattern_diversity = len(set(retrieval_patterns)) / len(retrieval_patterns) if retrieval_patterns else 0.0
        
        # Track temporal evolution
        current_timestamp = datetime.utcnow().timestamp()
        if hasattr(self, '_last_evolution_time'):
            time_delta = current_timestamp - self._last_evolution_time
            metrics.evolution_rate = (
                metrics.pattern_coherence - self._last_coherence_score
            ) / time_delta if time_delta > 0 else 0.0
        
        # Update historical tracking
        self._last_evolution_time = current_timestamp
        self._last_coherence_score = metrics.pattern_coherence
        
        # Calculate pattern flow dynamics
        pattern_flow = self._calculate_pattern_flow(
            query_patterns,
            retrieval_patterns,
            pattern_scores
        )
        metrics.flow_velocity = pattern_flow.velocity
        metrics.flow_direction = pattern_flow.direction
        
        return metrics
        
    def _calculate_pattern_similarity(
        self,
        patterns1: List[str],
        patterns2: List[str]
    ) -> float:
        """Calculate concrete similarity between pattern sets."""
        if not patterns1 or not patterns2:
            return 0.0
            
        # Use embeddings to get pattern vectors
        vectors1 = [self.embeddings.embed_text(p) for p in patterns1]
        vectors2 = [self.embeddings.embed_text(p) for p in patterns2]
        
        # Calculate average cosine similarity
        similarities = []
        for v1 in vectors1:
            for v2 in vectors2:
                similarity = self._cosine_similarity(v1, v2)
                similarities.append(similarity)
                
        return sum(similarities) / len(similarities)
        
    def _calculate_pattern_flow(
        self,
        query_patterns: List[str],
        retrieval_patterns: List[str],
        pattern_scores: List[float]
    ) -> FlowDynamics:
        """Calculate concrete pattern flow dynamics."""
        # Initialize flow dynamics
        flow = FlowDynamics()
        
        # Calculate velocity from pattern score changes
        if hasattr(self, '_last_pattern_scores'):
            score_deltas = [
                curr - prev 
                for curr, prev in zip(pattern_scores, self._last_pattern_scores)
            ]
            flow.velocity = sum(score_deltas) / len(score_deltas) if score_deltas else 0.0
            
        # Calculate direction from pattern alignment
        common_patterns = set(query_patterns) & set(retrieval_patterns)
        flow.direction = len(common_patterns) / len(set(query_patterns + retrieval_patterns))
        
        # Update historical tracking
        self._last_pattern_scores = pattern_scores
        
        return flow
