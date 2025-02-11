"""Test suite for pattern-aware RAG with concrete evolution metrics."""

import unittest
import pytest
import pytest_asyncio
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Mock classes to avoid circular imports
@dataclass
class MockTemporalContext:
    """Mock temporal context."""
    timestamp: str = "2025-01-01"
    duration: float = 1.0
    sequence: int = 1

@dataclass
class MockUncertaintyMetrics:
    """Mock uncertainty metrics."""
    confidence: float = 0.8
    stability: float = 0.7
    diversity: float = 0.6

@dataclass
class MockPatternEvidence:
    """Mock pattern evidence."""
    evidence_id: str = "test_evidence"
    timestamp: str = "2025-01-01"
    pattern_type: str = "test"
    source_data: Dict[str, Any] = None
    temporal_context: Optional[MockTemporalContext] = None
    uncertainty_metrics: Optional[MockUncertaintyMetrics] = None
    version: str = "1.0"
    
    def __post_init__(self):
        if self.source_data is None:
            self.source_data = {}
        if self.temporal_context is None:
            self.temporal_context = MockTemporalContext()
        if self.uncertainty_metrics is None:
            self.uncertainty_metrics = MockUncertaintyMetrics()

class MockPatternCore:
    """Mock pattern core."""
    def __init__(self):
        self.evidence_chains = {}
        self.temporal_maps = {}
        self.pattern_metrics = {}
        
    async def observe_evolution(self, structural_change, semantic_change, evidence=None):
        """Mock observe evolution."""
        return {
            "pattern_id": "test_pattern",
            "confidence": 0.8,
            "temporal_context": MockTemporalContext(),
            "uncertainty_metrics": MockUncertaintyMetrics()
        }
        
    def get_latest_evidence(self, limit=1):
        """Mock get latest evidence."""
        return [{
            "id": "test_pattern",
            "evidence": {
                "pattern_type": "test",
                "confidence": 0.8
            }
        }]

from habitat_test.core.learning_windows import LearningWindowInterface
from habitat_test.core.pattern_evolution import (
    PatternEvolution, 
    PatternEvolutionManager,
    EvolutionState
)
from habitat_test.core.pattern_aware_rag import RAGPatternContext

@pytest.fixture
def mock_pattern_core():
    """Create mock pattern core."""
    mock = MockPatternCore()
    return mock

@pytest.fixture
def mock_coherence_flow():
    """Create mock coherence flow."""
    mock = Mock()
    mock.process_flow = AsyncMock(return_value={
        "state": {
            "coherence": 0.85,
            "dynamics": {
                "direction": 0.7,
                "emergence_readiness": 0.8
            }
        }
    })
    return mock

@pytest.fixture
def mock_learning_window():
    """Create mock learning window interface."""
    mock = Mock()
    mock.register_window = Mock()
    mock.get_density_analysis = Mock(return_value={
        "global_density": 0.75,
        "density_centers": [
            {"density": 0.85, "domain": "climate", "alignments": [
                {"domain": "risk", "strength": 0.82}
            ]},
            {"density": 0.78, "domain": "risk", "alignments": [
                {"domain": "climate", "strength": 0.82}
            ]}
        ],
        "cross_domain_paths": [
            {"source": "climate", "target": "risk", "strength": 0.82, "stability": 0.79}
        ]
    })
    return mock

@pytest_asyncio.fixture
async def pattern_aware_rag(mock_pattern_core, mock_coherence_flow, mock_learning_window):
    """Create pattern-aware RAG with mocked dependencies."""
    rag = Mock()
    rag.evolution_manager = PatternEvolutionManager(
        pattern_core=mock_pattern_core,
        coherence_flow=mock_coherence_flow
    )
    rag.evolution_manager.learning_window_interface = mock_learning_window
    return rag

class MockEvolutionMetrics:
    """Mock class for evolution metrics tracking.
    
    Simulates the metrics tracking functionality with concrete attributes for:
    - pattern_coherence: Measure of pattern similarity and relationship strength
    - pattern_diversity: Measure of pattern variety and distribution
    - evolution_rate: Rate of pattern change over time
    - flow_velocity: Speed of pattern relationship formation
    - flow_direction: Directionality of pattern evolution
    """
    def __init__(self):
        self.pattern_coherence = 0.0
        self.pattern_diversity = 0.0
        self.evolution_rate = 0.0
        self.flow_velocity = 0.0
        self.flow_direction = 0.0

class MockFlowDynamics:
    """Mock class for flow dynamics tracking.
    
    Simulates the flow dynamics of pattern evolution with:
    - velocity: Speed of pattern relationship changes
    - direction: Orientation of pattern evolution
    """
    def __init__(self):
        self.velocity = 0.0
        self.direction = 0.0

class TestPatternEvolution:
    """Test suite for pattern evolution metrics."""
    
    @pytest.mark.asyncio
    async def test_document_pattern_evolution(self, pattern_aware_rag):
        """Test how patterns evolve with document adaptations.
        
        Validates:
        1. Pattern emergence from document updates
        2. Evolution metrics calculation
        3. Coherence improvement with pattern growth
        
        Args:
            pattern_aware_rag: Fixture providing mock RAG system
        """
        # Initial document state
        doc = {
            "id": "doc1",
            "content": "Sea levels rising 3mm/year",
            "adaptive_state": {
                "version": 1,
                "confidence": 0.8
            }
        }
        
        # Track initial patterns
        initial_patterns = pattern_aware_rag._extract_doc_patterns(doc)
        print(f"\nInitial patterns: {initial_patterns}")
        
        # Calculate initial pattern metrics
        initial_metrics = MockEvolutionMetrics()
        initial_metrics.pattern_coherence = len(initial_patterns)
        print(f"Initial pattern count: {initial_metrics.pattern_coherence}")

        # Document adapts
        doc["content"] += "\nProjected to accelerate to 4mm/year by 2050"
        doc["adaptive_state"]["version"] += 1
        doc["adaptive_state"]["confidence"] = 0.85

        # Track pattern evolution
        evolved_patterns = pattern_aware_rag._extract_doc_patterns(doc)
        print(f"Evolved patterns: {evolved_patterns}")
        
        # Calculate evolution metrics
        evolved_metrics = MockEvolutionMetrics()
        evolution_score = pattern_aware_rag._calculate_pattern_similarity(
            evolved_patterns, initial_patterns
        )
        evolved_metrics.pattern_coherence = evolution_score * len(evolved_patterns)
        print(f"Evolution score: {evolution_score}")
        print(f"Evolution-weighted pattern count: {evolved_metrics.pattern_coherence}")

        # Validate differentials
        pattern_delta = len(evolved_patterns) - len(initial_patterns)
        confidence_delta = evolved_metrics.pattern_coherence - initial_metrics.pattern_coherence
        print(f"Pattern delta: {pattern_delta}")
        print(f"Confidence delta: {confidence_delta}")
        
        assert pattern_delta > 0  # New patterns emerged
        assert confidence_delta > 0  # Pattern richness improved with coherence

    def test_pattern_relationship_formation(self, pattern_aware_rag):
        """Test how pattern evolution influences relationship formation.
        
        Validates:
        1. Relationship strengthening with pattern overlap
        2. Temporal context influence on relationship formation
        
        Args:
            pattern_aware_rag: Fixture providing mock RAG system
        """
        doc1 = {"content": "Sea level rise 3mm/year", "id": "doc1"}
        doc2 = {"content": "Coastal flooding increasing", "id": "doc2"}
        
        # Initial relationship state
        initial_similarity = pattern_aware_rag._calculate_pattern_similarity(
            pattern_aware_rag._extract_doc_patterns(doc1),
            pattern_aware_rag._extract_doc_patterns(doc2)
        )
        
        # Add temporal context
        doc1["content"] += "\nProjected impacts by 2050"
        doc2["content"] += "\nFlood frequency doubled since 1990"
        
        # Evolved relationship state
        evolved_similarity = pattern_aware_rag._calculate_pattern_similarity(
            pattern_aware_rag._extract_doc_patterns(doc1),
            pattern_aware_rag._extract_doc_patterns(doc2)
        )
        
        relationship_delta = evolved_similarity - initial_similarity
        assert relationship_delta > 0  # Relationship strengthened
        
    @pytest.mark.asyncio
    async def test_coherence_flow_dynamics(self, pattern_aware_rag):
        """Test how coherence changes affect flow dynamics.
        
        Validates:
        1. Flow acceleration with coherence improvement
        2. Flow direction alignment with pattern evolution
        
        Args:
            pattern_aware_rag: Fixture providing mock RAG system
        """
        # Initial flow state
        initial_flow = MockFlowDynamics()
        initial_patterns = ["temperature"]
        initial_related = ["arctic", "ice"]
        initial_scores = [0.7, 0.6]
        
        initial_flow.velocity = sum(initial_scores) / len(initial_scores)
        initial_flow.direction = pattern_aware_rag._calculate_pattern_similarity(
            initial_patterns, initial_related
        )
        
        # Add coherence-building content
        evolved_patterns = initial_patterns + ["warming"]
        evolved_related = initial_related + ["temperature"]
        evolved_scores = [0.8, 0.7, 0.9]
        
        # Evolved flow state
        evolved_flow = MockFlowDynamics()
        evolved_flow.velocity = sum(evolved_scores) / len(evolved_scores)
        evolved_flow.direction = pattern_aware_rag._calculate_pattern_similarity(
            evolved_patterns, evolved_related
        )
        
        # Validate flow differentials
        velocity_delta = evolved_flow.velocity - initial_flow.velocity
        direction_delta = evolved_flow.direction - initial_flow.direction
        
        assert velocity_delta > 0  # Flow accelerated
        assert direction_delta > 0  # Flow more coherent
        
    def test_coherence_metrics(self, pattern_aware_rag):
        """Test coherence metric calculations.
        
        Validates:
        1. Coherence score calculation
        2. Coherence score range [0,1]
        
        Args:
            pattern_aware_rag: Fixture providing mock RAG system
        """
        # Mock document patterns
        doc_patterns = [
            ["climate change", "sea level"],
            ["coastal erosion", "adaptation"]
        ]
        
        # Calculate coherence
        coherence_scores = []
        for patterns in doc_patterns:
            score = pattern_aware_rag._calculate_pattern_similarity(
                ["climate impact"], patterns
            )
            coherence_scores.append(score)
            
        avg_coherence = sum(coherence_scores) / len(coherence_scores)
        assert 0 <= avg_coherence <= 1

class TestDensityAnalysis(unittest.TestCase):
    """Test suite for density analysis in learning windows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interface = LearningWindowInterface()
        
    def test_density_analysis(self):
        """Test density analysis in learning windows."""
        # Create test windows with climate domain patterns
        window_data = {
            "score": 0.9117,
            "potential": 1.0000,
            "horizon": 0.8869,
            "viscosity": 0.3333,
            "channels": {
                "structural": {
                    "strength": 0.8869,
                    "sustainability": 1.0000
                },
                "semantic": {
                    "strength": 1.0000,
                    "sustainability": 0.5695
                }
            },
            "semantic_patterns": [
                {"domain": "climate", "strength": 0.8932},
                {"domain": "climate", "strength": 0.7845},
                {"domain": "oceanography", "strength": 0.6523}
            ]
        }
        
        # Register window
        window_id = self.interface.register_window(window_data)
        
        # Get density analysis
        density = self.interface.get_density_analysis(window_id)
        
        # Verify density metrics
        self.assertIsNotNone(density)
        self.assertIn("metrics", density)
        self.assertIn("gradients", density)
        
        metrics = density["metrics"]
        self.assertGreater(metrics["local"], 0)
        self.assertGreater(metrics["cross_domain"], 0)
        self.assertGreater(len(metrics["alignments"]), 0)
        
        # Verify domain alignments
        alignments = metrics["alignments"]
        climate_alignment = next(a for a in alignments if a["domain"] == "climate")
        self.assertGreater(climate_alignment["strength"], 0.8)  # Strong climate domain alignment
        
        # Register another window with shifted domain focus
        window_data_2 = {
            **window_data,
            "semantic_patterns": [
                {"domain": "oceanography", "strength": 0.8932},
                {"domain": "climate", "strength": 0.6523}
            ]
        }
        window_id_2 = self.interface.register_window(window_data_2)
        
        # Get field-wide density analysis
        field_density = self.interface.get_density_analysis()
        
        # Verify field analysis
        self.assertIn("global_density", field_density)
        self.assertIn("density_centers", field_density)
        self.assertIn("cross_domain_paths", field_density)
        
        # Verify cross-domain paths
        paths = field_density["cross_domain_paths"]
        self.assertTrue(len(paths) > 0)
        strongest_path = paths[0]
        self.assertGreater(strongest_path["strength"], 0.7)  # Strong cross-domain connection
        
        # Print detailed analysis
        print("\n=== Density Analysis ===")
        print("\nWindow Density Metrics:")
        print(f"Local Density: {metrics['local']:.4f}")
        print(f"Cross-Domain: {metrics['cross_domain']:.4f}")
        
        print("\nDomain Alignments:")
        for alignment in metrics["alignments"]:
            print(f"  {alignment['domain']}: {alignment['strength']:.4f}")
            
        print("\nField Analysis:")
        print(f"Global Density: {field_density['global_density']:.4f}")
        
        print("\nDensity Centers:")
        for center in field_density["density_centers"]:
            print(f"  Window {center['window_id']}: {center['density']:.4f}")
            
        print("\nCross-Domain Paths:")
        for path in paths:
            print(f"  {path['start']} → {path['end']}: {path['strength']:.4f} "
                  f"(stability: {path['stability']:.4f})")
            
        # Verify density insights
        self.assertGreater(field_density["global_density"], 0.5)  # Healthy overall density
        self.assertTrue(any(c["density"] > 0.8 for c in field_density["density_centers"]))  # Strong centers
        self.assertTrue(any(p["strength"] > 0.7 for p in paths))  # Strong cross-domain connections

class TestStateTransitions:
    """Test suite for observing state transitions in pattern-aware RAG."""
    
    @pytest.mark.asyncio
    async def test_query_retrieval_states(self, pattern_aware_rag):
        """Observe state changes between query and retrieval phases.
        
        This test specifically looks at:
        1. Initial query state formation
        2. State changes during retrieval
        3. Pattern presence in both states
        """
        # Set up initial document state
        doc = {
            "content": "Global temperature increased by 0.8°C",
            "metadata": {"source": "climate_report", "year": 2024}
        }
        await pattern_aware_rag.add_document(doc)
        
        # Capture query state
        query = "What is the temperature increase?"
        query_state = await pattern_aware_rag.process_query(query)
        
        # Record initial patterns
        initial_patterns = query_state.query_patterns
        assert len(initial_patterns) > 0, "Query should generate initial patterns"
        
        # Capture retrieval state
        retrieval_state = await pattern_aware_rag.retrieve_relevant(query_state)
        retrieved_patterns = retrieval_state.retrieval_patterns
        
        # Verify state differences
        assert len(retrieved_patterns) >= len(initial_patterns), "Retrieval should maintain or expand patterns"
        assert any(pattern in retrieved_patterns for pattern in initial_patterns), "Retrieved patterns should include query patterns"
        
        # Check pattern coherence
        coherence = pattern_aware_rag.coherence_analyzer.calculate_coherence(
            initial_patterns, retrieved_patterns
        )
        assert 0 <= coherence <= 1, "Coherence should be normalized"
        assert coherence > 0.5, "Retrieved patterns should maintain coherence with query"

    @pytest.mark.asyncio
    async def test_retrieval_augmentation_states(self, pattern_aware_rag):
        """Observe state changes between retrieval and augmentation phases.
        
        This test specifically looks at:
        1. Pattern state during retrieval
        2. State changes during augmentation
        3. Pattern evolution through the transition
        """
        # Set up test documents
        docs = [
            {
                "content": "Global temperature increased by 0.8°C",
                "metadata": {"source": "report_1", "year": 2024}
            },
            {
                "content": "Arctic ice declining 13% per decade",
                "metadata": {"source": "report_2", "year": 2024}
            }
        ]
        for doc in docs:
            await pattern_aware_rag.add_document(doc)
            
        # Initialize with query
        query = "How are temperatures affecting ice?"
        query_state = await pattern_aware_rag.process_query(query)
        
        # Capture retrieval state
        retrieval_state = await pattern_aware_rag.retrieve_relevant(query_state)
        pre_augment_patterns = retrieval_state.retrieval_patterns
        
        # Capture augmentation state
        augmented_state = await pattern_aware_rag.augment_context(retrieval_state)
        post_augment_patterns = augmented_state.augmentation_patterns
        
        # Verify pattern evolution
        assert len(post_augment_patterns) >= len(pre_augment_patterns), "Augmentation should maintain or expand patterns"
        
        # Check for relationship formation
        relationship_score = pattern_aware_rag.coherence_analyzer.calculate_relationship_strength(
            ["temperature"], ["ice"], post_augment_patterns
        )
        assert relationship_score > 0, "Augmentation should form relationships between concepts"
        
        # Verify state coherence maintained
        coherence = pattern_aware_rag.coherence_analyzer.calculate_coherence(
            pre_augment_patterns, post_augment_patterns
        )
        assert coherence > 0.5, "Augmentation should maintain coherence with retrieval"

    @pytest.mark.asyncio
    async def test_embedding_state_transitions(self, pattern_aware_rag):
        """Observe actual state transitions in embedding space, including edge dynamics."""
        docs = [
            {
                "content": "Global temperature increased by 0.8°C since 1980",
                "metadata": {"source": "report_1", "year": 2024}
            },
            {
                "content": "Arctic ice volume decreased 13% per decade",
                "metadata": {"source": "report_2", "year": 2024}
            },
            {
                "content": "Coastal flooding frequency doubled in same period",
                "metadata": {"source": "report_3", "year": 2024}
            },
            # Adding more states to observe longer evolution
            {
                "content": "Sea level rise accelerating in coastal regions",
                "metadata": {"source": "report_4", "year": 2024}
            },
            {
                "content": "Temperature anomalies affect ice sheet stability",
                "metadata": {"source": "report_5", "year": 2024}
            }
        ]
    
        # Each addition creates a real state in the system
        embedding_states = []
        vector_states = []
        augmented_states = []
        edge_states = []  # Track relationship edges
        
        print("\n=== Starting State Transition Observation ===")
        
        for i, doc in enumerate(docs, 1):
            print(f"\nProcessing Document {i}: {doc['content']}")
            
            # Capture the actual embedding state
            embedding = await pattern_aware_rag.embeddings.embed_text(doc["content"])
            embedding_states.append(np.array(embedding))
            print(f"Embedding State {i}:")
            print(f"  Shape: {embedding.shape}")
            print(f"  Values: {embedding[:5]}...")
            
            # Capture the actual vector state in the store
            await pattern_aware_rag.add_document(doc)
            vector = await pattern_aware_rag.vector_store.similarity_search(
                doc["content"], k=1
            )
            vector_states.append(vector[0])
            augmented_states.append({
                "original": doc["content"],
                "augmented": vector[0].page_content
            })
            
            # Track edges to all previous states
            if i > 1:
                edges = []
                for j in range(i-1):
                    edge = {
                        "from": j,
                        "to": i-1,
                        "embedding_similarity": compute_cosine_similarity(
                            embedding_states[j],
                            embedding_states[i-1]
                        ),
                        "semantic_similarity": compute_similarity(
                            vector_states[j].page_content,
                            vector_states[i-1].page_content
                        )
                    }
                    edges.append(edge)
                edge_states.append(edges)
            
            print(f"Vector State {i}:")
            print(f"  Original: {doc['content']}")
            print(f"  Augmented: {vector[0].page_content}")
            
            # Analyze augmentation impact
            augmentation_diff = set(vector[0].page_content.split()) - set(doc["content"].split())
            print(f"  Augmentation Added: {', '.join(augmentation_diff)}")
        
        # Analyze state transitions with edge dynamics
        print("\n=== State Transition Analysis ===")
        state_differences = []
        semantic_drifts = []
        edge_dynamics = []  # Track how edges evolve
        
        for i in range(len(embedding_states) - 1):
            # Basic transition metrics
            emb1, emb2 = embedding_states[i], embedding_states[i + 1]
            euclidean_dist = np.linalg.norm(emb1 - emb2)
            cosine_sim = compute_cosine_similarity(emb1, emb2)
            vec_sim = compute_similarity(
                vector_states[i].page_content,
                vector_states[i + 1].page_content
            )
            
            # Calculate semantic viscosity (resistance to meaning change)
            if i > 0:
                prev_meaning_change = semantic_drifts[-1]["meaning_change"]
                current_meaning_change = 1 - vec_sim
                viscosity = abs(current_meaning_change - prev_meaning_change)
            else:
                viscosity = 0
                
            # Track edge stability
            edge_stability = None
            delta_induction = None
            if i > 0:
                # Compare edges before and after transition
                prev_edges = edge_states[i-1]
                curr_edges = edge_states[i]
                edge_changes = []
                new_edges = []
                
                # Track both changes and new formations
                for p, c in zip(prev_edges, curr_edges):
                    edge_delta = {
                        "semantic_delta": abs(p["semantic_similarity"] - c["semantic_similarity"]),
                        "structural_delta": abs(p["embedding_similarity"] - c["embedding_similarity"])
                    }
                    edge_changes.append(edge_delta)
                    
                    # Detect delta induction points
                    # New edge forms (semantic > 0.3) despite high viscosity
                    if c["semantic_similarity"] > 0.3 and p["semantic_similarity"] < 0.2:
                        new_edges.append({
                            "from": c["from"],
                            "to": c["to"],
                            "induction_strength": c["semantic_similarity"],
                            "structural_support": c["embedding_similarity"]
                        })
                
                edge_stability = 1 - np.mean([
                    (e["semantic_delta"] + e["structural_delta"]) / 2 
                    for e in edge_changes
                ])
                
                # Calculate delta induction properties
                if new_edges:
                    # Calculate gradient properties
                    structural_supports = [e["structural_support"] for e in new_edges]
                    induction_strengths = [e["induction_strength"] for e in new_edges]
                    
                    # Gradient analysis
                    gradient_properties = {
                        "support_range": max(structural_supports) - min(structural_supports),
                        "strength_range": max(induction_strengths) - min(induction_strengths),
                        "gradient_ratio": (max(induction_strengths) - min(induction_strengths)) / 
                                        (max(structural_supports) - min(structural_supports) + 1e-10),
                        "correlation": np.corrcoef(structural_supports, induction_strengths)[0,1]
                    }
                    
                    # Calculate propensity scores
                    propensity = {
                        "formation": np.mean([
                            1 if s > 0.5 and i > 0.3 else 0 
                            for s, i in zip(structural_supports, induction_strengths)
                        ]),
                        "stability": np.mean([
                            min(s, i) / max(s, i) 
                            for s, i in zip(structural_supports, induction_strengths)
                        ]),
                        "growth_potential": np.mean([
                            (1 - s) * i 
                            for s, i in zip(structural_supports, induction_strengths)
                        ])
                    }
                    
                    delta_induction = {
                        "count": len(new_edges),
                        "edges": new_edges,
                        "avg_strength": np.mean(induction_strengths),
                        "structural_coherence": np.mean(structural_supports),
                        "gradient": gradient_properties,
                        "propensity": propensity
                    }
            
            # Calculate semantic drift with edge awareness
            drift = {
                "structure_change": 1 - cosine_sim,
                "meaning_change": 1 - vec_sim,
                "augmentation_impact": compute_similarity(
                    augmented_states[i]["augmented"],
                    augmented_states[i+1]["augmented"]
                ) - compute_similarity(
                    augmented_states[i]["original"],
                    augmented_states[i+1]["original"]
                ),
                "viscosity": viscosity,
                "edge_stability": edge_stability,
                "delta_induction": delta_induction
            }
            semantic_drifts.append(drift)
            
            state_differences.append({
                "embedding_distance": euclidean_dist,
                "cosine_similarity": cosine_sim,
                "vector_similarity": vec_sim
            })
            
            # Calculate learning window properties
            learning_window = None
            if i > 0:
                # Analyze conditions for learning receptivity
                receptivity_factors = {
                    "structural_stability": edge_stability if edge_stability is not None else 0,
                    "meaning_flux": abs(drift["meaning_change"] - semantic_drifts[-1]["meaning_change"]),
                    "viscosity_gradient": viscosity - semantic_drifts[-1]["viscosity"],
                    "augmentation_coherence": drift["augmentation_impact"] / (1 - cosine_sim + 1e-10)
                }
                
                # Calculate window properties
                window_score = (
                    receptivity_factors["structural_stability"] * 0.4 +
                    (1 - receptivity_factors["meaning_flux"]) * 0.3 +
                    (1 - abs(receptivity_factors["viscosity_gradient"])) * 0.2 +
                    receptivity_factors["augmentation_coherence"] * 0.1
                )
                
                # Determine window characteristics
                learning_window = {
                    "score": window_score,
                    "receptivity": receptivity_factors,
                    "propagation_potential": min(
                        1.0,
                        window_score * (1 + drift["augmentation_impact"])
                    ),
                    "stability_horizon": edge_stability * (1 - receptivity_factors["meaning_flux"]),
                    "growth_channels": [
                        {
                            "type": "structural",
                            "strength": receptivity_factors["structural_stability"],
                            "sustainability": 1 - abs(receptivity_factors["viscosity_gradient"])
                        },
                        {
                            "type": "semantic",
                            "strength": 1 - receptivity_factors["meaning_flux"],
                            "sustainability": receptivity_factors["augmentation_coherence"]
                        }
                    ]
                }
                
                # Calculate propagation dynamics
                if delta_induction:
                    propagation_dynamics = []
                    for edge in delta_induction["edges"]:
                        channel_alignment = np.mean([
                            ch["strength"] * ch["sustainability"]
                            for ch in learning_window["growth_channels"]
                        ])
                        
                        dynamics = {
                            "edge": (edge["from"], edge["to"]),
                            "propagation_strength": edge["induction_strength"] * channel_alignment,
                            "stability_factor": edge["structural_support"] * learning_window["stability_horizon"],
                            "growth_potential": (1 - edge["structural_support"]) * learning_window["propagation_potential"]
                        }
                        propagation_dynamics.append(dynamics)
                        
                    learning_window["propagation_dynamics"] = propagation_dynamics
            
            # Update drift analysis with learning window
            drift["learning_window"] = learning_window
            
            print(f"\nTransition {i+1} → {i+2}:")
            print(f"  Embedding Distance: {euclidean_dist:.4f}")
            print(f"  Cosine Similarity: {cosine_sim:.4f}")
            print(f"  Vector Similarity: {vec_sim:.4f}")
            print(f"  Semantic Drift:")
            print(f"    Structure Change: {drift['structure_change']:.4f}")
            print(f"    Meaning Change: {drift['meaning_change']:.4f}")
            print(f"    Augmentation Impact: {drift['augmentation_impact']:.4f}")
            print(f"    Viscosity: {drift['viscosity']:.4f}")
            if edge_stability is not None:
                print(f"    Edge Stability: {edge_stability:.4f}")
            if learning_window is not None:
                print(f"  Learning Window:")
                print(f"    Window Score: {learning_window['score']:.4f}")
                print(f"    Propagation Potential: {learning_window['propagation_potential']:.4f}")
                print(f"    Stability Horizon: {learning_window['stability_horizon']:.4f}")
                print(f"    Growth Channels:")
                for channel in learning_window["growth_channels"]:
                    print(f"      {channel['type'].title()}:")
                    print(f"        Strength: {channel['strength']:.4f}")
                    print(f"        Sustainability: {channel['sustainability']:.4f}")
                if "propagation_dynamics" in learning_window:
                    print(f"    Propagation Dynamics:")
                    for dynamics in learning_window["propagation_dynamics"]:
                        print(f"      Edge {dynamics['edge']}:")
                        print(f"        Propagation Strength: {dynamics['propagation_strength']:.4f}")
                        print(f"        Stability Factor: {dynamics['stability_factor']:.4f}")
                        print(f"        Growth Potential: {dynamics['growth_potential']:.4f}")
            
        # Analyze learning window evolution
        window_evolution = []
        propagation_patterns = []
        interface_data = {
            "metadata": {
                "timestamp": "2025-01-01T14:33:09-05:00",
                "version": "0.1.0",
                "schema": "habitat.learning.windows.v1"
            },
            "windows": [],
            "patterns": {
                "cross_window": [],
                "predictions": [],
                "recommendations": []
            },
            "metrics": {
                "temporal": {},
                "structural": {},
                "semantic": {}
            }
        }
        
        for drift in semantic_drifts:
            if drift["learning_window"]:
                window_data = {
                    "score": drift["learning_window"]["score"],
                    "potential": drift["learning_window"]["propagation_potential"],
                    "horizon": drift["learning_window"]["stability_horizon"],
                    "channels": {
                        ch["type"]: {
                            "strength": ch["strength"],
                            "sustainability": ch["sustainability"]
                        }
                        for ch in drift["learning_window"]["growth_channels"]
                    }
                }
                window_evolution.append(window_data)
                interface_data["windows"].append({
                    "id": len(interface_data["windows"]),
                    "state": window_data,
                    "metrics": {
                        "structure_change": drift["structure_change"],
                        "meaning_change": drift["meaning_change"],
                        "viscosity": drift["viscosity"],
                        "edge_stability": drift["edge_stability"]
                    }
                })
                
                if "propagation_dynamics" in drift["learning_window"]:
                    window_patterns = []
                    for dynamics in drift["learning_window"]["propagation_dynamics"]:
                        pattern = {
                            "edge": dynamics["edge"],
                            "strength": dynamics["propagation_strength"],
                            "stability": dynamics["stability_factor"],
                            "growth": dynamics["growth_potential"]
                        }
                        window_patterns.append(pattern)
                        propagation_patterns.append(pattern)
                    
                    interface_data["patterns"]["cross_window"].append({
                        "window_id": len(interface_data["windows"]) - 1,
                        "patterns": window_patterns
                    })
        
        if window_evolution:
            # Calculate cross-window patterns
            scores = np.array([w["score"] for w in window_evolution])
            potentials = np.array([w["potential"] for w in window_evolution])
            horizons = np.array([w["horizon"] for w in window_evolution])
            
            # Temporal patterns
            interface_data["metrics"]["temporal"] = {
                "score_trend": np.polyfit(np.arange(len(scores)), scores, 1).tolist(),
                "potential_trend": np.polyfit(np.arange(len(potentials)), potentials, 1).tolist(),
                "horizon_trend": np.polyfit(np.arange(len(horizons)), horizons, 1).tolist(),
                "cycle_length": len(scores)
            }
            
            # Structural patterns
            structural_metrics = np.array([
                [w["channels"]["structural"]["strength"] for w in window_evolution],
                [w["channels"]["structural"]["sustainability"] for w in window_evolution]
            ])
            interface_data["metrics"]["structural"] = {
                "strength_stability": np.corrcoef(structural_metrics)[0,1],
                "trend": np.polyfit(np.arange(len(structural_metrics[0])), 
                                  structural_metrics[0] * structural_metrics[1], 1).tolist()
            }
            
            # Semantic patterns
            semantic_metrics = np.array([
                [w["channels"]["semantic"]["strength"] for w in window_evolution],
                [w["channels"]["semantic"]["sustainability"] for w in window_evolution]
            ])
            interface_data["metrics"]["semantic"] = {
                "strength_stability": np.corrcoef(semantic_metrics)[0,1],
                "trend": np.polyfit(np.arange(len(semantic_metrics[0])), 
                                  semantic_metrics[0] * semantic_metrics[1], 1).tolist()
            }
            
            # Generate predictions
            next_window_prediction = {
                "score": np.poly1d(interface_data["metrics"]["temporal"]["score_trend"])(len(scores)),
                "potential": np.poly1d(interface_data["metrics"]["temporal"]["potential_trend"])(len(potentials)),
                "horizon": np.poly1d(interface_data["metrics"]["temporal"]["horizon_trend"])(len(horizons))
            }
            interface_data["patterns"]["predictions"].append({
                "type": "next_window",
                "prediction": next_window_prediction,
                "confidence": 1 - np.std([scores, potentials, horizons])
            })
            
            # Generate recommendations
            optimal_windows = np.where(scores > np.mean(scores) + np.std(scores))[0]
            interface_data["patterns"]["recommendations"].extend([
                {
                    "type": "learning_opportunity",
                    "window_id": int(i),
                    "score": float(scores[i]),
                    "reason": "Above average learning potential"
                }
                for i in optimal_windows
            ])
            
            # Print analysis in interface-friendly format
            print("\n=== Learning Window Interface Data ===")
            print(f"Schema Version: {interface_data['metadata']['schema']}")
            print("\nWindow Evolution:")
            for window in interface_data["windows"]:
                print(f"\nWindow {window['id']}:")
                print(f"  Score: {window['state']['score']:.4f}")
                print(f"  Potential: {window['state']['potential']:.4f}")
                print(f"  Horizon: {window['state']['horizon']:.4f}")
                print("  Channels:")
                for ch_type, ch_data in window['state']['channels'].items():
                    print(f"    {ch_type.title()}:")
                    print(f"      Strength: {ch_data['strength']:.4f}")
                    print(f"      Sustainability: {ch_data['sustainability']:.4f}")
            
            print("\nCross-Window Patterns:")
            print(f"Score Trend: {interface_data['metrics']['temporal']['score_trend']}")
            print(f"Structural Stability: {interface_data['metrics']['structural']['strength_stability']:.4f}")
            print(f"Semantic Evolution: {interface_data['metrics']['semantic']['strength_stability']:.4f}")
            
            print("\nPredictions:")
            for pred in interface_data["patterns"]["predictions"]:
                print(f"\n{pred['type'].title()}:")
                print(f"  Score: {pred['prediction']['score']:.4f}")
                print(f"  Potential: {pred['prediction']['potential']:.4f}")
                print(f"  Horizon: {pred['prediction']['horizon']:.4f}")
                print(f"  Confidence: {pred['confidence']:.4f}")
            
            print("\nRecommendations:")
            for rec in interface_data["patterns"]["recommendations"]:
                print(f"\n{rec['type'].title()}:")
                print(f"  Window: {rec['window_id']}")
                print(f"  Score: {rec['score']:.4f}")
                print(f"  Reason: {rec['reason']}")
            
            # Save interface data for future agent consumption
            interface_data_path = "habitat_test/data/learning_windows.json"
            os.makedirs(os.path.dirname(interface_data_path), exist_ok=True)
            with open(interface_data_path, 'w') as f:
                json.dump(interface_data, f, indent=2)
            print(f"\nInterface data saved to: {interface_data_path}")
        
        # Validate interface data structure
        if interface_data["windows"]:
            assert all(0 <= w["state"]["score"] <= 1 for w in interface_data["windows"])
            assert all(0 <= w["state"]["potential"] <= 1 for w in interface_data["windows"])
            assert all(isinstance(w["id"], int) for w in interface_data["windows"])
            assert all(isinstance(p["confidence"], float) for p in interface_data["patterns"]["predictions"])
        
        # Track edge persistence and delta induction patterns
        stable_edges = []
        induction_points = []
        for i, drift in enumerate(semantic_drifts):
            if drift["delta_induction"]:
                induction_points.append({
                    "transition": i,
                    "viscosity": drift["viscosity"],
                    "count": drift["delta_induction"]["count"],
                    "strength": drift["delta_induction"]["avg_strength"]
                })
            
            if i > 0:  # Skip first transition for edge tracking
                for edge in edge_states[i]:
                    if edge["semantic_similarity"] > 0.3 and edge["embedding_similarity"] > 0.5:
                        stable_edges.append((edge["from"], edge["to"]))
        
        print("\n=== Edge Evolution Analysis ===")
        print(f"Stable Edges (high similarity maintained): {stable_edges}")
        print("\n=== Delta Induction Analysis ===")
        for point in induction_points:
            print(f"Transition {point['transition']+1}:")
            print(f"  New Edges: {point['count']}")
            print(f"  System Viscosity: {point['viscosity']:.4f}")
            print(f"  Induction Strength: {point['strength']:.4f}")
        
        # Analyze relationship between viscosity and induction
        induction_transitions = [p["transition"] for p in induction_points]
        viscosity_at_induction = [semantic_drifts[t]["viscosity"] for t in induction_transitions]
        if viscosity_at_induction:
            print(f"\nAverage Viscosity at Induction Points: {np.mean(viscosity_at_induction):.4f}")
            print(f"Max Viscosity at Induction: {max(viscosity_at_induction):.4f}")
            print(f"Min Viscosity at Induction: {min(viscosity_at_induction):.4f}")
        
        # Validate evolution properties
        assert all(-1 <= d["viscosity"] <= 1 for d in semantic_drifts)
        for drift in semantic_drifts:
            if drift["edge_stability"] is not None:
                assert 0 <= drift["edge_stability"] <= 1
            if drift["delta_induction"] is not None:
                assert drift["delta_induction"]["avg_strength"] > 0
                assert 0 <= drift["delta_induction"]["structural_coherence"] <= 1

class TestIngestionEnhancement:
    """Test suite for document ingestion and pattern enhancement with density awareness."""

    @pytest_asyncio.fixture
    async def setup_test_docs(self):
        """Set up test documents for ingestion."""
        test_doc_path = "/Users/prphillips/Documents/GitHub/habitat_poc/habitat_test/data/test_docs/Climate-Risk-Assessment-Marthas-Vineyard-MA.txt"
        with open(test_doc_path, 'r') as f:
            climate_doc = f.read()
        return {"climate_doc": climate_doc}

    @pytest.mark.asyncio
    async def test_document_ingestion_with_density(self, pattern_aware_rag, setup_test_docs):
        """Test document ingestion with density-aware pattern evolution.
        
        Validates:
        1. Document ingestion creates appropriate density centers
        2. Pattern evolution responds to density changes
        3. Cross-domain relationships form based on density paths
        """
        # Mock density analysis results
        mock_density = {
            "global_density": 0.75,
            "density_centers": [
                {"density": 0.85, "domain": "climate", "alignments": [
                    {"domain": "risk", "strength": 0.82}
                ]},
                {"density": 0.78, "domain": "risk", "alignments": [
                    {"domain": "climate", "strength": 0.82}
                ]}
            ],
            "cross_domain_paths": [
                {"source": "climate", "target": "risk", "strength": 0.82, "stability": 0.79}
            ]
        }
        
        # Mock the learning window interface
        mock_window = Mock()
        mock_window.get_density_analysis.return_value = mock_density
        pattern_aware_rag.evolution_manager.learning_window_interface = mock_window
        
        # Ingest test document
        doc = setup_test_docs["climate_doc"]
        context = RAGPatternContext(
            query_patterns=[],
            retrieval_patterns=[],
            augmentation_patterns=[],
            coherence_level=0.0
        )
        
        # Process document and track evolution
        await pattern_aware_rag.process_document(doc, context)
        
        # Verify density-aware pattern evolution
        evolution_state = pattern_aware_rag.get_evolution_state()
        
        # Check density influence on evolution
        assert evolution_state["density_score"] > 0.7, "Density score should reflect strong centers"
        assert evolution_state["cross_domain_strength"] > 0.75, "Should show strong cross-domain relationships"
        
        # Verify pattern enhancement in high-density regions
        enhanced_patterns = evolution_state["enhanced_patterns"]
        assert any(p["domain"] == "climate" for p in enhanced_patterns), "Should enhance climate domain patterns"
        assert any(p["domain"] == "risk" for p in enhanced_patterns), "Should enhance risk domain patterns"
        
        # Check learning window registration
        calls = mock_window.register_window.call_args_list
        assert len(calls) > 0, "Should register learning windows during evolution"
        
        window_data = calls[0][0][0]  # Get first call's window data
        assert window_data["score"] > 0.7, "Window score should reflect high density"
        assert "channels" in window_data, "Should include channel information"
        assert len(window_data["semantic_patterns"]) > 0, "Should include semantic patterns"

    @pytest.mark.asyncio
    async def test_pattern_enhancement_with_density(self, pattern_aware_rag, setup_test_docs):
        """Test pattern enhancement in high-density regions.
        
        Validates:
        1. Patterns are enhanced in high-density centers
        2. Enhancement respects cross-domain relationships
        3. Coherence is maintained during enhancement
        """
        # Mock initial density state
        mock_density = {
            "global_density": 0.82,
            "density_centers": [
                {"density": 0.88, "domain": "climate", "alignments": [
                    {"domain": "adaptation", "strength": 0.85}
                ]},
                {"density": 0.84, "domain": "adaptation", "alignments": [
                    {"domain": "climate", "strength": 0.85}
                ]}
            ],
            "cross_domain_paths": [
                {
                    "source": "climate",
                    "target": "adaptation",
                    "strength": 0.85,
                    "stability": 0.83
                }
            ]
        }
        
        # Set up mock window interface
        mock_window = Mock()
        mock_window.get_density_analysis.return_value = mock_density
        pattern_aware_rag.evolution_manager.learning_window_interface = mock_window
        
        # Process document with enhancement focus
        doc = setup_test_docs["climate_doc"]
        context = RAGPatternContext(
            query_patterns=["climate adaptation"],
            retrieval_patterns=[],
            augmentation_patterns=[],
            coherence_level=0.8
        )
        
        # Trigger enhancement process
        await pattern_aware_rag.enhance_patterns(doc, context)
        
        # Verify enhancement results
        enhancement_state = pattern_aware_rag.get_enhancement_state()
        
        # Check density-guided enhancement
        assert enhancement_state["enhancement_score"] > 0.8, "Enhancement should be strong in high-density regions"
        
        # Verify cross-domain enhancement
        enhanced_domains = enhancement_state["enhanced_domains"]
        assert "climate" in enhanced_domains, "Should enhance climate domain"
        assert "adaptation" in enhanced_domains, "Should enhance adaptation domain"
        
        # Check coherence maintenance
        assert enhancement_state["coherence_level"] > 0.75, "Should maintain coherence during enhancement"
        
        # Verify learning window updates
        window_calls = mock_window.register_window.call_args_list
        assert len(window_calls) > 0, "Should update learning windows during enhancement"
        
        # Check enhanced pattern properties
        enhanced_patterns = enhancement_state["patterns"]
        for pattern in enhanced_patterns:
            assert pattern["density_score"] > 0.7, "Enhanced patterns should have high density"
            assert pattern["stability"] > 0.7, "Enhanced patterns should be stable"

def compute_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two text strings."""
    # Convert to sets of words for simple similarity
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
