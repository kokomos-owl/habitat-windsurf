"""Test suite for cyclical structure-meaning evolution."""

import pytest
from typing import Dict, Any, List
from datetime import datetime
from uuid import uuid4

from habitat_test.core.coherence_flow import CoherenceFlow, FlowState
from habitat_test.core.emergence_flow import EmergenceFlow, EmergenceType
from habitat_test.core.pattern_evolution import PatternEvolutionManager, EvolutionState
from habitat_test.core.graph_coherence import GraphCoherence, CoherenceVector
from habitat_test.core.coherence_monitor import CoherenceMonitor
from habitat_test.core.graph_schema import GraphSchema, NodeLabel, RelationType
from habitat_test.core.user_affinity import UserAffinityGraph, AffinityPattern

from adaptive_core.pattern_core import PatternCore, PatternEvidence
from adaptive_core.coherence_tracking import (
    CoherenceAssessment,
    CoherenceMetrics,
    CoherenceLevel,
    CoherenceEnhancer
)
from adaptive_core.relationship_model import (
    RelationshipModel,
    TemporalContext,
    UncertaintyMetrics
)

class TestStructureMeaningCycle:
    """
    Test the cyclical nature of structure-meaning evolution.
    
    The cycle follows:
    Structure → Meaning → New Structure → Enhanced Meaning → ...
    
    Each phase influences and is influenced by the others,
    creating a continuous flow of evolution.
    """
    
    @pytest.fixture
    def graph_schema(self):
        """Initialize graph schema."""
        return GraphSchema()
        
    @pytest.fixture
    def evolution_setup(self, graph_schema, relationship_coherence):
        """Setup evolution components with coherence infrastructure."""
        pattern_core = PatternCore()
        coherence_flow = CoherenceFlow()
        evolution_manager = PatternEvolutionManager(pattern_core)
        emergence_flow = EmergenceFlow(coherence_flow)
        
        # Initialize coherence infrastructure
        graph_coherence = GraphCoherence(
            graph_schema=graph_schema,
            relationship_coherence=relationship_coherence,
            metrics_dir="test_metrics"
        )
        coherence_monitor = CoherenceMonitor(metrics_dir="test_metrics")
        
        # Initialize adaptive core components
        coherence_assessment = CoherenceAssessment()
        coherence_enhancer = CoherenceEnhancer()
        relationship_model = RelationshipModel()
        
        # Initialize user affinity components
        user_affinity = UserAffinityGraph()
        
        return (
            pattern_core,
            coherence_flow,
            evolution_manager,
            emergence_flow,
            graph_coherence,
            coherence_monitor,
            coherence_assessment,
            coherence_enhancer,
            relationship_model,
            graph_schema,
            user_affinity
        )

    @pytest.mark.asyncio
    async def test_structure_meaning_cycle(self, evolution_setup):
        """Test complete structure-meaning evolution cycle."""
        (
            pattern_core,
            coherence_flow,
            evolution_manager,
            emergence_flow,
            graph_coherence,
            coherence_monitor,
            coherence_assessment,
            coherence_enhancer,
            relationship_model,
            graph_schema,
            user_affinity
        ) = evolution_setup
        
        # Track cycle progression
        cycle_states = []
        coherence_vectors = []
        affinity_patterns = []
        
        # Create test users
        user1_id = str(uuid4())
        user2_id = str(uuid4())
        
        # Phase 1: Initial Structure
        start_time = coherence_monitor.start_measurement()
        
        structural_patterns = [
            ("structure1", 0.6, "hierarchical", user1_id),
            ("structure2", 0.7, "relational", user2_id)
        ]
        
        nodes = []
        relationships = []
        for pattern_id, confidence, pattern_type, user_id in structural_patterns:
            evidence = self._create_pattern_evidence(
                pattern_type=pattern_type,
                confidence=confidence,
                suggestions={"type": "initial"}
            )
            evolution_manager.track_pattern(pattern_id, evidence)
            
            # Build graph elements
            node = self._create_pattern_node(pattern_id, pattern_type, confidence)
            nodes.append(node)
            
            # Track user affinity
            affinity = AffinityPattern(
                pattern_id=pattern_id,
                user_id=user_id,
                affinity_score=confidence,
                pattern_type=pattern_type
            )
            user_affinity.add_pattern(affinity)
            affinity_patterns.append(affinity)
        
        # Measure coherence
        vector = await graph_coherence.measure_coherence(nodes, relationships)
        coherence_vectors.append(vector)
        coherence_monitor.end_measurement(start_time)
            
        cycle_states.append(self._capture_cycle_state(
            evolution_manager,
            emergence_flow,
            vector,
            coherence_assessment,
            relationship_model,
            graph_schema
        ))

        # Phase 2: Meaning Formation with Co-Patterning
        meaning_patterns = [
            ("meaning1", 0.7, "semantic", "structure1", user2_id),  # Cross-user meaning formation
            ("meaning2", 0.6, "contextual", "structure2", user1_id)  # Cross-user meaning formation
        ]
        
        start_time = coherence_monitor.start_measurement()
        
        for pattern_id, confidence, pattern_type, related_to, user_id in meaning_patterns:
            evidence = self._create_pattern_evidence(
                pattern_type=pattern_type,
                confidence=confidence,
                suggestions={"related_to": related_to}
            )
            evolution_manager.track_pattern(pattern_id, evidence)
            
            # Build graph elements
            node = self._create_pattern_node(pattern_id, pattern_type, confidence)
            nodes.append(node)
            relationships.append(self._create_pattern_relationship(related_to, pattern_id, RelationType.MEANING, {}))
            
            # Track user affinity and co-patterning
            affinity = AffinityPattern(
                pattern_id=pattern_id,
                user_id=user_id,
                affinity_score=confidence,
                pattern_type=pattern_type
            )
            user_affinity.add_pattern(affinity)
            affinity_patterns.append(affinity)
            
            # Create co-pattern relationship
            user_affinity.add_co_pattern_relationship(
                source_user=user_id,
                target_user=user1_id if user_id == user2_id else user2_id,
                pattern_id=pattern_id,
                relationship_type="meaning_formation"
            )
            
        # Measure coherence
        vector = await graph_coherence.measure_coherence(nodes, relationships)
        coherence_vectors.append(vector)
        coherence_monitor.end_measurement(start_time)
            
        cycle_states.append(self._capture_cycle_state(
            evolution_manager,
            emergence_flow,
            vector,
            coherence_assessment,
            relationship_model,
            graph_schema
        ))

        # Phase 3: Structure Evolution with Collaborative Enhancement
        evolved_structures = [
            ("structure1", 0.8, "hierarchical", "meaning2", user1_id),  # Enhanced by other user's meaning
            ("structure2", 0.8, "relational", "meaning1", user2_id)     # Enhanced by other user's meaning
        ]
        
        start_time = coherence_monitor.start_measurement()
        
        for pattern_id, confidence, pattern_type, influenced_by, user_id in evolved_structures:
            evidence = self._create_pattern_evidence(
                pattern_type=pattern_type,
                confidence=confidence,
                suggestions={
                    "type": "evolved",
                    "influenced_by": influenced_by
                }
            )
            evolution_manager.track_pattern(pattern_id, evidence)
            
            # Build graph elements
            node = self._create_pattern_node(pattern_id, pattern_type, confidence)
            nodes.append(node)
            relationships.append(self._create_pattern_relationship(influenced_by, pattern_id, RelationType.INFLUENCE, {}))
            
            # Update user affinity with evolved pattern
            affinity = AffinityPattern(
                pattern_id=pattern_id,
                user_id=user_id,
                affinity_score=confidence,
                pattern_type=pattern_type
            )
            user_affinity.add_pattern(affinity)
            affinity_patterns.append(affinity)
            
            # Create collaborative enhancement relationship
            user_affinity.add_co_pattern_relationship(
                source_user=user_id,
                target_user=user1_id if user_id == user2_id else user2_id,
                pattern_id=pattern_id,
                relationship_type="collaborative_enhancement"
            )
            
        # Measure coherence
        vector = await graph_coherence.measure_coherence(nodes, relationships)
        coherence_vectors.append(vector)
        coherence_monitor.end_measurement(start_time)
            
        cycle_states.append(self._capture_cycle_state(
            evolution_manager,
            emergence_flow,
            vector,
            coherence_assessment,
            relationship_model,
            graph_schema
        ))

        # Phase 4: Meaning Enhancement through Collaborative Insight
        enhanced_meanings = [
            ("meaning1", 0.9, "semantic", "structure2", user2_id),      # Enhanced through collaboration
            ("meaning2", 0.8, "contextual", "structure1", user1_id)     # Enhanced through collaboration
        ]
        
        start_time = coherence_monitor.start_measurement()
        
        for pattern_id, confidence, pattern_type, enhanced_by, user_id in enhanced_meanings:
            evidence = self._create_pattern_evidence(
                pattern_type=pattern_type,
                confidence=confidence,
                suggestions={
                    "related_to": enhanced_by,
                    "enhancement": "collaborative_insight"
                }
            )
            evolution_manager.track_pattern(pattern_id, evidence)
            
            # Build graph elements
            node = self._create_pattern_node(pattern_id, pattern_type, confidence)
            nodes.append(node)
            relationships.append(self._create_pattern_relationship(enhanced_by, pattern_id, RelationType.ENHANCEMENT, {}))
            
            # Update user affinity with enhanced meaning
            affinity = AffinityPattern(
                pattern_id=pattern_id,
                user_id=user_id,
                affinity_score=confidence,
                pattern_type=pattern_type
            )
            user_affinity.add_pattern(affinity)
            affinity_patterns.append(affinity)
            
            # Create collaborative insight relationship
            user_affinity.add_co_pattern_relationship(
                source_user=user_id,
                target_user=user1_id if user_id == user2_id else user2_id,
                pattern_id=pattern_id,
                relationship_type="collaborative_insight"
            )
            
        # Measure coherence
        vector = await graph_coherence.measure_coherence(nodes, relationships)
        coherence_vectors.append(vector)
        coherence_monitor.end_measurement(start_time)
            
        cycle_states.append(self._capture_cycle_state(
            evolution_manager,
            emergence_flow,
            vector,
            coherence_assessment,
            relationship_model,
            graph_schema
        ))

        # Validate cycle progression
        self._validate_cycle_progression(cycle_states)
        
        # Validate co-pattern formation
        self._validate_co_pattern_formation(user_affinity, user1_id, user2_id)

    def _validate_co_pattern_formation(
        self,
        user_affinity: UserAffinityGraph,
        user1_id: str,
        user2_id: str
    ):
        """Validate the formation of co-patterns between users."""
        # Get co-pattern relationships
        co_patterns = user_affinity.get_co_pattern_relationships(user1_id, user2_id)
        
        # Verify progression of relationship types
        relationship_sequence = [rel.relationship_type for rel in co_patterns]
        expected_sequence = [
            "meaning_formation",
            "collaborative_enhancement",
            "collaborative_insight"
        ]
        
        assert all(rel_type in relationship_sequence for rel_type in expected_sequence), \
            "Co-pattern relationship progression should show increasing collaboration"
            
        # Verify pattern evolution through collaboration
        user1_patterns = user_affinity.get_user_patterns(user1_id)
        user2_patterns = user_affinity.get_user_patterns(user2_id)
        
        assert len(user1_patterns) > 0 and len(user2_patterns) > 0, \
            "Both users should have active patterns"
            
        # Verify increasing affinity scores through collaboration
        initial_scores = [p.affinity_score for p in user1_patterns[:2]]
        final_scores = [p.affinity_score for p in user1_patterns[-2:]]
        
        assert all(final > initial for final, initial in zip(final_scores, initial_scores)), \
            "Pattern affinity scores should increase through collaboration"

    def _create_pattern_node(
        self,
        pattern_id: str,
        pattern_type: str,
        confidence: float
    ) -> Dict[str, Any]:
        """Create a pattern node following schema."""
        return {
            "id": pattern_id,
            "label": NodeLabel.PATTERN.value,
            "type": pattern_type,
            "confidence": confidence,
            "emergence_time": datetime.now().isoformat(),
            "interface": "pattern",
            "evolution": True
        }
        
    def _create_pattern_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: RelationType,
        properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a relationship following schema."""
        return {
            "source": source_id,
            "target": target_id,
            "type": rel_type.value,
            "properties": {
                **properties,
                "created": datetime.now().isoformat()
            }
        }

    def _create_pattern_evidence(
        self,
        pattern_type: str,
        confidence: float,
        suggestions: Dict[str, Any]
    ) -> PatternEvidence:
        """Create pattern evidence with temporal context."""
        return PatternEvidence(
            evidence_id=str(uuid4()),
            timestamp=datetime.now().isoformat(),
            pattern_type=pattern_type,
            confidence=confidence,
            source_data=suggestions,
            temporal_context=TemporalContext(
                start_time=datetime.now().isoformat(),
                duration_ms=1000,
                sequence_number=1
            ),
            uncertainty_metrics=UncertaintyMetrics(
                confidence_score=confidence,
                variance=0.1,
                sample_size=10
            )
        )

    def _capture_cycle_state(
        self,
        evolution_manager: PatternEvolutionManager,
        emergence_flow: EmergenceFlow,
        coherence_vector: CoherenceVector,
        coherence_assessment: CoherenceAssessment,
        relationship_model: RelationshipModel,
        graph_schema: GraphSchema
    ) -> Dict[str, Any]:
        """Capture current state of the evolution cycle."""
        
        # Assess coherence
        coherence_metrics = coherence_assessment.assess_coherence(
            {
                "structure_data": evolution_manager.get_structure_data(),
                "meaning_data": evolution_manager.get_meaning_data()
            }
        )
        
        # Get relationship context
        relationship_context = relationship_model.get_relationship_context(
            list(evolution_manager.states.keys())
        )
        
        # Validate schema adherence
        schema_validation = graph_schema.validate_patterns(
            list(evolution_manager.states.keys())
        )
        
        return {
            "evolution_states": evolution_manager.states.copy(),
            "emergence_potentials": {
                pid: emergence_flow.get_emergence_potential(pid)
                for pid in evolution_manager.states
            },
            "coherence": {
                "structural": coherence_vector.structural,
                "semantic": coherence_vector.semantic,
                "temporal": coherence_vector.temporal,
                "interface": coherence_vector.interface,
                "total": coherence_vector.total()
            },
            "adaptive_coherence": {
                "structure_meaning_alignment": coherence_metrics.structure_meaning_alignment,
                "pattern_alignment": coherence_metrics.pattern_alignment,
                "temporal_consistency": coherence_metrics.temporal_consistency,
                "domain_consistency": coherence_metrics.domain_consistency,
                "overall_coherence": coherence_metrics.overall_coherence,
                "assessment_level": coherence_metrics.assessment_level
            },
            "relationships": relationship_context,
            "schema_validation": schema_validation
        }
        
    def _validate_cycle_progression(
        self,
        cycle_states: List[Dict[str, Any]]
    ) -> None:
        """Validate the progression of the structure-meaning cycle."""
        # Phase 1: Initial Structure
        assert cycle_states[0]["coherence"]["structural"] > 0.5
        assert cycle_states[0]["adaptive_coherence"]["structure_meaning_alignment"] > 0.3
        assert cycle_states[0]["adaptive_coherence"]["assessment_level"] != CoherenceLevel.WARNING
        assert cycle_states[0]["schema_validation"]["valid"]
        assert any(
            state["evolution_states"].get("structure1") == EvolutionState.EMERGING
            for state in cycle_states[:2]
        )
        
        # Phase 2: Meaning Formation
        assert cycle_states[1]["coherence"]["semantic"] > cycle_states[0]["coherence"]["semantic"]
        assert cycle_states[1]["adaptive_coherence"]["pattern_alignment"] > cycle_states[0]["adaptive_coherence"]["pattern_alignment"]
        assert cycle_states[1]["schema_validation"]["valid"]
        assert any(
            state["evolution_states"].get("meaning1") == EvolutionState.EMERGING
            for state in cycle_states[1:3]
        )
        
        # Phase 3: Structure Evolution
        assert cycle_states[2]["coherence"]["structural"] > cycle_states[0]["coherence"]["structural"]
        assert cycle_states[2]["adaptive_coherence"]["temporal_consistency"] > cycle_states[1]["adaptive_coherence"]["temporal_consistency"]
        assert cycle_states[2]["schema_validation"]["valid"]
        assert any(
            state["evolution_states"].get("structure1") == EvolutionState.EVOLVING
            for state in cycle_states[2:4]
        )
        
        # Phase 4: Meaning Enhancement
        assert cycle_states[3]["coherence"]["semantic"] > cycle_states[1]["coherence"]["semantic"]
        assert cycle_states[3]["coherence"]["total"] > cycle_states[0]["coherence"]["total"]
        assert cycle_states[3]["adaptive_coherence"]["overall_coherence"] > cycle_states[0]["adaptive_coherence"]["overall_coherence"]
        assert cycle_states[3]["adaptive_coherence"]["assessment_level"] in [CoherenceLevel.MODERATE, CoherenceLevel.HIGH]
        assert cycle_states[3]["schema_validation"]["valid"]
        assert any(
            state["evolution_states"].get("meaning1") == EvolutionState.STABLE
            for state in cycle_states[3:]
        )

    def _validate_emergence_progression(
        self,
        emergence_history: List[Dict[str, EmergenceType]]
    ) -> None:
        """Validate the progression of pattern emergence in the cycle."""
        # Should show natural → guided → potential → natural progression
        progression = [
            emergence_types.get("seed") for emergence_types in emergence_history
        ]
        
        assert EmergenceType.NATURAL in progression
        assert EmergenceType.GUIDED in progression
        assert EmergenceType.POTENTIAL in progression

    @pytest.mark.asyncio
    async def test_cycle_emergence(self, evolution_setup):
        """Test emergence of new patterns within the cycle."""
        pattern_core, coherence_flow, evolution_manager, emergence_flow, _, _, _, _, _ = evolution_setup
        
        # Initial pattern
        evidence = self._create_pattern_evidence(
            pattern_type="seed",
            confidence=0.6,
            suggestions={}
        )
        evolution_manager.track_pattern("seed", evidence)
        
        # Track emergent patterns through cycle
        emergence_history = []
        for _ in range(4):  # One complete cycle
            patterns = {
                "seed": {
                    "type": "seed",
                    "confidence": 0.6 + _ * 0.1
                }
            }
            
            emergence_types = emergence_flow.observe_emergence(
                patterns,
                evolution_manager.states
            )
            emergence_history.append(emergence_types)
            
        # Verify emergence progression
        self._validate_emergence_progression(emergence_history)
