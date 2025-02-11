"""Test suite for pattern detection during document ingestion.

This test suite validates the pattern detection system's ability to:
1. Detect patterns in real-time during document ingestion
2. Calculate accurate coherence metrics for new patterns
3. Track pattern relationships
4. Integrate with pattern evolution
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime

from habitat_test.core.pattern_detection import (
    PatternDetector,
    DetectionMode,
    PatternCandidate
)
from habitat_test.core.pattern_evolution import (
    PatternEvolutionManager,
    PatternType,
    DocumentStructureType,
    EvolutionState
)
from .test_pattern_evolution import MockPatternCore

@pytest.fixture
def evolution_manager():
    """Create pattern evolution manager for testing."""
    pattern_core = MockPatternCore()
    return PatternEvolutionManager(pattern_core=pattern_core)

@pytest.fixture
def pattern_detector(evolution_manager):
    """Create pattern detector for testing."""
    return PatternDetector(
        evolution_manager=evolution_manager,
        detection_mode=DetectionMode.HYBRID
    )

@pytest.fixture
def sample_document():
    """Create sample document content for testing."""
    return """
    # Pattern Analysis Report
    
    ## Overview
    This document contains several patterns that should be detected:
    
    1. Temporal Pattern:
       - Event A occurs every 2 hours
       - Event B follows 30 minutes later
       - Cycle repeats consistently
    
    2. Relational Pattern:
       - Entity X always connects to Entity Y
       - Entity Y influences Entity Z
       - Strong correlation observed
    
    3. System Pattern:
       - Components show emergent behavior
       - Feedback loops identified
       - Stable equilibrium maintained
    """

@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return {
        'source': 'test_suite',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0',
        'category': 'analysis'
    }

class TestPatternDetection:
    """Test pattern detection during document ingestion."""
    
    @pytest.mark.asyncio
    async def test_document_processing(self, pattern_detector, sample_document, sample_metadata):
        """Test basic document processing and pattern detection.
        
        This test verifies that:
        1. Document structure is correctly detected
        2. Patterns are identified in the content
        3. Pattern candidates have appropriate confidence scores
        4. Basic relationships are detected
        """
        candidates = await pattern_detector.process_document(
            sample_document,
            sample_metadata
        )
        
        assert len(candidates) > 0
        for candidate in candidates:
            assert isinstance(candidate, PatternCandidate)
            assert candidate.confidence > 0.5
            assert len(candidate.relationships) >= 0
            
    @pytest.mark.asyncio
    async def test_pattern_evolution_integration(self, pattern_detector, sample_document):
        """Test integration with pattern evolution system.
        
        This test verifies that:
        1. Detected patterns are tracked by evolution manager
        2. Evolution metrics are calculated correctly
        3. Pattern states are updated appropriately
        """
        candidates = await pattern_detector.process_document(sample_document)
        
        for candidate in candidates:
            # Check pattern evolution state
            evolution_result = await pattern_detector.evolution_manager.observe_evolution(
                candidate.pattern_id,
                pattern_detector._create_evolution_evidence(
                    candidate,
                    pattern_detector._calculate_coherence(
                        candidate,
                        DocumentStructureType.HIERARCHICAL,
                        0.8
                    )
                )
            )
            
            assert evolution_result['state'] in [EvolutionState.EMERGING, EvolutionState.STABLE]
            assert isinstance(evolution_result['timestamp'], str)
            
    def test_relationship_tracking(self, pattern_detector, sample_document):
        """Test pattern relationship tracking.
        
        This test verifies that:
        1. Pattern relationships are correctly identified
        2. Relationship graph is properly maintained
        3. Bidirectional relationships are tracked
        """
        candidates = []
        
        # Process document and collect candidates
        async def process():
            nonlocal candidates
            candidates = await pattern_detector.process_document(sample_document)
            
        import asyncio
        asyncio.run(process())
        
        # Verify relationships
        for candidate in candidates:
            if candidate.relationships:
                # Check that relationships are tracked
                assert candidate.pattern_id in pattern_detector.pattern_relationships
                
                # Check bidirectional relationships
                for related_id in candidate.relationships:
                    assert related_id in pattern_detector.pattern_relationships
                    assert candidate.pattern_id in pattern_detector.pattern_relationships[related_id]
                    
    @pytest.mark.asyncio
    async def test_detection_modes(self, evolution_manager):
        """Test different pattern detection modes.
        
        This test verifies that:
        1. Each detection mode operates correctly
        2. Different modes affect detection sensitivity
        3. Mode switching works properly
        """
        test_modes = [
            DetectionMode.PASSIVE,
            DetectionMode.ACTIVE,
            DetectionMode.HYBRID
        ]
        
        # Test content with basic patterns that should be detectable
        test_content = """
        The system shows regular behavior patterns.
        Component A interacts with process B.
        This forms a basic relationship between elements.
        """
        
        for mode in test_modes:
            detector = PatternDetector(
                evolution_manager=evolution_manager,
                detection_mode=mode
            )
            
            candidates = await detector.process_document(test_content)
            
            # Verify mode-specific behavior
            if mode == DetectionMode.PASSIVE:
                # Should only detect obvious patterns
                assert len(candidates) <= 2
            elif mode == DetectionMode.ACTIVE:
                # Should detect more potential patterns
                assert len(candidates) >= 1
                # Verify increased sensitivity
                assert any(c.confidence < 0.8 for c in candidates)
            else:  # HYBRID
                # Should balance detection
                assert len(candidates) >= 1
                # Should maintain higher confidence threshold
                assert all(c.confidence >= 0.7 for c in candidates)

    def test_coherence_adherence(self, pattern_detector, sample_document):
        """Test adherence to coherence principles.
        
        This test verifies that pattern detection naturally maintains
        coherence without requiring external enforcement:
        1. Patterns emerge with appropriate context
        2. Detection and coherence share state naturally
        3. Relationship tracking maintains system coherence
        4. Pattern confidence aligns with coherence measures
        """
        # Process document
        async def process():
            return await pattern_detector.process_document(sample_document)
            
        candidates = asyncio.run(process())
        
        # Verify temporal coherence adherence
        temporal_patterns = [c for c in candidates 
                           if c.pattern_type == PatternType.QUANTITATIVE]
        for pattern in temporal_patterns:
            # Temporal patterns should have temporal context
            assert "temporal_window" in pattern.metadata
            # Confidence should reflect temporal stability
            assert pattern.confidence >= 0.7
            
        # Verify system coherence adherence
        relational_patterns = [c for c in candidates 
                             if c.pattern_type == PatternType.RELATIONAL]
        for pattern in relational_patterns:
            # Relational patterns should have system state
            assert "system_state" in pattern.metadata
            # Should maintain relationship coherence
            if pattern.relationships:
                for related_id in pattern.relationships:
                    assert related_id in pattern_detector.pattern_relationships
                    
        # Verify knowledge coherence adherence
        system_patterns = [c for c in candidates 
                         if c.pattern_type == PatternType.SYSTEMIC]
        for pattern in system_patterns:
            # System patterns should have knowledge context
            assert "knowledge_context" in pattern.metadata
            # Higher confidence for established system patterns
            assert pattern.confidence >= 0.8
            
    def test_natural_emergence(self, pattern_detector):
        """Test natural pattern emergence through coherence.
        
        This test verifies that patterns emerge naturally through
        the interaction of detection and coherence, without forced
        classification or arbitrary thresholds.
        """
        # Test document with emerging patterns
        emerging_content = """
        Initial observation shows A occurring.
        Later, B follows A consistently.
        This forms a potential temporal relationship.
        
        Further observation reveals:
        - A and B form a stable cycle
        - C emerges as an influencing factor
        - The system maintains equilibrium
        """
        
        async def process():
            # First pass - initial detection
            candidates1 = await pattern_detector.process_document(emerging_content)
            
            # Update context to simulate time passage
            pattern_detector.context.temporal_window.append(datetime.now())
            
            # Second pass - pattern evolution
            candidates2 = await pattern_detector.process_document(emerging_content)
            
            return candidates1, candidates2
            
        candidates1, candidates2 = asyncio.run(process())
        
        # Verify natural emergence
        assert len(candidates2) >= len(candidates1)
        
        # Check confidence evolution
        for c2 in candidates2:
            matching_c1 = next((c for c in candidates1 if c.pattern_id == c2.pattern_id), None)
            if matching_c1:
                # Confidence should improve with context
                assert c2.confidence >= matching_c1.confidence
                
    def test_coherence_flow(self, pattern_detector):
        """Test coherence flow through detection contexts.
        
        This test verifies that coherence flows naturally between
        detection, context, and pattern evolution without requiring
        explicit synchronization.
        """
        # Test progressive pattern development
        stages = [
            # Stage 1: Initial observation
            """Entity A shows regular behavior.""",
            
            # Stage 2: Relationship emergence
            """
            Entity A shows regular behavior.
            Entity B responds to A's changes.
            """,
            
            # Stage 3: System pattern emergence
            """
            Entity A shows regular behavior.
            Entity B responds to A's changes.
            The A-B interaction creates feedback loops.
            System reaches equilibrium state.
            """
        ]
        
        patterns_by_stage = []
        
        async def process_stages():
            for stage in stages:
                candidates = await pattern_detector.process_document(stage)
                patterns_by_stage.append(candidates)
                
        asyncio.run(process_stages())
        
        # Verify coherence flow
        for i in range(1, len(patterns_by_stage)):
            current = patterns_by_stage[i]
            previous = patterns_by_stage[i-1]
            
            # Patterns should maintain or increase in complexity
            assert len(current) >= len(previous)
            
            # System understanding should deepen
            system_patterns = [p for p in current 
                             if p.pattern_type == PatternType.SYSTEMIC]
            prev_system_patterns = [p for p in previous 
                                  if p.pattern_type == PatternType.SYSTEMIC]
            
            if system_patterns and prev_system_patterns:
                # System coherence should strengthen
                current_confidence = sum(p.confidence for p in system_patterns)
                prev_confidence = sum(p.confidence for p in prev_system_patterns)
                assert current_confidence >= prev_confidence

    def test_coherence_indicates_stability(self, pattern_detector):
        """Test if coherence-adherence indicates stability.
        
        This test verifies that when patterns maintain coherence
        at interfaces, they demonstrate stability through:
        1. Maintaining coherence above threshold (0.7)
        2. Participating in the feedback loop stably
        3. Allowing for natural evolution
        """
        test_content = """
        Component A exhibits consistent behavior.
        Component B responds predictably to A.
        The relationship between A and B remains stable.
        """
        
        async def process():
            # First pass to establish patterns
            candidates = await pattern_detector.process_document(test_content)
            
            # Track stability metrics
            stability_metrics = []
            
            # Multiple passes to verify stability
            for _ in range(3):
                new_candidates = await pattern_detector.process_document(test_content)
                
                # Compare pattern metrics
                for new_c in new_candidates:
                    old_c = next((c for c in candidates if c.pattern_id == new_c.pattern_id), None)
                    if old_c:
                        metrics = {
                            'pattern_id': new_c.pattern_id,
                            'temporal_coherence': pattern_detector._calculate_temporal_coherence(new_c),
                            'system_coherence': pattern_detector._calculate_system_coherence(new_c),
                            'knowledge_coherence': pattern_detector._calculate_knowledge_coherence(new_c),
                            'stability': pattern_detector._calculate_stability(new_c),
                            'confidence': new_c.confidence
                        }
                        stability_metrics.append(metrics)
                
                candidates = new_candidates
            
            return stability_metrics
            
        metrics = asyncio.run(process())
        
        # Verify stability through coherence-adherence
        for m in metrics:
            # All coherence measures should be above threshold
            assert m['temporal_coherence'] >= pattern_detector.context.coherence_threshold
            assert m['system_coherence'] >= pattern_detector.context.coherence_threshold
            assert m['knowledge_coherence'] >= pattern_detector.context.coherence_threshold
            
            # Stability should be high but not perfect (allowing evolution)
            assert 0.8 <= m['stability'] <= 0.95

    def test_surface_tension_prevents_ripples(self, pattern_detector):
        """Test that surface tension (diversity penalty) prevents structure-meaning ripples.
        
        This test verifies that:
        1. Pattern jumps require energy cost (diversity penalty)
        2. Surface tension contains local changes
        3. System maintains coherence during evolution
        """
        # Test content with potential ripple points
        initial_content = """
        System A maintains stable equilibrium.
        Component B responds to changes in A.
        """
        
        evolved_content = """
        System A shows complex dynamics.  # Evolution point 1
        Component B adapts to A's changes.  # Evolution point 2
        The relationship maintains coherence.  # Stability check
        """
        
        async def process():
            # Get initial state
            initial_patterns = await pattern_detector.process_document(initial_content)
            
            # Track evolution metrics
            evolution_metrics = []
            
            # Multiple passes through evolved content
            for _ in range(3):
                new_patterns = await pattern_detector.process_document(evolved_content)
                
                # Calculate metrics for each pattern
                for new_p in new_patterns:
                    # Find matching initial pattern if exists
                    old_p = next((p for p in initial_patterns 
                                if p.pattern_id == new_p.pattern_id), None)
                    
                    # Calculate evolution metrics
                    metrics = pattern_detector._calculate_coherence(
                        new_p,
                        DocumentStructureType.HIERARCHICAL,
                        0.9
                    )
                    
                    # Calculate energy cost of evolution
                    if old_p:
                        evolution_cost = abs(metrics.coherence - 
                                          pattern_detector._calculate_coherence(
                                              old_p,
                                              DocumentStructureType.HIERARCHICAL,
                                              0.9
                                          ).coherence)
                    else:
                        evolution_cost = 0.0
                        
                    evolution_metrics.append({
                        'pattern_id': new_p.pattern_id,
                        'coherence': metrics.coherence,
                        'temporal_coherence': metrics.temporal_coherence,
                        'system_coherence': metrics.system_coherence,
                        'knowledge_coherence': metrics.knowledge_coherence,
                        'evolution_cost': evolution_cost
                    })
            
            return evolution_metrics
            
        metrics = asyncio.run(process())
        
        # Verify surface tension effects
        for m in metrics:
            # Evolution should have an energy cost
            if m['evolution_cost'] > 0:
                # Higher evolution cost should maintain higher coherence
                assert m['coherence'] >= 0.8, "High evolution cost should maintain coherence"
                
            # Local changes shouldn't ripple through all coherence measures
            coherence_measures = [
                m['temporal_coherence'],
                m['system_coherence'],
                m['knowledge_coherence']
            ]
            # At least one measure should remain highly stable
            assert any(c >= 0.9 for c in coherence_measures), "Surface tension should maintain stability in some dimensions"
            
            # Verify no destructive ripples
            assert all(c >= 0.7 for c in coherence_measures), "Changes shouldn't create destructive ripples"
