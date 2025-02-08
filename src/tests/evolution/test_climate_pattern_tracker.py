"""Tests for climate risk pattern tracking."""

import pytest
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Any

from src.core.evolution.domain_pattern_tracker import DomainPatternTracker
from src.core.types import (
    DensityMetrics,
    PatternEvolutionMetrics,
    PatternEvidence,
    TemporalContext,
    LearningWindow
)

class ClimatePatternGenerator:
    """Generates climate risk pattern evidence."""
    
    def __init__(self):
        self.base_time = datetime.now()
        
        # Historical baselines from Martha's Vineyard data
        self.drought_baseline = 0.085  # 8.5%
        self.drought_mid = 0.13       # 13% by mid-century
        self.drought_late = 0.26      # 26% by late-century
        
        self.precip_increase = 0.55   # 55% increase in heavy events
        self.fire_increase = 0.94     # 94% increase in danger days
    
    def generate_drought_evidence(
        self,
        time_offset: int,
        scenario: str = "historical"
    ) -> PatternEvidence:
        """Generate drought pattern evidence."""
        # Calculate drought probability based on scenario
        if scenario == "mid":
            target = self.drought_mid
        elif scenario == "late":
            target = self.drought_late
        else:
            target = self.drought_baseline
            
        progress = min(1.0, time_offset / 365)  # Year of evolution
        current_prob = self.drought_baseline + (target - self.drought_baseline) * progress
        
        # Calculate related metrics
        fire_danger = 1.0 + (current_prob * 2)  # Fire danger increases with drought
        precip_intensity = 1.0 + (progress * self.precip_increase)
        
        density = DensityMetrics(
            global_density=current_prob,
            local_density=current_prob * 1.2,
            cross_domain_strength=0.7 + (progress * 0.2),
            interface_recognition=0.8,
            viscosity=0.6 + (progress * 0.3)
        )
        
        evolution = PatternEvolutionMetrics(
            gradient=progress * 0.5,
            interface_strength=0.7 + (progress * 0.2),
            stability=max(0.5, 1.0 - (progress * 0.3)),
            emergence_rate=0.1 + (progress * 0.4),
            coherence_level=0.8
        )
        
        return PatternEvidence(
            evidence_id=f"drought_{time_offset}",
            timestamp=self.base_time + timedelta(days=time_offset),
            pattern_type="drought",
            source_data={
                "probability": current_prob,
                "fire_danger": fire_danger,
                "precip_intensity": precip_intensity
            },
            temporal_context=TemporalContext(
                start_time=self.base_time,
                learning_window=LearningWindow(
                    window_id="climate_risk",
                    start_time=self.base_time,
                    patterns=["drought", "fire", "precipitation"],
                    density_metrics=density,
                    coherence_level=0.8,
                    viscosity_gradient=0.4
                )
            ),
            uncertainty_metrics=None,
            density_metrics=density,
            evolution_metrics=evolution,
            stability_score=0.8,
            emergence_rate=0.1
        )

@pytest.fixture
def climate_tracker():
    """Create domain pattern tracker for climate risk."""
    return DomainPatternTracker(
        domain_name="climate_risk",
        dimensions=4,
        similarity_threshold=0.85,
        evolution_rate=0.05
    )

@pytest.fixture
def pattern_generator():
    """Create climate pattern generator."""
    return ClimatePatternGenerator()

@pytest.mark.asyncio
async def test_drought_pattern_evolution(
    climate_tracker: DomainPatternTracker,
    pattern_generator: ClimatePatternGenerator
):
    """Test evolution of drought patterns."""
    # Generate sequence of drought evidence
    time_steps = list(range(0, 365, 30))  # Monthly steps
    patterns = []
    states = []
    
    # Track pattern evolution through mid-century scenario
    for time_offset in time_steps:
        evidence = pattern_generator.generate_drought_evidence(
            time_offset,
            scenario="mid"
        )
        
        new_patterns, new_states = await climate_tracker.process_evidence(
            evidence,
            evidence.temporal_context
        )
        patterns.extend(new_patterns)
        states.extend(new_states)
    
    # Verify pattern emergence
    assert len(patterns) > 0, "No patterns emerged"
    
    # Analyze pattern relationships
    relationships = climate_tracker.analyze_pattern_relationships("drought")
    
    assert relationships["stability_score"] > 0.7, \
        "Pattern evolution not stable enough"
    assert relationships["mean_confidence"] > 0.7, \
        "Low pattern confidence"
    
    # Verify evolution matches projections
    final_patterns = climate_tracker.get_domain_patterns(
        pattern_type="drought",
        min_confidence=0.7
    )
    assert len(final_patterns) > 0, "No confident patterns found"
    
    # Check final state
    final_pattern, final_state = final_patterns[-1]
    source_data = final_pattern.elements[-1].source_data
    assert abs(source_data["probability"] - pattern_generator.drought_mid) < 0.02, \
        "Pattern didn't evolve to expected state"

@pytest.mark.asyncio
async def test_compound_pattern_detection(
    climate_tracker: DomainPatternTracker,
    pattern_generator: ClimatePatternGenerator
):
    """Test detection of compound patterns."""
    # Generate drought pattern sequence
    for time_offset in range(0, 180, 30):  # 6 months
        evidence = pattern_generator.generate_drought_evidence(
            time_offset,
            scenario="late"  # Use late century scenario for stronger effects
        )
        
        await climate_tracker.process_evidence(
            evidence,
            evidence.temporal_context
        )
    
    # Analyze evolved patterns
    drought_patterns = climate_tracker.get_domain_patterns("drought")
    assert len(drought_patterns) > 0, "No drought patterns found"
    
    # Find related patterns
    primary_pattern = drought_patterns[0][0]
    related = climate_tracker.find_related_patterns(
        primary_pattern.pattern_id,
        min_similarity=0.8
    )
    
    # Verify compound relationships
    assert len(related) > 0, "No related patterns found"
    
    # Check relationship strengths
    relationships = [r[1] for r in related]
    assert np.mean(relationships) > 0.8, \
        "Weak pattern relationships"

@pytest.mark.asyncio
async def test_pattern_stability_metrics(
    climate_tracker: DomainPatternTracker,
    pattern_generator: ClimatePatternGenerator
):
    """Test stability metrics during pattern evolution."""
    # Generate two scenarios of evidence
    scenarios = ["historical", "mid"]
    
    for scenario in scenarios:
        for time_offset in range(0, 90, 15):  # 3 months
            evidence = pattern_generator.generate_drought_evidence(
                time_offset,
                scenario=scenario
            )
            
            await climate_tracker.process_evidence(
                evidence,
                evidence.temporal_context
            )
    
    # Analyze pattern stability
    metrics = climate_tracker.analyze_pattern_relationships("drought")
    
    # Verify stability properties
    assert metrics["mean_transition"] < 0.2, \
        "Pattern transitions too large"
    assert metrics["std_transition"] < 0.1, \
        "Pattern evolution too volatile"
    assert metrics["evolution_rate"] < 0.3, \
        "Pattern evolution too rapid"
