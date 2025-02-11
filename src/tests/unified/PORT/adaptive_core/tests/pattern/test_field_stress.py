"""Stress tests for field-driven pattern evolution."""
import pytest
import asyncio
import numpy as np
from typing import List, Dict, Any

from .mocks.field_mock import (
    MockFieldState,
    MockPatternGenerator,
    create_stress_test_scenario
)
from src.core.flow.gradient.controller import GradientFlowController
from src.core.pattern_evolution import FieldDrivenPatternManager

class TestFieldStress:
    """Stress tests for field-driven pattern system."""
    
    @pytest.fixture
    def large_field_scenario(self):
        """Create a large field test scenario."""
        return create_stress_test_scenario(
            size=(1000, 1000),
            num_patterns=1000,
            turbulence_scale=2.0
        )
    
    async def test_large_scale_evolution(self, large_field_scenario):
        """Test pattern evolution with large number of patterns."""
        field_state, patterns = large_field_scenario
        pattern_manager = FieldDrivenPatternManager()
        
        # Evolve patterns in batches
        batch_size = 100
        for i in range(0, len(patterns), batch_size):
            batch = patterns[i:i + batch_size]
            evolved = await pattern_manager.evolve_patterns(
                field_state=field_state,
                patterns=batch
            )
            
            # Verify evolution results
            assert len(evolved) == len(batch)
            for pattern in evolved:
                assert 0 <= pattern['coherence'] <= 1
                assert pattern['state'] in ['ACTIVE', 'DISSIPATED']
    
    async def test_turbulent_field_stability(self, large_field_scenario):
        """Test pattern stability in highly turbulent field."""
        field_state, patterns = large_field_scenario
        pattern_manager = FieldDrivenPatternManager()
        
        # Create clusters of coherent and incoherent patterns
        coherent_patterns = MockPatternGenerator.create_pattern_cluster(
            num_patterns=50,
            base_coherence=0.8
        )
        incoherent_patterns = MockPatternGenerator.create_pattern_cluster(
            num_patterns=50,
            base_coherence=0.2
        )
        
        # Evolve both clusters
        evolved_coherent = await pattern_manager.evolve_patterns(
            field_state=field_state,
            patterns=coherent_patterns
        )
        evolved_incoherent = await pattern_manager.evolve_patterns(
            field_state=field_state,
            patterns=incoherent_patterns
        )
        
        # Verify stability characteristics
        coherent_active = sum(1 for p in evolved_coherent 
                            if p['state'] == 'ACTIVE')
        incoherent_active = sum(1 for p in evolved_incoherent 
                              if p['state'] == 'ACTIVE')
        
        # Coherent patterns should be more stable
        assert coherent_active > incoherent_active
    
    async def test_concurrent_evolution(self, large_field_scenario):
        """Test concurrent pattern evolution."""
        field_state, patterns = large_field_scenario
        pattern_manager = FieldDrivenPatternManager()
        
        # Split patterns into concurrent tasks
        num_tasks = 10
        batch_size = len(patterns) // num_tasks
        tasks = []
        
        for i in range(0, len(patterns), batch_size):
            batch = patterns[i:i + batch_size]
            task = asyncio.create_task(
                pattern_manager.evolve_patterns(
                    field_state=field_state,
                    patterns=batch
                )
            )
            tasks.append(task)
        
        # Run evolution concurrently
        results = await asyncio.gather(*tasks)
        
        # Verify results
        all_evolved = [p for batch in results for p in batch]
        assert len(all_evolved) == len(patterns)
        
        # Check for evolution consistency
        coherence_values = [p['coherence'] for p in all_evolved]
        assert 0 <= np.mean(coherence_values) <= 1
        assert 0 <= np.std(coherence_values) <= 0.5  # Reasonable spread
    
    @pytest.mark.parametrize('field_size,num_patterns', [
        ((100, 100), 100),
        ((500, 500), 500),
        ((1000, 1000), 1000),
    ])
    async def test_scalability(self, field_size, num_patterns):
        """Test system scalability with different sizes."""
        field_state, patterns = create_stress_test_scenario(
            size=field_size,
            num_patterns=num_patterns
        )
        pattern_manager = FieldDrivenPatternManager()
        
        # Measure evolution time
        start_time = asyncio.get_event_loop().time()
        evolved = await pattern_manager.evolve_patterns(
            field_state=field_state,
            patterns=patterns
        )
        end_time = asyncio.get_event_loop().time()
        
        # Verify performance
        duration = end_time - start_time
        patterns_per_second = num_patterns / duration
        
        # Log performance metrics
        print(f"Field size: {field_size}")
        print(f"Patterns: {num_patterns}")
        print(f"Duration: {duration:.2f}s")
        print(f"Patterns/second: {patterns_per_second:.2f}")
        
        # Basic scaling check (should be roughly linear)
        assert patterns_per_second > 10  # Minimum acceptable performance
