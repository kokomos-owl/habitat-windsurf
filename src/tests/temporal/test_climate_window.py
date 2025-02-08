"""Integration tests for climate pattern windows."""
import pytest
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional
from src.core.temporal.adaptive_window_v23 import (
    AdaptiveWindow,
    WindowState,
    WindowStateManager
)

@dataclass
class ClimatePattern:
    """Climate-specific pattern for testing."""
    pattern_id: str
    pattern_type: str  # drought, flood, fire
    intensity: float   # 0-1 scale
    duration: timedelta
    location: str
    timestamp: datetime = None
    
    def __post_init__(self):
        self.timestamp = self.timestamp or datetime.now()

class ClimatePatternGenerator:
    """Generates climate-specific test patterns."""
    def __init__(self):
        self.base_time = datetime.now()
        # Historical baselines from Martha's Vineyard data
        self.drought_baseline = 0.085  # 8.5%
        self.drought_mid = 0.13       # 13% by mid-century
        self.drought_late = 0.26      # 26% by late-century
        self.precip_increase = 0.55   # 55% increase in heavy events
        
    def generate_drought_sequence(
        self,
        duration_days: int,
        scenario: str = "historical"
    ) -> List[ClimatePattern]:
        """Generate a sequence of drought patterns."""
        patterns = []
        base_intensity = {
            "historical": self.drought_baseline,
            "mid": self.drought_mid,
            "late": self.drought_late
        }[scenario]
        
        for day in range(duration_days):
            # Add daily variation
            variation = (day % 7) * 0.02  # Weekly cycle
            intensity = base_intensity + variation
            
            pattern = ClimatePattern(
                pattern_id=f"drought_{day}",
                pattern_type="drought",
                intensity=intensity,
                duration=timedelta(days=1),
                location="MV_WEST",
                timestamp=self.base_time + timedelta(days=day)
            )
            patterns.append(pattern)
            
        return patterns
        
    def generate_compound_sequence(
        self,
        duration_days: int
    ) -> List[ClimatePattern]:
        """Generate compound event patterns (drought + fire risk)."""
        patterns = []
        for day in range(duration_days):
            # Drought intensity increases fire risk
            drought_intensity = self.drought_baseline + (day * 0.01)
            fire_intensity = drought_intensity * 1.5  # Fire risk amplification
            
            # Drought pattern
            patterns.append(ClimatePattern(
                pattern_id=f"drought_fire_{day}_d",
                pattern_type="drought",
                intensity=drought_intensity,
                duration=timedelta(days=1),
                location="MV_WEST",
                timestamp=self.base_time + timedelta(days=day)
            ))
            
            # Fire risk pattern
            patterns.append(ClimatePattern(
                pattern_id=f"drought_fire_{day}_f",
                pattern_type="fire",
                intensity=fire_intensity,
                duration=timedelta(days=1),
                location="MV_WEST",
                timestamp=self.base_time + timedelta(days=day)
            ))
            
        return patterns

@pytest.fixture
def climate_generator():
    """Provide climate pattern generator."""
    return ClimatePatternGenerator()

class TestClimateWindowIntegration:
    @pytest.mark.asyncio
    async def test_drought_intensification(
        self,
        climate_generator: ClimatePatternGenerator
    ):
        """Test window adaptation to drought intensification."""
        window = AdaptiveWindow(
            window_id="drought_window",
            opening_threshold=0.3,  # Open on moderate drought
            closing_threshold=0.1
        )
        
        # Generate historical to mid-century transition
        historical_patterns = climate_generator.generate_drought_sequence(
            duration_days=10,
            scenario="historical"
        )
        mid_patterns = climate_generator.generate_drought_sequence(
            duration_days=10,
            scenario="mid"
        )
        
        # Process patterns and track state transitions
        states = []
        densities = []
        
        for pattern in historical_patterns + mid_patterns:
            await window.process_pattern(pattern)
            states.append(window.state)
            densities.append(window.density_metrics.current_density)
            
        # Verify appropriate state transitions
        assert states.count(WindowState.OPENING) >= 1
        assert states.count(WindowState.ACTIVE) >= 1
        
        # Verify density increase detection
        assert densities[-1] > densities[0]
        
    @pytest.mark.asyncio
    async def test_compound_event_handling(
        self,
        climate_generator: ClimatePatternGenerator
    ):
        """Test window handling of compound climate events."""
        manager = WindowStateManager()
        
        # Create windows for different pattern types
        drought_window = await manager.create_window(
            window_id="drought_window",
            opening_threshold=0.3
        )
        fire_window = await manager.create_window(
            window_id="fire_window",
            opening_threshold=0.4
        )
        
        # Generate compound event patterns
        patterns = climate_generator.generate_compound_sequence(
            duration_days=15
        )
        
        # Track window states and IO notifications
        notifications = []
        
        for pattern in patterns:
            # Route patterns to appropriate windows
            window_id = (
                "drought_window"
                if pattern.pattern_type == "drought"
                else "fire_window"
            )
            
            success = await manager.process_pattern(window_id, pattern)
            assert success
            
            # Collect IO notifications
            while not manager.io_queue.empty():
                notification = await manager.io_queue.get()
                notifications.append(notification)
        
        # Verify compound event detection
        drought_notifications = [
            n for n in notifications
            if n['window_id'] == 'drought_window'
        ]
        fire_notifications = [
            n for n in notifications
            if n['window_id'] == 'fire_window'
        ]
        
        # Both windows should have opened
        assert any(
            n['state'] == 'opening'
            for n in drought_notifications
        )
        assert any(
            n['state'] == 'opening'
            for n in fire_notifications
        )
        
        # Fire window should open after drought window
        drought_opening = next(
            n['timestamp']
            for n in drought_notifications
            if n['state'] == 'opening'
        )
        fire_opening = next(
            n['timestamp']
            for n in fire_notifications
            if n['state'] == 'opening'
        )
        assert fire_opening > drought_opening
        
    @pytest.mark.asyncio
    async def test_climate_pattern_density(
        self,
        climate_generator: ClimatePatternGenerator
    ):
        """Test pattern density calculations for climate patterns."""
        window = AdaptiveWindow(
            window_id="density_test",
            opening_threshold=0.4,
            closing_threshold=0.2
        )
        
        # Generate mixed pattern sequence
        drought_patterns = climate_generator.generate_drought_sequence(
            duration_days=5,
            scenario="historical"
        )
        compound_patterns = climate_generator.generate_compound_sequence(
            duration_days=5
        )
        
        # Process patterns and track density metrics
        densities = []
        accelerations = []
        
        for pattern in drought_patterns + compound_patterns:
            await window.process_pattern(pattern)
            densities.append(window.density_metrics.current_density)
            accelerations.append(window.density_metrics.acceleration)
            
        # Verify density increases with compound events
        drought_avg = sum(densities[:5]) / 5
        compound_avg = sum(densities[-5:]) / 5
        assert compound_avg > drought_avg
        
        # Verify acceleration during transition
        transition_point = len(drought_patterns)
        assert accelerations[transition_point] > 0
