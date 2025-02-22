import unittest
from datetime import datetime
from habitat_evolution.adaptive_core.dimensional_context import (
    DimensionalContext, DimensionType, WindowState
)

class TestDimensionalContext(unittest.TestCase):
    def setUp(self):
        self.context = DimensionalContext()

    def test_coastal_erosion_pattern(self):
        """Test pattern evolution for coastal erosion case."""
        # Initial spatial observation
        observation1 = {
            "location": "MarthasVineyard",
            "boundary": "coastline",
            "spatial_metric": 0.5,
            "observation_type": "coastal_erosion"
        }
        self.context.observe_pattern(observation1)
        
        # Verify initial dimension detection
        active_dims = self.context.get_active_dimensions()
        self.assertIn(DimensionType.SPATIAL, active_dims)
        
        # Temporal evolution
        observation2 = {
            "location": "MarthasVineyard",
            "timestamp": datetime.now(),
            "frequency": "monthly",
            "erosion_rate": 0.8,
            "temporal_metric": 0.7,
            "spatial_metric": 0.6
        }
        self.context.observe_pattern(observation2)
        
        # Systemic coupling
        observation3 = {
            "location": "MarthasVineyard",
            "ecosystem": "coastal",
            "interaction": "erosion_habitat",
            "systemic_metric": 0.9,
            "spatial_metric": 0.7
        }
        self.context.observe_pattern(observation3)
        
        # Check evolution summary
        summary = self.context.get_evolution_summary()
        self.assertTrue(len(summary["active_dimensions"]) >= 2)
        self.assertGreater(summary["transcendence_scores"]["spatial"], 0)

    def test_storm_surge_pattern(self):
        """Test pattern evolution for storm surge case."""
        # Initial temporal observation
        observation1 = {
            "timestamp": datetime.now(),
            "frequency": "storm_surge",
            "temporal_metric": 0.6,
            "location": "MarthasVineyard"
        }
        self.context.observe_pattern(observation1)
        
        # Spatial transcendence
        observation2 = {
            "location": "MarthasVineyard",
            "boundary": "extended_coastal",
            "spatial_metric": 0.8,
            "temporal_metric": 0.7
        }
        self.context.observe_pattern(observation2)
        
        # System coupling
        observation3 = {
            "ecosystem": "coastal_marine",
            "interaction": "surge_adaptation",
            "systemic_metric": 0.9,
            "temporal_metric": 0.8
        }
        self.context.observe_pattern(observation3)
        
        # Verify pattern evolution
        summary = self.context.get_evolution_summary()
        self.assertGreater(
            summary["transcendence_scores"]["temporal"],
            summary["transcendence_scores"]["reference"]
        )

    def test_ecosystem_adaptation_pattern(self):
        """Test pattern evolution for ecosystem adaptation case."""
        # Initial systemic observation
        observation1 = {
            "ecosystem": "coastal",
            "interaction": "species_migration",
            "systemic_metric": 0.5
        }
        self.context.observe_pattern(observation1)
        
        # Temporal window
        observation2 = {
            "ecosystem": "coastal",
            "timestamp": datetime.now(),
            "frequency": "seasonal",
            "temporal_metric": 0.7,
            "systemic_metric": 0.6
        }
        self.context.observe_pattern(observation2)
        
        # Spatial boundary blur
        observation3 = {
            "ecosystem": "coastal",
            "boundary": "dynamic",
            "spatial_metric": 0.8,
            "systemic_metric": 0.7
        }
        self.context.observe_pattern(observation3)
        
        # Verify evolution
        summary = self.context.get_evolution_summary()
        active_dims = summary["active_dimensions"]
        self.assertIn(DimensionType.SYSTEMIC, active_dims)
        self.assertGreater(len(active_dims), 1)

    def test_dimension_weights_evolution(self):
        """Test how dimension weights evolve with observations."""
        initial_weights = dict(self.context.dimension_weights)
        
        # Series of observations that should strengthen systemic dimension
        for _ in range(3):
            self.context.observe_pattern({
                "ecosystem": "coastal",
                "interaction": "species_migration",
                "systemic_metric": 0.8
            })
        
        final_weights = dict(self.context.dimension_weights)
        self.assertGreater(
            final_weights[DimensionType.SYSTEMIC],
            initial_weights[DimensionType.SYSTEMIC]
        )

if __name__ == '__main__':
    unittest.main()
