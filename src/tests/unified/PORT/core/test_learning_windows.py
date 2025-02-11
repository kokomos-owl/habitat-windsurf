"""
Test suite for learning windows and density analysis.

This module tests the functionality of learning windows, including density metrics,
domain alignments, and field-wide density analysis.
"""

import unittest
from habitat_test.core.learning_windows import LearningWindowInterface

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
        self.assertIsNotNone(window_id)
        
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
        self.assertIsNotNone(window_id_2)
        
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
        
    def test_window_registration(self):
        """Test window registration with coherence validation."""
        # Test valid window
        window_data = {
            "score": 0.9500,
            "potential": 0.8500,
            "horizon": 0.9000,
            "channels": {
                "structural": {"strength": 0.9000, "sustainability": 0.8500},
                "semantic": {"strength": 0.8500, "sustainability": 0.9000}
            }
        }
        window_id = self.interface.register_window(window_data)
        self.assertIsNotNone(window_id)
        
        # Test invalid window (low score)
        invalid_window = {**window_data, "score": 0.7000}
        invalid_id = self.interface.register_window(invalid_window)
        self.assertIsNone(invalid_id)
        
        # Test missing required fields
        incomplete_window = {"score": 0.9500}
        incomplete_id = self.interface.register_window(incomplete_window)
        self.assertIsNone(incomplete_id)
        
    def test_window_retrieval(self):
        """Test window retrieval with coherence context."""
        window_data = {
            "score": 0.9500,
            "potential": 0.8500,
            "horizon": 0.9000,
            "channels": {
                "structural": {"strength": 0.9000, "sustainability": 0.8500},
                "semantic": {"strength": 0.8500, "sustainability": 0.9000}
            }
        }
        window_id = self.interface.register_window(window_data)
        
        # Get window with context
        window = self.interface.get_window(window_id)
        self.assertIsNotNone(window)
        self.assertIn("coherence_status", window)
        self.assertIn("patterns", window)
        
        # Verify coherence status
        status = window["coherence_status"]
        self.assertTrue(status["is_coherent"])
        self.assertEqual(status["structural_status"], "stable")
        self.assertEqual(status["semantic_status"], "stable")
        
        # Test non-existent window
        missing_window = self.interface.get_window("non-existent")
        self.assertIsNone(missing_window)

if __name__ == '__main__':
    unittest.main()
