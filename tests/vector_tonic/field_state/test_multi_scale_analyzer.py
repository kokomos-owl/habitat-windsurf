import unittest
import pandas as pd
import numpy as np
from src.habitat_evolution.vector_tonic.field_state.multi_scale_analyzer import MultiScaleAnalyzer

class TestMultiScaleAnalyzer(unittest.TestCase):
    
    def test_analyze_with_consistent_warming(self):
        # Create test data with consistent warming trend
        dates = pd.date_range(start='2020-01-01', periods=120, freq='D')
        base = np.linspace(10, 20, 120)  # 10 degree warming over 120 days
        # Add seasonal component
        seasonal = 3 * np.sin(2 * np.pi * np.arange(120) / 30)
        temps = base + seasonal
        data = pd.DataFrame({'date': dates, 'temperature': temps})
        
        # Analyze with multi-scale analyzer
        analyzer = MultiScaleAnalyzer(scales=[10, 30, 60])
        results = analyzer.analyze(data)
        
        # Verify results
        self.assertIn("scale_results", results)
        self.assertIn("cross_scale_patterns", results)
        self.assertGreater(len(results["cross_scale_patterns"]), 0)
        self.assertEqual(results["cross_scale_patterns"][0]["type"], "consistent_warming")
        
    def test_analyze_with_insufficient_data(self):
        # Create test data with only a few points
        dates = pd.date_range(start='2020-01-01', periods=3, freq='D')
        temps = [10, 11, 12]
        data = pd.DataFrame({'date': dates, 'temperature': temps})
        
        # Analyze with multi-scale analyzer
        analyzer = MultiScaleAnalyzer(scales=[10, 30, 60])
        results = analyzer.analyze(data)
        
        # Verify results
        self.assertIn("scale_results", results)
        self.assertIn("cross_scale_patterns", results)
        # Should have no cross-scale patterns due to insufficient data
        self.assertEqual(len(results["cross_scale_patterns"]), 0)
