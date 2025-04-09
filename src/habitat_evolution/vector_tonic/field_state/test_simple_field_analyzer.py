# tests/vector_tonic/field_state/test_simple_field_analyzer.py
import unittest
import pandas as pd
import numpy as np
from src.habitat_evolution.vector_tonic.field_state.simple_field_analyzer import SimpleFieldStateAnalyzer

class TestSimpleFieldAnalyzer(unittest.TestCase):
    
    def test_analyze_time_series_with_warming_trend(self):
        # Create test data with warming trend
        dates = pd.date_range(start='2020-01-01', periods=12, freq='MS')
        temps = np.linspace(10, 15, 12)  # 5 degree warming over 12 months
        data = pd.DataFrame({'date': dates, 'temperature': temps})
        
        # Analyze with field analyzer
        analyzer = SimpleFieldStateAnalyzer()
        results = analyzer.analyze_time_series(data)
        
        # Verify results
        self.assertIn("patterns", results)
        self.assertGreater(len(results["patterns"]), 0)
        self.assertEqual(results["patterns"][0]["type"], "warming_trend")
        self.assertAlmostEqual(results["patterns"][0]["magnitude"], 0.45, places=2)
        
    def test_analyze_time_series_with_cooling_trend(self):
        # Create test data with cooling trend
        dates = pd.date_range(start='2020-01-01', periods=12, freq='MS')
        temps = np.linspace(15, 10, 12)  # 5 degree cooling over 12 months
        data = pd.DataFrame({'date': dates, 'temperature': temps})
        
        # Analyze with field analyzer
        analyzer = SimpleFieldStateAnalyzer()
        results = analyzer.analyze_time_series(data)
        
        # Verify results
        self.assertIn("patterns", results)
        self.assertGreater(len(results["patterns"]), 0)
        self.assertEqual(results["patterns"][0]["type"], "cooling_trend")
        self.assertAlmostEqual(results["patterns"][0]["magnitude"], 0.45, places=2)