"""
Test script for the Field State Observer.

This script demonstrates how the Field State Observer detects natural resonances
between semantic patterns from climate risk documents and statistical patterns
from time series analysis without forcing outcomes.
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from src.habitat_evolution.vector_tonic.field_state.field_state_observer import FieldStateObserver
from src.habitat_evolution.vector_tonic.field_state.simple_field_analyzer import SimpleFieldStateAnalyzer

def load_semantic_patterns(file_path):
    """Load semantic patterns from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading semantic patterns: {e}")
        return {"patterns": []}

def generate_statistical_patterns():
    """Generate statistical patterns from time series data."""
    # Use SimpleFieldStateAnalyzer to generate basic patterns
    analyzer = SimpleFieldStateAnalyzer()
    
    # Create synthetic time series data
    import numpy as np
    time_points = 120  # 10 years of monthly data
    time = np.arange(time_points)
    
    # Create a warming trend with seasonal variations and some noise
    trend = 0.02 * time  # warming trend
    seasonal = 3 * np.sin(2 * np.pi * time / 12)  # seasonal cycle
    noise = np.random.normal(0, 1, time_points)  # random variations
    
    # Combine components
    temperature = 50 + trend + seasonal + noise
    
    # Create time series DataFrame
    dates = []
    temps = []
    for i, temp in enumerate(temperature):
        # Format as YYYYMM
        year = 2010 + i // 12
        month = i % 12 + 1
        dates.append(f"{year}{month:02d}")
        temps.append(temp)
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'temperature': temps
    })
    
    # Analyze the synthetic data
    ma_results = analyzer.analyze_time_series(data)
    
    # Create more sophisticated patterns based on the basic analysis
    patterns = []
    
    # Basic trend pattern
    if 'patterns' in ma_results and ma_results['patterns']:
        basic_pattern = ma_results['patterns'][0]
        patterns.append(basic_pattern)
        
        # Add more nuanced patterns based on the same data
        if basic_pattern['type'] == 'warming_trend':
            # Add acceleration pattern
            patterns.append({
                'id': 'ma_warming_acceleration',
                'type': 'accelerating_warming_trend',
                'magnitude': basic_pattern['magnitude'] * 1.2,
                'position': [basic_pattern['position'][0] * 1.2, basic_pattern['position'][1], 0.6],
                'confidence': 0.75
            })
            
            # Add seasonal variation pattern
            patterns.append({
                'id': 'ma_seasonal_variation',
                'type': 'seasonal_temperature_cycle',
                'magnitude': 0.8,
                'position': [0.2, basic_pattern['position'][1] * 1.5, 0.4],
                'confidence': 0.9
            })
            
            # Add extreme event pattern
            patterns.append({
                'id': 'ma_extreme_events',
                'type': 'increasing_temperature_extremes',
                'magnitude': 0.6,
                'position': [0.6, basic_pattern['position'][1] * 0.8, 0.7],
                'confidence': 0.7
            })
            
            # Add volatility pattern
            patterns.append({
                'id': 'ma_volatility',
                'type': 'increasing_temperature_volatility',
                'magnitude': basic_pattern['position'][1],
                'position': [0.3, basic_pattern['position'][1], 0.5],
                'confidence': 0.8
            })
    
    return {"patterns": patterns}

def main():
    """Run the Field State Observer test."""
    print("Running Field State Observer Test")
    print("--------------------------------")
    
    # Create output directory
    output_dir = Path(__file__).parents[5] / "data" / "field_state_logs" / "observer"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load semantic patterns from extracted patterns
    semantic_file = Path("/Users/prphillips/Documents/GitHub/habitat_alpha/data/extracted_patterns/climate_risk_marthas_vineyard_patterns.json")
    semantic_data = load_semantic_patterns(semantic_file)
    semantic_patterns = semantic_data.get("patterns", [])
    
    print(f"Loaded {len(semantic_patterns)} semantic patterns from climate risk documents")

    
    # Generate statistical patterns
    statistical_data = generate_statistical_patterns()
    statistical_patterns = statistical_data.get("patterns", [])
    
    print(f"Generated {len(statistical_patterns)} statistical patterns from time series data")
    
    # Initialize observer
    observer = FieldStateObserver(semantic_patterns, statistical_patterns)
    
    # Observe natural resonances
    print("\nObserving natural resonances between semantic and statistical patterns...")
    observations = observer.observe_natural_resonances()
    
    # Print summary
    print(f"\nObservation complete. Found:")
    print(f"- {len(observations['resonances'])} natural resonances")
    print(f"- {len(observations['dissonances'])} natural dissonances")
    print(f"- {len(observations['oscillations'])} meaning-structure oscillations")
    
    # Save observations to JSON
    output_path = output_dir / "field_state_observations.json"
    with open(output_path, "w") as f:
        json.dump(observations, f, indent=2)
    print(f"\nSaved observations to {output_path}")
    
    # Generate visualization
    print("\nGenerating visualization of field resonances...")
    viz_path = output_dir / "field_resonances.png"
    observer.visualize_field_resonances(str(viz_path))
    
    # Generate report
    print("\nGenerating resonance report...")
    report_path = output_dir / "field_resonance_report.md"
    observer.generate_resonance_report(str(report_path))
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
