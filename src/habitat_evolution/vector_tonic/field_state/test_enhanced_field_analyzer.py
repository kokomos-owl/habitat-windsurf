"""
Test script for the enhanced field state analyzer.

This script demonstrates how the enhanced field analyzer generates patterns and relationships
from real NOAA climate data.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from enhanced_field_analyzer import EnhancedFieldStateAnalyzer

def main():
    """Run the enhanced field analyzer test."""
    print("Running Enhanced Field State Analyzer Test")
    print("------------------------------------------")
    
    # Create output directory
    output_dir = Path(__file__).parents[5] / "data" / "field_state_logs" / "enhanced"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize analyzer
    analyzer = EnhancedFieldStateAnalyzer(dimensions=5)
    
    # Run complete analysis
    print("\nAnalyzing NOAA climate data...")
    results = analyzer.analyze_all()
    
    # Print summary
    print(f"\nAnalysis complete. Found:")
    print(f"- {len(results['ma_results']['patterns'])} patterns in Massachusetts data")
    print(f"- {len(results['ne_results']['patterns'])} patterns in Northeast data")
    print(f"- {len(results['diff_results']['patterns'])} regional difference patterns")
    print(f"- {len(results['all_patterns'])} total patterns")
    print(f"- {len(results['relationships'])} pattern relationships")
    
    # Save results to JSON
    output_path = output_dir / "climate_field_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved analysis results to {output_path}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Field space visualization
    field_viz_path = output_dir / "climate_field_space.png"
    analyzer.visualize_field_space(
        results['all_patterns'],
        dimensions=[0, 1, 2],  # Trend, Volatility, Acceleration
        output_path=str(field_viz_path)
    )
    
    # Relationship visualization
    rel_viz_path = output_dir / "climate_pattern_relationships.png"
    analyzer.visualize_relationships(
        results['all_patterns'],
        results['relationships'],
        output_path=str(rel_viz_path)
    )
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(results, output_dir)
    
    print("\nTest completed successfully!")

def generate_summary_report(results, output_dir):
    """Generate a summary report of the analysis."""
    # Count pattern types
    pattern_types = {}
    for pattern in results['all_patterns']:
        pattern_type = pattern.get('type', 'unknown')
        if pattern_type not in pattern_types:
            pattern_types[pattern_type] = []
        pattern_types[pattern_type].append(pattern)
    
    # Count relationship types
    relationship_types = {}
    for rel in results['relationships']:
        rel_type = rel.get('type', 'unknown')
        if rel_type not in relationship_types:
            relationship_types[rel_type] = []
        relationship_types[rel_type].append(rel)
    
    # Generate report
    report = [
        "# Enhanced Climate Field State Analysis Report",
        "",
        "## Overview",
        "",
        f"This report provides a summary of enhanced field state analysis for NOAA climate data.",
        f"The analysis includes patterns from Massachusetts and Northeast temperature data from 1991-2024.",
        "",
        "## Patterns Detected",
        "",
        f"Total patterns detected: {len(results['all_patterns'])}",
        "",
    ]
    
    # Add pattern type sections
    for pattern_type, patterns in pattern_types.items():
        report.append(f"### {pattern_type.replace('_', ' ').title()}")
        report.append("")
        report.append(f"Found {len(patterns)} patterns:")
        report.append("")
        
        for pattern in patterns:
            metadata = pattern.get('metadata', {})
            region = metadata.get('region', metadata.get('regions', ['unknown'])[0] if isinstance(metadata.get('regions', []), list) else 'unknown')
            time_period = metadata.get('time_period', 'unknown')
            magnitude = pattern.get('magnitude', 0)
            
            report.append(f"- {pattern['id']}: Magnitude {magnitude:.2f}, Region: {region}, Period: {time_period}")
        
        report.append("")
    
    # Add relationship section
    report.append("## Pattern Relationships")
    report.append("")
    report.append(f"Total relationships detected: {len(results['relationships'])}")
    report.append("")
    
    for rel_type, relationships in relationship_types.items():
        report.append(f"### {rel_type.replace('_', ' ').title()} Relationships")
        report.append("")
        report.append(f"Found {len(relationships)} relationships:")
        report.append("")
        
        for rel in relationships:
            pattern1_id = rel.get('pattern1_id', 'unknown')
            pattern2_id = rel.get('pattern2_id', 'unknown')
            strength = rel.get('strength', 0)
            
            report.append(f"- {pattern1_id} ↔ {pattern2_id}: Strength {strength:.2f}")
        
        report.append("")
    
    # Add conclusion
    report.append("## Conclusion")
    report.append("")
    report.append("This enhanced analysis demonstrates the power of field-state representation for climate data.")
    report.append("By analyzing real NOAA temperature data in a multi-dimensional field space, we can detect")
    report.append("complex patterns and relationships that provide insights into climate dynamics.")
    report.append("")
    report.append(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Write report to file
    report_path = output_dir / "enhanced_climate_analysis_summary.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    
    print(f"Saved summary report to {report_path}")

if __name__ == "__main__":
    main()
