"""
Field State Observer for detecting resonances between semantic and statistical patterns.

This module implements an observer pattern that watches for natural resonances between
semantic projections and statistical time series without forcing direct mappings.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats, signal

class FieldStateObserver:
    """
    Observer that detects natural resonances between semantic and statistical patterns
    without forcing direct mappings between them.
    """
    
    def __init__(self, semantic_patterns: List[Dict] = None, statistical_patterns: List[Dict] = None):
        """
        Initialize the field state observer.
        
        Args:
            semantic_patterns: List of semantic patterns from climate risk documents
            statistical_patterns: List of statistical patterns from time series analysis
        """
        self.semantic_patterns = semantic_patterns or []
        self.statistical_patterns = statistical_patterns or []
        self.resonances = []
        self.dissonances = []
        self.oscillations = []
        
    def load_patterns(self, semantic_path: str = None, statistical_path: str = None):
        """
        Load patterns from files.
        
        Args:
            semantic_path: Path to semantic patterns JSON file
            statistical_path: Path to statistical patterns JSON file
        """
        if semantic_path:
            try:
                with open(semantic_path, 'r') as f:
                    data = json.load(f)
                    self.semantic_patterns = data.get('patterns', [])
                print(f"Loaded {len(self.semantic_patterns)} semantic patterns")
            except Exception as e:
                print(f"Error loading semantic patterns: {e}")
        
        if statistical_path:
            try:
                with open(statistical_path, 'r') as f:
                    data = json.load(f)
                    self.statistical_patterns = data.get('patterns', [])
                print(f"Loaded {len(self.statistical_patterns)} statistical patterns")
            except Exception as e:
                print(f"Error loading statistical patterns: {e}")
    
    def observe_natural_resonances(self):
        """
        Observe natural resonances between semantic and statistical patterns
        without forcing direct mappings.
        
        Returns:
            Dictionary of observed resonances, dissonances, and oscillations
        """
        # Reset observations
        self.resonances = []
        self.dissonances = []
        self.oscillations = []
        
        # Extract key terms from semantic patterns
        semantic_terms = {}
        for pattern in self.semantic_patterns:
            pattern_id = pattern.get('id', '')
            pattern_name = pattern.get('name', '')
            pattern_desc = pattern.get('description', '')
            
            # Extract key terms from name and description
            terms = self._extract_key_terms(f"{pattern_name} {pattern_desc}")
            semantic_terms[pattern_id] = terms
        
        # Extract properties from statistical patterns
        stat_properties = {}
        for pattern in self.statistical_patterns:
            pattern_id = pattern.get('id', '')
            pattern_type = pattern.get('type', '')
            pattern_magnitude = pattern.get('magnitude', 0)
            pattern_position = pattern.get('position', [])
            
            stat_properties[pattern_id] = {
                'type': pattern_type,
                'magnitude': pattern_magnitude,
                'position': pattern_position
            }
        
        # Observe resonances (natural alignments)
        for sem_id, terms in semantic_terms.items():
            sem_pattern = next((p for p in self.semantic_patterns if p.get('id') == sem_id), {})
            
            for stat_id, props in stat_properties.items():
                stat_pattern = next((p for p in self.statistical_patterns if p.get('id') == stat_id), {})
                
                # Check for natural term resonance
                resonance_score = self._calculate_term_resonance(terms, props['type'])
                
                # If there's a natural resonance
                if resonance_score > 0.3:
                    resonance = {
                        'semantic_id': sem_id,
                        'statistical_id': stat_id,
                        'resonance_type': 'natural_term_alignment',
                        'resonance_score': resonance_score,
                        'description': f"Natural term alignment between '{sem_pattern.get('name', '')}' and '{props['type'].replace('_', ' ')}'"
                    }
                    self.resonances.append(resonance)
                
                # Check for temporal resonance (if semantic pattern has temporal indicators)
                temporal_terms = [term for term in terms if term in ['increasing', 'decreasing', 'accelerating', 'slowing', 'trend']]
                if temporal_terms and 'trend' in props['type']:
                    temporal_alignment = self._check_temporal_alignment(sem_pattern, stat_pattern)
                    if temporal_alignment['aligned']:
                        resonance = {
                            'semantic_id': sem_id,
                            'statistical_id': stat_id,
                            'resonance_type': 'temporal_alignment',
                            'resonance_score': temporal_alignment['score'],
                            'description': temporal_alignment['description']
                        }
                        self.resonances.append(resonance)
                    else:
                        # If temporal terms exist but don't align, it's a dissonance
                        dissonance = {
                            'semantic_id': sem_id,
                            'statistical_id': stat_id,
                            'dissonance_type': 'temporal_misalignment',
                            'dissonance_score': 1 - temporal_alignment['score'],
                            'description': f"Temporal misalignment: {temporal_alignment['description']}"
                        }
                        self.dissonances.append(dissonance)
        
        # Observe oscillations (patterns that shift between alignment and misalignment)
        for sem_id, terms in semantic_terms.items():
            sem_pattern = next((p for p in self.semantic_patterns if p.get('id') == sem_id), {})
            
            # Look for terms that suggest oscillation
            oscillation_terms = [term for term in terms if term in ['oscillating', 'varying', 'fluctuating', 'cycle', 'periodic']]
            
            if oscillation_terms:
                # Find statistical patterns with seasonal or cyclical components
                for stat_id, props in stat_properties.items():
                    stat_pattern = next((p for p in self.statistical_patterns if p.get('id') == stat_id), {})
                    
                    if any(term in props['type'] for term in ['seasonal', 'cycle', 'periodic', 'oscillation']):
                        oscillation = {
                            'semantic_id': sem_id,
                            'statistical_id': stat_id,
                            'oscillation_type': 'meaning_structure_oscillation',
                            'oscillation_score': 0.7,  # Default score
                            'description': f"Oscillation between '{sem_pattern.get('name', '')}' and '{props['type'].replace('_', ' ')}'"
                        }
                        self.oscillations.append(oscillation)
        
        return {
            'resonances': self.resonances,
            'dissonances': self.dissonances,
            'oscillations': self.oscillations
        }
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key climate-related terms from text.
        
        Args:
            text: Text to extract terms from
            
        Returns:
            List of key terms
        """
        # List of climate-related terms to look for
        climate_terms = [
            'warming', 'cooling', 'temperature', 'precipitation', 'rainfall',
            'drought', 'flood', 'sea level', 'rise', 'extreme', 'storm',
            'hurricane', 'cyclone', 'erosion', 'heat wave', 'cold wave',
            'increasing', 'decreasing', 'accelerating', 'slowing', 'trend',
            'oscillating', 'varying', 'fluctuating', 'cycle', 'periodic',
            'seasonal', 'annual', 'decadal', 'projection', 'forecast',
            'uncertainty', 'confidence', 'likelihood', 'probability',
            'impact', 'risk', 'hazard', 'vulnerability', 'resilience',
            'adaptation', 'mitigation', 'feedback', 'threshold', 'tipping point'
        ]
        
        # Convert text to lowercase
        text = text.lower()
        
        # Find terms in text
        found_terms = []
        for term in climate_terms:
            if term in text:
                found_terms.append(term)
        
        return found_terms
    
    def _calculate_term_resonance(self, semantic_terms: List[str], statistical_type: str) -> float:
        """
        Calculate natural term resonance between semantic terms and statistical type.
        
        Args:
            semantic_terms: List of semantic terms
            statistical_type: Type of statistical pattern
            
        Returns:
            Resonance score (0-1)
        """
        # Convert statistical type to terms
        stat_terms = statistical_type.replace('_', ' ').split()
        
        # Count matching terms
        matching_terms = set(semantic_terms).intersection(set(stat_terms))
        
        # Calculate resonance score
        if not semantic_terms:
            return 0
        
        return len(matching_terms) / len(semantic_terms)
    
    def _check_temporal_alignment(self, semantic_pattern: Dict, statistical_pattern: Dict) -> Dict:
        """
        Check if semantic and statistical patterns align temporally.
        
        Args:
            semantic_pattern: Semantic pattern
            statistical_pattern: Statistical pattern
            
        Returns:
            Dictionary with alignment information
        """
        # Extract semantic temporal direction
        sem_desc = semantic_pattern.get('description', '').lower()
        sem_name = semantic_pattern.get('name', '').lower()
        
        semantic_direction = None
        if any(term in sem_desc or term in sem_name for term in ['increase', 'increasing', 'rise', 'rising', 'higher']):
            semantic_direction = 'increasing'
        elif any(term in sem_desc or term in sem_name for term in ['decrease', 'decreasing', 'fall', 'falling', 'lower']):
            semantic_direction = 'decreasing'
        
        # Extract statistical direction
        stat_type = statistical_pattern.get('type', '').lower()
        statistical_direction = None
        
        if 'warming' in stat_type or 'increasing' in stat_type:
            statistical_direction = 'increasing'
        elif 'cooling' in stat_type or 'decreasing' in stat_type:
            statistical_direction = 'decreasing'
        
        # Check alignment
        if semantic_direction and statistical_direction:
            aligned = semantic_direction == statistical_direction
            score = 1.0 if aligned else 0.0
            
            description = f"Semantic direction '{semantic_direction}' {'matches' if aligned else 'conflicts with'} statistical direction '{statistical_direction}'"
            
            return {
                'aligned': aligned,
                'score': score,
                'description': description
            }
        
        # If either direction is unknown
        return {
            'aligned': False,
            'score': 0.0,
            'description': "Unable to determine temporal alignment due to missing direction indicators"
        }
    
    def visualize_field_resonances(self, output_path: Optional[str] = None):
        """
        Visualize resonances in field space.
        
        Args:
            output_path: Path to save visualization
        """
        if not self.resonances and not self.dissonances and not self.oscillations:
            print("No resonances to visualize. Run observe_natural_resonances() first.")
            return
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Create semantic pattern positions (random for now)
        semantic_positions = {}
        for i, pattern in enumerate(self.semantic_patterns):
            # Position in a circle
            angle = 2 * np.pi * i / len(self.semantic_patterns)
            x = 0.8 * np.cos(angle)
            y = 0.8 * np.sin(angle)
            semantic_positions[pattern.get('id', f'sem_{i}')] = (x, y)
        
        # Create statistical pattern positions (based on first two dimensions if available)
        statistical_positions = {}
        for i, pattern in enumerate(self.statistical_patterns):
            pos = pattern.get('position', [])
            if len(pos) >= 2:
                # Normalize to similar scale as semantic positions
                x = pos[0] * 0.5
                y = pos[1] * 0.2
                statistical_positions[pattern.get('id', f'stat_{i}')] = (x, y)
            else:
                # Position in inner circle
                angle = 2 * np.pi * i / len(self.statistical_patterns)
                x = 0.4 * np.cos(angle)
                y = 0.4 * np.sin(angle)
                statistical_positions[pattern.get('id', f'stat_{i}')] = (x, y)
        
        # Plot semantic patterns
        for pattern in self.semantic_patterns:
            pattern_id = pattern.get('id', '')
            if pattern_id in semantic_positions:
                pos = semantic_positions[pattern_id]
                plt.scatter(pos[0], pos[1], s=100, c='blue', alpha=0.7, marker='o')
                plt.text(pos[0], pos[1], pattern.get('name', '').replace('_', ' '),
                       fontsize=8, ha='center', va='bottom')
        
        # Plot statistical patterns
        for pattern in self.statistical_patterns:
            pattern_id = pattern.get('id', '')
            if pattern_id in statistical_positions:
                pos = statistical_positions[pattern_id]
                plt.scatter(pos[0], pos[1], s=100, c='red', alpha=0.7, marker='s')
                plt.text(pos[0], pos[1], pattern.get('type', '').replace('_', ' '),
                       fontsize=8, ha='center', va='bottom')
        
        # Plot resonances
        for resonance in self.resonances:
            sem_id = resonance.get('semantic_id', '')
            stat_id = resonance.get('statistical_id', '')
            
            if sem_id in semantic_positions and stat_id in statistical_positions:
                sem_pos = semantic_positions[sem_id]
                stat_pos = statistical_positions[stat_id]
                
                # Draw line with width proportional to resonance score
                plt.plot([sem_pos[0], stat_pos[0]], [sem_pos[1], stat_pos[1]],
                       linestyle='-', color='green',
                       linewidth=resonance.get('resonance_score', 0.5) * 3,
                       alpha=0.7)
        
        # Plot dissonances
        for dissonance in self.dissonances:
            sem_id = dissonance.get('semantic_id', '')
            stat_id = dissonance.get('statistical_id', '')
            
            if sem_id in semantic_positions and stat_id in statistical_positions:
                sem_pos = semantic_positions[sem_id]
                stat_pos = statistical_positions[stat_id]
                
                # Draw line with width proportional to dissonance score
                plt.plot([sem_pos[0], stat_pos[0]], [sem_pos[1], stat_pos[1]],
                       linestyle='--', color='red',
                       linewidth=dissonance.get('dissonance_score', 0.5) * 3,
                       alpha=0.7)
        
        # Plot oscillations
        for oscillation in self.oscillations:
            sem_id = oscillation.get('semantic_id', '')
            stat_id = oscillation.get('statistical_id', '')
            
            if sem_id in semantic_positions and stat_id in statistical_positions:
                sem_pos = semantic_positions[sem_id]
                stat_pos = statistical_positions[stat_id]
                
                # Draw wavy line for oscillations
                x = np.linspace(0, 1, 100)
                line_x = sem_pos[0] * (1 - x) + stat_pos[0] * x
                line_y = sem_pos[1] * (1 - x) + stat_pos[1] * x
                
                # Add wave
                amplitude = 0.05
                frequency = 10
                wave_y = line_y + amplitude * np.sin(frequency * np.pi * x)
                
                plt.plot(line_x, wave_y,
                       linestyle='-', color='purple',
                       linewidth=oscillation.get('oscillation_score', 0.5) * 3,
                       alpha=0.7)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=10, label='Semantic Pattern'),
            Line2D([0], [0], color='red', marker='s', linestyle='None', markersize=10, label='Statistical Pattern'),
            Line2D([0], [0], color='green', linestyle='-', linewidth=2, label='Resonance'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Dissonance'),
            Line2D([0], [0], color='purple', linestyle='-', linewidth=2, label='Oscillation')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Set labels and title
        plt.xlabel("Field Dimension 1")
        plt.ylabel("Field Dimension 2")
        plt.title("Semantic-Statistical Field Resonances")
        
        # Equal aspect ratio
        plt.axis('equal')
        
        # Save or show
        if output_path:
            plt.savefig(output_path)
            print(f"Saved field resonance visualization to {output_path}")
        else:
            plt.tight_layout()
            plt.show()
        
        plt.close()
    
    def generate_resonance_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a report of observed resonances, dissonances, and oscillations.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Report text
        """
        if not self.resonances and not self.dissonances and not self.oscillations:
            return "No resonances observed. Run observe_natural_resonances() first."
        
        # Generate report
        report = [
            "# Semantic-Statistical Field Resonance Report",
            "",
            "## Overview",
            "",
            "This report documents the natural resonances, dissonances, and oscillations",
            "observed between semantic patterns (from climate risk documents) and",
            "statistical patterns (from time series analysis).",
            "",
            f"- Total semantic patterns: {len(self.semantic_patterns)}",
            f"- Total statistical patterns: {len(self.statistical_patterns)}",
            f"- Observed resonances: {len(self.resonances)}",
            f"- Observed dissonances: {len(self.dissonances)}",
            f"- Observed oscillations: {len(self.oscillations)}",
            ""
        ]
        
        # Add resonances section
        if self.resonances:
            report.append("## Natural Resonances")
            report.append("")
            
            for i, resonance in enumerate(self.resonances, 1):
                sem_id = resonance.get('semantic_id', '')
                stat_id = resonance.get('statistical_id', '')
                
                sem_pattern = next((p for p in self.semantic_patterns if p.get('id') == sem_id), {})
                stat_pattern = next((p for p in self.statistical_patterns if p.get('id') == stat_id), {})
                
                sem_name = sem_pattern.get('name', 'Unknown')
                stat_type = stat_pattern.get('type', 'Unknown').replace('_', ' ')
                
                report.append(f"### Resonance {i}: {sem_name} ↔ {stat_type}")
                report.append("")
                report.append(f"- Type: {resonance.get('resonance_type', 'Unknown').replace('_', ' ')}")
                report.append(f"- Score: {resonance.get('resonance_score', 0):.2f}")
                report.append(f"- Description: {resonance.get('description', '')}")
                report.append("")
                report.append("**Semantic Pattern:**")
                report.append(f"- Name: {sem_name}")
                report.append(f"- Description: {sem_pattern.get('description', 'No description')}")
                report.append("")
                report.append("**Statistical Pattern:**")
                report.append(f"- Type: {stat_type}")
                report.append(f"- Magnitude: {stat_pattern.get('magnitude', 0):.2f}")
                report.append("")
        
        # Add dissonances section
        if self.dissonances:
            report.append("## Natural Dissonances")
            report.append("")
            
            for i, dissonance in enumerate(self.dissonances, 1):
                sem_id = dissonance.get('semantic_id', '')
                stat_id = dissonance.get('statistical_id', '')
                
                sem_pattern = next((p for p in self.semantic_patterns if p.get('id') == sem_id), {})
                stat_pattern = next((p for p in self.statistical_patterns if p.get('id') == stat_id), {})
                
                sem_name = sem_pattern.get('name', 'Unknown')
                stat_type = stat_pattern.get('type', 'Unknown').replace('_', ' ')
                
                report.append(f"### Dissonance {i}: {sem_name} ↔ {stat_type}")
                report.append("")
                report.append(f"- Type: {dissonance.get('dissonance_type', 'Unknown').replace('_', ' ')}")
                report.append(f"- Score: {dissonance.get('dissonance_score', 0):.2f}")
                report.append(f"- Description: {dissonance.get('description', '')}")
                report.append("")
                report.append("**Semantic Pattern:**")
                report.append(f"- Name: {sem_name}")
                report.append(f"- Description: {sem_pattern.get('description', 'No description')}")
                report.append("")
                report.append("**Statistical Pattern:**")
                report.append(f"- Type: {stat_type}")
                report.append(f"- Magnitude: {stat_pattern.get('magnitude', 0):.2f}")
                report.append("")
        
        # Add oscillations section
        if self.oscillations:
            report.append("## Meaning-Structure Oscillations")
            report.append("")
            
            for i, oscillation in enumerate(self.oscillations, 1):
                sem_id = oscillation.get('semantic_id', '')
                stat_id = oscillation.get('statistical_id', '')
                
                sem_pattern = next((p for p in self.semantic_patterns if p.get('id') == sem_id), {})
                stat_pattern = next((p for p in self.statistical_patterns if p.get('id') == stat_id), {})
                
                sem_name = sem_pattern.get('name', 'Unknown')
                stat_type = stat_pattern.get('type', 'Unknown').replace('_', ' ')
                
                report.append(f"### Oscillation {i}: {sem_name} ↔ {stat_type}")
                report.append("")
                report.append(f"- Type: {oscillation.get('oscillation_type', 'Unknown').replace('_', ' ')}")
                report.append(f"- Score: {oscillation.get('oscillation_score', 0):.2f}")
                report.append(f"- Description: {oscillation.get('description', '')}")
                report.append("")
                report.append("**Semantic Pattern:**")
                report.append(f"- Name: {sem_name}")
                report.append(f"- Description: {sem_pattern.get('description', 'No description')}")
                report.append("")
                report.append("**Statistical Pattern:**")
                report.append(f"- Type: {stat_type}")
                report.append(f"- Magnitude: {stat_pattern.get('magnitude', 0):.2f}")
                report.append("")
        
        # Add conclusion
        report.append("## Conclusion")
        report.append("")
        report.append("This analysis demonstrates the natural interplay between semantic and statistical")
        report.append("patterns in climate data. By observing rather than forcing connections, we can")
        report.append("identify meaningful resonances while preserving the productive tension between")
        report.append("meaning and structure, temporality and topology.")
        report.append("")
        report.append(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Join report lines
        report_text = "\n".join(report)
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Saved resonance report to {output_path}")
        
        return report_text
