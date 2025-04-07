"""
Analyze climate risk patterns and integrate them into the Habitat Evolution system.

This module analyzes the patterns extracted from climate risk documents and
integrates them into the Habitat Evolution system to enhance pattern extraction
capabilities and build a comprehensive climate lexicon.
"""

import json
import os
import sys
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[3]))

# Create a simplified field state service for our analysis
class SimpleFieldStateService:
    """
    A simplified implementation of field state service for climate pattern analysis.
    """
    
    def __init__(self):
        """
        Initialize the simple field state service.
        """
        self.patterns = {}
        self.relationships = {}
        print("Initialized SimpleFieldStateService for climate pattern analysis")
    
    def add_pattern(self, pattern):
        """
        Add a pattern to the field state.
        
        Args:
            pattern: The pattern to add
        """
        pattern_id = pattern.get("id")
        if pattern_id:
            self.patterns[pattern_id] = pattern
            print(f"Added pattern: {pattern.get('name')}")
    
    def add_pattern_relationship(self, pattern1_id, pattern2_id, relationship_type, strength, description):
        """
        Add a relationship between two patterns.
        
        Args:
            pattern1_id: ID of the first pattern
            pattern2_id: ID of the second pattern
            relationship_type: Type of relationship
            strength: Strength of relationship
            description: Description of relationship
        """
        relationship_id = f"{pattern1_id}_{pattern2_id}"
        
        self.relationships[relationship_id] = {
            "pattern1_id": pattern1_id,
            "pattern2_id": pattern2_id,
            "relationship_type": relationship_type,
            "strength": strength,
            "description": description
        }
        
        print(f"Added relationship between {pattern1_id} and {pattern2_id}")
    
    def get_patterns(self):
        """
        Get all patterns in the field state.
        
        Returns:
            Dictionary of patterns
        """
        return self.patterns
    
    def get_relationships(self):
        """
        Get all relationships in the field state.
        
        Returns:
            Dictionary of relationships
        """
        return self.relationships
    
    def export_field_state(self, output_path):
        """
        Export the field state to a JSON file.
        
        Args:
            output_path: Path to save the field state
        """
        import json
        from pathlib import Path
        
        field_state = {
            "patterns": self.patterns,
            "relationships": self.relationships
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(field_state, f, indent=2)
        
        print(f"Exported field state to {output_path}")
from src.habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter


class ClimatePatternAnalyzer:
    """Analyze climate risk patterns and integrate them into the Habitat Evolution system."""
    
    def __init__(self):
        """Initialize the climate pattern analyzer."""
        # Initialize paths
        self.project_root = Path(__file__).parents[4]
        self.patterns_dir = self.project_root / "data" / "extracted_patterns"
        self.output_dir = self.project_root / "data" / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self.field_state_service = SimpleFieldStateService()
        self.claude_adapter = ClaudeAdapter()
    
    def load_extracted_patterns(self) -> List[Dict[str, Any]]:
        """
        Load all extracted patterns from the patterns directory.
        
        Returns:
            List of pattern dictionaries
        """
        patterns = []
        
        # Find all pattern files
        pattern_files = list(self.patterns_dir.glob("*_patterns.json"))
        
        for file_path in pattern_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    
                # Add source file information
                for pattern in data.get("patterns", []):
                    pattern["source_file"] = file_path.stem.replace("_patterns", "")
                
                patterns.extend(data.get("patterns", []))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return patterns
    
    def load_climate_lexicon(self) -> Dict[str, Any]:
        """
        Load the climate lexicon.
        
        Returns:
            Climate lexicon dictionary
        """
        lexicon_path = self.patterns_dir / "climate_lexicon.json"
        
        if not lexicon_path.exists():
            return {"terms": [], "analysis": "", "timestamp": ""}
        
        try:
            with open(lexicon_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading climate lexicon: {e}")
            return {"terms": [], "analysis": "", "timestamp": ""}
    
    def load_ner_patterns(self) -> Dict[str, Any]:
        """
        Load the NER patterns.
        
        Returns:
            NER patterns dictionary
        """
        ner_path = self.patterns_dir / "climate_ner_patterns.json"
        
        if not ner_path.exists():
            return {"analysis": "", "timestamp": ""}
        
        try:
            with open(ner_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading NER patterns: {e}")
            return {"analysis": "", "timestamp": ""}
    
    def analyze_pattern_quality(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the quality of extracted patterns.
        
        Args:
            patterns: List of pattern dictionaries
            
        Returns:
            Dictionary containing quality analysis
        """
        # Count patterns by quality state
        quality_counts = Counter(p.get("quality_state", "unknown") for p in patterns)
        
        # Count patterns by source file
        source_counts = Counter(p.get("source_file", "unknown") for p in patterns)
        
        # Analyze pattern length
        name_lengths = [len(p.get("name", "")) for p in patterns]
        desc_lengths = [len(p.get("description", "")) for p in patterns]
        evidence_lengths = [len(p.get("evidence", "")) for p in patterns]
        
        return {
            "total_patterns": len(patterns),
            "quality_distribution": dict(quality_counts),
            "source_distribution": dict(source_counts),
            "name_length": {
                "min": min(name_lengths) if name_lengths else 0,
                "max": max(name_lengths) if name_lengths else 0,
                "avg": sum(name_lengths) / len(name_lengths) if name_lengths else 0
            },
            "description_length": {
                "min": min(desc_lengths) if desc_lengths else 0,
                "max": max(desc_lengths) if desc_lengths else 0,
                "avg": sum(desc_lengths) / len(desc_lengths) if desc_lengths else 0
            },
            "evidence_length": {
                "min": min(evidence_lengths) if evidence_lengths else 0,
                "max": max(evidence_lengths) if evidence_lengths else 0,
                "avg": sum(evidence_lengths) / len(evidence_lengths) if evidence_lengths else 0
            }
        }
    
    def generate_pattern_visualizations(self, patterns: List[Dict[str, Any]], quality_analysis: Dict[str, Any]):
        """
        Generate visualizations for pattern analysis.
        
        Args:
            patterns: List of pattern dictionaries
            quality_analysis: Dictionary containing quality analysis
        """
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Quality distribution
        quality_data = quality_analysis["quality_distribution"]
        axs[0, 0].bar(quality_data.keys(), quality_data.values())
        axs[0, 0].set_title("Pattern Quality Distribution")
        axs[0, 0].set_xlabel("Quality State")
        axs[0, 0].set_ylabel("Count")
        
        # Plot 2: Source distribution
        source_data = quality_analysis["source_distribution"]
        axs[0, 1].bar(source_data.keys(), source_data.values())
        axs[0, 1].set_title("Pattern Source Distribution")
        axs[0, 1].set_xlabel("Source File")
        axs[0, 1].set_ylabel("Count")
        axs[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Pattern length statistics
        length_categories = ["Name", "Description", "Evidence"]
        avg_lengths = [
            quality_analysis["name_length"]["avg"],
            quality_analysis["description_length"]["avg"],
            quality_analysis["evidence_length"]["avg"]
        ]
        axs[1, 0].bar(length_categories, avg_lengths)
        axs[1, 0].set_title("Average Pattern Length")
        axs[1, 0].set_xlabel("Pattern Component")
        axs[1, 0].set_ylabel("Average Length (characters)")
        
        # Plot 4: Pattern relationships (placeholder)
        axs[1, 1].text(0.5, 0.5, "Pattern Relationship Analysis\n(Requires additional processing)", 
                     horizontalalignment='center', verticalalignment='center', transform=axs[1, 1].transAxes)
        axs[1, 1].set_title("Pattern Relationships")
        axs[1, 1].axis('off')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.output_dir / "climate_pattern_analysis.png")
        plt.close()
    
    def integrate_patterns_with_field_state(self, patterns: List[Dict[str, Any]]):
        """
        Integrate patterns with the field state service.
        
        Args:
            patterns: List of pattern dictionaries
        """
        # Convert patterns to field state format
        field_patterns = []
        
        for pattern in patterns:
            field_pattern = {
                "id": f"climate_{pattern.get('source_file')}_{len(field_patterns)}",
                "name": pattern.get("name", "Unnamed Pattern"),
                "description": pattern.get("description", ""),
                "evidence": pattern.get("evidence", ""),
                "quality_state": pattern.get("quality_state", "hypothetical"),
                "domain": "climate_risk",
                "source": pattern.get("source_file", "unknown"),
                "created_at": datetime.now().isoformat(),
                "relationships": []
            }
            
            field_patterns.append(field_pattern)
        
        # Update field state with new patterns
        for pattern in field_patterns:
            self.field_state_service.add_pattern(pattern)
        
        print(f"Integrated {len(field_patterns)} climate patterns with field state service")
    
    def generate_pattern_relationships(self, patterns: List[Dict[str, Any]]):
        """
        Generate relationships between patterns.
        
        Args:
            patterns: List of pattern dictionaries
        """
        # Group patterns by source file
        patterns_by_source = {}
        for pattern in patterns:
            source = pattern.get("source_file", "unknown")
            if source not in patterns_by_source:
                patterns_by_source[source] = []
            patterns_by_source[source].append(pattern)
        
        # For each source file, generate relationships between patterns
        for source, source_patterns in patterns_by_source.items():
            if len(source_patterns) < 2:
                continue
            
            # Create a query to analyze pattern relationships
            pattern_texts = []
            for i, pattern in enumerate(source_patterns):
                pattern_texts.append(f"{i+1}. {pattern.get('name')}: {pattern.get('description')}")
            
            query = f"""
            Analyze the relationships between the following climate risk patterns:
            
            {chr(10).join(pattern_texts)}
            
            For each pair of patterns, identify:
            1. If they are related (yes/no)
            2. The type of relationship (e.g., causal, correlational, hierarchical)
            3. The strength of the relationship (weak, moderate, strong)
            
            Format your response as a list of relationships:
            [
                {{
                    "pattern1_index": 1,
                    "pattern2_index": 2,
                    "related": true,
                    "relationship_type": "causal",
                    "strength": "strong",
                    "description": "Pattern 1 causes Pattern 2 because..."
                }},
                ...
            ]
            """
            
            # Process the query with Claude
            try:
                result = asyncio.run(self.claude_adapter.process_query(
                    query=query,
                    context={"task": "pattern_relationship_analysis", "source": source},
                    patterns=[]
                ))
                
                # Parse the relationships from the response
                response = result.get("response", "")
                
                # Extract JSON from the response (may be embedded in text)
                import re
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
                
                if json_match:
                    relationships_json = json_match.group(0)
                    relationships = json.loads(relationships_json)
                    
                    # Update field state with relationships
                    for rel in relationships:
                        if rel.get("related", False):
                            pattern1_idx = rel.get("pattern1_index", 0) - 1
                            pattern2_idx = rel.get("pattern2_index", 0) - 1
                            
                            if 0 <= pattern1_idx < len(source_patterns) and 0 <= pattern2_idx < len(source_patterns):
                                pattern1 = source_patterns[pattern1_idx]
                                pattern2 = source_patterns[pattern2_idx]
                                
                                # Create relationship IDs
                                pattern1_id = f"climate_{source}_{pattern1_idx}"
                                pattern2_id = f"climate_{source}_{pattern2_idx}"
                                
                                # Add relationship to field state
                                self.field_state_service.add_pattern_relationship(
                                    pattern1_id, 
                                    pattern2_id,
                                    rel.get("relationship_type", "related"),
                                    rel.get("strength", "moderate"),
                                    rel.get("description", "")
                                )
                
                print(f"Generated relationships for patterns in {source}")
            except Exception as e:
                print(f"Error generating relationships for {source}: {e}")
    
    def generate_report(self, patterns: List[Dict[str, Any]], quality_analysis: Dict[str, Any], 
                       lexicon: Dict[str, Any], ner_patterns: Dict[str, Any]):
        """
        Generate a comprehensive report of the climate pattern analysis.
        
        Args:
            patterns: List of pattern dictionaries
            quality_analysis: Dictionary containing quality analysis
            lexicon: Climate lexicon dictionary
            ner_patterns: NER patterns dictionary
        """
        # Create report content
        report = f"""# Climate Risk Pattern Analysis Report

## Overview

This report provides an analysis of patterns extracted from climate risk documents
using the Claude API. The analysis includes pattern quality assessment, lexicon
building, and NER pattern extraction.

## Pattern Statistics

- **Total Patterns**: {quality_analysis["total_patterns"]}
- **Quality Distribution**:
  - Hypothetical: {quality_analysis["quality_distribution"].get("hypothetical", 0)}
  - Emergent: {quality_analysis["quality_distribution"].get("emergent", 0)}
  - Stable: {quality_analysis["quality_distribution"].get("stable", 0)}
  - Unknown: {quality_analysis["quality_distribution"].get("unknown", 0)}

## Pattern Sources

The patterns were extracted from the following sources:

{chr(10).join([f"- {source}: {count} patterns" for source, count in quality_analysis["source_distribution"].items()])}

## Pattern Quality Analysis

### Length Statistics

- **Name Length**: Min: {quality_analysis["name_length"]["min"]}, Max: {quality_analysis["name_length"]["max"]}, Avg: {quality_analysis["name_length"]["avg"]:.2f}
- **Description Length**: Min: {quality_analysis["description_length"]["min"]}, Max: {quality_analysis["description_length"]["max"]}, Avg: {quality_analysis["description_length"]["avg"]:.2f}
- **Evidence Length**: Min: {quality_analysis["evidence_length"]["min"]}, Max: {quality_analysis["evidence_length"]["max"]}, Avg: {quality_analysis["evidence_length"]["avg"]:.2f}

## Climate Lexicon

The climate lexicon contains {len(lexicon["terms"])} terms extracted from the patterns.

### Lexicon Analysis

{lexicon["analysis"]}

## NER Patterns

The following NER patterns were extracted from the climate risk documents:

{ner_patterns["analysis"]}

## Conclusion

This analysis provides valuable insights into the patterns extracted from climate risk
documents. The patterns, lexicon, and NER patterns can be integrated into the Habitat
Evolution system to enhance pattern extraction capabilities and build a comprehensive
climate lexicon.

## Next Steps

1. Refine pattern extraction prompts based on quality analysis
2. Integrate climate lexicon with pattern extraction process
3. Implement NER patterns for climate risk document processing
4. Develop pattern relationship visualization tools
5. Expand climate risk document corpus for more comprehensive pattern extraction

Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        # Save the report
        with open(self.output_dir / "climate_pattern_analysis_report.md", "w") as f:
            f.write(report)
        
        print(f"Generated climate pattern analysis report")
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        # Load extracted patterns
        patterns = self.load_extracted_patterns()
        print(f"Loaded {len(patterns)} patterns from extracted pattern files")
        
        # Load climate lexicon
        lexicon = self.load_climate_lexicon()
        print(f"Loaded climate lexicon with {len(lexicon['terms'])} terms")
        
        # Load NER patterns
        ner_patterns = self.load_ner_patterns()
        print(f"Loaded NER patterns")
        
        # Analyze pattern quality
        quality_analysis = self.analyze_pattern_quality(patterns)
        print(f"Analyzed pattern quality")
        
        # Generate visualizations
        self.generate_pattern_visualizations(patterns, quality_analysis)
        print(f"Generated pattern visualizations")
        
        # Integrate patterns with field state
        self.integrate_patterns_with_field_state(patterns)
        
        # Generate pattern relationships
        self.generate_pattern_relationships(patterns)
        
        # Generate report
        self.generate_report(patterns, quality_analysis, lexicon, ner_patterns)
        
        print(f"Analysis complete. Results saved to {self.output_dir}")


def main():
    """Main function."""
    analyzer = ClimatePatternAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
