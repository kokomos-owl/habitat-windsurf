"""
Climate Pattern Navigator

This module demonstrates how to use the PatternRetrieverAgent to navigate
climate patterns in the tonic-harmonic field for a specific use case:
analyzing sea level rise impacts across different coastal regions.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from tonic_harmonic_pattern_retriever import TonicHarmonicFieldAPI, PatternRetrieverAgent

class ClimatePatternNavigator:
    """
    Navigator for climate patterns that uses the PatternRetrieverAgent
    to explore and analyze patterns related to climate risks.
    """
    
    def __init__(self, agent: PatternRetrieverAgent):
        """
        Initialize the climate pattern navigator.
        
        Args:
            agent: Pattern retriever agent
        """
        self.agent = agent
    
    def analyze_sea_level_rise_impacts(self, region: str, time_horizon: str = "2050") -> Dict[str, Any]:
        """
        Analyze sea level rise impacts for a specific coastal region.
        
        Args:
            region: Coastal region to analyze
            time_horizon: Future time horizon for projections
            
        Returns:
            Analysis of sea level rise impacts
        """
        print(f"\n=== Analyzing Sea Level Rise Impacts for {region} through {time_horizon} ===\n")
        
        # Step 1: Find sea level rise patterns for the region
        # In a real implementation, we would query the tonic-harmonic field
        # for patterns related to sea level rise in the specified region
        print(f"Step 1: Finding sea level rise patterns for {region}...")
        slr_pattern_id = f"slr_{region.lower().replace(' ', '_')}"
        slr_pattern = self.agent.get_pattern(slr_pattern_id)
        print(f"  Found pattern: {slr_pattern['name']}")
        
        # Step 2: Find related impact patterns
        print(f"\nStep 2: Finding impact patterns related to sea level rise in {region}...")
        impact_patterns = self.agent.find_related_patterns(slr_pattern_id, ["causal", "impact"])
        print(f"  Found {len(impact_patterns)} impact patterns:")
        for i, pattern in enumerate(impact_patterns, 1):
            print(f"  {i}. {pattern['name']} - {pattern['description']}")
        
        # Step 3: Analyze constructive dissonance
        print(f"\nStep 3: Analyzing constructive dissonance in {region}...")
        current_year = datetime.now().year
        horizon_year = int(time_horizon) if time_horizon.isdigit() else 2050
        dissonance = self.agent.analyze_constructive_dissonance(
            region, 
            start_date=f"{current_year}-01-01", 
            end_date=f"{horizon_year}-12-31"
        )
        print(f"  Found {len(dissonance)} instances of constructive dissonance:")
        for i, instance in enumerate(dissonance, 1):
            print(f"  {i}. Between '{instance['pattern1']['name']}' and '{instance['pattern2']['name']}':")
            print(f"     {instance['dissonance_description']}")
        
        # Step 4: Explore pattern evolution through sliding windows
        print(f"\nStep 4: Exploring pattern evolution through sliding windows...")
        window_analysis = self.agent.explore_sliding_windows(region, window_size="5y", step_size="1y")
        print(f"  Analyzed {window_analysis['window_count']} sliding windows")
        print(f"  Quality state distribution: {window_analysis['evolution_analysis']['quality_state_distribution']}")
        print(f"  Evolution summary: {window_analysis['evolution_analysis']['evolution_summary']}")
        
        # Step 5: Compile comprehensive analysis
        print(f"\nStep 5: Compiling comprehensive analysis...")
        analysis = {
            "region": region,
            "time_horizon": time_horizon,
            "sea_level_rise_pattern": slr_pattern,
            "impact_patterns": impact_patterns,
            "constructive_dissonance": dissonance,
            "pattern_evolution": window_analysis,
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": self._generate_summary(region, slr_pattern, impact_patterns, dissonance, window_analysis)
        }
        
        print(f"\n=== Analysis Complete ===\n")
        print(f"Summary: {analysis['summary']}")
        
        return analysis
    
    def compare_regional_impacts(self, regions: List[str], time_horizon: str = "2050") -> Dict[str, Any]:
        """
        Compare sea level rise impacts across multiple coastal regions.
        
        Args:
            regions: List of coastal regions to compare
            time_horizon: Future time horizon for projections
            
        Returns:
            Comparative analysis of sea level rise impacts
        """
        print(f"\n=== Comparing Sea Level Rise Impacts Across Regions through {time_horizon} ===\n")
        
        # Analyze each region
        regional_analyses = {}
        for region in regions:
            print(f"Analyzing {region}...")
            regional_analyses[region] = self.analyze_sea_level_rise_impacts(region, time_horizon)
        
        # Perform comparative analysis
        print(f"\nPerforming comparative analysis across regions...")
        comparison = self._compare_analyses(regional_analyses)
        
        print(f"\n=== Comparative Analysis Complete ===\n")
        print(f"Key findings: {comparison['key_findings']}")
        
        return {
            "regions": regions,
            "time_horizon": time_horizon,
            "regional_analyses": regional_analyses,
            "comparison": comparison,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _generate_summary(
        self, 
        region: str, 
        slr_pattern: Dict[str, Any],
        impact_patterns: List[Dict[str, Any]],
        dissonance: List[Dict[str, Any]],
        window_analysis: Dict[str, Any]
    ) -> str:
        """
        Generate a summary of the sea level rise impact analysis.
        
        Args:
            region: Analyzed region
            slr_pattern: Sea level rise pattern
            impact_patterns: Related impact patterns
            dissonance: Constructive dissonance instances
            window_analysis: Sliding window analysis
            
        Returns:
            Summary text
        """
        # In a real implementation, this would generate a more sophisticated summary
        # based on the patterns and analyses
        impact_count = len(impact_patterns)
        dissonance_count = len(dissonance)
        
        return (
            f"Analysis of sea level rise impacts in {region} reveals {impact_count} related impact patterns "
            f"and {dissonance_count} instances of constructive dissonance. Pattern evolution analysis "
            f"through sliding windows shows {window_analysis['evolution_analysis']['evolution_summary'].lower()}."
        )
    
    def _compare_analyses(self, regional_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare analyses across regions.
        
        Args:
            regional_analyses: Dictionary of regional analyses
            
        Returns:
            Comparative analysis
        """
        # In a real implementation, this would perform more sophisticated comparison
        # For now, we'll return a simple comparison
        regions = list(regional_analyses.keys())
        impact_counts = {region: len(analysis["impact_patterns"]) for region, analysis in regional_analyses.items()}
        
        # Find region with most impacts
        most_impacts_region = max(impact_counts, key=impact_counts.get)
        
        # Generate key findings
        key_findings = (
            f"Comparison across {len(regions)} regions shows that {most_impacts_region} has the highest "
            f"number of identified impact patterns ({impact_counts[most_impacts_region]}). "
            f"All regions show evidence of pattern evolution through quality states, with constructive "
            f"dissonance providing insights into complex climate dynamics."
        )
        
        return {
            "impact_counts": impact_counts,
            "most_impacts_region": most_impacts_region,
            "key_findings": key_findings,
            "region_count": len(regions)
        }


# Example usage
if __name__ == "__main__":
    # Initialize the tonic-harmonic field API client
    field_api = TonicHarmonicFieldAPI(field_service_url="https://api.habitat-evolution.example/tonic-harmonic")
    
    # Initialize the pattern retriever agent
    agent = PatternRetrieverAgent(field_api)
    
    # Initialize the climate pattern navigator
    navigator = ClimatePatternNavigator(agent)
    
    # Example 1: Analyze sea level rise impacts for a single region
    cape_cod_analysis = navigator.analyze_sea_level_rise_impacts("Cape Cod", "2050")
    
    # Example 2: Compare impacts across multiple regions
    regions = ["Cape Cod", "Boston Harbor", "Martha's Vineyard"]
    comparative_analysis = navigator.compare_regional_impacts(regions, "2050")
    
    # Save the analyses to files for reference
    with open("cape_cod_analysis.json", "w") as f:
        json.dump(cape_cod_analysis, f, indent=2)
    
    with open("comparative_analysis.json", "w") as f:
        json.dump(comparative_analysis, f, indent=2)
    
    print("\nAnalyses saved to cape_cod_analysis.json and comparative_analysis.json")
