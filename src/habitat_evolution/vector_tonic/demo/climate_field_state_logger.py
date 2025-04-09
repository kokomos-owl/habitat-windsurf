"""
Real-time climate data logger using field state analysis.

This module monitors climate risk documents and logs the results of field state analysis
in real-time, providing insights into climate patterns and their relationships.
"""

import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
project_root = Path(__file__).parents[4]
sys.path.append(str(project_root))

from src.habitat_evolution.vector_tonic.field_state.simple_field_analyzer import SimpleFieldStateAnalyzer
from src.habitat_evolution.vector_tonic.field_state.multi_scale_analyzer import MultiScaleAnalyzer
from src.habitat_evolution.vector_tonic.bridge.field_pattern_bridge import FieldPatternBridge
from src.habitat_evolution.infrastructure.services.pattern_evolution_service import PatternEvolutionService
from src.habitat_evolution.infrastructure.services.event_service import EventService
from src.habitat_evolution.infrastructure.services.bidirectional_flow_service import BidirectionalFlowService
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClimateFieldStateLogger:
    """
    Real-time climate data logger using field state analysis.
    """
    
    def __init__(self, use_mock_services: bool = False):
        """
        Initialize the climate field state logger.
        
        Args:
            use_mock_services: Whether to use mock services for testing
        """
        self.project_root = project_root
        self.climate_risk_dir = self.project_root / "data" / "climate_risk"
        self.extracted_patterns_dir = self.project_root / "data" / "extracted_patterns"
        self.output_dir = self.project_root / "data" / "field_state_logs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        if use_mock_services:
            from unittest.mock import MagicMock
            self.pattern_evolution_service = MagicMock()
            self.pattern_evolution_service.create_pattern.return_value = {
                "id": "mock_pattern_id",
                "type": "climate_warming_trend",
                "quality_state": "emergent"
            }
        else:
            # Use real services
            arangodb_connection = ArangoDBConnection(
                host="localhost",
                port=8529,
                username="root",
                password="habitat",
                database_name="habitat_evolution"
            )
            
            # Use mock services for components we don't need full functionality from
            from unittest.mock import MagicMock
            event_service = EventService()
            pattern_aware_rag_service = MagicMock()
            
            bidirectional_flow_service = BidirectionalFlowService(
                event_service=event_service,
                pattern_aware_rag_service=pattern_aware_rag_service,
                arangodb_connection=arangodb_connection
            )
            
            self.pattern_evolution_service = PatternEvolutionService(
                event_service=event_service,
                bidirectional_flow_service=bidirectional_flow_service,
                arangodb_connection=arangodb_connection
            )
            
            # Initialize pattern evolution service
            self.pattern_evolution_service.initialize()
        
        # Initialize field state analyzers
        self.field_analyzer = SimpleFieldStateAnalyzer()
        self.multi_scale_analyzer = MultiScaleAnalyzer()
        self.field_bridge = FieldPatternBridge(self.pattern_evolution_service)
        
        logger.info("Initialized ClimateFieldStateLogger")
    
    def load_climate_document(self, document_path: Path) -> Dict[str, Any]:
        """
        Load and parse a climate risk document.
        
        Args:
            document_path: Path to the climate risk document
            
        Returns:
            Parsed document data
        """
        try:
            with open(document_path, "r") as f:
                content = f.read()
            
            # Extract document name from filename
            document_name = document_path.stem
            
            # Parse document content
            # For this demo, we'll create a simple time series from the document
            # In a real implementation, you would extract actual time series data
            
            # Generate synthetic time series based on document length
            dates = pd.date_range(start='2020-01-01', periods=min(120, len(content) // 100), freq='D')
            
            # Use document content to seed the random generator for reproducibility
            np.random.seed(hash(content) % 2**32)
            
            # Create base trend with some randomness
            base = np.linspace(10, 15, len(dates))
            noise = np.random.normal(0, 1, len(dates))
            seasonal = 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
            temps = base + noise + seasonal
            
            # Create DataFrame
            data = pd.DataFrame({
                'date': dates,
                'temperature': temps,
                'region': document_name
            })
            
            logger.info(f"Loaded climate document: {document_name} ({len(content)} bytes)")
            
            return {
                "document_name": document_name,
                "content": content,
                "time_series": data,
                "metadata": {
                    "source": "climate_risk_document",
                    "path": str(document_path),
                    "size": len(content),
                    "created_at": datetime.now().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error loading climate document {document_path}: {e}")
            return {
                "document_name": document_path.stem,
                "content": "",
                "time_series": pd.DataFrame(),
                "metadata": {
                    "source": "climate_risk_document",
                    "path": str(document_path),
                    "error": str(e)
                }
            }
    
    def load_extracted_patterns(self, document_name: str) -> List[Dict[str, Any]]:
        """
        Load extracted patterns for a specific document.
        
        Args:
            document_name: Name of the document
            
        Returns:
            List of extracted patterns
        """
        pattern_file = self.extracted_patterns_dir / f"{document_name}_patterns.json"
        
        if not pattern_file.exists():
            logger.warning(f"No extracted patterns found for {document_name}")
            return []
        
        try:
            with open(pattern_file, "r") as f:
                data = json.load(f)
            
            patterns = data.get("patterns", [])
            logger.info(f"Loaded {len(patterns)} extracted patterns for {document_name}")
            
            return patterns
        except Exception as e:
            logger.error(f"Error loading extracted patterns for {document_name}: {e}")
            return []
    
    def analyze_document(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a climate risk document using field state analysis.
        
        Args:
            document_data: Document data
            
        Returns:
            Analysis results
        """
        document_name = document_data["document_name"]
        time_series = document_data["time_series"]
        
        if time_series.empty:
            logger.warning(f"No time series data available for {document_name}")
            return {
                "document_name": document_name,
                "field_analysis": {},
                "multi_scale_analysis": {},
                "bridge_results": {},
                "extracted_patterns": [],
                "status": "error",
                "error": "No time series data available"
            }
        
        # Load extracted patterns
        extracted_patterns = self.load_extracted_patterns(document_name)
        
        # Analyze with field state analyzer
        logger.info(f"Analyzing {document_name} with field state analyzer")
        field_results = self.field_analyzer.analyze_time_series(time_series)
        
        # Analyze at multiple scales
        logger.info(f"Performing multi-scale analysis for {document_name}")
        scale_results = self.multi_scale_analyzer.analyze(time_series)
        
        # Bridge to pattern evolution
        logger.info(f"Bridging {document_name} to pattern evolution")
        bridge_results = self.field_bridge.process_time_series(
            time_series,
            metadata={"document_name": document_name, "source": "climate_field_state_logger"}
        )
        
        # Log results
        logger.info(f"Field State Analysis Results for {document_name}:")
        logger.info(f"Found {len(field_results['patterns'])} field patterns")
        for pattern in field_results["patterns"]:
            logger.info(f"  - {pattern['type']} (magnitude: {pattern['magnitude']:.2f})")
        
        logger.info(f"Multi-Scale Analysis Results for {document_name}:")
        logger.info(f"Found {len(scale_results['cross_scale_patterns'])} cross-scale patterns")
        for pattern in scale_results["cross_scale_patterns"]:
            logger.info(f"  - {pattern['type']} (magnitude: {pattern['magnitude']:.2f})")
        
        logger.info(f"Pattern Evolution Bridge Results for {document_name}:")
        logger.info(f"Created {len(bridge_results['patterns'])} patterns in evolution service")
        
        # Generate visualizations
        self.generate_visualizations(document_name, field_results, scale_results, extracted_patterns)
        
        return {
            "document_name": document_name,
            "field_analysis": field_results,
            "multi_scale_analysis": scale_results,
            "bridge_results": bridge_results,
            "extracted_patterns": extracted_patterns,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_visualizations(self, document_name: str, field_results: Dict[str, Any],
                               scale_results: Dict[str, Any], extracted_patterns: List[Dict[str, Any]]):
        """
        Generate visualizations for the analysis results.
        
        Args:
            document_name: Name of the document
            field_results: Field analysis results
            scale_results: Multi-scale analysis results
            extracted_patterns: Extracted patterns
        """
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Climate Field State Analysis: {document_name}", fontsize=16)
        
        # Plot 1: Field patterns
        axs[0, 0].bar(
            [p["type"] for p in field_results["patterns"]],
            [p["magnitude"] for p in field_results["patterns"]]
        )
        axs[0, 0].set_title("Field Patterns")
        axs[0, 0].set_xlabel("Pattern Type")
        axs[0, 0].set_ylabel("Magnitude")
        axs[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Multi-scale patterns
        axs[0, 1].bar(
            [p["type"] for p in scale_results["cross_scale_patterns"]],
            [p["magnitude"] for p in scale_results["cross_scale_patterns"]]
        )
        axs[0, 1].set_title("Multi-Scale Patterns")
        axs[0, 1].set_xlabel("Pattern Type")
        axs[0, 1].set_ylabel("Magnitude")
        axs[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Extracted patterns
        axs[1, 0].bar(
            [p["name"][:20] + "..." if len(p["name"]) > 20 else p["name"] for p in extracted_patterns],
            [1 for _ in extracted_patterns]
        )
        axs[1, 0].set_title("Extracted Patterns")
        axs[1, 0].set_xlabel("Pattern Name")
        axs[1, 0].set_ylabel("Count")
        axs[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Pattern quality
        quality_counts = {}
        for p in extracted_patterns:
            quality = p.get("quality_state", "unknown")
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        axs[1, 1].bar(quality_counts.keys(), quality_counts.values())
        axs[1, 1].set_title("Pattern Quality Distribution")
        axs[1, 1].set_xlabel("Quality State")
        axs[1, 1].set_ylabel("Count")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{document_name}_field_state_analysis.png")
        plt.close()
        
        logger.info(f"Generated visualizations for {document_name}")
    
    def run_logger(self):
        """
        Run the climate field state logger on all climate risk documents.
        """
        logger.info("Starting Climate Field State Logger")
        
        # Find all climate risk documents
        climate_docs = list(self.climate_risk_dir.glob("*.txt"))
        logger.info(f"Found {len(climate_docs)} climate risk documents")
        
        # Process each document
        results = []
        for doc_path in climate_docs:
            # Load document
            document_data = self.load_climate_document(doc_path)
            
            # Analyze document
            analysis_results = self.analyze_document(document_data)
            
            # Store results
            results.append(analysis_results)
            
            # Save results to JSON
            output_file = self.output_dir / f"{document_data['document_name']}_field_state_analysis.json"
            with open(output_file, "w") as f:
                json.dump(analysis_results, f, indent=2)
            
            logger.info(f"Saved analysis results for {document_data['document_name']} to {output_file}")
        
        # Generate summary report
        self.generate_summary_report(results)
        
        logger.info(f"Climate Field State Logger complete. Results saved to {self.output_dir}")
        
        return results
    
    def generate_summary_report(self, results: List[Dict[str, Any]]):
        """
        Generate a summary report of all analysis results.
        
        Args:
            results: List of analysis results
        """
        # Create report content
        report = f"""# Climate Field State Analysis Summary Report

## Overview

This report provides a summary of field state analysis for {len(results)} climate risk documents.
The analysis includes field state patterns, multi-scale patterns, and integration with extracted patterns.

## Documents Analyzed

{chr(10).join([f"- {result['document_name']}" for result in results])}

## Field State Patterns

The following field state patterns were detected across all documents:

"""
        
        # Aggregate field patterns
        all_field_patterns = []
        for result in results:
            field_patterns = result.get("field_analysis", {}).get("patterns", [])
            for pattern in field_patterns:
                pattern["document"] = result["document_name"]
                all_field_patterns.append(pattern)
        
        # Group by type
        field_patterns_by_type = {}
        for pattern in all_field_patterns:
            pattern_type = pattern["type"]
            if pattern_type not in field_patterns_by_type:
                field_patterns_by_type[pattern_type] = []
            field_patterns_by_type[pattern_type].append(pattern)
        
        # Add to report
        for pattern_type, patterns in field_patterns_by_type.items():
            report += f"### {pattern_type.title()}\n\n"
            report += f"Found in {len(patterns)} documents:\n\n"
            for pattern in patterns:
                report += f"- {pattern['document']}: Magnitude {pattern['magnitude']:.2f}\n"
            report += "\n"
        
        # Add multi-scale patterns
        report += "## Multi-Scale Patterns\n\n"
        
        # Aggregate multi-scale patterns
        all_multi_scale_patterns = []
        for result in results:
            multi_scale_patterns = result.get("multi_scale_analysis", {}).get("cross_scale_patterns", [])
            for pattern in multi_scale_patterns:
                pattern["document"] = result["document_name"]
                all_multi_scale_patterns.append(pattern)
        
        # Group by type
        multi_scale_patterns_by_type = {}
        for pattern in all_multi_scale_patterns:
            pattern_type = pattern["type"]
            if pattern_type not in multi_scale_patterns_by_type:
                multi_scale_patterns_by_type[pattern_type] = []
            multi_scale_patterns_by_type[pattern_type].append(pattern)
        
        # Add to report
        for pattern_type, patterns in multi_scale_patterns_by_type.items():
            report += f"### {pattern_type.title()}\n\n"
            report += f"Found in {len(patterns)} documents:\n\n"
            for pattern in patterns:
                report += f"- {pattern['document']}: Magnitude {pattern['magnitude']:.2f}\n"
            report += "\n"
        
        # Add extracted patterns
        report += "## Extracted Patterns\n\n"
        
        # Aggregate extracted patterns
        all_extracted_patterns = []
        for result in results:
            extracted_patterns = result.get("extracted_patterns", [])
            for pattern in extracted_patterns:
                pattern["document"] = result["document_name"]
                all_extracted_patterns.append(pattern)
        
        # Group by quality state
        extracted_patterns_by_quality = {}
        for pattern in all_extracted_patterns:
            quality = pattern.get("quality_state", "unknown")
            if quality not in extracted_patterns_by_quality:
                extracted_patterns_by_quality[quality] = []
            extracted_patterns_by_quality[quality].append(pattern)
        
        # Add to report
        for quality, patterns in extracted_patterns_by_quality.items():
            report += f"### {quality.title()} Patterns\n\n"
            report += f"Found {len(patterns)} patterns:\n\n"
            for pattern in patterns:
                report += f"- {pattern['name']} ({pattern['document']})\n"
            report += "\n"
        
        # Add conclusion
        report += f"""## Conclusion

This analysis demonstrates the integration of field state analysis with extracted patterns
from climate risk documents. The field state approach provides a quantitative framework
for analyzing climate patterns, while the extracted patterns provide semantic context.

Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        # Save the report
        with open(self.output_dir / "climate_field_state_analysis_summary.md", "w") as f:
            f.write(report)
        
        logger.info("Generated summary report")


def main():
    """Main function."""
    logger = ClimateFieldStateLogger()
    logger.run_logger()


if __name__ == "__main__":
    main()
