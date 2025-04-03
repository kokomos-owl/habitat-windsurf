#!/usr/bin/env python
"""
Modified test harness for context-aware pattern extraction.

This script sets up the Python path correctly to work with the existing import structure.
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_dir = str(project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import with relative paths that match the established code pattern
from src.habitat_evolution.adaptive_core.emergence.context_aware_extraction.context_aware_extractor import ContextAwareExtractor
from src.habitat_evolution.pattern_aware_rag.context.quality_aware_context import QualityAwarePatternContext
from src.habitat_evolution.pattern_aware_rag.quality_rag.context_aware_rag import ContextAwareRAG
from src.habitat_evolution.pattern_aware_rag.quality_rag.quality_enhanced_retrieval import QualityEnhancedRetrieval
from src.habitat_evolution.adaptive_core.persistence.adapters.in_memory_pattern_repository import InMemoryPatternRepository

class ContextAwareExtractionTest:
    """Test harness for context-aware pattern extraction."""
    
    def __init__(self, data_dir: Path = None):
        """Initialize the test harness.
        
        Args:
            data_dir: Directory containing test data
        """
        self.data_dir = data_dir or project_root / "src" / "habitat_evolution" / "adaptive_core" / "demos" / "test_data"
        self.results_dir = project_root / "src" / "habitat_evolution" / "adaptive_core" / "demos" / "analysis_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize pattern repository
        self.pattern_repository = InMemoryPatternRepository()
        
        # Initialize context-aware RAG
        self.context_aware_rag = ContextAwareRAG(
            pattern_repository=self.pattern_repository,
            window_sizes=[2, 3, 4, 5],
            quality_threshold=0.7,
            data_dir=self.data_dir
        )
        
        logger.info(f"Initialized ContextAwareExtractionTest with data_dir={self.data_dir}")
    
    def run(self):
        """Run the context-aware extraction test."""
        logger.info("Starting context-aware extraction test...")
        
        # Load test document
        test_doc_path = self.data_dir / "climate_risk_sample.txt"
        if not test_doc_path.exists():
            logger.warning(f"Test document not found at {test_doc_path}. Using sample text.")
            test_doc = """
            Climate risk assessment identifies Salt marsh complexes as critical natural barriers.
            Sea level rise threatens coastal infrastructure and Salt marsh habitats.
            Adaptation strategies must consider both engineered and natural solutions.
            Salt marsh restoration provides co-benefits for biodiversity and carbon sequestration.
            """
        else:
            with open(test_doc_path, "r") as f:
                test_doc = f.read()
        
        # Process document to extract patterns
        logger.info("Processing document for pattern extraction...")
        extraction_results = self.context_aware_rag.process_document_for_patterns(test_doc)
        
        # Log extraction statistics
        total_patterns = len(extraction_results.get("patterns", []))
        good_patterns = len([p for p in extraction_results.get("patterns", []) 
                           if p.get("quality_state") == "good"])
        
        logger.info(f"Extracted {total_patterns} patterns, {good_patterns} classified as 'good'")
        logger.info(f"Good pattern ratio: {good_patterns/total_patterns:.1%}" if total_patterns else "No patterns extracted")
        
        # Save results
        results_file = self.results_dir / f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump(extraction_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        return extraction_results

if __name__ == "__main__":
    # Create and run the test
    test = ContextAwareExtractionTest()
    results = test.run()
    
    # Print summary
    print("\nContext-Aware Extraction Test Summary:")
    print("-------------------------------------")
    total_patterns = len(results.get("patterns", []))
    good_patterns = len([p for p in results.get("patterns", []) 
                       if p.get("quality_state") == "good"])
    print(f"Total patterns extracted: {total_patterns}")
    print(f"Good quality patterns: {good_patterns} ({good_patterns/total_patterns:.1%})" if total_patterns else "No patterns extracted")
    print(f"Uncertain quality patterns: {len([p for p in results.get('patterns', []) if p.get('quality_state') == 'uncertain'])}")
    print(f"Poor quality patterns: {len([p for p in results.get('patterns', []) if p.get('quality_state') == 'poor'])}")
    print("-------------------------------------")
    print("Test completed.")
