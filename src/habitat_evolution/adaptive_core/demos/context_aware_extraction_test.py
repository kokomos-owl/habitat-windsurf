"""
Test harness for context-aware pattern extraction with quality assessment paths.

This module demonstrates the context-aware pattern extraction and quality assessment
capabilities, showing how the self-reinforcing feedback mechanism improves pattern
extraction and retrieval capabilities over time.
"""

import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import os

# Import the fix for handling both import styles
from .import_fix import *

from src.habitat_evolution.adaptive_core.emergence.context_aware_extraction.context_aware_extractor import ContextAwareExtractor
from src.habitat_evolution.pattern_aware_rag.context.quality_aware_context import QualityAwarePatternContext
from src.habitat_evolution.pattern_aware_rag.quality_rag.context_aware_rag import ContextAwareRAG
from src.habitat_evolution.pattern_aware_rag.quality_rag.quality_enhanced_retrieval import QualityEnhancedRetrieval
from src.habitat_evolution.adaptive_core.persistence.interfaces.repository_adapter import InMemoryPatternRepository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ContextAwareExtractionTest:
    """Test harness for context-aware pattern extraction."""
    
    def __init__(self, data_dir: Path = None):
        """Initialize the test harness.
        
        Args:
            data_dir: Directory containing test data
        """
        self.data_dir = data_dir or Path(__file__).parent / "test_data"
        self.results_dir = Path(__file__).parent / "analysis_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.pattern_repository = InMemoryPatternRepository()
        self.extractor = ContextAwareExtractor(
            window_sizes=[2, 3, 4, 5],
            quality_threshold=0.7,
            data_dir=self.data_dir
        )
        self.context_aware_rag = ContextAwareRAG(
            pattern_repository=self.pattern_repository,
            window_sizes=[2, 3, 4, 5],
            quality_threshold=0.7,
            quality_weight=0.7,
            coherence_threshold=0.6,
            data_dir=self.data_dir
        )
        
        logger.info(f"Initialized ContextAwareExtractionTest with data_dir={self.data_dir}")
    
    def run_single_document_test(self, document: str) -> Dict[str, Any]:
        """Run test on a single document.
        
        Args:
            document: Document text to process
            
        Returns:
            Dictionary with test results
        """
        logger.info("Running single document test")
        
        # Process document with context-aware extractor
        extraction_results = self.extractor.process_document(document)
        
        # Create quality-aware context
        quality_context = QualityAwarePatternContext()
        quality_context.context_aware_extraction_results = extraction_results
        quality_context.update_from_quality_assessment(self.extractor.quality_assessor)
        
        # Get quality summary
        quality_summary = quality_context.get_quality_summary()
        
        # Log results
        logger.info(f"Extracted {len(extraction_results['entities'])} entities")
        logger.info(f"Quality distribution: good={quality_summary['quality_state_distribution']['good']}, "
                   f"uncertain={quality_summary['quality_state_distribution']['uncertain']}, "
                   f"poor={quality_summary['quality_state_distribution']['poor']}")
        
        return {
            "extraction_results": extraction_results,
            "quality_context": quality_context.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
    
    def run_multi_pass_test(self, document: str, passes: int = 3) -> Dict[str, Any]:
        """Run multi-pass test to demonstrate self-reinforcing feedback.
        
        Args:
            document: Document text to process
            passes: Number of passes to run
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Running multi-pass test with {passes} passes")
        
        results = []
        
        for i in range(passes):
            logger.info(f"Pass {i+1}/{passes}")
            
            # Process document
            pass_result = self.context_aware_rag.process_document_for_patterns(document)
            
            # Store results
            results.append({
                "pass": i+1,
                "extraction_results": pass_result["extraction_results"],
                "quality_context": pass_result["quality_context"],
                "stored_patterns_count": pass_result["stored_patterns_count"],
                "timestamp": datetime.now().isoformat()
            })
            
            # Log pass results
            quality_summary = pass_result["extraction_results"]["quality_summary"]
            logger.info(f"Pass {i+1} results:")
            logger.info(f"  Good entities: {quality_summary['good_entities_count']}")
            logger.info(f"  Uncertain entities: {quality_summary['uncertain_entities_count']}")
            logger.info(f"  Poor entities: {quality_summary['poor_entities_count']}")
            logger.info(f"  Stored patterns: {pass_result['stored_patterns_count']}")
        
        # Calculate improvement metrics
        improvement_metrics = self._calculate_improvement_metrics(results)
        
        return {
            "passes": results,
            "improvement_metrics": improvement_metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_improvement_metrics(self, pass_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate improvement metrics across passes.
        
        Args:
            pass_results: List of pass results
            
        Returns:
            Dictionary with improvement metrics
        """
        if not pass_results or len(pass_results) < 2:
            return {"insufficient_data": True}
        
        first_pass = pass_results[0]
        last_pass = pass_results[-1]
        
        first_good = first_pass["extraction_results"]["quality_summary"]["good_entities_count"]
        last_good = last_pass["extraction_results"]["quality_summary"]["good_entities_count"]
        
        first_uncertain = first_pass["extraction_results"]["quality_summary"]["uncertain_entities_count"]
        last_uncertain = last_pass["extraction_results"]["quality_summary"]["uncertain_entities_count"]
        
        first_poor = first_pass["extraction_results"]["quality_summary"].get("poor_entities_count", 0)
        last_poor = last_pass["extraction_results"]["quality_summary"].get("poor_entities_count", 0)
        
        # Calculate improvement percentages
        good_improvement = ((last_good - first_good) / max(1, first_good)) * 100 if first_good > 0 else 0
        uncertain_reduction = ((first_uncertain - last_uncertain) / max(1, first_uncertain)) * 100 if first_uncertain > 0 else 0
        poor_reduction = ((first_poor - last_poor) / max(1, first_poor)) * 100 if first_poor > 0 else 0
        
        return {
            "good_entities_improvement": good_improvement,
            "uncertain_entities_reduction": uncertain_reduction,
            "poor_entities_reduction": poor_reduction,
            "first_pass_good": first_good,
            "last_pass_good": last_good,
            "first_pass_uncertain": first_uncertain,
            "last_pass_uncertain": last_uncertain,
            "first_pass_poor": first_poor,
            "last_pass_poor": last_poor,
            "passes_count": len(pass_results)
        }
    
    def run_query_test(self, query: str, document: str) -> Dict[str, Any]:
        """Run test with a query to demonstrate RAG capabilities.
        
        Args:
            query: Query string
            document: Document text to process
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Running query test with query: '{query}'")
        
        # Process query with context-aware RAG
        result = self.context_aware_rag.process_with_context_aware_patterns(
            query=query,
            document=document
        )
        
        # Log results
        logger.info(f"Generated response with confidence: {result['retrieval_result']['confidence']}")
        logger.info(f"Quality distribution: {result['retrieval_result']['quality_distribution']}")
        
        return result
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save test results to file.
        
        Args:
            results: Test results to save
            filename: Filename to save to
        """
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {filepath}")
    
    def load_climate_risk_documents(self) -> List[str]:
        """Load climate risk documents from the data directory.
        
        Returns:
            List of document texts
        """
        climate_risk_dir = Path("/Users/prphillips/Documents/GitHub/habitat-windsurf/data/climate_risk")
        documents = []
        
        if not climate_risk_dir.exists():
            logger.warning(f"Climate risk directory not found: {climate_risk_dir}")
            # Fall back to sample document if directory doesn't exist
            return [self._get_sample_document()]
        
        # Find all text files in the climate risk directory
        text_files = list(climate_risk_dir.glob("*.txt"))
        
        if not text_files:
            logger.warning(f"No text files found in {climate_risk_dir}")
            return [self._get_sample_document()]
        
        # Load each document
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
                    logger.info(f"Loaded document: {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _get_sample_document(self) -> str:
        """Get a sample climate risk document as fallback.
        
        Returns:
            Sample document text
        """
        # Sample climate risk document
        return """
        Climate Risk Assessment for Coastal Infrastructure
        
        Executive Summary
        
        This assessment evaluates the climate risks facing coastal infrastructure in the Northeast region. 
        Sea level rise and increased storm intensity pose significant threats to coastal communities.
        Salt marsh complexes provide natural buffers against storm surge but are themselves vulnerable to erosion.
        
        Key Findings
        
        The Northeast region faces multiple climate hazards including sea level rise, coastal flooding, and erosion.
        Salt marsh degradation has accelerated in recent years, reducing natural protection.
        Critical infrastructure including roads, bridges, and water treatment facilities are at high risk.
        Adaptation strategies must address both immediate vulnerabilities and long-term resilience.
        
        Methodology
        
        This assessment used the Coastal Infrastructure Vulnerability Index (CIVI) to evaluate risk levels.
        The CIVI incorporates factors such as elevation, distance from shore, structural integrity, and adaptive capacity.
        Salt marsh health was evaluated using the Marsh Resilience Assessment Protocol.
        Projections are based on IPCC RCP 8.5 scenario through 2050.
        
        Vulnerable Systems
        
        Transportation networks including coastal highways and bridges show high vulnerability scores.
        Water infrastructure including treatment plants and stormwater systems face contamination risks.
        Salt marsh ecosystems are experiencing rapid degradation due to sea level rise and human impacts.
        Energy distribution systems including substations in low-lying areas require immediate attention.
        
        Adaptation Recommendations
        
        Implement living shoreline approaches to reduce erosion while enhancing salt marsh health.
        Elevate critical transportation infrastructure above projected flood levels.
        Relocate vulnerable water treatment facilities to higher ground where feasible.
        Develop salt marsh migration corridors to allow for ecosystem adaptation.
        Establish early warning systems for storm surge and flooding events.
        
        Implementation Timeline
        
        Immediate actions (0-2 years): Conduct detailed vulnerability assessments of all critical infrastructure.
        Short-term actions (2-5 years): Implement priority protection measures for highest-risk assets.
        Medium-term actions (5-10 years): Begin strategic relocation of vulnerable infrastructure.
        Long-term actions (10+ years): Transform coastal development patterns to accommodate climate realities.
        
        Conclusion
        
        Climate risks to coastal infrastructure in the Northeast region are substantial and increasing.
        Salt marsh protection must be integrated with infrastructure adaptation strategies.
        A coordinated, multi-sectoral approach is essential for effective climate resilience.
        """
    
    def run_full_demo(self) -> None:
        """Run full demonstration of context-aware extraction and RAG capabilities."""
        logger.info("Running full demonstration")
        
        # Load climate risk documents
        documents = self.load_climate_risk_documents()
        logger.info(f"Loaded {len(documents)} climate risk documents")
        
        if not documents:
            logger.error("No documents to process")
            return
        
        # Run single document test on first document
        single_doc_results = self.run_single_document_test(documents[0])
        self.save_results(single_doc_results, "context_aware_extraction_results.json")
        
        # Track specific entities of interest (like "Salt")
        entities_of_interest = ["Salt", "Salt marsh", "Salt marsh complexes", "Sea level rise"]
        entity_quality_evolution = {entity: [] for entity in entities_of_interest}
        
        # Run multi-pass test with all documents to show quality evolution
        all_documents = "\n\n".join(documents)
        multi_pass_results = self.run_multi_pass_test(all_documents, passes=3)
        self.save_results(multi_pass_results, "self_reinforcing_feedback_results.json")
        
        # Extract quality evolution for entities of interest
        for pass_result in multi_pass_results["passes"]:
            extraction_results = pass_result["extraction_results"]
            quality_states = extraction_results.get("quality_states", {})
            
            for entity in entities_of_interest:
                # Check each quality state
                for state in ["good", "uncertain", "poor"]:
                    if entity in quality_states.get(state, {}):
                        entity_quality_evolution[entity].append({
                            "pass": pass_result["pass"],
                            "state": state,
                            "data": quality_states[state][entity]
                        })
        
        # Save entity quality evolution
        self.save_results(entity_quality_evolution, "entity_quality_evolution.json")
        
        # Run query test
        query_results = self.run_query_test(
            query="What are the risks to salt marshes from climate change?",
            document=all_documents
        )
        self.save_results(query_results, "quality_aware_rag_results.json")
        
        logger.info("Full demonstration completed")
        
        # Print summary of entity quality evolution
        logger.info("Entity Quality Evolution Summary:")
        for entity, evolution in entity_quality_evolution.items():
            if evolution:
                initial_state = evolution[0]["state"] if evolution else "unknown"
                final_state = evolution[-1]["state"] if evolution else "unknown"
                logger.info(f"  {entity}: {initial_state} -> {final_state}")


if __name__ == "__main__":
    # Run the test
    test = ContextAwareExtractionTest()
    test.run_full_demo()
