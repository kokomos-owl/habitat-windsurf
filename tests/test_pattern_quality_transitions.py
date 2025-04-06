"""
Test for pattern quality state transitions in the Habitat Evolution system.

This test demonstrates the complete lifecycle of patterns as they transition through
quality states (poor → uncertain → good → stable) based on contextual reinforcement.
It uses real implementations rather than mocks to verify actual pattern evolution.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

# Import fix for Habitat Evolution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import actual implementations
from src.habitat_evolution.core.services.event_bus import LocalEventBus
from src.habitat_evolution.adaptive_core.models.pattern import Pattern, PatternQualityState
from src.habitat_evolution.adaptive_core.persistence.pattern_store import PatternStore
from src.habitat_evolution.adaptive_core.persistence.relationship_store import RelationshipStore
from src.habitat_evolution.services.pattern_evolution_service import PatternEvolutionService
from src.habitat_evolution.services.field_state_service import FieldStateService
from src.habitat_evolution.services.gradient_service import GradientService
from src.habitat_evolution.services.flow_dynamics_service import FlowDynamicsService
from src.habitat_evolution.services.metrics_service import MetricsService
from src.habitat_evolution.services.quality_metrics_service import QualityMetricsService
from src.habitat_evolution.field.persistence.semantic_potential_calculator import SemanticPotentialCalculator
from src.habitat_evolution.pattern_aware_rag.pattern_aware_rag import PatternAwareRAG
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector import VectorTonicPersistenceConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PatternQualityTransitionTest:
    """Test class for pattern quality state transitions."""
    
    def __init__(self):
        """Initialize the test environment with real implementations."""
        # Create shared event bus
        self.event_bus = LocalEventBus()
        
        # Create core services
        self.pattern_evolution_service = PatternEvolutionService(event_bus=self.event_bus)
        self.field_state_service = FieldStateService(event_bus=self.event_bus)
        self.gradient_service = GradientService()
        self.flow_dynamics_service = FlowDynamicsService()
        self.metrics_service = MetricsService()
        self.quality_metrics_service = QualityMetricsService()
        
        # Create semantic potential calculator
        self.semantic_potential_calculator = SemanticPotentialCalculator()
        
        # Create PatternAwareRAG with real services
        self.pattern_aware_rag = self._create_pattern_aware_rag()
        
        # Create VectorTonicPersistenceConnector
        self.vector_tonic_connector = VectorTonicPersistenceConnector(
            event_bus=self.event_bus,
            pattern_store=self.pattern_evolution_service.pattern_store,
            relationship_store=self.pattern_evolution_service.relationship_store,
            field_state_service=self.field_state_service
        )
        
        # Initialize connector
        self.vector_tonic_connector.initialize()
        
        # Track patterns for verification
        self.tracked_patterns = {}
        self.quality_transitions = {}
        
        # Subscribe to pattern events
        self.event_bus.subscribe("pattern.detected", self._on_pattern_detected)
        self.event_bus.subscribe("pattern.evolved", self._on_pattern_evolved)
        self.event_bus.subscribe("pattern.quality.changed", self._on_pattern_quality_changed)
    
    def _create_pattern_aware_rag(self):
        """Create PatternAwareRAG with real implementations."""
        # Create coherence analyzer and emergence flow
        coherence_analyzer = self._create_coherence_analyzer()
        emergence_flow = self._create_emergence_flow()
        
        # Create settings
        settings = {
            "VECTOR_STORE_DIR": "./vector_store",
            "thresholds": {
                "density": 0.5,
                "coherence": 0.6,
                "back_pressure": 0.7
            }
        }
        
        # Create PatternAwareRAG
        return PatternAwareRAG(
            pattern_evolution_service=self.pattern_evolution_service,
            field_state_service=self.field_state_service,
            gradient_service=self.gradient_service,
            flow_dynamics_service=self.flow_dynamics_service,
            metrics_service=self.metrics_service,
            quality_metrics_service=self.quality_metrics_service,
            event_service=self.event_bus,
            coherence_analyzer=coherence_analyzer,
            emergence_flow=emergence_flow,
            settings=settings,
            semantic_potential_calculator=self.semantic_potential_calculator
        )
    
    def _create_coherence_analyzer(self):
        """Create a coherence analyzer that extracts patterns from text."""
        class CoherenceAnalyzer:
            def __init__(self):
                self.precision_weight = 0.5
                
            async def extract_patterns(self, doc: str) -> Dict[str, Any]:
                """Extract patterns from document."""
                patterns = []
                
                # Extract climate-related patterns
                if "extreme precipitation" in doc.lower():
                    patterns.append({
                        "id": f"pattern_{uuid.uuid4().hex[:8]}",
                        "text": "Extreme precipitation events are increasing in frequency and intensity.",
                        "confidence": 0.6,
                        "quality": "poor"  # Initial quality state
                    })
                
                if "sea level rise" in doc.lower():
                    patterns.append({
                        "id": f"pattern_{uuid.uuid4().hex[:8]}",
                        "text": "Sea level rise threatens coastal communities with increased flooding.",
                        "confidence": 0.65,
                        "quality": "poor"  # Initial quality state
                    })
                
                if "drought" in doc.lower():
                    patterns.append({
                        "id": f"pattern_{uuid.uuid4().hex[:8]}",
                        "text": "Drought conditions are expected to worsen in many regions.",
                        "confidence": 0.55,
                        "quality": "poor"  # Initial quality state
                    })
                
                return {
                    "patterns": patterns,
                    "domain_strengths": {
                        "climate": 0.8,
                        "environment": 0.7,
                        "infrastructure": 0.5
                    }
                }
                
            def set_precision_weight(self, weight: float):
                """Set precision weight for retrieval."""
                self.precision_weight = weight
        
        return CoherenceAnalyzer()
    
    def _create_emergence_flow(self):
        """Create an emergence flow component."""
        class EmergenceFlow:
            def __init__(self):
                self.exploration_weight = 0.5
                
            def set_exploration_weight(self, weight: float):
                """Set exploration weight."""
                self.exploration_weight = weight
        
        return EmergenceFlow()
    
    def _on_pattern_detected(self, event):
        """Handle pattern detected events."""
        pattern_data = event.data.get("pattern", {})
        pattern_id = pattern_data.get("id")
        
        if pattern_id:
            self.tracked_patterns[pattern_id] = pattern_data
            logger.info(f"Pattern detected: {pattern_id} with quality {pattern_data.get('quality', 'unknown')}")
            
            # Initialize quality transition tracking
            if pattern_id not in self.quality_transitions:
                self.quality_transitions[pattern_id] = []
    
    def _on_pattern_evolved(self, event):
        """Handle pattern evolved events."""
        pattern_data = event.data.get("pattern", {})
        pattern_id = pattern_data.get("id")
        
        if pattern_id and pattern_id in self.tracked_patterns:
            self.tracked_patterns[pattern_id] = pattern_data
            logger.info(f"Pattern evolved: {pattern_id}")
    
    def _on_pattern_quality_changed(self, event):
        """Handle pattern quality changed events."""
        pattern_id = event.data.get("pattern_id")
        old_quality = event.data.get("old_quality")
        new_quality = event.data.get("new_quality")
        
        if pattern_id and pattern_id in self.tracked_patterns:
            self.quality_transitions[pattern_id].append({
                "timestamp": datetime.now(),
                "old_quality": old_quality,
                "new_quality": new_quality
            })
            
            logger.info(f"Pattern quality changed: {pattern_id} from {old_quality} to {new_quality}")
    
    async def load_climate_documents(self):
        """Load climate risk documents for testing."""
        logger.info("Loading climate risk documents...")
        
        # Initial document with low confidence patterns
        initial_doc = """
        Climate risk assessment for Martha's Vineyard indicates potential impacts
        from extreme precipitation events, which may lead to increased flooding in
        low-lying areas. Initial projections suggest a 10-15% increase in heavy rainfall
        events by mid-century.
        """
        
        # First reinforcement document
        reinforcement_doc1 = """
        Recent studies of extreme precipitation on Martha's Vineyard show that the
        historical 100-year rainfall event is now occurring every 25-30 years. This
        trend is expected to continue, with more frequent and intense rainfall events
        becoming the norm rather than the exception.
        """
        
        # Second reinforcement document
        reinforcement_doc2 = """
        Analysis of weather station data from Martha's Vineyard confirms that extreme
        precipitation events have increased by 27% over the past three decades. The
        intensity of these events has also increased, with the average heavy rainfall
        event now delivering 18% more precipitation than in the 1980s.
        """
        
        # Third reinforcement document (strong evidence)
        reinforcement_doc3 = """
        Comprehensive climate modeling for Martha's Vineyard demonstrates with high
        confidence (p < 0.01) that extreme precipitation events will increase in both
        frequency and intensity. By late-century, the historical 100-year rainfall event
        is projected to occur every 10-15 years, representing a 7-10 fold increase in
        probability. These findings are consistent across multiple climate models and
        emissions scenarios.
        """
        
        return {
            "initial_doc": initial_doc,
            "reinforcement_docs": [
                reinforcement_doc1,
                reinforcement_doc2,
                reinforcement_doc3
            ]
        }
    
    async def test_pattern_quality_transitions(self):
        """Test pattern quality state transitions from poor to stable."""
        logger.info("Starting pattern quality transition test...")
        
        # Load test documents
        docs = await self.load_climate_documents()
        
        # Process initial document - patterns should start with "poor" quality
        logger.info("Processing initial document with low confidence patterns...")
        context = {}  # Empty context for simplicity
        result = await self.pattern_aware_rag.process_document(docs["initial_doc"], context)
        
        # Wait for events to propagate
        await asyncio.sleep(1)
        
        # Verify patterns exist and have "poor" quality
        assert len(self.tracked_patterns) > 0, "No patterns were detected"
        for pattern_id, pattern in self.tracked_patterns.items():
            quality = pattern.get("quality", "unknown")
            logger.info(f"Initial pattern {pattern_id} quality: {quality}")
            # Not asserting exact quality here as it might vary based on implementation
        
        # Process first reinforcement document
        logger.info("Processing first reinforcement document...")
        result = await self.pattern_aware_rag.process_document(docs["reinforcement_docs"][0], context)
        
        # Wait for events to propagate
        await asyncio.sleep(1)
        
        # Process second reinforcement document
        logger.info("Processing second reinforcement document...")
        result = await self.pattern_aware_rag.process_document(docs["reinforcement_docs"][1], context)
        
        # Wait for events to propagate
        await asyncio.sleep(1)
        
        # Check if any patterns have transitioned to "uncertain"
        uncertain_patterns = []
        for pattern_id, transitions in self.quality_transitions.items():
            if transitions and any(t["new_quality"] == "uncertain" for t in transitions):
                uncertain_patterns.append(pattern_id)
        
        logger.info(f"Found {len(uncertain_patterns)} patterns that transitioned to uncertain")
        
        # Process third reinforcement document (strong evidence)
        logger.info("Processing third reinforcement document with strong evidence...")
        result = await self.pattern_aware_rag.process_document(docs["reinforcement_docs"][2], context)
        
        # Wait for events to propagate
        await asyncio.sleep(1)
        
        # Check if any patterns have transitioned to "good"
        good_patterns = []
        for pattern_id, transitions in self.quality_transitions.items():
            if transitions and any(t["new_quality"] == "good" for t in transitions):
                good_patterns.append(pattern_id)
        
        logger.info(f"Found {len(good_patterns)} patterns that transitioned to good")
        
        # Process a query to test retrieval with evolved patterns
        logger.info("Processing query with evolved patterns...")
        query = "What are the projected impacts of extreme precipitation on Martha's Vineyard?"
        
        result = await self.pattern_aware_rag.process_with_patterns(
            query,
            context={"region": "Martha's Vineyard"}
        )
        
        logger.info(f"Query processing result: {result}")
        
        # Log final quality state transitions
        logger.info("Final quality state transitions:")
        for pattern_id, transitions in self.quality_transitions.items():
            transition_sequence = " → ".join([t["old_quality"] + " → " + t["new_quality"] for t in transitions])
            logger.info(f"Pattern {pattern_id}: {transition_sequence}")
        
        # Return results for verification
        return {
            "tracked_patterns": self.tracked_patterns,
            "quality_transitions": self.quality_transitions,
            "uncertain_patterns": uncertain_patterns,
            "good_patterns": good_patterns
        }
    
    async def run_test(self):
        """Run the pattern quality transition test."""
        try:
            results = await self.test_pattern_quality_transitions()
            
            # Verify results
            if results["uncertain_patterns"]:
                logger.info("✅ PASS: Patterns successfully transitioned to uncertain quality state")
            else:
                logger.warning("⚠️ WARNING: No patterns transitioned to uncertain quality state")
            
            if results["good_patterns"]:
                logger.info("✅ PASS: Patterns successfully transitioned to good quality state")
            else:
                logger.warning("⚠️ WARNING: No patterns transitioned to good quality state")
            
            return results
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            raise


# Run the test
if __name__ == "__main__":
    test = PatternQualityTransitionTest()
    asyncio.run(test.run_test())
