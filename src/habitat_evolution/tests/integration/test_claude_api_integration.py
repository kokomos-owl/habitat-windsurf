"""
Integration tests for the Claude API integration with Habitat Evolution.

This module contains comprehensive tests for the Claude API integration,
focusing on pattern extraction from climate risk documents, enhanced query
processing, and constructive dissonance detection.
"""

import asyncio
import json
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[4]))

from src.habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter
from src.habitat_evolution.pattern_aware_rag.services.claude_baseline_service import ClaudeBaselineService
from src.habitat_evolution.pattern_aware_rag.services.enhanced_claude_baseline_service import EnhancedClaudeBaselineService


class TestClaudeAPIIntegration(unittest.TestCase):
    """Test the Claude API integration with Habitat Evolution."""

    def setUp(self):
        """Set up the test environment."""
        # Check if the API key is available
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            self.skipTest("ANTHROPIC_API_KEY not set. Skipping integration tests.")
        
        # Initialize the Claude adapter
        self.claude_adapter = ClaudeAdapter(api_key=self.api_key)
        
        # Initialize the services
        self.claude_baseline_service = ClaudeBaselineService(claude_adapter=self.claude_adapter)
        self.enhanced_claude_baseline_service = EnhancedClaudeBaselineService(claude_adapter=self.claude_adapter)
        
        # Test data paths
        self.test_data_dir = Path(__file__).parents[3] / "tests" / "data"
        self.climate_risk_doc_path = self.test_data_dir / "climate_risk_marthas_vineyard.txt"
        
        # Create test data directory if it doesn't exist
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a sample climate risk document if it doesn't exist
        if not self.climate_risk_doc_path.exists():
            self._create_sample_climate_risk_document()

    def _create_sample_climate_risk_document(self):
        """Create a sample climate risk document for testing."""
        climate_risk_content = """
        # Climate Risk Assessment: Martha's Vineyard
        
        ## Executive Summary
        
        Martha's Vineyard faces significant climate risks over the next 50 years, including sea level rise, 
        increased storm intensity, coastal erosion, and changes in precipitation patterns. This assessment 
        outlines the key risks and potential adaptation strategies.
        
        ## Key Climate Risks
        
        ### Sea Level Rise
        
        Martha's Vineyard is projected to experience sea level rise of 1.5 to 3.1 feet by 2070. This will 
        impact coastal properties, infrastructure, and ecosystems. Areas most at risk include:
        
        - Low-lying coastal areas in Edgartown and Oak Bluffs
        - Barrier beaches and coastal ponds
        - Transportation infrastructure near the coast
        
        ### Extreme Weather Events
        
        Climate models project an increase in the frequency and intensity of storms, including:
        
        - More frequent nor'easters with higher storm surge
        - Potential increase in hurricane intensity
        - Longer periods of drought followed by intense precipitation events
        
        ### Ecosystem Impacts
        
        Climate change will affect the island's ecosystems in several ways:
        
        - Salt marsh migration and potential loss due to sea level rise
        - Changes in marine species composition due to warming waters
        - Stress on freshwater ponds and aquifers from saltwater intrusion
        - Shifts in forest composition as temperatures warm
        
        ## Adaptation Strategies
        
        ### Infrastructure Resilience
        
        - Elevate critical infrastructure above projected flood levels
        - Implement green infrastructure for stormwater management
        - Develop redundant systems for power and transportation
        
        ### Natural Systems Protection
        
        - Preserve and enhance salt marshes to buffer storm impacts
        - Implement living shorelines where appropriate
        - Protect migration corridors for shifting ecosystems
        
        ### Community Preparedness
        
        - Update emergency response plans for more frequent extreme weather
        - Develop community resilience hubs
        - Implement water conservation measures
        
        ## Conclusion
        
        Martha's Vineyard faces significant but manageable climate risks. By implementing proactive 
        adaptation strategies now, the island can reduce vulnerability and build resilience to future 
        climate impacts.
        """
        
        with open(self.climate_risk_doc_path, "w") as f:
            f.write(climate_risk_content)

    async def _test_pattern_extraction(self):
        """Test pattern extraction from a climate risk document."""
        # Load the document
        with open(self.climate_risk_doc_path, "r") as f:
            content = f.read()
        
        # Create a document object
        document = {
            "id": "climate_risk_marthas_vineyard",
            "title": "Climate Risk Assessment: Martha's Vineyard",
            "content": content,
            "metadata": {
                "type": "climate_risk_assessment",
                "region": "Martha's Vineyard",
                "date": datetime.now().isoformat()
            }
        }
        
        # Process the document with Claude
        result = await self.claude_adapter.process_document(document)
        
        # Validate the result
        self.assertIn("patterns", result)
        self.assertGreater(len(result["patterns"]), 0)
        
        # Check pattern structure
        for pattern in result["patterns"]:
            self.assertIn("id", pattern)
            self.assertIn("name", pattern)
            self.assertIn("description", pattern)
            self.assertIn("evidence", pattern)
            self.assertIn("quality_state", pattern)
        
        return result["patterns"]

    async def _test_query_enhancement(self, patterns):
        """Test query enhancement with patterns."""
        # Create a test query
        query = "What are the main climate risks for Martha's Vineyard related to sea level rise?"
        
        # Create context
        context = {
            "region": "Martha's Vineyard",
            "document_type": "climate_risk_assessment"
        }
        
        # Process the query with Claude
        result = await self.claude_adapter.process_query(query, context, patterns)
        
        # Validate the result
        self.assertIn("response", result)
        self.assertIn("query_id", result)
        self.assertIn("timestamp", result)
        self.assertIn("model", result)
        
        return result

    async def _test_constructive_dissonance(self, patterns):
        """Test constructive dissonance detection."""
        # Create a query with potential constructive dissonance
        query = "How might rising temperatures create new opportunities for agriculture on Martha's Vineyard?"
        
        # Process with the enhanced service
        query_id = "test_dissonance_" + datetime.now().strftime("%Y%m%d%H%M%S")
        result = await self.enhanced_claude_baseline_service.enhance_query(
            query_id=query_id,
            query_text=query,
            significance_vector=None
        )
        
        # Validate the result
        self.assertIn("enhanced_query", result)
        self.assertIn("dissonance_detected", result)
        
        return result

    async def _run_all_tests(self):
        """Run all tests in sequence."""
        # Extract patterns from a climate risk document
        patterns = await self._test_pattern_extraction()
        print(f"Extracted {len(patterns)} patterns from climate risk document")
        
        # Test query enhancement with the extracted patterns
        query_result = await self._test_query_enhancement(patterns)
        print(f"Enhanced query with {len(patterns)} patterns")
        print(f"Response: {query_result['response'][:100]}...")
        
        # Test constructive dissonance detection
        dissonance_result = await self._test_constructive_dissonance(patterns)
        print(f"Dissonance detected: {dissonance_result.get('dissonance_detected', False)}")
        print(f"Enhanced query: {dissonance_result.get('enhanced_query', '')[:100]}...")

    def test_claude_api_integration(self):
        """Test the complete Claude API integration."""
        asyncio.run(self._run_all_tests())


if __name__ == "__main__":
    unittest.main()
