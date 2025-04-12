"""
Test for the PKM Repository with Claude responses about Boston Harbor.

This test demonstrates how the PKM repository can capture knowledge from
Claude's responses about Boston Harbor sea level rise, creating a structured
knowledge medium that preserves patterns and relationships.
"""

import unittest
import logging
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter
from src.habitat_evolution.pkm.pkm_repository import PKMFile, PKMRepository, create_pkm_from_claude_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPKMRepository(unittest.TestCase):
    """Test cases for the PKM Repository with Claude responses."""
    
    def setUp(self):
        """Set up the test environment."""
        # Set up ArangoDB connection
        self.db_name = "habitat_evolution_test"
        self.arangodb_connection = ArangoDBConnection(
            host="localhost",
            port=8529,
            username="root",
            password="habitat",
            database_name=self.db_name
        )
        
        # Initialize ArangoDB connection
        self.arangodb_connection.initialize()
        
        # Create PKM repository
        self.pkm_repository = PKMRepository(self.arangodb_connection)
        
        # Create Claude adapter
        self.claude_adapter = ClaudeAdapter()
        
        # Sample patterns for Boston Harbor
        self.boston_harbor_patterns = [
            {
                "id": "pattern-1",
                "type": "semantic",
                "content": "Sea level rise in Boston Harbor",
                "metadata": {
                    "confidence": 0.9,
                    "source": "climate_risk_assessment_2023.pdf"
                },
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "pattern-2",
                "type": "statistical",
                "content": "9-21 inches of sea level rise by 2050",
                "metadata": {
                    "confidence": 0.85,
                    "source": "boston_harbor_measurements.csv"
                },
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "pattern-3",
                "type": "semantic",
                "content": "Infrastructure vulnerability in coastal areas",
                "metadata": {
                    "confidence": 0.8,
                    "source": "infrastructure_vulnerability_report.pdf"
                },
                "created_at": datetime.now().isoformat()
            }
        ]
        
        # Sample source documents
        self.source_documents = [
            {
                "id": "doc-1",
                "title": "Climate Risk Assessment 2023",
                "url": "https://example.com/climate_risk_assessment_2023.pdf",
                "metadata": {
                    "author": "Boston Climate Task Force",
                    "year": 2023
                }
            },
            {
                "id": "doc-2",
                "title": "Boston Harbor Measurements Dataset",
                "url": "https://example.com/boston_harbor_measurements.csv",
                "metadata": {
                    "source": "NOAA",
                    "years": "1950-2023"
                }
            },
            {
                "id": "doc-3",
                "title": "Infrastructure Vulnerability Report",
                "url": "https://example.com/infrastructure_vulnerability_report.pdf",
                "metadata": {
                    "author": "Boston City Planning Department",
                    "year": 2024
                }
            }
        ]
        
        # Sample Claude response about Boston Harbor
        self.claude_response = {
            "query_id": "137840fa-8c96-47bb-a793-22256ac53daf",
            "timestamp": "2025-04-12T09:16:38.249262",
            "response": """Here are some of the main impacts of sea level rise on Boston Harbor:

1. Increased flooding - As sea levels rise, low-lying areas around Boston Harbor will be more prone to flooding during high tides and storm surges. This puts homes, businesses, infrastructure, and historic sites at greater risk.

2. Erosion and land loss - Higher sea levels and stronger wave action will accelerate erosion of shorelines, beaches, and coastal wetlands around the harbor. Some land areas may be permanently inundated and lost.

3. Impacts on coastal ecosystems - The harbor's salt marshes, estuaries, and other coastal habitats will be altered as salinity levels change with rising seas. This will affect fish, birds, and other wildlife dependent on these ecosystems.

4. Damage to infrastructure - Critical infrastructure like roads, bridges, sewers, subway and rail lines, and the airport are vulnerable to flooding as seas rise. The cost to protect or relocate this infrastructure will be substantial. 

5. Contamination risks - Many contaminated and hazardous waste sites are located near the harbor. Flooding of these sites could release pollutants into the water and environment.

6. Economic impacts - Businesses in flood-prone areas may need to relocate or invest in costly protection measures. Property values may decline in high-risk locations. Tourism and recreation could also be negatively impacted.

Significant planning and adaptation measures will be needed to make Boston Harbor and its communities more resilient to rising sea levels in the coming decades. This is a major long-term challenge for the region.""",
            "patterns": [],
            "model": "claude-3-opus-20240229",
            "tokens_used": 343
        }
        
        # Sample Claude response with 2050 context
        self.claude_response_with_context = {
            "query_id": "2e6dcae9-4703-450f-92f8-0297bef74542",
            "timestamp": "2025-04-12T09:16:50.289102",
            "response": """Based on the context provided about Boston Harbor in 2050, some of the potential impacts of sea level rise could include:

1. Increased coastal flooding: As sea levels rise, low-lying areas around Boston Harbor will be more prone to flooding during high tides, storm surges, and heavy rainfall events. This can damage infrastructure, homes, and businesses.

2. Erosion and shoreline changes: Rising seas can accelerate erosion along the coastline, altering beaches, wetlands, and other natural habitats. This may impact recreational activities and require costly shoreline protection measures.

3. Saltwater intrusion: As sea levels rise, saltwater can intrude into freshwater aquifers and rivers, compromising water quality and threatening drinking water supplies. This can also affect vegetation and wildlife in coastal ecosystems.

4. Impacts on transportation infrastructure: Key transportation routes and facilities near Boston Harbor, such as roads, bridges, tunnels, and public transit systems, may be increasingly vulnerable to flooding and damage from sea level rise.

5. Economic consequences: The costs associated with adapting to and mitigating the impacts of sea level rise, such as building sea walls or relocating infrastructure, could be significant. Additionally, sea level rise may affect property values, insurance rates, and key industries like tourism and fishing.

Proactive planning, adaptation strategies, and reducing greenhouse gas emissions are important to help minimize the risks posed by sea level rise in Boston Harbor and other coastal areas.""",
            "patterns": [],
            "model": "claude-3-opus-20240229",
            "tokens_used": 315
        }
        
        logger.info("Test setup complete")
    
    def tearDown(self):
        """Clean up after the test."""
        # No need to clean up collections for now
        # The database will be reused for integration with PatternAwareRAG
        logger.info("Test teardown complete")
    
    def test_create_pkm_from_claude_response(self):
        """Test creating a PKM file from a Claude response."""
        # Create PKM file from Claude response
        query = "What are the impacts of sea level rise on Boston Harbor?"
        pkm_file = create_pkm_from_claude_response(
            response=self.claude_response,
            query=query,
            source_documents=self.source_documents,
            patterns=self.boston_harbor_patterns,
            creator_id="test_user"
        )
        
        # Verify PKM file properties
        self.assertEqual(pkm_file.title, f"PKM: {query}")
        self.assertEqual(pkm_file.creator_id, "test_user")
        self.assertEqual(len(pkm_file.source_documents), 3)
        self.assertEqual(len(pkm_file.patterns), 4)  # 3 original patterns + 1 response pattern
        
        # Verify metadata
        self.assertEqual(pkm_file.metadata["query"], query)
        self.assertEqual(pkm_file.metadata["model"], "claude-3-opus-20240229")
        
        # Log PKM file details
        logger.info(f"Created PKM file: {pkm_file.title} (ID: {pkm_file.pkm_id})")
        logger.info(f"PKM file contains {len(pkm_file.patterns)} patterns and references {len(pkm_file.source_documents)} source documents")
        
        return pkm_file
    
    def test_store_and_retrieve_pkm_file(self):
        """Test storing and retrieving a PKM file from the repository."""
        # Create PKM file
        pkm_file = self.test_create_pkm_from_claude_response()
        
        # Store PKM file in repository
        pkm_id = self.pkm_repository.create_pkm_file(pkm_file)
        
        # Verify PKM ID
        self.assertEqual(pkm_id, pkm_file.pkm_id)
        
        # Retrieve PKM file from repository
        retrieved_pkm_file = self.pkm_repository.get_pkm_file(pkm_id)
        
        # Verify retrieved PKM file
        self.assertIsNotNone(retrieved_pkm_file)
        self.assertEqual(retrieved_pkm_file.pkm_id, pkm_file.pkm_id)
        self.assertEqual(retrieved_pkm_file.title, pkm_file.title)
        self.assertEqual(len(retrieved_pkm_file.patterns), len(pkm_file.patterns))
        
        # Log retrieved PKM file details
        logger.info(f"Retrieved PKM file: {retrieved_pkm_file.title} (ID: {retrieved_pkm_file.pkm_id})")
        
        return retrieved_pkm_file
    
    def test_create_multiple_related_pkm_files(self):
        """Test creating multiple related PKM files from Claude responses."""
        # Create first PKM file
        query1 = "What are the impacts of sea level rise on Boston Harbor?"
        pkm_file1 = create_pkm_from_claude_response(
            response=self.claude_response,
            query=query1,
            source_documents=self.source_documents,
            patterns=self.boston_harbor_patterns,
            creator_id="test_user"
        )
        
        # Store first PKM file
        pkm_id1 = self.pkm_repository.create_pkm_file(pkm_file1)
        
        # Create second PKM file with context
        query2 = "What are the impacts of sea level rise?"
        context = {
            "location": "Boston Harbor",
            "time_period": "2050"
        }
        
        pkm_file2 = create_pkm_from_claude_response(
            response=self.claude_response_with_context,
            query=query2,
            source_documents=self.source_documents,
            patterns=self.boston_harbor_patterns,
            creator_id="test_user"
        )
        
        # Add context to metadata
        pkm_file2.metadata["context"] = context
        
        # Store second PKM file
        pkm_id2 = self.pkm_repository.create_pkm_file(pkm_file2)
        
        # Create relationship between PKM files
        relationship_id = self.pkm_repository.create_pkm_relationship(
            from_pkm_id=pkm_id1,
            to_pkm_id=pkm_id2,
            relationship_type="contextual_refinement",
            metadata={
                "context_added": context,
                "created_at": datetime.now().isoformat()
            }
        )
        
        # Get related PKM files
        related_pkm_files = self.pkm_repository.get_related_pkm_files(pkm_id1)
        
        # Verify related PKM files
        self.assertEqual(len(related_pkm_files), 1)
        related_pkm_file, relationship_type = related_pkm_files[0]
        self.assertEqual(related_pkm_file.pkm_id, pkm_id2)
        self.assertEqual(relationship_type, "contextual_refinement")
        
        # Log relationship details
        logger.info(f"Created relationship between PKM files: {pkm_id1} -> {pkm_id2} ({relationship_type})")
        logger.info(f"Related PKM file: {related_pkm_file.title} (ID: {related_pkm_file.pkm_id})")
        
        return pkm_id1, pkm_id2, relationship_id
    
    def test_export_and_import_pkm_file(self):
        """Test exporting and importing a PKM file."""
        # Create and store PKM file
        pkm_file = self.test_create_pkm_from_claude_response()
        pkm_id = self.pkm_repository.create_pkm_file(pkm_file)
        
        # Create export directory if it doesn't exist
        export_dir = os.path.join(os.getcwd(), "pkm_exports")
        os.makedirs(export_dir, exist_ok=True)
        
        # Export PKM file
        export_path = os.path.join(export_dir, f"{pkm_id}.pkm.json")
        export_result = self.pkm_repository.export_pkm_file(pkm_id, export_path)
        
        # Verify export result
        self.assertTrue(export_result)
        self.assertTrue(os.path.exists(export_path))
        
        # Delete PKM file from repository
        delete_result = self.pkm_repository.delete_pkm_file(pkm_id)
        self.assertTrue(delete_result)
        
        # Verify PKM file is deleted
        deleted_pkm_file = self.pkm_repository.get_pkm_file(pkm_id)
        self.assertIsNone(deleted_pkm_file)
        
        # Import PKM file
        import_result = self.pkm_repository.import_pkm_file(export_path)
        
        # Verify import result
        self.assertEqual(import_result, pkm_id)
        
        # Retrieve imported PKM file
        imported_pkm_file = self.pkm_repository.get_pkm_file(pkm_id)
        
        # Verify imported PKM file
        self.assertIsNotNone(imported_pkm_file)
        self.assertEqual(imported_pkm_file.pkm_id, pkm_id)
        self.assertEqual(imported_pkm_file.title, pkm_file.title)
        
        # Log import details
        logger.info(f"Exported and imported PKM file: {imported_pkm_file.title} (ID: {imported_pkm_file.pkm_id})")
        
        # Clean up export file
        os.remove(export_path)
        
        return imported_pkm_file

if __name__ == "__main__":
    unittest.main()
