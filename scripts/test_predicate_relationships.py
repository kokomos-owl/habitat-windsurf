"""
Script to test the PredicateRelationshipRepository with mock data.

This script demonstrates how the PredicateRelationshipRepository handles
different types of predicate relationships between actants.
"""

import os
import logging
import sys
import json
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.predicate_relationship_repository import PredicateRelationshipRepository

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the PredicateRelationshipRepository with mock data."""
    logger.info("Testing PredicateRelationshipRepository with mock data")
    
    # Create a mock for ArangoDBConnectionManager
    with patch('src.habitat_evolution.pattern_aware_rag.persistence.arangodb.predicate_relationship_repository.ArangoDBConnectionManager') as mock_connection_manager:
        # Set up the mock database
        mock_db = MagicMock()
        mock_connection_manager.return_value.get_db.return_value = mock_db
        
        # Create mock collections
        mock_collections = {}
        
        # Create repository
        repository = PredicateRelationshipRepository()
        
        # Mock collections for the repository
        for predicate in repository.specific_predicates + [repository.generic_predicate]:
            mock_collections[predicate] = MagicMock()
            mock_db.collection.side_effect = lambda name: mock_collections.get(name, MagicMock())
        
        # Sample data from regional_observations.json
        sample_actants = [
            {"id": "Pattern/mangrove_forests", "name": "Mangrove Forests"},
            {"id": "Pattern/coastal_communities", "name": "Coastal Communities"},
            {"id": "Pattern/storm_surge", "name": "Storm Surge"},
            {"id": "Pattern/biodiversity", "name": "Biodiversity"},
            {"id": "Pattern/carbon_sequestration", "name": "Carbon Sequestration"}
        ]
        
        # Sample predicate relationships
        sample_relationships = [
            {
                "source": "Pattern/mangrove_forests",
                "predicate": "protects",
                "target": "Pattern/coastal_communities",
                "properties": {
                    "confidence": 0.95,
                    "harmonic_properties": {
                        "frequency": 0.35,
                        "amplitude": 0.88,
                        "phase": 0.12
                    },
                    "observation_count": 12
                }
            },
            {
                "source": "Pattern/mangrove_forests",
                "predicate": "enhances",
                "target": "Pattern/biodiversity",
                "properties": {
                    "confidence": 0.92,
                    "harmonic_properties": {
                        "frequency": 0.28,
                        "amplitude": 0.75,
                        "phase": 0.22
                    },
                    "observation_count": 8
                }
            },
            {
                "source": "Pattern/mangrove_forests",
                "predicate": "reduces_dependence_on",
                "target": "Pattern/storm_surge",
                "properties": {
                    "confidence": 0.88,
                    "harmonic_properties": {
                        "frequency": 0.42,
                        "amplitude": 0.65,
                        "phase": 0.18
                    },
                    "observation_count": 6
                }
            },
            {
                "source": "Pattern/mangrove_forests",
                "predicate": "maintains",
                "target": "Pattern/carbon_sequestration",
                "properties": {
                    "confidence": 0.90,
                    "harmonic_properties": {
                        "frequency": 0.15,
                        "amplitude": 0.82,
                        "phase": 0.05
                    },
                    "observation_count": 9
                }
            }
        ]
        
        # Mock the insert method to return an ID
        for predicate in repository.specific_predicates + [repository.generic_predicate]:
            mock_collections[predicate].insert.side_effect = lambda doc: {"_id": f"{predicate}/{doc['_from']}_{doc['_to']}"}
        
        # Test saving relationships
        logger.info("Testing save_relationship method")
        for relationship in sample_relationships:
            # Mock the insert method to return an ID
            predicate = repository._normalize_predicate(relationship["predicate"])
            if predicate in repository.specific_predicates:
                collection_name = predicate
            else:
                collection_name = repository.generic_predicate
            
            # Save the relationship
            edge_id = repository.save_relationship(
                relationship["source"],
                relationship["predicate"],
                relationship["target"],
                relationship["properties"]
            )
            
            logger.info(f"Created {relationship['predicate']} relationship: {relationship['source']} -> {relationship['target']}")
            logger.info(f"Edge ID: {edge_id}")
            
            # Verify the edge document
            args, kwargs = mock_collections[collection_name].insert.call_args
            edge_doc = args[0]
            logger.info(f"Edge document: {json.dumps(edge_doc, indent=2, default=str)}")
            
            # Add a separator for readability
            logger.info("-" * 80)
        
        # Test finding relationships
        logger.info("Testing find_by_source_and_target method")
        
        # Mock AQL execution to return documents
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{
            "_id": "Protects/Pattern/mangrove_forests_Pattern/coastal_communities",
            "_from": "Pattern/mangrove_forests",
            "_to": "Pattern/coastal_communities",
            "confidence": 0.95,
            "harmonic_properties": json.dumps({
                "frequency": 0.35,
                "amplitude": 0.88,
                "phase": 0.12
            }),
            "observation_count": 12
        }]
        
        mock_db.aql.execute.return_value = mock_cursor
        
        # Find relationships
        results = repository.find_by_source_and_target(
            "Pattern/mangrove_forests",
            "Pattern/coastal_communities",
            "protects"
        )
        
        logger.info(f"Found {len(results)} relationships")
        for result in results:
            logger.info(f"Relationship: {result['_from']} -> {result['_to']}")
            logger.info(f"Confidence: {result['confidence']}")
            logger.info(f"Observation count: {result['observation_count']}")
            logger.info(f"Harmonic properties: {result['harmonic_properties']}")
            
            # Add a separator for readability
            logger.info("-" * 80)
        
        logger.info("PredicateRelationshipRepository mock testing completed successfully")
        
        return True

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("Script completed successfully")
    else:
        logger.error("Script failed")
