"""
Vector-Tonic Persistence Connector Usage Example.

This module demonstrates how to use the Vector-Tonic Persistence Connector
in a real-world scenario, processing documents and capturing patterns,
field states, and relationships in ArangoDB.
"""

import logging
import json
import os
import time
from datetime import datetime

from src.habitat_evolution.adaptive_core.event_bus import LocalEventBus
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import VectorTonicWindowIntegrator
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector import create_connector
from src.habitat_evolution.adaptive_core.persistence.arangodb.connection import ArangoDBConnectionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sample_documents(directory_path):
    """Load sample documents from a directory.
    
    Args:
        directory_path: Path to directory containing sample documents
        
    Returns:
        List of document dictionaries
    """
    documents = []
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        logger.warning(f"Directory not found: {directory_path}")
        # Create sample documents in memory
        documents = [
            {
                "id": "climate_risk_report_2025",
                "content": """
                Climate Risk Assessment Report 2025
                
                This report analyzes the impact of climate change on global food security,
                with a focus on vulnerable regions. Rising temperatures and changing precipitation
                patterns are affecting crop yields and water availability. Adaptation strategies
                must be implemented to ensure food security for future generations.
                
                Key findings:
                1. Temperature increases of 1.5°C will reduce crop yields by 5-10% in tropical regions
                2. Water scarcity will affect 40% more agricultural land by 2040
                3. Sustainable farming practices can mitigate 30% of climate-related yield losses
                """
            },
            {
                "id": "renewable_energy_transition_2025",
                "content": """
                Renewable Energy Transition Report 2025
                
                This report examines the global transition to renewable energy sources and its impact
                on climate change mitigation. Solar and wind power capacity has increased significantly,
                reducing carbon emissions and creating new economic opportunities. However, challenges
                remain in energy storage and grid integration.
                
                Key findings:
                1. Renewable energy capacity grew by 15% in the past year
                2. Solar power is now the cheapest form of electricity in many regions
                3. Energy storage technology is advancing rapidly but remains a bottleneck
                4. Policy support is critical for continued renewable energy adoption
                """
            },
            {
                "id": "biodiversity_conservation_2025",
                "content": """
                Biodiversity Conservation Status Report 2025
                
                This report assesses the current state of global biodiversity and conservation efforts.
                Habitat loss, climate change, and pollution continue to threaten species worldwide.
                Protected areas have expanded, but more action is needed to halt biodiversity decline.
                
                Key findings:
                1. 15% of assessed species are at risk of extinction
                2. Protected areas now cover 18% of land and 10% of marine environments
                3. Climate change is becoming a leading driver of biodiversity loss
                4. Nature-based solutions can address both biodiversity loss and climate change
                """
            }
        ]
        return documents
    
    # Load documents from directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r') as file:
                    document = json.load(file)
                    documents.append(document)
            except Exception as e:
                logger.error(f"Error loading document {filename}: {e}")
    
    return documents

def run_connector_example():
    """Run a complete example of the connector usage."""
    # Create event bus
    event_bus = LocalEventBus()
    
    # Create vector-tonic window integrator
    vector_tonic_integrator = VectorTonicWindowIntegrator(event_bus=event_bus)
    vector_tonic_integrator.initialize()
    
    # Create and initialize connector
    connector = create_connector(event_bus)
    
    # Connect to integrator
    connector.connect_to_integrator(vector_tonic_integrator)
    
    logger.info("Connector setup complete")
    
    # Load sample documents
    documents = load_sample_documents("sample_documents")
    
    logger.info(f"Loaded {len(documents)} sample documents")
    
    # Process each document
    for document in documents:
        logger.info(f"Processing document: {document['id']}")
        
        # Process document through connector
        document_id = connector.process_document(document)
        
        # Process document through vector-tonic integrator
        vector_tonic_integrator.process_document(document)
        
        logger.info(f"Document processed: {document_id}")
        
        # Allow time for event processing
        time.sleep(1)
    
    # Wait for all events to be processed
    logger.info("Waiting for event processing to complete...")
    time.sleep(5)
    
    logger.info("Example completed successfully")

def analyze_results():
    """Analyze the results of the connector usage example."""
    # Connect to ArangoDB
    db = ArangoDBConnectionManager().get_db()
    
    # Query patterns
    pattern_count = db.collection("AdaptiveID").count() if db.has_collection("AdaptiveID") else 0
    logger.info(f"Persisted {pattern_count} AdaptiveID documents")
    
    # Query relationships
    relationship_collections = [
        coll for coll in db.collections() 
        if coll["name"].startswith("leads_to") or coll["name"].startswith("depends_on") or coll["name"].startswith("impacts") or coll["name"].startswith("mitigates")
    ]
    
    for coll in relationship_collections:
        rel_count = db.collection(coll["name"]).count()
        logger.info(f"Persisted {rel_count} relationships in collection {coll['name']}")
    
    # Query field states
    field_state_count = db.collection("TonicHarmonicFieldState").count() if db.has_collection("TonicHarmonicFieldState") else 0
    logger.info(f"Persisted {field_state_count} field state documents")
    
    logger.info("Analysis complete")

if __name__ == "__main__":
    run_connector_example()
    analyze_results()
