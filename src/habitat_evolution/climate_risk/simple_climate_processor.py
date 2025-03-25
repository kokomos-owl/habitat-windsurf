"""
Simple Climate Processor

This script provides a simplified version of the climate processor that focuses on
processing climate risk data through the harmonic I/O system without complex dependencies.
"""

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Any, Optional, Tuple

# Add src to path if not already there
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService, OperationType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('climate_processor.log')
    ]
)

logger = logging.getLogger(__name__)


class SimpleClimateProcessor:
    """
    A simplified processor for climate risk data using the harmonic I/O system.
    
    This processor focuses on the core functionality of processing climate data
    through the harmonic I/O system without complex dependencies.
    """
    
    def __init__(self, io_service: HarmonicIOService, data_dir: str):
        """
        Initialize the simple climate processor.
        
        Args:
            io_service: Harmonic I/O service to use for scheduling operations
            data_dir: Directory containing climate risk data
        """
        self.io_service = io_service
        self.data_dir = Path(data_dir)
        self.processed_files = []
        self.discovered_entities = {}
        self.discovered_domains = {}
        self.discovered_relationships = []
        
    def process_data(self) -> Dict[str, Any]:
        """
        Process climate risk data using the harmonic I/O system.
        
        Returns:
            Dictionary of processing metrics
        """
        start_time = datetime.now()
        logger.info(f"Starting climate data processing from {self.data_dir}")
        
        # Get list of data files
        data_files = list(self.data_dir.glob('*.txt')) + list(self.data_dir.glob('*.csv')) + list(self.data_dir.glob('*.json'))
        logger.info(f"Found {len(data_files)} data files to process")
        
        # Process each file
        for file_path in data_files:
            self._process_file(file_path)
            
        # Calculate metrics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        metrics = {
            "files_processed": len(self.processed_files),
            "entities_discovered": len(self.discovered_entities),
            "domains_discovered": len(self.discovered_domains),
            "relationships_discovered": len(self.discovered_relationships),
            "processing_time_seconds": processing_time,
            "files": self.processed_files
        }
        
        logger.info(f"Processing complete. Processed {len(self.processed_files)} files in {processing_time:.2f} seconds")
        return metrics
    
    def _process_file(self, file_path: Path):
        """
        Process a single data file.
        
        Args:
            file_path: Path to the data file
        """
        logger.info(f"Processing file: {file_path.name}")
        
        try:
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Schedule read operation through harmonic I/O
            self.io_service.schedule_operation(
                OperationType.READ.value,
                self,
                "_process_content",
                (content, file_path.name),
                {},
                {"stability": 0.7, "data_type": "climate_risk"}
            )
            
            # Allow time for processing
            time.sleep(0.5)
            
            # Record processed file
            self.processed_files.append({
                "name": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")
    
    def _process_content(self, content: str, filename: str) -> Dict[str, Any]:
        """
        Process file content through the harmonic I/O system.
        
        Args:
            content: File content to process
            filename: Name of the source file
            
        Returns:
            Processing results
        """
        logger.info(f"Processing content from {filename} ({len(content)} bytes)")
        
        # Extract entities (simplified)
        entities = self._extract_entities(content)
        
        # Extract domains (simplified)
        domains = self._extract_domains(content)
        
        # Extract relationships (simplified)
        relationships = self._extract_relationships(content, entities, domains)
        
        # Update discovered items
        self.discovered_entities.update(entities)
        self.discovered_domains.update(domains)
        self.discovered_relationships.extend(relationships)
        
        results = {
            "entities_count": len(entities),
            "domains_count": len(domains),
            "relationships_count": len(relationships)
        }
        
        logger.info(f"Extracted {len(entities)} entities, {len(domains)} domains, and {len(relationships)} relationships from {filename}")
        return results
    
    def _extract_entities(self, content: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract entities from content (simplified implementation).
        
        Args:
            content: Text content to process
            
        Returns:
            Dictionary of extracted entities
        """
        # Simple entity extraction based on common climate terms
        climate_terms = [
            "temperature", "precipitation", "humidity", "wind", "storm", 
            "flood", "drought", "sea level", "carbon", "emission", "greenhouse",
            "climate change", "global warming", "extreme weather", "adaptation",
            "mitigation", "resilience", "vulnerability"
        ]
        
        entities = {}
        for term in climate_terms:
            if term.lower() in content.lower():
                entity_id = f"entity_{len(entities)}"
                entities[entity_id] = {
                    "name": term,
                    "type": "climate_factor",
                    "mentions": content.lower().count(term.lower())
                }
                
        return entities
    
    def _extract_domains(self, content: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract domains from content (simplified implementation).
        
        Args:
            content: Text content to process
            
        Returns:
            Dictionary of extracted domains
        """
        # Simple domain extraction based on common climate domains
        domain_terms = [
            "atmosphere", "ocean", "land", "cryosphere", "biosphere",
            "urban", "rural", "coastal", "agriculture", "forestry",
            "water resources", "energy", "health", "infrastructure"
        ]
        
        domains = {}
        for term in domain_terms:
            if term.lower() in content.lower():
                domain_id = f"domain_{len(domains)}"
                domains[domain_id] = {
                    "name": term,
                    "type": "climate_domain",
                    "mentions": content.lower().count(term.lower())
                }
                
        return domains
    
    def _extract_relationships(self, content: str, entities: Dict[str, Dict[str, Any]], 
                              domains: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities and domains (simplified implementation).
        
        Args:
            content: Text content to process
            entities: Extracted entities
            domains: Extracted domains
            
        Returns:
            List of extracted relationships
        """
        # Simple relationship extraction based on co-occurrence
        relationships = []
        
        # Create entity-domain relationships based on proximity
        for entity_id, entity in entities.items():
            for domain_id, domain in domains.items():
                # If both terms appear in the content, create a relationship
                if entity["name"].lower() in content.lower() and domain["name"].lower() in content.lower():
                    relationships.append({
                        "source_id": entity_id,
                        "source_type": "entity",
                        "target_id": domain_id,
                        "target_type": "domain",
                        "type": "appears_in",
                        "strength": min(entity["mentions"], domain["mentions"]) / max(entity["mentions"], domain["mentions"])
                    })
        
        return relationships


def create_simple_processor(data_dir: str) -> Tuple[SimpleClimateProcessor, HarmonicIOService]:
    """
    Create a simple climate processor with the harmonic I/O service.
    
    Args:
        data_dir: Directory containing climate risk data
        
    Returns:
        Tuple of (processor, io_service)
    """
    # Create harmonic I/O service
    io_service = HarmonicIOService(base_frequency=0.2, harmonics=3)
    io_service.start()
    
    # Create processor
    processor = SimpleClimateProcessor(
        io_service=io_service,
        data_dir=data_dir
    )
    
    return processor, io_service


def main():
    """Run the simple climate processor."""
    # Set up data directory
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/climate_risk'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create processor and I/O service
    logger.info(f"Creating simple climate processor for data directory: {data_dir}")
    processor, io_service = create_simple_processor(data_dir)
    
    try:
        # Process data
        logger.info("Starting data processing")
        metrics = processor.process_data()
        
        # Log metrics
        logger.info(f"Processing complete. Metrics: {metrics}")
        
        # Save metrics to output directory
        metrics_file = os.path.join(output_dir, f"processing_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_file}")
        
    finally:
        # Ensure I/O service is stopped
        logger.info("Stopping I/O service")
        io_service.stop()


if __name__ == "__main__":
    main()
