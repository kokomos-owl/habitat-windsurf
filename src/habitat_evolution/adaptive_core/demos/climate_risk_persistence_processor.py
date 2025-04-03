#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Climate Risk Persistence Processor

This script processes climate risk data from the data/climate_risk directory
through the Habitat Evolution persistence layer to detect patterns, establish
relationships, and make sense of the climate risk information.
"""

import os
import json
import logging
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Habitat Evolution components
from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector import create_connector
from src.habitat_evolution.adaptive_core.persistence.factory import create_repositories
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.connection_manager import ArangoDBConnectionManager
from src.habitat_evolution.adaptive_core.resonance.tonic_harmonic_metrics import TonicHarmonicMetrics

class ClimateRiskPersistenceProcessor:
    """Processor for climate risk data through the persistence layer."""
    
    def __init__(self):
        """Initialize the processor with necessary components."""
        self.event_bus = LocalEventBus()
        self.connection_manager = ArangoDBConnectionManager()
        self.db = self.connection_manager.get_db()
        
        # Create repositories
        self.repositories = create_repositories(self.db)
        
        # Create persistence connector
        self.connector = create_connector(
            event_bus=self.event_bus,
            db=self.db,
            field_state_repository=self.repositories["field_state_repository"],
            pattern_repository=self.repositories["pattern_repository"],
            relationship_repository=self.repositories["relationship_repository"],
            topology_repository=self.repositories["topology_repository"],
            boundary_repository=self.repositories.get("boundary_repository"),
            predicate_relationship_repository=self.repositories.get("predicate_relationship_repository")
        )
        
        # Subscribe connector to events
        self.event_bus.subscribe("pattern.detected", self.connector.on_pattern_detected)
        self.event_bus.subscribe("pattern.relationship.detected", self.connector.on_pattern_relationship_detected)
        self.event_bus.subscribe("field.state.changed", self.connector.on_field_state_change)
        self.event_bus.subscribe("topology.changed", self.connector.on_topology_change)
        
        # Create metrics service for tonic-harmonic analysis
        self.metrics = TonicHarmonicMetrics()
        
        # Data paths
        self.data_dir = Path(__file__).parent.parent.parent.parent.parent / "data" / "climate_risk"
        
        # Statistics for reporting
        self.stats = {
            "files_processed": 0,
            "paragraphs_processed": 0,
            "patterns_detected": 0,
            "relationships_detected": 0,
            "field_states_updated": 0
        }
    
    def load_text_file(self, filename):
        """Load a text file from the data directory."""
        file_path = self.data_dir / filename
        with open(file_path, 'r') as f:
            return f.read()
    
    def get_climate_risk_files(self):
        """Get a list of all climate risk text files."""
        return [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
    
    def extract_paragraphs(self, text):
        """Extract paragraphs from text content."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs
    
    def extract_entities(self, paragraph):
        """Extract key entities from a paragraph using simple NLP techniques."""
        # This is a simplified approach - in a real system, you would use NLP
        # Look for capitalized phrases as potential entities
        entity_pattern = r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b'
        entities = re.findall(entity_pattern, paragraph)
        
        # Filter out common words that might be capitalized at start of sentences
        common_words = {"The", "A", "An", "This", "That", "These", "Those", "It", "They"}
        entities = [e for e in entities if e.split()[0] not in common_words]
        
        # Deduplicate
        entities = list(set(entities))
        
        return entities[:5]  # Limit to top 5 entities
    
    def extract_relationships(self, paragraph, entities):
        """Extract potential relationships between entities."""
        relationships = []
        
        if len(entities) < 2:
            return relationships
        
        # Simple relationship extraction based on proximity
        for i in range(len(entities) - 1):
            for j in range(i + 1, len(entities)):
                source = entities[i]
                target = entities[j]
                
                # Check if both entities appear in the paragraph
                if source in paragraph and target in paragraph:
                    # Extract text between entities to infer relationship
                    pattern = f"{re.escape(source)}(.*?){re.escape(target)}"
                    matches = re.findall(pattern, paragraph)
                    
                    if matches:
                        # Use connecting text as predicate, or default to "related_to"
                        predicate = matches[0].strip()
                        if len(predicate) > 50 or len(predicate) < 2:
                            predicate = "related_to"
                        else:
                            # Clean up predicate
                            predicate = re.sub(r'[^\w\s]', '', predicate).strip()
                            predicate = re.sub(r'\s+', '_', predicate)
                        
                        relationships.append({
                            "source": source,
                            "predicate": predicate,
                            "target": target
                        })
        
        return relationships
    
    def generate_vector_from_text(self, text):
        """Generate a vector representation from text."""
        # This is a simplified approach - in a real system, you would use embeddings
        # Here we're just creating a deterministic vector based on the text
        import hashlib
        
        # Create a hash of the text
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert hash bytes to floats between 0 and 1
        vector = [float(b) / 255.0 for b in hash_bytes[:5]]
        
        return vector
        
    def calculate_tonic_value(self, vector):
        """Calculate a tonic value from a vector."""
        # Simple calculation based on vector values
        # In a real system, this would use more sophisticated metrics
        if not vector:
            return 0.5
            
        # Calculate average of vector components
        avg = sum(vector) / len(vector)
        
        # Calculate variance
        variance = sum((x - avg) ** 2 for x in vector) / len(vector)
        
        # Combine average and variance for tonic value
        tonic_value = (avg + (1.0 - variance)) / 2.0
        
        return min(1.0, max(0.0, tonic_value))
    
    def generate_harmonic_properties(self, text):
        """Generate harmonic properties from text."""
        # This is a simplified approach - in a real system, you would use actual analysis
        import hashlib
        
        # Create a hash of the text
        hash_obj = hashlib.md5(text.encode())
        hash_int = int.from_bytes(hash_obj.digest(), byteorder='big')
        
        # Generate harmonic properties
        return {
            "frequency": (hash_int % 100) / 100.0,
            "amplitude": ((hash_int // 100) % 100) / 100.0,
            "phase": ((hash_int // 10000) % 100) / 100.0
        }
    
    def process_file(self, filename):
        """Process a single climate risk file."""
        logger.info(f"Processing file: {filename}")
        
        # Load file content
        content = self.load_text_file(filename)
        
        # Extract paragraphs
        paragraphs = self.extract_paragraphs(content)
        
        # Create field state for this document
        field_id = f"field_{filename.replace('.txt', '')}"
        
        # Extract key themes as resonance centers
        all_entities = []
        for paragraph in paragraphs:
            all_entities.extend(self.extract_entities(paragraph))
        
        # Get unique entities, sorted by frequency
        from collections import Counter
        entity_counts = Counter(all_entities)
        resonance_centers = [entity for entity, _ in entity_counts.most_common(10)]
        
        # Create density centers (simplified approach)
        density_centers = resonance_centers[:5] if len(resonance_centers) >= 5 else resonance_centers
        
        # Create flow vectors (simplified approach)
        flow_vectors = []
        if len(resonance_centers) >= 2:
            for i in range(min(5, len(resonance_centers) - 1)):
                flow_vectors.append({
                    "source": resonance_centers[i],
                    "target": resonance_centers[i + 1],
                    "strength": 0.7 + (i * 0.05)  # Varying strengths
                })
        
        # Create field state
        field_state = {
            "id": field_id,
            "name": filename.replace('.txt', '').replace('_', ' ').title(),
            "description": f"Field state derived from {filename}",
            "resonance_centers": resonance_centers,
            "dimensionality": 5,  # Fixed dimensionality for simplicity
            "density_centers": density_centers,
            "flow_vectors": flow_vectors,
            "timestamp": datetime.now().isoformat()
        }
        
        # Process field state through connector
        self.event_bus.publish(Event.create(
            "field.state.changed",
            {
                "field_id": field_id,
                "previous_state": {},  # No previous state for new fields
                "new_state": field_state,
                "metadata": {"source": "climate_risk_processor"}
            }
        ))
        
        # Create topology from flow vectors
        topology = {
            "field_id": field_id,
            "nodes": resonance_centers,
            "edges": [
                {
                    "source": vector["source"],
                    "target": vector["target"],
                    "weight": vector["strength"]
                }
                for vector in flow_vectors
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        # Process topology through connector
        self.event_bus.publish(Event.create(
            "topology.changed",
            {
                "field_id": field_id,
                "previous_topology": {},
                "new_topology": topology,
                "metadata": {"source": "climate_risk_processor"}
            }
        ))
        
        self.stats["field_states_updated"] += 1
        
        # Process each paragraph to extract patterns and relationships
        for paragraph in paragraphs:
            # Skip very short paragraphs
            if len(paragraph) < 50:
                continue
                
            # Extract entities
            entities = self.extract_entities(paragraph)
            
            # Skip paragraphs with no entities
            if not entities:
                continue
            
            # Process entities as patterns
            for entity in entities:
                pattern_id = f"pattern_{uuid.uuid4()}"
                
                # Generate vector from entity and context
                vector = self.generate_vector_from_text(entity + " " + paragraph[:100])
                
                # Generate harmonic properties
                harmonic_properties = self.generate_harmonic_properties(entity + " " + paragraph[:100])
                
                # Create pattern data
                pattern_data = {
                    "id": pattern_id,
                    "name": entity,
                    "description": paragraph[:100] + "...",
                    "vector": vector,
                    "harmonic_properties": harmonic_properties,
                    "confidence": 0.7 + (0.2 * self.calculate_tonic_value(vector)),
                    "tonic_value": self.calculate_tonic_value(vector),
                    "perspective": "climate_risk_assessment",
                    "timestamp": datetime.now().isoformat(),
                    "source_file": filename,
                    "source_paragraph": paragraph[:100] + "..."
                }
                
                # Process pattern through connector
                self.event_bus.publish(Event.create(
                    "pattern.detected",
                    {
                        "pattern_id": pattern_id,
                        "pattern_data": pattern_data,
                        "metadata": {"source": "climate_risk_processor"}
                    }
                ))
                
                self.stats["patterns_detected"] += 1
            
            # Extract relationships between entities
            relationships = self.extract_relationships(paragraph, entities)
            
            # Process relationships
            for rel in relationships:
                relationship_id = f"relationship_{uuid.uuid4()}"
                
                # Create source pattern if not exists
                source_id = f"entity_{uuid.uuid4()}"
                source_pattern = {
                    "id": source_id,
                    "name": rel["source"],
                    "description": f"Entity extracted from {filename}",
                    "vector": self.generate_vector_from_text(rel["source"]),
                    "confidence": 0.8,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Create target pattern if not exists
                target_id = f"entity_{uuid.uuid4()}"
                target_pattern = {
                    "id": target_id,
                    "name": rel["target"],
                    "description": f"Entity extracted from {filename}",
                    "vector": self.generate_vector_from_text(rel["target"]),
                    "confidence": 0.8,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Process patterns through connector
                self.connector.on_pattern_detected(
                    pattern_id=source_id,
                    pattern_data=source_pattern,
                    metadata={"source": "climate_risk_processor"}
                )
                
                self.connector.on_pattern_detected(
                    pattern_id=target_id,
                    pattern_data=target_pattern,
                    metadata={"source": "climate_risk_processor"}
                )
                
                # Create relationship data
                relationship_data = {
                    "id": relationship_id,
                    "source_id": source_id,
                    "target_id": target_id,
                    "type": rel["predicate"],
                    "strength": 0.75,
                    "harmonic_properties": self.generate_harmonic_properties(rel["predicate"]),
                    "perspective": "climate_risk_assessment",
                    "timestamp": datetime.now().isoformat(),
                    "source_file": filename,
                    "source_paragraph": paragraph[:100] + "..."
                }
                
                # Process relationship through connector
                self.event_bus.publish(Event.create(
                    "pattern.relationship.detected",
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "relationship_data": relationship_data,
                        "metadata": {"source": "climate_risk_processor"}
                    }
                ))
                
                self.stats["patterns_detected"] += 2
                self.stats["relationships_detected"] += 1
            
            self.stats["paragraphs_processed"] += 1
        
        self.stats["files_processed"] += 1
    
    def verify_persistence(self):
        """Verify that data was correctly persisted."""
        logger.info("Verifying persistence...")
        
        # Check patterns
        pattern_count = len(self.repositories["pattern_repository"].find_all())
        logger.info(f"Found {pattern_count} patterns in the database")
        
        # Check relationships
        relationship_count = len(self.repositories["relationship_repository"].find_all())
        logger.info(f"Found {relationship_count} relationships in the database")
        
        # Check field states
        field_state_count = len(self.repositories["field_state_repository"].find_all())
        logger.info(f"Found {field_state_count} field states in the database")
        
        # Check topology
        topology_count = len(self.repositories["topology_repository"].find_all())
        logger.info(f"Found {topology_count} topologies in the database")
    
    def run(self):
        """Run the complete processor."""
        logger.info("Starting Climate Risk Persistence Processor")
        
        try:
            # Get all climate risk files
            files = self.get_climate_risk_files()
            logger.info(f"Found {len(files)} climate risk files to process")
            
            # Process each file
            for filename in files:
                self.process_file(filename)
            
            # Verify persistence
            self.verify_persistence()
            
            # Report statistics
            logger.info("Processing completed successfully")
            logger.info(f"Statistics: {json.dumps(self.stats, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error processing climate risk data: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    processor = ClimateRiskPersistenceProcessor()
    processor.run()
