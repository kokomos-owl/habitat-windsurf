#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct Climate Risk Persistence

This script directly processes climate risk data from the data/climate_risk directory
using the repository interfaces to store patterns, relationships, and field states
in ArangoDB without relying on the event system.
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
from src.habitat_evolution.adaptive_core.persistence.factory import create_repositories
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.connection_manager import ArangoDBConnectionManager

class DirectClimateRiskPersistence:
    """Direct persistence for climate risk data using repository interfaces."""
    
    def __init__(self):
        """Initialize with necessary components."""
        self.connection_manager = ArangoDBConnectionManager()
        self.db = self.connection_manager.get_db()
        
        # Create repositories
        self.repositories = create_repositories(self.db)
        
        # Data paths
        self.data_dir = Path(__file__).parent.parent.parent.parent.parent / "data" / "climate_risk"
        
        # Statistics for reporting
        self.stats = {
            "files_processed": 0,
            "paragraphs_processed": 0,
            "patterns_detected": 0,
            "relationships_detected": 0,
            "field_states_updated": 0,
            "topologies_created": 0
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
        
        # Directly save field state to repository
        try:
            self.repositories["field_state_repository"].save(field_state)
            logger.info(f"Field state {field_id} saved successfully")
            self.stats["field_states_updated"] += 1
        except Exception as e:
            logger.error(f"Error saving field state: {str(e)}")
        
        # Create topology from flow vectors
        topology = {
            "id": f"topology_{field_id}",
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
        
        # Directly save topology to repository
        try:
            self.repositories["topology_repository"].save(topology)
            logger.info(f"Topology for field {field_id} saved successfully")
            self.stats["topologies_created"] += 1
        except Exception as e:
            logger.error(f"Error saving topology: {str(e)}")
        
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
                
                # Calculate tonic value
                tonic_value = self.calculate_tonic_value(vector)
                
                # Create pattern data
                pattern_data = {
                    "id": pattern_id,
                    "name": entity,
                    "description": paragraph[:100] + "...",
                    "vector": vector,
                    "harmonic_properties": harmonic_properties,
                    "confidence": 0.7 + (0.2 * tonic_value),
                    "tonic_value": tonic_value,
                    "perspective": "climate_risk_assessment",
                    "timestamp": datetime.now().isoformat(),
                    "source_file": filename,
                    "source_paragraph": paragraph[:100] + "..."
                }
                
                # Directly save pattern to repository
                try:
                    self.repositories["pattern_repository"].save(pattern_data)
                    logger.info(f"Pattern {pattern_id} saved successfully")
                    self.stats["patterns_detected"] += 1
                except Exception as e:
                    logger.error(f"Error saving pattern: {str(e)}")
            
            # Extract relationships between entities
            relationships = self.extract_relationships(paragraph, entities)
            
            # Process relationships
            for rel in relationships:
                # Create source pattern
                source_id = f"entity_{uuid.uuid4()}"
                source_pattern = {
                    "id": source_id,
                    "name": rel["source"],
                    "description": f"Entity extracted from {filename}",
                    "vector": self.generate_vector_from_text(rel["source"]),
                    "confidence": 0.8,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Create target pattern
                target_id = f"entity_{uuid.uuid4()}"
                target_pattern = {
                    "id": target_id,
                    "name": rel["target"],
                    "description": f"Entity extracted from {filename}",
                    "vector": self.generate_vector_from_text(rel["target"]),
                    "confidence": 0.8,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Save source and target patterns
                try:
                    self.repositories["pattern_repository"].save(source_pattern)
                    logger.info(f"Pattern {source_id} saved successfully")
                    self.stats["patterns_detected"] += 1
                    
                    self.repositories["pattern_repository"].save(target_pattern)
                    logger.info(f"Pattern {target_id} saved successfully")
                    self.stats["patterns_detected"] += 1
                except Exception as e:
                    logger.error(f"Error saving pattern: {str(e)}")
                    continue
                
                # Create relationship
                relationship_id = f"relationship_{uuid.uuid4()}"
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
                
                # Save relationship
                try:
                    self.repositories["relationship_repository"].save(relationship_data)
                    logger.info(f"Relationship {relationship_id} saved successfully")
                    self.stats["relationships_detected"] += 1
                except Exception as e:
                    logger.error(f"Error saving relationship: {str(e)}")
            
            self.stats["paragraphs_processed"] += 1
        
        self.stats["files_processed"] += 1
    
    def query_patterns(self, limit=10):
        """Query patterns from the repository."""
        logger.info("Querying patterns...")
        
        try:
            patterns = self.repositories["pattern_repository"].find_all(limit=limit)
            logger.info(f"Found {len(patterns)} patterns")
            return patterns
        except Exception as e:
            logger.error(f"Error querying patterns: {str(e)}")
            return []
    
    def query_relationships(self, limit=10):
        """Query relationships from the repository."""
        logger.info("Querying relationships...")
        
        try:
            relationships = self.repositories["relationship_repository"].find_all(limit=limit)
            logger.info(f"Found {len(relationships)} relationships")
            return relationships
        except Exception as e:
            logger.error(f"Error querying relationships: {str(e)}")
            return []
    
    def query_field_states(self):
        """Query field states from the repository."""
        logger.info("Querying field states...")
        
        try:
            field_states = self.repositories["field_state_repository"].find_all()
            logger.info(f"Found {len(field_states)} field states")
            return field_states
        except Exception as e:
            logger.error(f"Error querying field states: {str(e)}")
            return []
    
    def analyze_patterns(self):
        """Analyze patterns to identify key themes."""
        logger.info("Analyzing patterns...")
        
        try:
            patterns = self.query_patterns(limit=100)
            
            # Group patterns by name
            pattern_groups = {}
            for pattern in patterns:
                name = pattern.get("name", "")
                if name not in pattern_groups:
                    pattern_groups[name] = []
                pattern_groups[name].append(pattern)
            
            # Sort by frequency
            sorted_patterns = sorted(pattern_groups.items(), key=lambda x: len(x[1]), reverse=True)
            
            logger.info("Top patterns by frequency:")
            for name, patterns in sorted_patterns[:10]:
                logger.info(f"  {name}: {len(patterns)} occurrences")
            
            return sorted_patterns
        except Exception as e:
            logger.error(f"Error analyzing patterns: {str(e)}")
            return []
    
    def analyze_relationships(self):
        """Analyze relationships to identify key connections."""
        logger.info("Analyzing relationships...")
        
        try:
            relationships = self.query_relationships(limit=100)
            
            # Group relationships by type
            relationship_groups = {}
            for rel in relationships:
                rel_type = rel.get("type", "")
                if rel_type not in relationship_groups:
                    relationship_groups[rel_type] = []
                relationship_groups[rel_type].append(rel)
            
            # Sort by frequency
            sorted_relationships = sorted(relationship_groups.items(), key=lambda x: len(x[1]), reverse=True)
            
            logger.info("Top relationship types by frequency:")
            for rel_type, rels in sorted_relationships[:10]:
                logger.info(f"  {rel_type}: {len(rels)} occurrences")
            
            return sorted_relationships
        except Exception as e:
            logger.error(f"Error analyzing relationships: {str(e)}")
            return []
    
    def run(self):
        """Run the complete persistence process."""
        logger.info("Starting Direct Climate Risk Persistence")
        
        try:
            # Get all climate risk files
            files = self.get_climate_risk_files()
            logger.info(f"Found {len(files)} climate risk files to process")
            
            # Process each file
            for filename in files:
                self.process_file(filename)
            
            # Analyze the persisted data
            self.analyze_patterns()
            self.analyze_relationships()
            
            # Report statistics
            logger.info("Processing completed successfully")
            logger.info(f"Statistics: {json.dumps(self.stats, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error processing climate risk data: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    processor = DirectClimateRiskPersistence()
    processor.run()
