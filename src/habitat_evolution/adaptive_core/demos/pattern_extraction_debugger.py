#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pattern Extraction Debugger

This script analyzes the pattern extraction results from the climate risk data
to identify areas for improvement in the extraction algorithm.
"""

import os
import json
import logging
import re
# import spacy - removed due to dependency issues
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Habitat Evolution components
from src.habitat_evolution.adaptive_core.persistence.factory import create_repositories
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.connection_manager import ArangoDBConnectionManager

class PatternExtractionDebugger:
    """Debugger for pattern extraction from climate risk data."""
    
    def __init__(self):
        """Initialize with necessary components."""
        self.connection_manager = ArangoDBConnectionManager()
        self.db = self.connection_manager.get_db()
        
        # Create repositories
        self.repositories = create_repositories(self.db)
        
        # Data paths
        self.data_dir = Path(__file__).parent.parent.parent.parent.parent / "data" / "climate_risk"
        
        # spaCy model loading removed due to dependency issues
        self.spacy_available = False
        logger.info("Using basic NLP techniques for entity extraction.")
        
        # Statistics
        self.stats = {
            "total_patterns": 0,
            "valid_patterns": 0,
            "invalid_patterns": 0,
            "uncertain_patterns": 0,
            "relationship_quality": {
                "good": 0,
                "uncertain": 0,
                "poor": 0
            }
        }
        
        # Pattern quality metrics
        self.pattern_quality = {}
        
        # Entity validation sets
        self.known_entities = set()
        self.domain_specific_terms = set()
        self.load_domain_knowledge()
    
    def load_domain_knowledge(self):
        """Load domain-specific knowledge for validation."""
        # Climate risk domain entities (simplified list)
        self.domain_specific_terms = {
            "Sea Level Rise", "Storm Surge", "Coastal Erosion", "Flooding",
            "Climate Change", "Carbon Emissions", "Greenhouse Gases", "Global Warming",
            "Adaptation", "Mitigation", "Vulnerability", "Resilience",
            "Martha's Vineyard", "Cape Cod", "Nantucket", "Boston Harbor",
            "Ecosystem Services", "Biodiversity", "Habitat Loss", "Species Migration",
            "Infrastructure", "Economic Impact", "Social Vulnerability", "Public Health",
            "Policy Response", "Governance", "Stakeholder Engagement", "Community Planning"
        }
        
        # Add these to known entities
        self.known_entities.update(self.domain_specific_terms)
    
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
    
    def extract_entities_basic(self, paragraph):
        """Extract entities using basic regex (original method)."""
        entity_pattern = r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b'
        entities = re.findall(entity_pattern, paragraph)
        
        # Filter out common words that might be capitalized at start of sentences
        common_words = {"The", "A", "An", "This", "That", "These", "Those", "It", "They"}
        entities = [e for e in entities if e.split()[0] not in common_words]
        
        # Deduplicate
        entities = list(set(entities))
        
        return entities
    
    def extract_entities_spacy(self, paragraph):
        """Extract entities using advanced NLP (placeholder)."""
        # This method is a placeholder since spaCy is not available
        # In a real implementation, this would use more sophisticated NLP
        return []
    
    def validate_entity(self, entity):
        """Validate if an entity is likely to be relevant."""
        # Check if it's a known entity
        if entity in self.known_entities:
            return True, "known_entity"
            
        # Check length (very short entities are often not meaningful)
        if len(entity.split()) == 1 and len(entity) < 4:
            return False, "too_short"
            
        # Check for common false positives
        common_false_positives = {"The", "A", "An", "This", "That", "These", "Those", "It", "They", "I", "We", "You"}
        if entity in common_false_positives:
            return False, "common_word"
            
        # Default to uncertain
        return None, "uncertain"
    
    def analyze_entity_extraction(self, paragraph):
        """Analyze entity extraction quality for a paragraph."""
        # Get entities using basic method only
        basic_entities = self.extract_entities_basic(paragraph)
        spacy_entities = []
        
        # Use only basic entities
        all_entities = basic_entities
        
        # Validate each entity
        valid_entities = []
        invalid_entities = []
        uncertain_entities = []
        
        for entity in all_entities:
            is_valid, reason = self.validate_entity(entity)
            
            if is_valid:
                valid_entities.append((entity, reason))
            elif is_valid is False:
                invalid_entities.append((entity, reason))
            else:
                uncertain_entities.append((entity, reason))
        
        # Calculate metrics (simplified since we only have basic entities)
        only_in_basic = basic_entities
        only_in_spacy = []
        in_both = []
        
        # Update statistics
        self.stats["total_patterns"] += len(all_entities)
        self.stats["valid_patterns"] += len(valid_entities)
        self.stats["invalid_patterns"] += len(invalid_entities)
        
        # Return analysis results
        return {
            "paragraph": paragraph[:100] + "..." if len(paragraph) > 100 else paragraph,
            "basic_entities": basic_entities,
            "spacy_entities": spacy_entities,
            "valid_entities": valid_entities,
            "invalid_entities": invalid_entities,
            "uncertain_entities": uncertain_entities,
            "only_in_basic": only_in_basic,
            "only_in_spacy": only_in_spacy,
            "in_both": in_both
        }
    
    def extract_relationships(self, paragraph, entities):
        """Extract relationships between entities in a paragraph."""
        relationships = []
        
        # Simple relationship extraction based on proximity
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Find positions of entities in the paragraph
                pos1 = paragraph.find(entity1)
                pos2 = paragraph.find(entity2)
                
                if pos1 >= 0 and pos2 >= 0:
                    # Extract text between entities if they're close enough
                    start = min(pos1 + len(entity1), pos2 + len(entity2))
                    end = max(pos1, pos2)
                    
                    if abs(end - start) < 100:  # Arbitrary threshold
                        between_text = paragraph[start:end].strip()
                        
                        # Look for relationship verbs
                        for verb in ["is", "was", "are", "were", "has", "have", "causes", "affects", "impacts", "increases", "decreases"]:
                            if verb in between_text.lower():
                                if pos1 < pos2:
                                    relationships.append((entity1, verb, entity2))
                                else:
                                    relationships.append((entity2, verb, entity1))
                                break
        
        return {"relationships": relationships, "quality_assessment": {"good": [], "uncertain": [], "poor": []}}
    
    def analyze_relationship_extraction(self, paragraph, entities):
        """Analyze relationship extraction quality for a paragraph."""
        relationships = self.extract_relationships(paragraph, entities)
        
        # Assess quality of each relationship
        quality_assessment = {
            "good": [],
            "uncertain": [],
            "poor": []
        }
        
        for rel in relationships["relationships"]:
            # Simple heuristic for relationship quality
            if rel[0] in entities and rel[2] in entities:
                # Both entities are valid
                quality_assessment["good"].append(rel)
            elif rel[0] in entities or rel[2] in entities:
                # One entity is valid
                quality_assessment["uncertain"].append(rel)
            else:
                # Neither entity is valid
                quality_assessment["poor"].append(rel)
        
        # Update statistics
        self.stats["relationship_quality"]["good"] += len(quality_assessment["good"])
        self.stats["relationship_quality"]["uncertain"] += len(quality_assessment["uncertain"])
        self.stats["relationship_quality"]["poor"] += len(quality_assessment["poor"])
        
        return {
            "relationships": relationships["relationships"],
            "quality_assessment": quality_assessment
        }
    
    def analyze_file(self, filename):
        """Analyze pattern extraction for a single file."""
        logger.info(f"Analyzing pattern extraction for: {filename}")
        
        # Load file content
        content = self.load_text_file(filename)
        
        # Extract paragraphs
        paragraphs = self.extract_paragraphs(content)
        
        file_analysis = {
            "filename": filename,
            "paragraph_count": len(paragraphs),
            "paragraph_analyses": [],
            "summary": {
                "total_entities": 0,
                "valid_entities": 0,
                "invalid_entities": 0,
                "uncertain_entities": 0,
                "relationships": {
                    "total": 0,
                    "good": 0,
                    "uncertain": 0,
                    "poor": 0
                }
            }
        }
        
        # Analyze each paragraph
        for paragraph in paragraphs:
            # Skip very short paragraphs
            if len(paragraph) < 50:
                continue
                
            # Analyze entity extraction
            entity_analysis = self.analyze_entity_extraction(paragraph)
            valid_entities = [e[0] for e in entity_analysis["valid_entities"]]
            
            # Analyze relationships
            relationship_analysis = self.analyze_relationship_extraction(paragraph, valid_entities)
            
            # Return analysis results
            file_analysis["paragraph_analyses"].append({
                "entity_analysis": entity_analysis,
                "relationship_analysis": relationship_analysis,
                "quality_assessment": relationship_analysis.get("quality_assessment", {"good": [], "uncertain": [], "poor": []})
            })
            
            # Update file summary
            file_analysis["summary"]["total_entities"] += len(entity_analysis["entity_analysis"][0]["basic_entities"])
            file_analysis["summary"]["valid_entities"] += len(entity_analysis["entity_analysis"][0]["valid_entities"])
            file_analysis["summary"]["invalid_entities"] += len(entity_analysis["entity_analysis"][0]["invalid_entities"])
            file_analysis["summary"]["uncertain_entities"] += len(entity_analysis["entity_analysis"][0]["uncertain_entities"])
            file_analysis["summary"]["relationships"]["total"] += len(relationship_analysis.get("relationships", []))
            file_analysis["summary"]["relationships"]["good"] += len(relationship_analysis.get("quality_assessment", {}).get("good", []))
            file_analysis["summary"]["relationships"]["uncertain"] += len(relationship_analysis.get("quality_assessment", {}).get("uncertain", []))
            file_analysis["summary"]["relationships"]["poor"] += len(relationship_analysis.get("quality_assessment", {}).get("poor", []))
        
        return file_analysis
    
    def generate_improvement_recommendations(self, analysis_results):
        """Generate recommendations for improving pattern extraction."""
        recommendations = {
            "entity_extraction": [],
            "relationship_extraction": [],
            "general": []
        }
        
        # Entity extraction recommendations
        valid_ratio = self.stats["valid_patterns"] / max(1, self.stats["total_patterns"])
        if valid_ratio < 0.5:
            recommendations["entity_extraction"].append(
                "Improve entity validation by incorporating domain-specific terminology"
            )
        
        # Add recommendation for advanced NLP
        recommendations["entity_extraction"].append(
            "Implement advanced NLP using libraries like spaCy or NLTK for better entity extraction"
        )
        
        # Relationship extraction recommendations
        good_ratio = self.stats["relationship_quality"]["good"] / max(1, sum(self.stats["relationship_quality"].values()))
        if good_ratio < 0.5:
            recommendations["relationship_extraction"].append(
                "Improve relationship extraction by considering syntactic parsing"
            )
        
        if self.stats["relationship_quality"]["uncertain"] > self.stats["relationship_quality"]["good"]:
            recommendations["relationship_extraction"].append(
                "Reduce uncertain relationships by implementing better predicate extraction"
            )
        
        if self.stats["relationship_quality"]["poor"] > 0:
            recommendations["relationship_extraction"].append(
                "Filter out poor relationships by validating context length and relevance"
            )
        
        # General recommendations
        recommendations["general"].append(
            "Implement a feedback loop mechanism to improve entity and relationship extraction over time"
        )
        
        recommendations["general"].append(
            "Create a domain-specific entity recognition model trained on climate risk terminology"
        )
        
        return recommendations
    
    def run(self):
        """Run the pattern extraction analysis."""
        logger.info("Starting Pattern Extraction Analysis")
        
        try:
            # Get all climate risk files
            files = self.get_climate_risk_files()
            logger.info(f"Found {len(files)} climate risk files to analyze")
            
            # Analyze each file
            analysis_results = []
            for filename in files:
                file_analysis = self.analyze_file(filename)
                analysis_results.append(file_analysis)
                logger.info(f"Completed analysis for {filename}")
            
            # Generate improvement recommendations
            recommendations = self.generate_improvement_recommendations(analysis_results)
            
            # Output results
            logger.info("=== Pattern Extraction Analysis Results ===")
            logger.info(f"Total patterns analyzed: {self.stats['total_patterns']}")
            logger.info(f"Valid entities: {self.stats['valid_entities']} ({self.stats['valid_entities']/max(1, self.stats['total_patterns'])*100:.1f}%)")
            logger.info(f"Invalid entities: {self.stats['invalid_entities']} ({self.stats['invalid_entities']/max(1, self.stats['total_patterns'])*100:.1f}%)")
            
            logger.info("\n=== Relationship Quality ===")
            total_rels = sum(self.stats["relationship_quality"].values())
            logger.info(f"Good relationships: {self.stats['relationship_quality']['good']} ({self.stats['relationship_quality']['good']/max(1, total_rels)*100:.1f}%)")
            logger.info(f"Ambiguous relationships: {self.stats['relationship_quality']['ambiguous']} ({self.stats['relationship_quality']['ambiguous']/max(1, total_rels)*100:.1f}%)")
            logger.info(f"Incorrect relationships: {self.stats['relationship_quality']['incorrect']} ({self.stats['relationship_quality']['incorrect']/max(1, total_rels)*100:.1f}%)")
            
            logger.info("\n=== Improvement Recommendations ===")
            logger.info("Entity Extraction:")
            for rec in recommendations["entity_extraction"]:
                logger.info(f"  - {rec}")
            
            logger.info("Relationship Extraction:")
            for rec in recommendations["relationship_extraction"]:
                logger.info(f"  - {rec}")
            
            logger.info("General:")
            for rec in recommendations["general"]:
                logger.info(f"  - {rec}")
            
            # Save detailed analysis to file
            output_dir = Path(__file__).parent / "analysis_results"
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / "pattern_extraction_analysis.json", "w") as f:
                json.dump({
                    "stats": self.stats,
                    "recommendations": recommendations,
                    "file_analysis": analysis_results
                }, f, indent=2)
            
            logger.info(f"Detailed analysis saved to {output_dir / 'pattern_extraction_analysis.json'}")
            
        except Exception as e:
            logger.error(f"Error analyzing pattern extraction: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    debugger = PatternExtractionDebugger()
    debugger.run()
