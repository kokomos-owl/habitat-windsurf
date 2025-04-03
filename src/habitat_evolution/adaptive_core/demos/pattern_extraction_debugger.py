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
import spacy
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
        
        # Load spaCy model for NLP analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_available = True
        except:
            logger.warning("spaCy model not available. Using basic NLP techniques.")
            self.spacy_available = False
        
        # Statistics
        self.stats = {
            "total_patterns": 0,
            "valid_entities": 0,
            "invalid_entities": 0,
            "missed_entities": 0,
            "entity_types": Counter(),
            "relationship_quality": {
                "good": 0,
                "ambiguous": 0,
                "incorrect": 0
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
        """Extract entities using spaCy NLP."""
        if not self.spacy_available:
            return []
            
        doc = self.nlp(paragraph)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ["ORG", "GPE", "LOC", "PRODUCT", "EVENT", "FAC", "NORP"]:
                entities.append(ent.text)
        
        # Extract noun phrases that might be domain concepts
        for chunk in doc.noun_chunks:
            # Check if the chunk starts with a capital letter and isn't already in entities
            if chunk.text[0].isupper() and chunk.text not in entities:
                entities.append(chunk.text)
        
        return list(set(entities))
    
    def validate_entity(self, entity):
        """Validate if an entity is likely to be relevant."""
        # Check if it's a known entity
        if entity in self.known_entities:
            return True, "known_entity"
            
        # Check length (very short entities are often not meaningful)
        if len(entity.split()) == 1 and len(entity) < 4:
            return False, "too_short"
            
        # Check for common false positives
        common_false_positives = ["The", "A", "An", "This", "That", "These", "Those", "It", "They", "I", "We", "You"]
        if entity in common_false_positives:
            return False, "common_word"
            
        # Default to uncertain
        return None, "uncertain"
    
    def analyze_entity_extraction(self, paragraph):
        """Analyze entity extraction quality for a paragraph."""
        # Get entities from both methods
        basic_entities = self.extract_entities_basic(paragraph)
        spacy_entities = self.extract_entities_spacy(paragraph) if self.spacy_available else []
        
        # Combine unique entities
        all_entities = list(set(basic_entities + spacy_entities))
        
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
        
        # Calculate metrics
        only_in_basic = [e for e in basic_entities if e not in spacy_entities]
        only_in_spacy = [e for e in spacy_entities if e not in basic_entities]
        in_both = [e for e in basic_entities if e in spacy_entities]
        
        # Update statistics
        self.stats["total_patterns"] += len(all_entities)
        self.stats["valid_entities"] += len(valid_entities)
        self.stats["invalid_entities"] += len(invalid_entities)
        
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
    
    def analyze_relationship_extraction(self, paragraph, entities):
        """Analyze relationship extraction quality."""
        # Original relationship extraction method
        relationships = []
        
        if len(entities) < 2:
            return {"relationships": [], "quality_assessment": {}}
            
        # Extract relationships using original method
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
                            "target": target,
                            "context": matches[0]
                        })
        
        # Analyze relationship quality
        quality_assessment = {
            "good": [],
            "ambiguous": [],
            "incorrect": []
        }
        
        for rel in relationships:
            # Assess relationship quality
            predicate = rel["predicate"]
            context = rel["context"]
            
            # Simple heuristics for quality assessment
            if predicate == "related_to":
                quality_assessment["ambiguous"].append(rel)
            elif len(predicate.split('_')) <= 1:
                quality_assessment["ambiguous"].append(rel)
            elif len(context) > 50:
                quality_assessment["incorrect"].append(rel)
            else:
                quality_assessment["good"].append(rel)
        
        # Update statistics
        self.stats["relationship_quality"]["good"] += len(quality_assessment["good"])
        self.stats["relationship_quality"]["ambiguous"] += len(quality_assessment["ambiguous"])
        self.stats["relationship_quality"]["incorrect"] += len(quality_assessment["incorrect"])
        
        return {
            "relationships": relationships,
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
            "entity_analysis": [],
            "relationship_analysis": [],
            "summary": {
                "total_entities": 0,
                "valid_entities": 0,
                "invalid_entities": 0,
                "uncertain_entities": 0,
                "relationships": {
                    "total": 0,
                    "good": 0,
                    "ambiguous": 0,
                    "incorrect": 0
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
            file_analysis["entity_analysis"].append(entity_analysis)
            
            # Get all entities for relationship analysis
            all_entities = entity_analysis["basic_entities"] + entity_analysis["spacy_entities"]
            all_entities = list(set(all_entities))
            
            # Analyze relationship extraction
            relationship_analysis = self.analyze_relationship_extraction(paragraph, all_entities)
            file_analysis["relationship_analysis"].append(relationship_analysis)
            
            # Update file summary
            file_analysis["summary"]["total_entities"] += len(all_entities)
            file_analysis["summary"]["valid_entities"] += len(entity_analysis["valid_entities"])
            file_analysis["summary"]["invalid_entities"] += len(entity_analysis["invalid_entities"])
            file_analysis["summary"]["uncertain_entities"] += len(entity_analysis["uncertain_entities"])
            file_analysis["summary"]["relationships"]["total"] += len(relationship_analysis["relationships"])
            file_analysis["summary"]["relationships"]["good"] += len(relationship_analysis["quality_assessment"]["good"])
            file_analysis["summary"]["relationships"]["ambiguous"] += len(relationship_analysis["quality_assessment"]["ambiguous"])
            file_analysis["summary"]["relationships"]["incorrect"] += len(relationship_analysis["quality_assessment"]["incorrect"])
        
        return file_analysis
    
    def generate_improvement_recommendations(self, analysis_results):
        """Generate recommendations for improving pattern extraction."""
        recommendations = {
            "entity_extraction": [],
            "relationship_extraction": [],
            "general": []
        }
        
        # Entity extraction recommendations
        valid_ratio = self.stats["valid_entities"] / max(1, self.stats["total_patterns"])
        if valid_ratio < 0.5:
            recommendations["entity_extraction"].append(
                "Improve entity validation by incorporating domain-specific terminology"
            )
        
        if self.spacy_available and len(analysis_results) > 0:
            spacy_only_count = sum(len(file["entity_analysis"][0]["only_in_spacy"]) for file in analysis_results)
            basic_only_count = sum(len(file["entity_analysis"][0]["only_in_basic"]) for file in analysis_results)
            
            if spacy_only_count > basic_only_count:
                recommendations["entity_extraction"].append(
                    "Prioritize spaCy-based entity extraction as it identifies more valid entities"
                )
            else:
                recommendations["entity_extraction"].append(
                    "Combine regex and spaCy approaches for better entity coverage"
                )
        
        # Relationship extraction recommendations
        good_ratio = self.stats["relationship_quality"]["good"] / max(1, sum(self.stats["relationship_quality"].values()))
        if good_ratio < 0.3:
            recommendations["relationship_extraction"].append(
                "Improve relationship predicate extraction with more sophisticated NLP techniques"
            )
        
        if self.stats["relationship_quality"]["ambiguous"] > self.stats["relationship_quality"]["good"]:
            recommendations["relationship_extraction"].append(
                "Reduce ambiguous 'related_to' relationships by implementing dependency parsing"
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
