"""
Test climate risk pattern extraction using Claude API.

This module tests the extraction of patterns from climate risk documents
using the Claude API, with a focus on building a climate lexicon and
introducing NER patterns.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[4]))

from src.habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter
from src.habitat_evolution.infrastructure.adapters.claude_cache import claude_cache


class ClimateRiskPatternExtractor:
    """Extract patterns from climate risk documents using Claude API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the climate risk pattern extractor.
        
        Args:
            api_key: Optional API key for Claude (if None, will look for ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.claude_adapter = ClaudeAdapter(api_key=self.api_key)
        
        # Initialize output directories
        self.output_dir = Path(__file__).parents[4] / "data" / "extracted_patterns"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Climate risk documents directory
        self.climate_risk_dir = Path(__file__).parents[4] / "data" / "climate_risk"
    
    async def extract_patterns_from_document(self, document_path: Path) -> Dict[str, Any]:
        """
        Extract patterns from a climate risk document.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Dictionary containing the extracted patterns
        """
        # Read the document
        with open(document_path, "r") as f:
            content = f.read()
        
        # Create a document object
        document = {
            "id": document_path.stem,
            "title": document_path.stem.replace("_", " ").title(),
            "content": content,
            "metadata": {
                "type": "climate_risk_assessment",
                "source": "local",
                "date": datetime.now().isoformat()
            }
        }
        
        # Process the document with Claude
        result = await self.claude_adapter.process_document(document)
        
        # Save the result
        output_file = self.output_dir / f"{document_path.stem}_patterns.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        return result
    
    async def extract_patterns_from_all_documents(self) -> List[Dict[str, Any]]:
        """
        Extract patterns from all climate risk documents.
        
        Returns:
            List of dictionaries containing the extracted patterns
        """
        results = []
        
        # Get all climate risk documents
        documents = list(self.climate_risk_dir.glob("*.txt"))
        
        print(f"Found {len(documents)} climate risk documents")
        
        # Process each document
        for doc_path in documents:
            print(f"Processing {doc_path.name}...")
            result = await self.extract_patterns_from_document(doc_path)
            results.append(result)
            
            # Print the extracted patterns
            patterns = result.get("patterns", [])
            print(f"Extracted {len(patterns)} patterns from {doc_path.name}")
            for i, pattern in enumerate(patterns):
                print(f"  Pattern {i+1}: {pattern.get('name')}")
        
        return results
    
    async def analyze_climate_lexicon(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the extracted patterns to build a climate lexicon.
        
        Args:
            results: List of pattern extraction results
            
        Returns:
            Dictionary containing the climate lexicon
        """
        # Combine all patterns
        all_patterns = []
        for result in results:
            all_patterns.extend(result.get("patterns", []))
        
        # Extract unique terms from pattern names and descriptions
        terms = set()
        for pattern in all_patterns:
            name = pattern.get("name", "")
            description = pattern.get("description", "")
            
            # Add terms from name
            for term in name.split():
                term = term.strip(",.()[]{}").lower()
                if term and len(term) > 3:  # Filter out short terms
                    terms.add(term)
            
            # Add terms from description
            for term in description.split():
                term = term.strip(",.()[]{}").lower()
                if term and len(term) > 3:  # Filter out short terms
                    terms.add(term)
        
        # Create a query to analyze the climate lexicon
        query = f"""
        Based on the following climate-related terms, identify the key categories
        of climate risk terminology and group these terms accordingly:
        
        {', '.join(sorted(terms))}
        
        Group these terms into meaningful categories related to climate risk assessment.
        """
        
        # Process the query with Claude
        result = await self.claude_adapter.process_query(
            query=query,
            context={"task": "climate_lexicon_analysis"},
            patterns=[]
        )
        
        # Save the lexicon
        lexicon = {
            "terms": sorted(terms),
            "analysis": result.get("response", ""),
            "timestamp": datetime.now().isoformat()
        }
        
        output_file = self.output_dir / "climate_lexicon.json"
        with open(output_file, "w") as f:
            json.dump(lexicon, f, indent=2)
        
        return lexicon
    
    async def extract_ner_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract NER patterns from the climate risk documents.
        
        Args:
            results: List of pattern extraction results
            
        Returns:
            Dictionary containing the NER patterns
        """
        # Combine all patterns
        all_patterns = []
        for result in results:
            all_patterns.extend(result.get("patterns", []))
        
        # Create a query to extract NER patterns
        pattern_texts = []
        for pattern in all_patterns:
            pattern_texts.append(f"- {pattern.get('name')}: {pattern.get('description')}")
        
        query = f"""
        Based on the following climate risk patterns, identify named entity types
        that would be valuable for a climate risk NER system:
        
        {chr(10).join(pattern_texts)}
        
        For each entity type, provide:
        1. A name for the entity type (e.g., CLIMATE_HAZARD)
        2. A description of what this entity type represents
        3. Examples of entities that would fall into this category
        4. Patterns or rules that could help identify this entity type
        """
        
        # Process the query with Claude
        result = await self.claude_adapter.process_query(
            query=query,
            context={"task": "ner_pattern_extraction"},
            patterns=[]
        )
        
        # Save the NER patterns
        ner_patterns = {
            "analysis": result.get("response", ""),
            "timestamp": datetime.now().isoformat()
        }
        
        output_file = self.output_dir / "climate_ner_patterns.json"
        with open(output_file, "w") as f:
            json.dump(ner_patterns, f, indent=2)
        
        return ner_patterns
    
    async def run_extraction_pipeline(self):
        """Run the complete extraction pipeline."""
        # Extract patterns from all documents
        results = await self.extract_patterns_from_all_documents()
        
        # Analyze the climate lexicon
        lexicon = await self.analyze_climate_lexicon(results)
        print(f"\nExtracted {len(lexicon['terms'])} terms for the climate lexicon")
        
        # Extract NER patterns
        ner_patterns = await self.extract_ner_patterns(results)
        print("\nExtracted NER patterns for climate risk documents")
        
        # Print cache statistics
        cache_stats = claude_cache.get_cache_stats()
        print("\nCache Statistics:")
        print(f"  Hits: {cache_stats['hits']}")
        print(f"  Misses: {cache_stats['misses']}")
        print(f"  Hit Rate: {cache_stats['hit_rate']:.2%}")
        print(f"  Cache Entries: {cache_stats['cache_entries']}")
        print(f"  Cache Size: {cache_stats['cache_size_mb']:.2f} MB")


async def main():
    """Main function."""
    extractor = ClimateRiskPatternExtractor()
    await extractor.run_extraction_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
