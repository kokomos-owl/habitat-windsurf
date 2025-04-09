"""
Real Climate Data Loader for Habitat Evolution.

This module provides specialized loaders for real climate time series data
and climate risk documents, with specific handling for the JSON temperature
datasets and text-based climate risk documents.
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class RealClimateDataLoader:
    """
    Loader for real climate data in JSON format and climate risk documents.
    
    This class provides methods for loading real climate data from JSON files
    and climate risk documents, converting them to formats suitable for
    pattern detection and correlation analysis.
    """
    
    def __init__(self, 
                climate_data_dir: str = None,
                climate_risk_dir: str = None):
        """
        Initialize the real climate data loader.
        
        Args:
            climate_data_dir: Directory containing climate data JSON files
            climate_risk_dir: Directory containing climate risk documents
        """
        # Set default directories if not provided
        self.climate_data_dir = climate_data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))),
            "docs", "untitled folder"
        )
        
        self.climate_risk_dir = climate_risk_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))),
            "data", "climate_risk"
        )
        
        logger.info(f"Climate data directory: {self.climate_data_dir}")
        logger.info(f"Climate risk directory: {self.climate_risk_dir}")
    
    def load_temperature_data(self, region: str) -> pd.DataFrame:
        """
        Load temperature data for a specific region from JSON files.
        
        Args:
            region: Region to load data for (e.g., "MA" for Massachusetts,
                  "NE" for Northeast)
            
        Returns:
            DataFrame with temperature data
        """
        # Map region to filename pattern
        region_map = {
            "massachusetts": "MA_AvgTemp",
            "ma": "MA_AvgTemp",
            "northeast": "NE_AvgTemp",
            "ne": "NE_AvgTemp"
        }
        
        region_key = region.lower()
        if region_key not in region_map:
            raise ValueError(f"Unsupported region: {region}. Supported regions: {list(region_map.keys())}")
        
        file_pattern = region_map[region_key]
        
        # Find matching files
        matching_files = []
        for filename in os.listdir(self.climate_data_dir):
            if file_pattern in filename and filename.endswith(".json"):
                matching_files.append(os.path.join(self.climate_data_dir, filename))
        
        if not matching_files:
            raise FileNotFoundError(f"No temperature data files found for region: {region}")
        
        # Use the most recent file if multiple matches
        data_file = sorted(matching_files)[-1]
        logger.info(f"Loading temperature data from: {data_file}")
        
        # Load JSON data
        with open(data_file, 'r') as f:
            json_data = json.load(f)
        
        # Extract metadata
        description = json_data.get("description", {})
        title = description.get("title", "")
        units = description.get("units", "")
        base_period = description.get("base_period", "")
        
        # Extract time series data
        data_points = json_data.get("data", {})
        
        # Convert to DataFrame
        records = []
        for date_str, values in data_points.items():
            year = int(date_str[:4])
            month = int(date_str[4:6])
            date = datetime(year, month, 1)
            
            record = {
                "date": date,
                "temperature": values.get("value"),
                "anomaly": values.get("anomaly"),
                "region": region
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Add metadata as attributes
        df.attrs["title"] = title
        df.attrs["units"] = units
        df.attrs["base_period"] = base_period
        
        return df
    
    def load_climate_risk_documents(self, region: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load climate risk documents, optionally filtered by region.
        
        Args:
            region: Optional region to filter documents by
            
        Returns:
            List of document dictionaries with text and metadata
        """
        documents = []
        
        # Get all text files in the climate risk directory
        for filename in os.listdir(self.climate_risk_dir):
            if not filename.endswith(".txt"):
                continue
            
            file_path = os.path.join(self.climate_risk_dir, filename)
            
            # Read document content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract document metadata
            doc_region = self._extract_region_from_document(content, filename)
            doc_date = self._extract_date_from_document(content)
            doc_title = self._extract_title_from_document(content, filename)
            
            # Filter by region if specified
            if region and region.lower() not in doc_region.lower():
                continue
            
            # Create document dictionary
            document = {
                "id": os.path.splitext(filename)[0],
                "title": doc_title,
                "content": content,
                "region": doc_region,
                "date": doc_date,
                "source": "climate_risk_document",
                "file_path": file_path
            }
            
            documents.append(document)
        
        logger.info(f"Loaded {len(documents)} climate risk documents")
        return documents
    
    def _extract_region_from_document(self, content: str, filename: str) -> str:
        """
        Extract region information from document content or filename.
        
        Args:
            content: Document content
            filename: Document filename
            
        Returns:
            Region string
        """
        # Try to extract from content
        region_match = re.search(r"Region:\s*([^,\n]+)", content)
        if region_match:
            return region_match.group(1).strip()
        
        # Try to extract from first few lines
        first_lines = content.split("\n")[:5]
        for line in first_lines:
            if "cape cod" in line.lower():
                return "Cape Cod, Massachusetts"
            if "massachusetts" in line.lower() or "ma" in line.lower():
                return "Massachusetts"
            if "northeast" in line.lower() or "new england" in line.lower():
                return "Northeast"
        
        # Extract from filename
        if "cape_code" in filename.lower() or "cape_cod" in filename.lower():
            return "Cape Cod, Massachusetts"
        if "marthas_vineyard" in filename.lower() or "vineyard" in filename.lower():
            return "Martha's Vineyard, Massachusetts"
        if "boston" in filename.lower():
            return "Boston, Massachusetts"
        if "nantucket" in filename.lower():
            return "Nantucket, Massachusetts"
        if "plum_island" in filename.lower():
            return "Plum Island, Massachusetts"
        
        # Default
        return "Massachusetts"
    
    def _extract_date_from_document(self, content: str) -> str:
        """
        Extract date information from document content.
        
        Args:
            content: Document content
            
        Returns:
            Date string in YYYY-MM-DD format
        """
        # Try to extract creation date
        date_match = re.search(r"Creation Date:\s*(\d{4}-\d{2}-\d{2})", content)
        if date_match:
            return date_match.group(1)
        
        date_match = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", content)
        if date_match:
            return date_match.group(1)
        
        # Default to current date
        return datetime.now().strftime("%Y-%m-%d")
    
    def _extract_title_from_document(self, content: str, filename: str) -> str:
        """
        Extract title from document content or filename.
        
        Args:
            content: Document content
            filename: Document filename
            
        Returns:
            Document title
        """
        # Try to get first non-empty line as title
        lines = content.split("\n")
        for line in lines:
            if line.strip():
                return line.strip()
        
        # Fall back to filename
        return os.path.splitext(filename)[0].replace("_", " ").title()
    
    def extract_sentences_from_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract sentences from a document for semantic pattern detection.
        
        Args:
            document: Document dictionary
            
        Returns:
            List of sentence dictionaries with text and metadata
        """
        content = document["content"]
        
        # Split into sentences (simple approach)
        # In a production system, this would use NLP for better sentence segmentation
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        # Filter and process sentences
        processed_sentences = []
        for i, sentence in enumerate(sentences):
            # Skip short sentences or headings
            if len(sentence.strip()) < 20 or sentence.strip().isupper():
                continue
            
            # Extract temporal markers
            temporal_markers = self._extract_temporal_markers(sentence)
            
            # Create sentence dictionary
            sentence_dict = {
                "id": f"{document['id']}_sent_{i}",
                "text": sentence.strip(),
                "document_id": document["id"],
                "region": document["region"],
                "temporal_markers": temporal_markers,
                "source": document["source"]
            }
            
            processed_sentences.append(sentence_dict)
        
        return processed_sentences
    
    def _extract_temporal_markers(self, text: str) -> List[Dict[str, str]]:
        """
        Extract temporal markers from text.
        
        Args:
            text: Text to extract markers from
            
        Returns:
            List of temporal markers
        """
        markers = []
        text_lower = text.lower()
        
        # Check for year mentions
        for year in range(1990, 2025):
            year_str = str(year)
            if year_str in text_lower:
                markers.append({"time": f"{year}01", "text": year_str})
        
        # Check for decade mentions
        if "1990s" in text_lower or "nineties" in text_lower:
            markers.append({"time": "199001", "text": "1990s"})
        if "2000s" in text_lower:
            markers.append({"time": "200001", "text": "2000s"})
        if "2010s" in text_lower:
            markers.append({"time": "201001", "text": "2010s"})
        if "2020s" in text_lower:
            markers.append({"time": "202001", "text": "2020s"})
            
        # Check for specific time periods
        if "mid-century" in text_lower or "mid century" in text_lower:
            markers.append({"time": "205001", "text": "mid-century"})
        if "late-century" in text_lower or "late century" in text_lower:
            markers.append({"time": "208001", "text": "late-century"})
        if "by 2050" in text_lower:
            markers.append({"time": "205001", "text": "by 2050"})
        if "by 2030" in text_lower:
            markers.append({"time": "203001", "text": "by 2030"})
        if "present" in text_lower or "current" in text_lower:
            markers.append({"time": "202401", "text": "present"})
        
        return markers


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create loader
    loader = RealClimateDataLoader()
    
    # Load Massachusetts temperature data
    try:
        ma_data = loader.load_temperature_data("MA")
        print(f"Loaded {len(ma_data)} temperature records for Massachusetts")
        print(f"Temperature range: {ma_data['temperature'].min():.1f} to {ma_data['temperature'].max():.1f} {ma_data.attrs.get('units', '')}")
        print(f"Date range: {ma_data['date'].min()} to {ma_data['date'].max()}")
    except Exception as e:
        print(f"Error loading Massachusetts data: {e}")
    
    # Load climate risk documents
    documents = loader.load_climate_risk_documents()
    print(f"Loaded {len(documents)} climate risk documents")
    
    # Extract sentences from first document
    if documents:
        sentences = loader.extract_sentences_from_document(documents[0])
        print(f"Extracted {len(sentences)} sentences from document: {documents[0]['title']}")
        
        # Print first few sentences
        for i, sentence in enumerate(sentences[:3]):
            print(f"Sentence {i+1}: {sentence['text'][:100]}...")
            if sentence['temporal_markers']:
                print(f"  Temporal markers: {sentence['temporal_markers']}")
