#!/usr/bin/env python
"""
Habitat Evolution Image-to-PKM Demo

This script demonstrates the Pattern Knowledge Medium capabilities of Habitat Evolution
by associating an image with patterns stored in ArangoDB and generating a response.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Habitat Evolution components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.habitat_evolution.infrastructure.services.event_service import EventService
from src.habitat_evolution.infrastructure.db.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.infrastructure.services.pattern_aware_rag_service import PatternAwareRAGService
from src.habitat_evolution.pkm.pkm_factory import PKMFactory
from src.habitat_evolution.climate_risk.field_pattern_bridge import FieldPatternBridge

# Optional: Import image processing libraries if available
try:
    import cv2
    import numpy as np
    from PIL import Image
    HAS_IMAGE_LIBS = True
except ImportError:
    logger.warning("Image processing libraries not found. Using simplified image handling.")
    HAS_IMAGE_LIBS = False

class ImagePKMDemo:
    """Demo class for associating images with the Pattern Knowledge Medium."""
    
    def __init__(self, db_host: str = "localhost", db_port: int = 8529, 
                 db_name: str = "habitat_evolution", db_user: str = "root", 
                 db_password: str = ""):
        """Initialize the demo with database connection parameters."""
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "database": db_name,
            "username": db_user,
            "password": db_password
        }
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all required Habitat Evolution components."""
        logger.info("Initializing Habitat Evolution components...")
        
        # Initialize EventService (global singleton)
        self.event_service = EventService()
        if not self.event_service._initialized:
            self.event_service.initialize()
            os.environ["EVENT_SERVICE_INITIALIZED"] = "True"
        
        # Initialize ArangoDB connection
        self.db_connection = ArangoDBConnection(**self.db_config)
        self.db_connection.connect()
        
        # Initialize PKM Factory
        self.pkm_factory = PKMFactory(self.db_connection, self.event_service)
        
        # Initialize Field-Pattern Bridge
        self.field_pattern_bridge = FieldPatternBridge(self.event_service)
        
        # Initialize Pattern-Aware RAG Service
        self.pattern_rag = self.pkm_factory.create_pattern_aware_rag()
        
        logger.info("Components initialized successfully")
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image and extract basic features.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary of extracted features
        """
        logger.info(f"Processing image: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Extract basic image metadata
        image_metadata = {
            "filename": os.path.basename(image_path),
            "path": image_path,
            "size": os.path.getsize(image_path),
            "last_modified": os.path.getmtime(image_path)
        }
        
        # Extract image features if libraries are available
        if HAS_IMAGE_LIBS:
            try:
                # Read image
                img = cv2.imread(image_path)
                
                # Extract basic features
                height, width, channels = img.shape
                
                # Calculate color histogram
                hist_b = cv2.calcHist([img], [0], None, [8], [0, 256])
                hist_g = cv2.calcHist([img], [1], None, [8], [0, 256])
                hist_r = cv2.calcHist([img], [2], None, [8], [0, 256])
                
                # Detect edges (simplified feature extraction)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_percentage = np.count_nonzero(edges) / (height * width)
                
                # Add features to metadata
                image_metadata.update({
                    "dimensions": {
                        "width": width,
                        "height": height,
                        "channels": channels
                    },
                    "features": {
                        "color_distribution": {
                            "blue": hist_b.flatten().tolist(),
                            "green": hist_g.flatten().tolist(),
                            "red": hist_r.flatten().tolist()
                        },
                        "edge_percentage": edge_percentage
                    }
                })
                
                # Determine dominant colors (simplified)
                pixels = np.float32(img.reshape(-1, 3))
                n_colors = 5
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
                flags = cv2.KMEANS_RANDOM_CENTERS
                _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
                _, counts = np.unique(labels, return_counts=True)
                
                # Convert colors to hex for easier interpretation
                dominant_colors = []
                for color in palette:
                    dominant_colors.append('#{:02x}{:02x}{:02x}'.format(
                        int(color[2]), int(color[1]), int(color[0])))
                
                image_metadata["features"]["dominant_colors"] = dominant_colors
                
            except Exception as e:
                logger.warning(f"Error extracting image features: {e}")
                logger.warning("Using simplified image metadata only")
        
        logger.info("Image processing complete")
        return image_metadata
    
    def associate_with_patterns(self, image_metadata: Dict[str, Any], 
                               context_description: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Associate image metadata with patterns in ArangoDB.
        
        Args:
            image_metadata: Dictionary of image features
            context_description: Optional description to provide context
            
        Returns:
            List of related patterns
        """
        logger.info("Associating image with patterns in ArangoDB...")
        
        # Create a query context from the image metadata and description
        query_context = {
            "image_metadata": image_metadata,
            "description": context_description or "",
            "query_type": "image_association"
        }
        
        # Use the field-pattern bridge to create a field representation
        field_representation = self.field_pattern_bridge.create_field_from_data(query_context)
        
        # Use the pattern-aware RAG to find related patterns
        related_patterns = self.pattern_rag.retrieve_patterns(
            query=json.dumps(query_context),
            field_representation=field_representation,
            max_results=10
        )
        
        logger.info(f"Found {len(related_patterns)} related patterns")
        return related_patterns
    
    def generate_response(self, image_metadata: Dict[str, Any], 
                         related_patterns: List[Dict[str, Any]],
                         context_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response based on the image and related patterns.
        
        Args:
            image_metadata: Dictionary of image features
            related_patterns: List of related patterns from ArangoDB
            context_description: Optional description to provide context
            
        Returns:
            Response dictionary
        """
        logger.info("Generating response...")
        
        # Create a query for the pattern-aware RAG
        query = {
            "image_metadata": image_metadata,
            "description": context_description or "",
            "related_patterns": related_patterns
        }
        
        # Generate a response using the pattern-aware RAG
        response = self.pattern_rag.generate_response(
            query=json.dumps(query),
            patterns=related_patterns
        )
        
        # Create a structured response
        structured_response = {
            "image": {
                "filename": image_metadata.get("filename", ""),
                "dimensions": image_metadata.get("dimensions", {})
            },
            "patterns": {
                "count": len(related_patterns),
                "top_patterns": [p.get("name", "Unnamed Pattern") for p in related_patterns[:3]]
            },
            "response": response,
            "context": context_description
        }
        
        logger.info("Response generation complete")
        return structured_response
    
    def run_demo(self, image_path: str, context_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete demo pipeline.
        
        Args:
            image_path: Path to the image file
            context_description: Optional description to provide context
            
        Returns:
            Complete response dictionary
        """
        logger.info(f"Running Image PKM Demo with image: {image_path}")
        
        # Process the image
        image_metadata = self.process_image(image_path)
        
        # Associate with patterns
        related_patterns = self.associate_with_patterns(image_metadata, context_description)
        
        # Generate response
        response = self.generate_response(image_metadata, related_patterns, context_description)
        
        # Persist the result in ArangoDB for future reference
        self._persist_demo_result(image_metadata, related_patterns, response)
        
        return response
    
    def _persist_demo_result(self, image_metadata: Dict[str, Any], 
                           related_patterns: List[Dict[str, Any]],
                           response: Dict[str, Any]):
        """Persist the demo result in ArangoDB for future reference."""
        logger.info("Persisting demo result in ArangoDB...")
        
        # Create a document to store in ArangoDB
        demo_result = {
            "type": "image_pkm_demo",
            "timestamp": os.path.getmtime(image_metadata.get("path", "")),
            "image_metadata": {
                "filename": image_metadata.get("filename", ""),
                "size": image_metadata.get("size", 0)
            },
            "related_pattern_ids": [p.get("id") for p in related_patterns if "id" in p],
            "response_summary": response.get("response", "")[:500]  # Store a summary
        }
        
        # Store in ArangoDB
        try:
            collection = self.db_connection.db.collection("demo_results")
            if not collection:
                collection = self.db_connection.db.create_collection("demo_results")
            
            result = collection.insert(demo_result)
            logger.info(f"Demo result persisted with key: {result['_key']}")
        except Exception as e:
            logger.error(f"Error persisting demo result: {e}")


def main():
    """Main function to run the demo from command line."""
    parser = argparse.ArgumentParser(description="Habitat Evolution Image-to-PKM Demo")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--context", help="Optional context description", default="")
    parser.add_argument("--db-host", help="ArangoDB host", default="localhost")
    parser.add_argument("--db-port", help="ArangoDB port", type=int, default=8529)
    parser.add_argument("--db-name", help="ArangoDB database name", default="habitat_evolution")
    parser.add_argument("--db-user", help="ArangoDB username", default="root")
    parser.add_argument("--db-password", help="ArangoDB password", default="")
    parser.add_argument("--output", help="Output file path (JSON)", default="")
    
    args = parser.parse_args()
    
    # Create and run the demo
    demo = ImagePKMDemo(
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password
    )
    
    # Run the demo
    response = demo.run_demo(args.image_path, args.context)
    
    # Print the response
    print("\n=== Image PKM Demo Response ===")
    print(f"Image: {response['image']['filename']}")
    print(f"Related Patterns: {', '.join(response['patterns']['top_patterns'])}")
    print("\nResponse:")
    print(response['response'])
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(response, f, indent=2)
        print(f"\nResponse saved to {args.output}")


if __name__ == "__main__":
    main()
