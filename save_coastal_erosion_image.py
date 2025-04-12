#!/usr/bin/env python
"""
Simple script to save the coastal erosion image from the conversation to a file.
This is needed to run the demo with the image shown in the conversation.
"""

import os
import sys
import requests
from pathlib import Path

def save_image():
    """Save the coastal erosion image to the current directory."""
    # URL of the coastal erosion image
    image_url = "https://i.imgur.com/MQwbRYE.jpg"
    
    # Path to save the image
    output_path = Path(os.getcwd()) / "coastal_erosion.jpg"
    
    print(f"Downloading coastal erosion image from {image_url}")
    
    try:
        # Download the image
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        # Save the image
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Image saved to {output_path}")
        return str(output_path)
    
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

if __name__ == "__main__":
    save_image()
