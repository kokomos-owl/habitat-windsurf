#!/usr/bin/env python
"""
Habitat Evolution Image PKM Demo Wrapper

A simple wrapper script to run the Image-to-PKM demo with the coastal erosion image.
This script provides a more user-friendly interface for demonstrating the Pattern
Knowledge Medium capabilities of Habitat Evolution.
"""

import os
import sys
import argparse
import tempfile
import webbrowser
from pathlib import Path
import json
from datetime import datetime

# Import the demo module
from demo_image_pkm import ImagePKMDemo

def save_html_report(response, image_path, output_path):
    """Generate an HTML report from the demo response."""
    
    # Get the absolute path to the image
    abs_image_path = os.path.abspath(image_path)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Habitat Evolution PKM Demo - {response['image']['filename']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f7fa;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }}
        .image-container {{
            flex: 1;
            min-width: 400px;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .response-container {{
            flex: 2;
            min-width: 500px;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .pattern-container {{
            margin-top: 20px;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .pattern-tag {{
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            margin-right: 10px;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        img {{
            max-width: 100%;
            border-radius: 4px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }}
        .logo {{
            font-weight: bold;
            font-size: 24px;
            color: #2c3e50;
        }}
        .logo span {{
            color: #3498db;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">Habitat <span>Evolution</span> | Pattern Knowledge Medium</div>
        <div class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
    
    <h1>Image Analysis and Pattern Association</h1>
    
    <div class="container">
        <div class="image-container">
            <h2>Source Image</h2>
            <img src="file://{abs_image_path}" alt="{response['image']['filename']}">
            <div style="margin-top: 15px;">
                <strong>Filename:</strong> {response['image']['filename']}<br>
                <strong>Dimensions:</strong> {response['image']['dimensions'].get('width', 'N/A')} x {response['image']['dimensions'].get('height', 'N/A')}
            </div>
        </div>
        
        <div class="response-container">
            <h2>PKM Response</h2>
            <p>{response['response'].replace('\\n', '<br>').replace('\n', '<br>')}</p>
            
            <h3>Context</h3>
            <p>{response['context'] or 'No context provided'}</p>
        </div>
    </div>
    
    <div class="pattern-container">
        <h2>Related Patterns</h2>
        <div>
            {' '.join([f'<span class="pattern-tag">{pattern}</span>' for pattern in response['patterns']['top_patterns']])}
        </div>
        
        <h3>Pattern Relationships</h3>
        <p>The image has been associated with {response['patterns']['count']} patterns in the ArangoDB knowledge base.</p>
        
        <h3>Technical Details</h3>
        <pre>{json.dumps({'image_metadata': response['image'], 'patterns_found': response['patterns']['count']}, indent=2)}</pre>
    </div>
    
    <div style="margin-top: 40px; text-align: center; color: #7f8c8d; font-size: 14px;">
        Habitat Evolution - Pattern Knowledge Medium Demo | © {datetime.now().year}
    </div>
</body>
</html>
"""
    
    # Write the HTML content to the output file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return output_path

def main():
    """Main function to run the demo wrapper."""
    parser = argparse.ArgumentParser(description="Run the Habitat Evolution Image PKM Demo")
    parser.add_argument("--image", help="Path to the image file (default: use the coastal erosion image)",
                       default="")
    parser.add_argument("--context", help="Context description for the image", 
                       default="Coastal erosion with damaged stairs on a beach cliff, showing environmental impact of climate change.")
    parser.add_argument("--db-host", help="ArangoDB host", default="localhost")
    parser.add_argument("--db-port", help="ArangoDB port", type=int, default=8529)
    parser.add_argument("--db-name", help="ArangoDB database name", default="habitat_evolution")
    parser.add_argument("--db-user", help="ArangoDB username", default="root")
    parser.add_argument("--db-password", help="ArangoDB password", default="")
    parser.add_argument("--output-dir", help="Directory to save output files", default="")
    
    args = parser.parse_args()
    
    # Determine the image path
    image_path = args.image
    if not image_path:
        # Check if there's a coastal_erosion.jpg in the current directory
        if os.path.exists("coastal_erosion.jpg"):
            image_path = "coastal_erosion.jpg"
        else:
            print("No image specified and coastal_erosion.jpg not found.")
            print("Please provide an image path with --image")
            sys.exit(1)
    
    # Create output directory if needed
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), "demo_output")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and run the demo
    print(f"Initializing Habitat Evolution Image PKM Demo...")
    demo = ImagePKMDemo(
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password
    )
    
    print(f"Processing image: {image_path}")
    print(f"Context: {args.context}")
    
    # Run the demo
    response = demo.run_demo(image_path, args.context)
    
    # Save JSON output
    json_output_path = os.path.join(output_dir, "demo_response.json")
    with open(json_output_path, 'w') as f:
        json.dump(response, f, indent=2)
    
    # Generate and save HTML report
    html_output_path = os.path.join(output_dir, "demo_report.html")
    html_path = save_html_report(response, image_path, html_output_path)
    
    print("\n=== Demo Completed Successfully ===")
    print(f"JSON output saved to: {json_output_path}")
    print(f"HTML report saved to: {html_output_path}")
    
    # Open the HTML report in the default browser
    print("\nOpening HTML report in browser...")
    webbrowser.open(f"file://{os.path.abspath(html_path)}")
    
    print("\nDemo complete!")

if __name__ == "__main__":
    main()
