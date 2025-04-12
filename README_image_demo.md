# Habitat Evolution Image PKM Demo

This demo showcases the Pattern Knowledge Medium (PKM) capabilities of Habitat Evolution by associating images with patterns stored in ArangoDB and generating contextual responses.

## Overview

The demo implements a complete pipeline that:

1. Processes an image to extract features (dimensions, color distribution, edge detection)
2. Associates the image with patterns in ArangoDB using the field-pattern bridge
3. Generates contextual responses based on the image and related patterns
4. Persists the demo results in ArangoDB for future reference
5. Creates an interactive HTML report with the results

## Prerequisites

- Python 3.11+
- ArangoDB running locally or remotely
- OpenCV and Pillow for image processing (optional but recommended)
- Habitat Evolution system with initialized components

## Installation

Ensure you have the required Python packages:

```bash
pip install opencv-python pillow
```

## Running the Demo

The simplest way to run the demo is:

```bash
python run_image_demo.py
```

This will:
- Use the coastal erosion image if available in the current directory
- Connect to a local ArangoDB instance
- Generate a response based on patterns in the database
- Create an HTML report and open it in your browser

### Command-line Options

```
python run_image_demo.py --help
```

Options:
- `--image`: Path to the image file (default: uses coastal_erosion.jpg)
- `--context`: Context description for the image
- `--db-host`: ArangoDB host (default: localhost)
- `--db-port`: ArangoDB port (default: 8529)
- `--db-name`: ArangoDB database name (default: habitat_evolution)
- `--db-user`: ArangoDB username (default: root)
- `--db-password`: ArangoDB password
- `--output-dir`: Directory to save output files (default: demo_output in current directory)

## Example

```bash
python run_image_demo.py --image path/to/coastal_erosion.jpg --context "Coastal erosion showing climate change impact"
```

## Understanding the Results

The demo produces:

1. **JSON Response**: Contains the structured data including image metadata, related patterns, and generated response
2. **HTML Report**: An interactive visualization of the results with the image and related patterns

## How It Works

The demo leverages key Habitat Evolution components:

- **EventService**: For event-based communication
- **ArangoDBConnection**: For pattern storage and retrieval
- **PatternAwareRAGService**: For generating contextual responses
- **PKMFactory**: For component initialization
- **FieldPatternBridge**: For creating field representations

## Extending the Demo

You can extend this demo by:

1. Adding more sophisticated image processing techniques
2. Implementing custom pattern detection algorithms
3. Enhancing the HTML report with interactive visualizations
4. Adding support for video or other media types

## Troubleshooting

- **Missing Image Libraries**: The demo will fall back to basic metadata if OpenCV and Pillow are not installed
- **Database Connection Issues**: Ensure ArangoDB is running and accessible
- **Missing Patterns**: The quality of responses depends on having relevant patterns in your ArangoDB instance

---

This demo serves as a practical example of how Habitat Evolution can function as a Pattern Knowledge Medium that bridges visual data with semantic patterns, showcasing the system's ability to work with multimodal data.
