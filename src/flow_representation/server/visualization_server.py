"""FastAPI server for serving flow visualizations."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path

app = FastAPI(title="Flow Visualization Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the output directories as static files
examples_output_dir = Path(__file__).parent.parent.parent.parent / "examples" / "output"
app.mount("/visualizations", StaticFiles(directory=str(examples_output_dir)), name="visualizations")

# Mount the visualization output directory
visualization_output_dir = Path(__file__).parent.parent / "visualization" / "output"
app.mount("/data", StaticFiles(directory=str(visualization_output_dir)), name="data")

# Get absolute paths to output directories
base_dir = Path(__file__).resolve().parent.parent
town_metrics_dir = base_dir / "visualization" / "output"
print(f"Town metrics directory: {town_metrics_dir}")
print(f"Directory exists: {town_metrics_dir.exists()}")
print(f"Directory contents: {list(town_metrics_dir.glob('*'))}")

# Serve the static image file
app.mount("/static", StaticFiles(directory=str(town_metrics_dir)), name="static")

# Serve an HTML page that displays the visualization
@app.get("/town-metrics/visualization")
async def get_town_metrics_visualization():
    """Return the interactive town metrics visualization."""
    visualization_path = town_metrics_dir / "climate_risk_visualization.html"
    if not visualization_path.exists():
        raise HTTPException(status_code=404, detail="Visualization not found")
    return FileResponse(visualization_path)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Return visualization index page."""
    visualizations = [
        "coherence",
        "cross_pattern_flow",
        "emergence_rate",
        "geographic"
    ]
    
    special_visualizations = [
        {
            "name": "Town Metrics Overview",
            "url": "/town-metrics/visualization",
            "description": "Interactive visualization of climate risk metrics across Martha's Vineyard towns"
        }
    ]
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flow Visualizations</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                margin-bottom: 20px;
            }
            .visualization-list {
                list-style: none;
                padding: 0;
            }
            .visualization-item {
                margin: 10px 0;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 4px;
                transition: background-color 0.2s;
            }
            .visualization-item:hover {
                background-color: #e9ecef;
            }
            a {
                color: #007bff;
                text-decoration: none;
                font-weight: bold;
            }
            a:hover {
                color: #0056b3;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Flow Visualizations</h1>
            <ul class="visualization-list">
    """
    
    for viz in visualizations:
        html_content += f"""
                <li class="visualization-item">
                    <a href="/visualizations/mv_climate_risk_{viz}.html" target="_blank">
                        {viz.replace('_', ' ').title()} View
                    </a>
                </li>
        """
    
    html_content += """
            </ul>
            
            <h2>Special Visualizations</h2>
            <ul class="visualization-list">
    """
    
    for viz in special_visualizations:
        html_content += f"""
                <li class=\"visualization-item\">
                    <a href=\"{viz['url']}\" target=\"_blank\">
                        {viz['name']}
                    </a>
                    <p style=\"color: #666; margin: 5px 0 0;\">{viz['description']}</p>
                </li>
        """
    
    html_content += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    return html_content

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
