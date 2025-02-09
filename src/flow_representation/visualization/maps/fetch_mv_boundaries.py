"""Script to fetch Martha's Vineyard town boundaries from ArcGIS REST API."""

import requests
import json
from pathlib import Path

def fetch_mv_boundaries():
    """Fetch Martha's Vineyard town boundaries from ArcGIS REST API."""
    # ArcGIS REST endpoint for MV town boundaries
    url = "https://services1.arcgis.com/FNsEJ848HT5uDOHD/ArcGIS/rest/services/MV_Interior_Boundary/FeatureServer/0/query"
    
    # Query parameters
    params = {
        'where': '1=1',  # Get all features
        'outFields': '*',  # Get all fields
        'geometryPrecision': 6,
        'outSR': '4326',  # WGS84 coordinate system
        'f': 'geojson'  # Request GeoJSON format
    }
    
    print("Fetching Martha's Vineyard boundaries...")
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        geojson_data = response.json()
        
        # Create data directory if it doesn't exist
        data_dir = Path(__file__).parent / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save GeoJSON file
        output_file = data_dir / 'marthas_vineyard.geojson'
        with open(output_file, 'w') as f:
            json.dump(geojson_data, f, indent=2)
        
        print(f"Saved GeoJSON to {output_file}")
        return geojson_data
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

if __name__ == "__main__":
    fetch_mv_boundaries()
