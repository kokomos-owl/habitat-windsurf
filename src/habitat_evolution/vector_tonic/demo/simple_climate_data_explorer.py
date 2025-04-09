"""
Simple Climate Data Explorer

This script loads and displays information about the real climate data
without complex visualizations or event processing.
"""

import json
import os
import logging
from typing import Dict, Any
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_json_climate_data(file_path: str) -> Dict[str, Any]:
    """
    Load climate data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary with climate data
    """
    logger.info(f"Loading climate data from: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data


def analyze_climate_data(data: Dict[str, Any]) -> None:
    """
    Analyze and print information about climate data.
    
    Args:
        data: Dictionary with climate data
    """
    # Extract metadata
    description = data.get("description", {})
    title = description.get("title", "Unknown")
    units = description.get("units", "Unknown")
    base_period = description.get("base_period", "Unknown")
    
    logger.info(f"Climate Data: {title}")
    logger.info(f"Units: {units}")
    logger.info(f"Base Period: {base_period}")
    
    # Extract time series data
    data_points = data.get("data", {})
    logger.info(f"Number of data points: {len(data_points)}")
    
    # Convert to DataFrame for analysis
    records = []
    for date_str, values in data_points.items():
        year = int(date_str[:4])
        month = int(date_str[4:6])
        date = datetime(year, month, 1)
        
        record = {
            "date": date,
            "value": values.get("value"),
            "anomaly": values.get("anomaly")
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Calculate statistics
    min_value = df["value"].min()
    max_value = df["value"].max()
    mean_value = df["value"].mean()
    min_anomaly = df["anomaly"].min()
    max_anomaly = df["anomaly"].max()
    mean_anomaly = df["anomaly"].mean()
    
    logger.info(f"Temperature range: {min_value:.2f} to {max_value:.2f} {units}")
    logger.info(f"Mean temperature: {mean_value:.2f} {units}")
    logger.info(f"Anomaly range: {min_anomaly:.2f} to {max_anomaly:.2f} {units}")
    logger.info(f"Mean anomaly: {mean_anomaly:.2f} {units}")
    
    # Find years with highest and lowest values
    max_year = df.loc[df["value"].idxmax(), "date"].year
    min_year = df.loc[df["value"].idxmin(), "date"].year
    
    logger.info(f"Highest temperature: {max_value:.2f} {units} in {max_year}")
    logger.info(f"Lowest temperature: {min_value:.2f} {units} in {min_year}")
    
    # Find years with highest and lowest anomalies
    max_anomaly_year = df.loc[df["anomaly"].idxmax(), "date"].year
    min_anomaly_year = df.loc[df["anomaly"].idxmin(), "date"].year
    
    logger.info(f"Highest anomaly: {max_anomaly:.2f} {units} in {max_anomaly_year}")
    logger.info(f"Lowest anomaly: {min_anomaly:.2f} {units} in {min_anomaly_year}")
    
    # Check for warming trend
    first_decade = df[df["date"].dt.year < 2000]["value"].mean()
    last_decade = df[df["date"].dt.year >= 2015]["value"].mean()
    
    if last_decade > first_decade:
        logger.info(f"Warming trend detected: {last_decade - first_decade:.2f} {units} increase")
    else:
        logger.info(f"No warming trend detected")


def read_climate_risk_document(file_path: str) -> None:
    """
    Read and display information about a climate risk document.
    
    Args:
        file_path: Path to the document file
    """
    logger.info(f"Reading climate risk document: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Print basic information
    lines = content.split('\n')
    title = lines[0] if lines else "Unknown"
    
    logger.info(f"Document: {title}")
    logger.info(f"Length: {len(content)} characters, {len(lines)} lines")
    
    # Extract key sections
    sections = {}
    current_section = "Header"
    section_content = []
    
    for line in lines:
        if line.strip() and not line[0].isspace() and line.strip().endswith(':'):
            # This is a section header
            sections[current_section] = '\n'.join(section_content)
            current_section = line.strip()
            section_content = []
        else:
            section_content.append(line)
    
    # Add the last section
    sections[current_section] = '\n'.join(section_content)
    
    # Print sections
    logger.info(f"Document sections: {len(sections)}")
    for section, _ in sections.items():
        if section != "Header":
            logger.info(f"  - {section}")
    
    # Look for climate-related keywords
    keywords = ["sea level rise", "flooding", "erosion", "temperature", 
                "warming", "climate change", "extreme weather", "precipitation"]
    
    keyword_counts = {}
    for keyword in keywords:
        count = content.lower().count(keyword.lower())
        if count > 0:
            keyword_counts[keyword] = count
    
    logger.info("Climate-related keywords found:")
    for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  - {keyword}: {count} occurrences")


def main():
    """Main function to explore climate data and risk documents."""
    # Define paths to data files
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    
    ma_temp_path = os.path.join(base_dir, "docs", "untitled folder", "MA_AvgTemp_91_24.json")
    ne_temp_path = os.path.join(base_dir, "docs", "untitled folder", "NE_AvgTemp_91_24.json")
    
    climate_risk_dir = os.path.join(base_dir, "data", "climate_risk")
    
    # Analyze Massachusetts temperature data
    logger.info("\n" + "="*50)
    logger.info("MASSACHUSETTS TEMPERATURE DATA ANALYSIS")
    logger.info("="*50)
    
    try:
        ma_data = load_json_climate_data(ma_temp_path)
        analyze_climate_data(ma_data)
    except Exception as e:
        logger.error(f"Error analyzing Massachusetts data: {e}")
    
    # Analyze Northeast temperature data
    logger.info("\n" + "="*50)
    logger.info("NORTHEAST TEMPERATURE DATA ANALYSIS")
    logger.info("="*50)
    
    try:
        ne_data = load_json_climate_data(ne_temp_path)
        analyze_climate_data(ne_data)
    except Exception as e:
        logger.error(f"Error analyzing Northeast data: {e}")
    
    # Analyze climate risk documents
    logger.info("\n" + "="*50)
    logger.info("CLIMATE RISK DOCUMENT ANALYSIS")
    logger.info("="*50)
    
    try:
        # Get all text files in the climate risk directory
        doc_files = [os.path.join(climate_risk_dir, f) for f in os.listdir(climate_risk_dir) 
                    if f.endswith(".txt")]
        
        logger.info(f"Found {len(doc_files)} climate risk documents")
        
        # Analyze first document as an example
        if doc_files:
            read_climate_risk_document(doc_files[0])
    except Exception as e:
        logger.error(f"Error analyzing climate risk documents: {e}")
    
    logger.info("\n" + "="*50)
    logger.info("CLIMATE DATA EXPLORATION COMPLETE")
    logger.info("="*50)


if __name__ == "__main__":
    main()
